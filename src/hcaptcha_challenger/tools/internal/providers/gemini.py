# -*- coding: utf-8 -*-
"""
GeminiProvider - Google Gemini API implementation.

Wraps the google-genai SDK for image-based content generation, with built-in
resilience for free-tier use:

- **Multiple API keys.** ``api_key`` may be one key, a comma-separated string, or
  a list. Requests spread round-robin across keys; a key that returns 429 /
  RESOURCE_EXHAUSTED is rotated out for the next.
- **Model fallback chain.** ``model`` may be a single id or a comma-separated,
  accuracy-ordered chain (e.g. ``gemini-3.5-flash,gemini-3-flash,...``). When a
  model is rate-limited on every key, or is unavailable on the key/tier, the
  provider drops to the next model in the chain.

Images are inlined as bytes (not uploaded via the Files API) so a request works
with any key without re-uploading on failover.
"""
import asyncio
import json
import mimetypes
from pathlib import Path
from typing import List, Type, TypeVar, cast

from google import genai
from google.genai import types
from loguru import logger
from pydantic import BaseModel

from hcaptcha_challenger.models import THINKING_LEVEL_MODELS

ResponseT = TypeVar("ResponseT", bound=BaseModel)

# How many times to retry a single (model, key) call on a transient error
# (5xx / network / deadline) before giving up on that combination.
_TRANSIENT_RETRIES = 3
_TRANSIENT_WAIT_S = 3.0


def extract_first_json_block(text: str) -> dict | None:
    """Extract the first JSON code block from text."""
    import re

    pattern = r"```json\s*([\s\S]*?)```"
    matches = re.findall(pattern, text)
    if matches:
        return json.loads(matches[0])
    return None


def _normalize_list(value) -> List[str]:
    """Normalize a str / comma-separated str / list into a clean list."""
    raw: List[str] = []
    if isinstance(value, str):
        raw = value.split(",")
    elif isinstance(value, (list, tuple)):
        for item in value:
            raw.extend(str(item).split(","))
    return [v.strip() for v in raw if v and v.strip()]


class GeminiProvider:
    """Gemini-based chat provider with key rotation and model fallback."""

    def __init__(self, api_key, model, *, base_url: str | None = None):
        """
        Args:
            api_key: One key, a comma-separated string, or a list of keys.
            model: A model id or a comma-separated, accuracy-ordered fallback chain.
            base_url: Optional custom endpoint (proxy/gateway).
        """
        self._keys = _normalize_list(api_key)
        if not self._keys:
            raise ValueError("At least one Gemini API key is required.")
        self._models = _normalize_list(model) or [str(model)]
        self._base_url = base_url or None
        self._key_idx = -1
        self._clients: dict[str, genai.Client] = {}
        self._response: types.GenerateContentResponse | None = None

    # -- back-compat: first key/model exposed as the "current" one ----------
    @property
    def _api_key(self) -> str:
        return self._keys[0]

    @property
    def _model(self) -> str:
        return self._models[0]

    @property
    def last_response(self) -> types.GenerateContentResponse | None:
        return self._response

    def _client_for(self, key: str) -> genai.Client:
        if key not in self._clients:
            http_options = types.HttpOptions(base_url=self._base_url) if self._base_url else None
            self._clients[key] = genai.Client(api_key=key, http_options=http_options)
        return self._clients[key]

    def _next_key(self) -> str:
        self._key_idx = (self._key_idx + 1) % len(self._keys)
        return self._keys[self._key_idx]

    @staticmethod
    def _build_image_parts(images: List[Path]) -> List[types.Part]:
        """Inline each existing image file as bytes (works with any key)."""
        parts: List[types.Part] = []
        for f in images or []:
            p = Path(f) if f else None
            if not p or not p.exists():
                continue
            mime = mimetypes.guess_type(str(p))[0] or "image/png"
            parts.append(types.Part.from_bytes(data=p.read_bytes(), mime_type=mime))
        return parts

    def _build_config(
        self, model: str, description: str | None, response_schema: Type[ResponseT]
    ) -> types.GenerateContentConfig:
        config = types.GenerateContentConfig(
            system_instruction=description,
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            response_mime_type="application/json",
            response_schema=response_schema,
        )
        # Thinking config (per-model capability).
        config.thinking_config = types.ThinkingConfig(include_thoughts=True)
        if model in THINKING_LEVEL_MODELS:
            config.thinking_config = types.ThinkingConfig(
                include_thoughts=False, thinking_level=types.ThinkingLevel.HIGH
            )
        return config

    @staticmethod
    def _classify(exc: Exception) -> str:
        """Map an SDK exception to 'rate_limit' | 'unavailable' | 'transient' | 'other'."""
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        msg = str(exc).lower()
        if code == 429 or "resource_exhausted" in msg or "rate limit" in msg or "quota" in msg:
            return "rate_limit"
        if code in (404, 400) or "not found" in msg or "not supported" in msg or (
            "invalid argument" in msg and "model" in msg
        ):
            return "unavailable"
        if (isinstance(code, int) and 500 <= code < 600) or any(
            t in msg for t in ("unavailable", "deadline", "timeout", "internal error")
        ):
            return "transient"
        return "other"

    async def generate_with_images(
        self,
        *,
        images: List[Path],
        response_schema: Type[ResponseT],
        user_prompt: str | None = None,
        description: str | None = None,
        **kwargs,
    ) -> ResponseT:
        parts = self._build_image_parts(images)
        if user_prompt and isinstance(user_prompt, str):
            parts.append(types.Part.from_text(text=user_prompt))
        contents = [types.Content(role="user", parts=parts)]

        last_exc: Exception | None = None
        for model in self._models:
            # Give every key a chance on this model before dropping to the next.
            for _ in range(len(self._keys)):
                key = self._next_key()
                client = self._client_for(key)
                config = self._build_config(model, description, response_schema)
                try:
                    return await self._generate_once(
                        client, model, contents, config, response_schema
                    )
                except Exception as e:  # noqa: BLE001 - classified below
                    last_exc = e
                    kind = self._classify(e)
                    if kind == "rate_limit":
                        logger.warning(
                            f"Gemini model={model} key #{self._key_idx + 1}/{len(self._keys)} "
                            f"rate-limited; rotating key."
                        )
                        continue
                    if kind == "unavailable":
                        logger.warning(
                            f"Gemini model={model} unavailable on this key/tier "
                            f"({e}); falling back to the next model."
                        )
                        break  # stop trying keys for this model; go to next model
                    # 'other' (e.g. parse/validation) — do not mask it.
                    raise
            else:
                # for-loop completed without break => every key was rate-limited.
                logger.warning(
                    f"Gemini model={model} exhausted across all keys; trying next model."
                )

        raise last_exc or ValueError("Gemini request failed across all keys and models.")

    async def _generate_once(
        self, client, model, contents, config, response_schema: Type[ResponseT]
    ) -> ResponseT:
        """One (model, key) call with a small transient-error retry, then parse."""
        attempt = 0
        while True:
            attempt += 1
            try:
                self._response = await client.aio.models.generate_content(
                    model=model, contents=contents, config=config
                )
                break
            except Exception as e:  # noqa: BLE001
                if self._classify(e) == "transient" and attempt < _TRANSIENT_RETRIES:
                    logger.warning(
                        f"Transient Gemini error on {model} "
                        f"({attempt}/{_TRANSIENT_RETRIES}): {e}; retrying in {_TRANSIENT_WAIT_S}s."
                    )
                    await asyncio.sleep(_TRANSIENT_WAIT_S)
                    continue
                raise
        return self._parse(self._response, response_schema)

    @staticmethod
    def _parse(response, response_schema: Type[ResponseT]) -> ResponseT:
        if response.parsed:
            parsed = response.parsed
            if isinstance(parsed, BaseModel):
                return response_schema(**parsed.model_dump())
            if isinstance(parsed, dict):
                return response_schema(**cast(dict[str, object], parsed))
        if response_text := response.text:
            json_data = extract_first_json_block(response_text)
            if json_data:
                return response_schema(**json_data)
        raise ValueError(f"Failed to parse response: {getattr(response, 'text', response)}")

    def cache_response(self, path: Path) -> None:
        """Cache the last response to a file."""
        if not self._response:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self._response.model_dump(mode="json"), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")
