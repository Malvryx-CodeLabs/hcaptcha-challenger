# -*- coding: utf-8 -*-
"""
GroqProvider - Groq API implementation.

Groq exposes an OpenAI-compatible Chat Completions endpoint, so this provider
talks to it directly with httpx (already a project dependency) instead of
pulling in an extra SDK.

Key differences from the Gemini provider:
- Images are inlined as base64 data URIs (Groq has no file-upload API).
  Limits: <=4MB per base64 image, <=5 images per request.
  https://console.groq.com/docs/vision
- Structured output uses ``response_format`` with ``json_schema`` (best-effort,
  non-strict) and falls back to ``json_object`` + schema-in-prompt when a model
  does not support schema mode.
  https://console.groq.com/docs/structured-outputs
- There is no Gemini-style "thinking_level"; SCoT reasoning is prompt-driven.
"""
import base64
import json
import mimetypes
from pathlib import Path
from typing import Any, List, Type, TypeVar

import httpx
from loguru import logger
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed

ResponseT = TypeVar("ResponseT", bound=BaseModel)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Groq rejects base64 images larger than 4MB with a 413.
MAX_BASE64_IMAGE_BYTES = 4 * 1024 * 1024
# Groq accepts at most 5 images per request.
MAX_IMAGES_PER_REQUEST = 5


def extract_first_json_block(text: str) -> dict | None:
    """Extract the first JSON code block from text."""
    import re

    pattern = r"```json\s*([\s\S]*?)```"
    matches = re.findall(pattern, text)
    if matches:
        return json.loads(matches[0])
    # Fall back to the first bare {...} object.
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        return json.loads(brace_match.group(0))
    return None


class GroqProvider:
    """
    Groq-based chat provider implementation.

    Mirrors the ``ChatProvider`` protocol so it is a drop-in alternative to
    :class:`GeminiProvider`.
    """

    def __init__(self, api_key, model: str, *, base_url: str = GROQ_BASE_URL):
        """
        Initialize the Groq provider.

        Args:
            api_key: One or more Groq API keys (``gsk_...``). Accepts a single
                string, a comma-separated string, or a list. When more than one
                key is given, requests are spread round-robin across keys and a
                key that returns 429 (rate/usage limit) is rotated out for the
                next, so no single key is exhausted quickly.
            model: Vision-capable model id, e.g. ``meta-llama/llama-4-scout-17b-16e-instruct``.
            base_url: Override for the OpenAI-compatible endpoint.
        """
        self._keys = self.normalize_keys(api_key)
        # Back-compat: expose the first key as the "current" key.
        self._api_key = self._keys[0]
        self._key_idx = -1
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._response: dict | None = None

    @staticmethod
    def normalize_keys(api_key) -> List[str]:
        """Normalize a str / comma-separated str / list into a list of keys."""
        raw: List[str] = []
        if isinstance(api_key, str):
            raw = api_key.split(",")
        elif isinstance(api_key, (list, tuple)):
            for item in api_key:
                raw.extend(str(item).split(","))
        keys = [k.strip() for k in raw if k and k.strip()]
        if not keys:
            raise ValueError("At least one API key is required.")
        return keys

    def _next_key(self) -> str:
        """Advance round-robin and return the next key."""
        self._key_idx = (self._key_idx + 1) % len(self._keys)
        self._api_key = self._keys[self._key_idx]
        return self._api_key

    @property
    def last_response(self) -> dict | None:
        """Get the last raw response for debugging/caching purposes."""
        return self._response

    @staticmethod
    def _encode_image(path: Path) -> str:
        """Read an image file and return an OpenAI-style base64 data URI."""
        raw = path.read_bytes()
        b64 = base64.b64encode(raw)
        if len(b64) > MAX_BASE64_IMAGE_BYTES:
            raise ValueError(
                f"Image {path.name} is {len(b64) / 1024 / 1024:.2f}MB base64-encoded, "
                f"exceeding Groq's 4MB limit. Downscale the screenshot before sending."
            )
        mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
        return f"data:{mime_type};base64,{b64.decode('utf-8')}"

    def _build_image_parts(self, images: List[Path]) -> List[dict]:
        """Convert image paths into Groq ``image_url`` content parts."""
        valid_files = [Path(f) for f in images if f and Path(f).exists()]
        if len(valid_files) > MAX_IMAGES_PER_REQUEST:
            raise ValueError(
                f"Groq accepts at most {MAX_IMAGES_PER_REQUEST} images per request, "
                f"got {len(valid_files)}."
            )
        return [
            {"type": "image_url", "image_url": {"url": self._encode_image(f)}}
            for f in valid_files
        ]

    def _build_messages(
        self,
        *,
        image_parts: List[dict],
        user_prompt: str | None,
        description: str | None,
        schema_hint: str | None = None,
    ) -> List[dict]:
        """Assemble the OpenAI-style messages list."""
        messages: List[dict] = []
        if description:
            messages.append({"role": "system", "content": description})

        user_content: List[dict] = list(image_parts)

        text_blocks: List[str] = []
        if user_prompt and isinstance(user_prompt, str):
            text_blocks.append(user_prompt)
        if schema_hint:
            text_blocks.append(schema_hint)
        if text_blocks:
            user_content.append({"type": "text", "text": "\n\n".join(text_blocks)})

        messages.append({"role": "user", "content": user_content})
        return messages

    async def _post(self, payload: dict) -> dict:
        """
        Send a chat completion request and return the parsed JSON body.

        Spreads load round-robin across the key pool and, on a 429 (rate/usage
        limit), rotates to the next key and retries. Raises the last 429 only if
        every key is rate-limited.
        """
        url = f"{self._base_url}/chat/completions"
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            resp = None
            for _ in range(len(self._keys)):
                key = self._next_key()
                headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code == 429 and len(self._keys) > 1:
                    logger.warning(
                        f"Key #{self._key_idx + 1}/{len(self._keys)} rate-limited (429); "
                        f"rotating to the next key."
                    )
                    continue
                resp.raise_for_status()
                return resp.json()
            # Every key was rate-limited: surface the last 429.
            resp.raise_for_status()
            return resp.json()  # pragma: no cover - unreachable

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(3),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry request ({retry_state.attempt_number}/3) - "
            f"Wait 3 seconds - Exception: {retry_state.outcome.exception()}"
        ),
    )
    async def generate_with_images(
        self,
        *,
        images: List[Path],
        response_schema: Type[ResponseT],
        user_prompt: str | None = None,
        description: str | None = None,
        **kwargs,
    ) -> ResponseT:
        """
        Generate content with image inputs.

        Args:
            images: List of image file paths to include in the request.
            response_schema: Pydantic model class for structured output.
            user_prompt: User-provided prompt/instructions.
            description: System instruction/description for the model.
            **kwargs: Provider-specific options. Recognized: ``temperature``.
                Gemini-only options (e.g. ``thinking_level``) are ignored.

        Returns:
            Parsed response matching the response_schema type.
        """
        image_parts = self._build_image_parts(images)
        temperature = kwargs.get("temperature", 1.0)
        json_schema = response_schema.model_json_schema()

        base_payload: dict[str, Any] = {
            "model": self._model,
            "temperature": temperature,
        }

        # First attempt: native json_schema structured output.
        messages = self._build_messages(
            image_parts=image_parts, user_prompt=user_prompt, description=description
        )
        payload = {
            **base_payload,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": response_schema.__name__, "schema": json_schema},
            },
        }

        try:
            data = await self._post(payload)
        except httpx.HTTPStatusError as e:
            # Model may not support json_schema mode — fall back to json_object
            # with the schema embedded in the prompt.
            if e.response.status_code not in (400, 422):
                raise
            logger.warning(
                f"Groq json_schema mode rejected ({e.response.status_code}); "
                f"falling back to json_object mode."
            )
            schema_hint = (
                "Respond with a single JSON object that strictly conforms to this JSON Schema:\n"
                f"{json.dumps(json_schema, ensure_ascii=False)}"
            )
            messages = self._build_messages(
                image_parts=image_parts,
                user_prompt=user_prompt,
                description=description,
                schema_hint=schema_hint,
            )
            payload = {
                **base_payload,
                "messages": messages,
                "response_format": {"type": "json_object"},
            }
            data = await self._post(payload)

        self._response = data
        return self._parse(data, response_schema)

    @staticmethod
    def _parse(data: dict, response_schema: Type[ResponseT]) -> ResponseT:
        """Parse the model's JSON content into the response schema."""
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise ValueError(f"Unexpected Groq response shape: {data}") from e

        if not content:
            raise ValueError(f"Empty content in Groq response: {data}")

        # Content should be a JSON string; try direct load, then block extraction.
        try:
            return response_schema(**json.loads(content))
        except (json.JSONDecodeError, TypeError):
            json_data = extract_first_json_block(content)
            if json_data:
                return response_schema(**json_data)

        raise ValueError(f"Failed to parse Groq response content: {content}")

    def cache_response(self, path: Path) -> None:
        """Cache the last response to a file."""
        if not self._response:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self._response, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")
