# -*- coding: utf-8 -*-
"""
OmegatechProvider - vision via the Omegatech gateway (GPT-4o-mini class).

The endpoint is a single GET request, NOT an OpenAI-compatible chat API:

    GET {base}/{model}?message=<text>&imageUrl=<url-or-data-uri>
    -> {"success": true, "answer": "<model text>", ...}

Notes that shape this implementation:

- **One image only.** The API takes a single ``imageUrl`` query parameter, so for
  multi-image tools (the spatial reasoners pass the clean screenshot *and* a grid
  overlay) we send the LAST image, which is the grid-annotated one carrying the
  coordinate axes.
- **URL, not base64.** A real challenge screenshot base64-encodes to hundreds of
  KB; that overflows the GET query string (URL too long) and a POST body is
  rejected (413). So each image is uploaded to a temp host and the short CDN URL
  is sent, then deleted after the request (shared :class:`TempUploader`).
- **No structured-output mode.** The JSON Schema is embedded in the prompt and the
  free-text ``answer`` is parsed with the same lenient extractor Groq uses.
- **No auth.** ``api_key`` is accepted for interface uniformity but unused.
"""
import json
from pathlib import Path
from typing import List, Type, TypeVar

import httpx
from loguru import logger
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed

from ._upload import TempUploader
from .groq import GroqProvider

ResponseT = TypeVar("ResponseT", bound=BaseModel)

OMEGATECH_BASE_URL = "https://omegatech-api.dixonomega.tech/api/ai"
DEFAULT_OMEGATECH_MODEL = "Gpt-4-mini"


class OmegatechProvider:
    """Vision provider for the Omegatech GET gateway."""

    def __init__(
        self,
        api_key=None,
        model: str = DEFAULT_OMEGATECH_MODEL,
        *,
        base_url: str = OMEGATECH_BASE_URL,
        auto_refresh: bool = True,  # accepted for interface parity; unused
    ):
        # The model name is the final URL path segment (e.g. "Gpt-4-mini").
        self._model = (model or DEFAULT_OMEGATECH_MODEL).strip("/")
        self._base_url = base_url.rstrip("/")
        self._uploader = TempUploader()
        self._response: dict | None = None

    @property
    def last_response(self) -> dict | None:
        return self._response

    @staticmethod
    def _build_message(
        *, user_prompt: str | None, description: str | None, json_schema: dict
    ) -> str:
        """Fold the system description, user prompt, and schema into one message."""
        parts: List[str] = []
        if description:
            parts.append(description)
        if user_prompt:
            parts.append(user_prompt)
        parts.append(
            "Respond with ONLY a single JSON object that strictly conforms to this "
            "JSON Schema. Do not add any prose, explanation, or markdown fences other "
            "than the JSON itself.\n" + json.dumps(json_schema, ensure_ascii=False)
        )
        return "\n\n".join(parts)

    async def generate_with_images(
        self,
        *,
        images: List[Path],
        response_schema: Type[ResponseT],
        user_prompt: str | None = None,
        description: str | None = None,
        **kwargs,
    ) -> ResponseT:
        json_schema = response_schema.model_json_schema()
        message = self._build_message(
            user_prompt=user_prompt, description=description, json_schema=json_schema
        )

        valid = [Path(f) for f in (images or []) if f and Path(f).exists()]
        if not valid:
            return await self._complete(message, image_url=None, response_schema=response_schema)

        if len(valid) > 1:
            logger.debug(
                f"Omegatech accepts a single image; sending the last of {len(valid)} "
                f"(the grid-annotated one)."
            )
        chosen = valid[-1]

        if not self._uploader.enabled:
            raise ValueError(
                "Omegatech vision needs an image URL, but image upload is disabled "
                "(AIKIT_IMAGE_UPLOAD=false). Enable it to use LLM_PROVIDER='omegatech'."
            )

        upload = None
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as up_client:
            try:
                upload = await self._uploader.upload(up_client, chosen)
                return await self._complete(
                    message, image_url=upload["url"], response_schema=response_schema
                )
            finally:
                if upload is not None:
                    await self._uploader.delete(up_client, upload)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(3),
        before_sleep=lambda rs: logger.warning(
            f"Retry Omegatech request ({rs.attempt_number}/3) - "
            f"Wait 3 seconds - Exception: {rs.outcome.exception()}"
        ),
    )
    async def _complete(
        self, message: str, *, image_url: str | None, response_schema: Type[ResponseT]
    ) -> ResponseT:
        url = f"{self._base_url}/{self._model}"
        params = {"message": message}
        if image_url:
            params["imageUrl"] = image_url

        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        self._response = data
        if not data.get("success", True):
            raise ValueError(f"Omegatech returned an error: {data}")

        answer = data.get("answer")
        if not answer:
            raise ValueError(f"Empty 'answer' in Omegatech response: {data}")

        # Reuse Groq's lenient JSON extraction on the free-text answer.
        shaped = {"choices": [{"message": {"content": answer}}]}
        return GroqProvider._parse(shaped, response_schema)

    def cache_response(self, path: Path) -> None:
        if not self._response:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self._response, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as e:  # pragma: no cover
            logger.warning(f"Failed to cache response: {e}")
