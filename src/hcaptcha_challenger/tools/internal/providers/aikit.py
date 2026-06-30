# -*- coding: utf-8 -*-
"""
AikitProvider - Qwen via the aikit.club proxy (https://qwen.aikit.club).

This is an unofficial OpenAI-compatible proxy for Qwen with vision support
(`qwen-max-latest` / `qwen2.5-max` accept images). It is kept separate from the
generic ``openai`` provider because it adds a token-refresh mechanism:

- Auth uses a single compressed token (starts with ``H4sIAAAA...``) passed as
  ``Authorization: Bearer <token>``.
- Tokens expire. ``POST /v1/refresh`` with ``{"token": <current>}`` returns a new
  ``access_token`` and an ``expires_at`` Unix timestamp — no re-login needed.

Docs: https://qwen-api.readme.io/docs/getting-started
"""
import re
import time
from typing import List, Type, TypeVar

import httpx
from loguru import logger
from pydantic import BaseModel

from .groq import GroqProvider

ResponseT = TypeVar("ResponseT", bound=BaseModel)

AIKIT_BASE_URL = "https://qwen.aikit.club/v1"

# Refresh this many seconds before the token's stated expiry.
_REFRESH_LEEWAY_SECONDS = 60

# The proxy appends a metadata block to every answer; strip it before parsing.
# e.g. "...real answer...\n\n<details>\n<summary></summary>\n```\nResponse ID: ...\n```\n</details>"
_DETAILS_RE = re.compile(r"\n*<details>.*?</details>\s*$", re.DOTALL)


class _TokenSlot:
    """One aikit token with its own (independently refreshable) expiry."""

    __slots__ = ("token", "expires_at")

    def __init__(self, token: str, expires_at: int | None = None):
        self.token = token
        self.expires_at = expires_at


class AikitProvider(GroqProvider):
    """
    Qwen-over-aikit.club provider with multi-token rotation.

    Reuses the OpenAI-compatible request/parse pipeline from :class:`GroqProvider`
    (base64 image inlining, json_schema -> json_object fallback) and layers, on
    top of the HTTP transport:

    - **Rotation:** with several tokens, requests are spread round-robin and a
      token that returns 429 (rate/usage limit) is rotated out for the next.
    - **Refresh:** each token is refreshed independently via ``/v1/refresh`` —
      proactively when near its expiry, and reactively once on a 401.
    """

    def __init__(
        self,
        api_key,
        model: str,
        *,
        base_url: str = AIKIT_BASE_URL,
        auto_refresh: bool = True,
        expires_at: int | None = None,
    ):
        """
        Args:
            api_key: One or more compressed aikit tokens (``H4sIAAAA...``).
                Accepts a single string, a comma-separated string, or a list.
            model: A vision-capable Qwen model, e.g. ``qwen-max-latest``.
            base_url: Endpoint root (``.../v1``).
            auto_refresh: When True, proactively refresh near expiry and retry
                once on a 401.
            expires_at: Optional known Unix expiry applied to every token; if
                omitted it is learned from each token's first refresh.
        """
        super().__init__(api_key=api_key, model=model, base_url=base_url)
        self._auto_refresh = auto_refresh
        self._slots = [_TokenSlot(token=k, expires_at=expires_at) for k in self._keys]
        self._slot_idx = -1

    async def generate_with_images(
        self, *, images, response_schema, user_prompt=None, description=None, **kwargs
    ):
        """
        aikit.club only accepts image *URLs*, not inline/base64 images, which this
        codebase always produces (it inlines local screenshots). Fail loudly with
        guidance instead of silently returning empty answers.
        """
        if images:
            raise ValueError(
                "aikit.club vision only accepts image URLs, not inline/base64 images, "
                "so LLM_PROVIDER='aikit' cannot solve screenshot-based hCaptcha challenges. "
                "Use LLM_PROVIDER='groq', 'gemini', or 'openai' (e.g. DashScope qwen-vl) instead."
            )
        return await super().generate_with_images(
            images=images,
            response_schema=response_schema,
            user_prompt=user_prompt,
            description=description,
            **kwargs,
        )

    @staticmethod
    def _parse(data: dict, response_schema):
        """Strip the proxy's trailing <details> block before normal parsing."""
        try:
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, str):
                data["choices"][0]["message"]["content"] = _DETAILS_RE.sub("", content).strip()
        except (KeyError, IndexError, TypeError):
            pass
        return GroqProvider._parse(data, response_schema)

    def _next_slot(self) -> _TokenSlot:
        """Advance round-robin and return the next token slot."""
        self._slot_idx = (self._slot_idx + 1) % len(self._slots)
        slot = self._slots[self._slot_idx]
        # Keep the back-compat single-key mirror pointing at the active slot.
        self._api_key = slot.token
        return slot

    async def _refresh_slot(self, slot: _TokenSlot) -> bool:
        """Exchange one slot's token for a fresh one. Returns True on success."""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
                resp = await client.post(f"{self._base_url}/refresh", json={"token": slot.token})
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.warning(
                f"aikit token refresh failed: {e}. If this persists, regenerate the "
                f"token from the browser (see https://qwen-api.readme.io/docs/getting-started)."
            )
            return False

        new_token = data.get("access_token")
        if new_token:
            slot.token = new_token
            if slot is self._slots[self._slot_idx]:
                self._api_key = new_token
        if data.get("expires_at"):
            try:
                slot.expires_at = int(data["expires_at"])
            except (TypeError, ValueError):
                slot.expires_at = None
        logger.debug(f"aikit token refreshed (expires_at={slot.expires_at}).")
        return bool(new_token)

    async def _ensure_fresh(self, slot: _TokenSlot) -> None:
        """Proactively refresh a slot when it is known to be near expiry."""
        if not self._auto_refresh or slot.expires_at is None:
            return
        if time.time() >= slot.expires_at - _REFRESH_LEEWAY_SECONDS:
            await self._refresh_slot(slot)

    async def _post(self, payload: dict) -> dict:
        """
        POST with token rotation + refresh.

        Round-robins across tokens; refreshes a token proactively near expiry and
        once reactively on a 401; rotates to the next token on a 429.
        """
        url = f"{self._base_url}/chat/completions"
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            resp = None
            for _ in range(len(self._slots)):
                slot = self._next_slot()
                await self._ensure_fresh(slot)

                resp = await self._send(client, url, payload, slot)
                if resp.status_code == 401 and self._auto_refresh:
                    logger.info("aikit returned 401 - refreshing token and retrying.")
                    if await self._refresh_slot(slot):
                        resp = await self._send(client, url, payload, slot)

                if resp.status_code == 429 and len(self._slots) > 1:
                    logger.warning(
                        f"aikit token #{self._slot_idx + 1}/{len(self._slots)} "
                        f"rate-limited (429); rotating to the next token."
                    )
                    continue

                resp.raise_for_status()
                return resp.json()
            resp.raise_for_status()
            return resp.json()  # pragma: no cover - unreachable

    @staticmethod
    async def _send(client, url: str, payload: dict, slot: _TokenSlot):
        headers = {"Authorization": f"Bearer {slot.token}", "Content-Type": "application/json"}
        return await client.post(url, headers=headers, json=payload)
