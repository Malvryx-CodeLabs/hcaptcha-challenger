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


class AikitProvider(GroqProvider):
    """
    Qwen-over-aikit.club provider.

    Reuses the OpenAI-compatible request/parse pipeline from :class:`GroqProvider`
    (base64 image inlining, json_schema -> json_object fallback) and layers token
    refresh on top of the HTTP transport.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = AIKIT_BASE_URL,
        auto_refresh: bool = True,
        expires_at: int | None = None,
    ):
        """
        Args:
            api_key: The compressed aikit token (``H4sIAAAA...``).
            model: A vision-capable Qwen model, e.g. ``qwen-max-latest``.
            base_url: Endpoint root (``.../v1``).
            auto_refresh: When True, proactively refresh near expiry and retry
                once on a 401.
            expires_at: Optional known Unix expiry; if omitted it is learned from
                the first refresh response.
        """
        super().__init__(api_key=api_key, model=model, base_url=base_url)
        self._auto_refresh = auto_refresh
        self._expires_at = expires_at

    async def _refresh_token(self) -> bool:
        """Exchange the current token for a fresh one. Returns True on success."""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
                resp = await client.post(
                    f"{self._base_url}/refresh", json={"token": self._api_key}
                )
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
            self._api_key = new_token
        if data.get("expires_at"):
            try:
                self._expires_at = int(data["expires_at"])
            except (TypeError, ValueError):
                self._expires_at = None
        logger.debug(f"aikit token refreshed (expires_at={self._expires_at}).")
        return bool(new_token)

    async def _ensure_fresh(self) -> None:
        """Proactively refresh when the token is known to be near expiry."""
        if not self._auto_refresh or self._expires_at is None:
            return
        if time.time() >= self._expires_at - _REFRESH_LEEWAY_SECONDS:
            await self._refresh_token()

    async def _post(self, payload: dict) -> dict:
        """POST with proactive refresh and one reactive refresh-retry on 401."""
        await self._ensure_fresh()
        try:
            return await super()._post(payload)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401 and self._auto_refresh:
                logger.info("aikit returned 401 - attempting token refresh and retry.")
                if await self._refresh_token():
                    return await super()._post(payload)
            raise
