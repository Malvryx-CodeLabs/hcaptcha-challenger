# -*- coding: utf-8 -*-
"""
Reasoner - Abstract base class for all reasoning tools.

This module provides the base class that all tool classes inherit from.
Using ABC allows us to share common implementation code while enforcing
that subclasses implement the required methods.

Design principles:
1. Provider-agnostic: Uses ChatProvider protocol for flexibility
2. Description-driven: Loads prompts from .md files
3. Standalone-friendly: Can be used without agent context
"""
import json
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Generic, TypeVar, Union

from loguru import logger
from pydantic import BaseModel

from .providers.gemini import GeminiProvider
from .providers.protocol import ChatProvider

ModelT = TypeVar("ModelT", bound=str)
ResponseT = TypeVar("ResponseT", bound=Union[BaseModel, Enum])


class Reasoner(ABC, Generic[ModelT, ResponseT]):
    """
    Abstract base class for all reasoning tools.

    Subclasses must:
    1. Define a `description` class attribute with the system prompt
    2. Implement __call__() with their specific async logic

    Attributes:
        description: The system prompt for the tool.
            Subclasses should define this using `load_desc(Path(__file__).parent / 'xxx.md')`.
    """

    description: str = ""
    """The description of the tool."""

    def __init__(
        self,
        gemini_api_key: str,
        model: ModelT | None = None,
        *,
        provider: ChatProvider | None = None,
        provider_type: str = "gemini",
        api_key: str | None = None,
        base_url: str | None = None,
        auto_refresh: bool = True,
        **kwargs,
    ):
        """
        Initialize the reasoner.

        Args:
            gemini_api_key: API key for the selected provider. Kept under this
                name for backwards compatibility; ``api_key`` takes precedence
                when given.
            model: Model name to use.
            provider: Optional custom provider instance (overrides provider_type).
            provider_type: Which built-in provider to create when no explicit
                ``provider`` is passed. One of ``"gemini"``, ``"groq"`` or
                ``"openai"`` (OpenAI-compatible endpoint, e.g. Qwen).
            api_key: API key for the selected provider (preferred over
                ``gemini_api_key``).
            base_url: Optional custom endpoint. For ``gemini`` it routes the SDK
                to a proxy; for ``openai`` it is the required base URL.
            **kwargs: Additional options for subclasses.
        """
        self._api_key = api_key or gemini_api_key
        self._model = model
        self._provider_type = provider_type
        self._base_url = base_url
        self._auto_refresh = auto_refresh
        self._provider: ChatProvider = provider or self._create_default_provider()
        self._response = None

    def _create_default_provider(self) -> ChatProvider:
        """Create the provider selected by ``provider_type``."""
        # Qwen via the aikit.club proxy: OpenAI-compatible + token refresh.
        if self._provider_type == "aikit":
            from .providers.aikit import AIKIT_BASE_URL, AikitProvider

            return AikitProvider(
                api_key=self._api_key,
                model=self._model,
                base_url=self._base_url or AIKIT_BASE_URL,
                auto_refresh=self._auto_refresh,
            )
        # Omegatech: single-GET gateway (gpt-4o-mini), image-URL only.
        if self._provider_type == "omegatech":
            from .providers.omegatech import OMEGATECH_BASE_URL, OmegatechProvider

            return OmegatechProvider(
                api_key=self._api_key,
                model=self._model,
                base_url=self._base_url or OMEGATECH_BASE_URL,
            )
        # Both Groq and generic "openai" backends speak the OpenAI-compatible
        # Chat Completions protocol, so they share one provider implementation.
        if self._provider_type in ("groq", "openai"):
            # Imported lazily so Gemini-only users never touch the module.
            from .providers.groq import GROQ_BASE_URL, GroqProvider

            base_url = self._base_url or GROQ_BASE_URL
            return GroqProvider(api_key=self._api_key, model=self._model, base_url=base_url)
        return GeminiProvider(
            api_key=self._api_key, model=self._model, base_url=self._base_url
        )

    @abstractmethod
    async def __call__(self, *args, **kwargs) -> ResponseT:
        """
        Invoke the reasoning tool asynchronously.

        Subclasses must implement this method with their specific logic.

        Usage:
            result = await tool(challenge_screenshot=path)

        Returns:
            The parsed response from the provider.
        """
        raise NotImplementedError

    def cache_response(self, path: Path) -> None:
        """
        Cache the last response to a file.

        Args:
            path: Path to save the response JSON.
        """
        cache_fn = getattr(self._provider, "cache_response", None)
        if cache_fn is not None and callable(cache_fn):
            cache_fn(path)
        elif self._response:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        self._response.model_dump(mode="json"), indent=2, ensure_ascii=False
                    ),
                    encoding="utf-8",
                )
            except Exception as e:
                logger.warning(f"Failed to cache response: {e}")
