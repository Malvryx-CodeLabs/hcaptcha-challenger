# -*- coding: utf-8 -*-
"""
BrowserManager - one shared Chromium process, an isolated context per solve.

Launching a browser process per request is slow; sharing a single process and
giving each solve a fresh ``BrowserContext`` keeps solves isolated (cookies,
storage, the hCaptcha widget state) while amortising launch cost.
"""
from contextlib import asynccontextmanager
from typing import List, Optional

from loguru import logger
from playwright.async_api import Browser, BrowserContext, Playwright, async_playwright


class BrowserManager:
    def __init__(self, *, headless: bool, args: List[str]):
        self._headless = headless
        self._args = args
        self._pw: Optional[Playwright] = None
        self._browser: Optional[Browser] = None

    @property
    def is_running(self) -> bool:
        return self._browser is not None and self._browser.is_connected()

    async def start(self) -> None:
        if self._browser is not None:
            return
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(headless=self._headless, args=self._args)
        logger.success(f"Chromium launched (headless={self._headless}).")

    async def stop(self) -> None:
        try:
            if self._browser is not None:
                await self._browser.close()
        finally:
            self._browser = None
            if self._pw is not None:
                await self._pw.stop()
                self._pw = None
        logger.info("Chromium stopped.")

    @asynccontextmanager
    async def context(self):
        """Yield a fresh, isolated browser context, closed on exit."""
        if self._browser is None:
            raise RuntimeError("BrowserManager not started")
        ctx: BrowserContext = await self._browser.new_context(
            locale="en-US",
            viewport={"width": 1280, "height": 720},
        )
        try:
            yield ctx
        finally:
            try:
                await ctx.close()
            except Exception as e:  # pragma: no cover - best-effort cleanup
                logger.warning(f"Failed to close browser context: {e}")
