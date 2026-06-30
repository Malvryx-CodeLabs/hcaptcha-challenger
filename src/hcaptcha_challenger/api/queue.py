# -*- coding: utf-8 -*-
"""
SolveManager - bounded concurrency with an explicit queue.

Admission control has two layers:

1. **Admission cap** (``max_concurrent + max_queue``): the total number of
   requests allowed in the system at once. When full, new requests are rejected
   immediately with :class:`QueueFull` (HTTP 429) instead of piling up unbounded.
2. **Concurrency cap** (``max_concurrent``): how many solves actually run at the
   same time. Admitted-but-not-running requests wait on a semaphore — this is the
   queue. A request that waits longer than ``wait_timeout`` raises
   :class:`QueueTimeout` (HTTP 429).

A running solve longer than ``solve_timeout`` raises :class:`asyncio.TimeoutError`
(mapped to HTTP 504 by the route).

This keeps a memory-constrained box (each solve = one Chromium context) from
being overwhelmed: it runs at most ``max_concurrent`` browsers and bounds the
backlog.
"""
import asyncio
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


class QueueFull(Exception):
    """Raised when the admission cap (concurrent + queued) is exceeded."""


class QueueTimeout(Exception):
    """Raised when a queued request waits past the queue timeout."""


class SolveManager:
    def __init__(self, max_concurrent: int, max_queue: int):
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self.max_concurrent = max_concurrent
        self.max_queue = max_queue
        self.max_admitted = max_concurrent + max_queue

        self._sem = asyncio.Semaphore(max_concurrent)
        self._lock = asyncio.Lock()
        self._admitted = 0
        self._active = 0

    @property
    def active(self) -> int:
        return self._active

    @property
    def queued(self) -> int:
        # Admitted requests that have not yet acquired a concurrency slot.
        return max(0, self._admitted - self._active)

    def stats(self) -> dict:
        return {
            "active": self._active,
            "queued": self.queued,
            "max_concurrent": self.max_concurrent,
            "max_queue": self.max_queue,
            "capacity_used": round(self._active / self.max_concurrent, 3),
        }

    async def submit(
        self,
        factory: Callable[[], Awaitable[T]],
        *,
        wait_timeout: float,
        solve_timeout: float,
    ) -> T:
        """
        Admit, queue, and run ``factory()`` under the concurrency limit.

        Raises:
            QueueFull: admission cap reached.
            QueueTimeout: waited too long for a slot.
            asyncio.TimeoutError: the solve itself exceeded ``solve_timeout``.
        """
        async with self._lock:
            if self._admitted >= self.max_admitted:
                raise QueueFull()
            self._admitted += 1

        slot_acquired = False
        try:
            try:
                await asyncio.wait_for(self._sem.acquire(), timeout=wait_timeout)
                slot_acquired = True
            except asyncio.TimeoutError:
                raise QueueTimeout()

            async with self._lock:
                self._active += 1
            try:
                return await asyncio.wait_for(factory(), timeout=solve_timeout)
            finally:
                async with self._lock:
                    self._active -= 1
        finally:
            if slot_acquired:
                self._sem.release()
            async with self._lock:
                self._admitted -= 1
