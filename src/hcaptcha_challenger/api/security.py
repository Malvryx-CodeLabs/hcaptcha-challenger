# -*- coding: utf-8 -*-
"""API-key authentication and a lightweight in-process rate limiter."""
import time
from collections import defaultdict, deque
from typing import Deque, Dict

from fastapi import Header, HTTPException, Request, status


def extract_api_key(authorization: str | None, x_api_key: str | None) -> str | None:
    """Pull the key from 'Authorization: Bearer <key>' or 'X-API-Key: <key>'."""
    if x_api_key:
        return x_api_key.strip()
    if authorization:
        parts = authorization.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
        return authorization.strip()
    return None


class RateLimiter:
    """Fixed-window-ish sliding counter keyed by identity. Per-process only."""

    def __init__(self, rpm: int):
        self.rpm = rpm
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    def check(self, identity: str) -> bool:
        """Return True if allowed; record the hit. No-op when rpm <= 0."""
        if self.rpm <= 0:
            return True
        now = time.monotonic()
        window_start = now - 60.0
        bucket = self._hits[identity]
        while bucket and bucket[0] < window_start:
            bucket.popleft()
        if len(bucket) >= self.rpm:
            return False
        bucket.append(now)
        return True


def build_auth_dependency(settings, rate_limiter: RateLimiter):
    """Create the FastAPI dependency enforcing auth + rate limiting."""

    async def dependency(
        request: Request,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> str:
        identity = "anonymous"
        if settings.REQUIRE_AUTH:
            key = extract_api_key(authorization, x_api_key)
            if not key or key not in set(settings.API_KEYS):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Missing or invalid API key.",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            identity = key
        else:
            # Unauthenticated mode: rate-limit by client IP.
            identity = request.client.host if request.client else "anonymous"

        if not rate_limiter.check(identity):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded.",
                headers={"Retry-After": "60"},
            )
        return identity

    return dependency
