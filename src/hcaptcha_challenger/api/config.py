# -*- coding: utf-8 -*-
"""
ApiSettings - configuration for the solver API.

All settings are read from environment variables (prefix ``HCAPTCHA_API_``) or a
``.env`` file, so the same image can be tuned per machine without code changes.

Defaults target a small box: **Linux, 4 GB RAM, 2 CPU**. Each concurrent solve
runs its own isolated Chromium browser context (~300-500 MB), so the most
important knob is ``MAX_CONCURRENT_SOLVES``. See deploy/README.md for a tuning
table.

Note on scaling: the solver shares one browser process and one in-memory queue,
so the service runs as a **single process** (one Uvicorn worker). To scale out,
run more containers behind a load balancer rather than more workers.
"""
from typing import Annotated, List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HCAPTCHA_API_", env_file=".env", env_ignore_empty=True, extra="ignore"
    )

    # -- Network --
    HOST: str = Field(default="0.0.0.0", description="Bind address.")
    PORT: int = Field(default=8000, description="Bind port.")

    # -- Capacity / queueing (tune to the machine) --
    MAX_CONCURRENT_SOLVES: int = Field(
        default=2,
        ge=1,
        description="Max browser solves running at once. ~300-500 MB RAM each. "
        "Default 2 fits 4 GB / 2 CPU.",
    )
    MAX_QUEUE: int = Field(
        default=32,
        ge=0,
        description="Max requests waiting for a free solve slot. Beyond "
        "MAX_CONCURRENT_SOLVES + MAX_QUEUE, new requests are rejected with 429.",
    )
    QUEUE_WAIT_TIMEOUT_S: float = Field(
        default=120.0,
        gt=0,
        description="Max seconds a request may wait in the queue before 429.",
    )
    SOLVE_TIMEOUT_S: float = Field(
        default=180.0,
        gt=0,
        description="Max seconds for a single solve before 504.",
    )
    NAV_TIMEOUT_S: float = Field(
        default=60.0, gt=0, description="Per-page navigation timeout."
    )

    # -- Browser --
    HEADLESS: bool = Field(default=True, description="Run Chromium headless.")
    BROWSER_ARGS: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: [
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
        ],
        description="Extra Chromium launch flags (defaults are container-friendly).",
    )

    # -- Security --
    REQUIRE_AUTH: bool = Field(
        default=True,
        description="Require an API key. When True, API_KEYS must be non-empty or "
        "the service refuses to start (fail closed).",
    )
    API_KEYS: Annotated[List[str], NoDecode] = Field(
        default_factory=list,
        description="Accepted API keys (comma-separated in env). Sent as "
        "'Authorization: Bearer <key>' or 'X-API-Key: <key>'.",
    )
    RATE_LIMIT_RPM: int = Field(
        default=120,
        ge=0,
        description="Per-identity requests per minute (0 disables). In-process, "
        "so it is per-container.",
    )
    CORS_ORIGINS: Annotated[List[str], NoDecode] = Field(
        default_factory=list,
        description="Allowed CORS origins (comma-separated). Empty = none.",
    )
    ENABLE_DOCS: bool = Field(
        default=True, description="Expose /docs and /openapi.json. Disable in prod if unused."
    )

    # -- Debug live stream --
    # When enabled, the server exposes a continuous MJPEG video stream of the
    # browser on STREAM_PORT (open it in VLC / a browser). The stream starts
    # immediately showing "No Task" and switches to the live page the moment a
    # solve begins, without the stream connection breaking. This is a debugging
    # aid: it FORCES MAX_CONCURRENT_SOLVES to 1 (only one task can be streamed).
    STREAM_ENABLED: bool = Field(
        default=False,
        description="Expose a live MJPEG debug stream of the browser. Forces "
        "concurrency to 1. Intended for debugging behind a private network (e.g. Tailscale).",
    )
    STREAM_PORT: int = Field(default=8089, description="Port for the MJPEG debug stream.")
    STREAM_FPS: int = Field(
        default=6, ge=1, le=30, description="Frames per second pushed to the stream."
    )
    STREAM_QUALITY: int = Field(
        default=60, ge=10, le=100, description="JPEG quality (1-100) of stream frames."
    )
    STREAM_MAX_WIDTH: int = Field(
        default=1280, ge=320, le=1920, description="Max width of stream frames (downscaled)."
    )

    # -- Misc --
    LOG_LEVEL: str = Field(default="info", description="Uvicorn/log level.")

    @field_validator("API_KEYS", "CORS_ORIGINS", "BROWSER_ARGS", mode="before")
    @classmethod
    def _split_csv(cls, v):
        """Allow comma-separated strings in env vars for list fields."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @property
    def max_admitted(self) -> int:
        """Total requests allowed in the system (running + queued)."""
        return self.MAX_CONCURRENT_SOLVES + self.MAX_QUEUE

    def validate_security(self) -> None:
        """Fail closed: refuse to run with auth required but no keys configured."""
        if self.REQUIRE_AUTH and not self.API_KEYS:
            raise RuntimeError(
                "REQUIRE_AUTH is true but no API_KEYS are configured. Set "
                "HCAPTCHA_API_API_KEYS=<key1,key2> or explicitly set "
                "HCAPTCHA_API_REQUIRE_AUTH=false for an unauthenticated service."
            )
