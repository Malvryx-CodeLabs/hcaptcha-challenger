# -*- coding: utf-8 -*-
"""Request/response models for the solver API."""
import uuid
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


class SolveRequest(BaseModel):
    sitekey: str = Field(
        ...,
        description="hCaptcha sitekey (UUID) of the target widget.",
        examples=["a5f74b19-9e45-40e0-b45d-47ff91b7a6c2"],
    )
    siteurl: str = Field(
        ...,
        description="URL of the page the token should be bound to.",
        examples=["https://example.com/login"],
    )
    rqdata: Optional[str] = Field(
        default=None, description="Optional enterprise hCaptcha rqdata blob."
    )

    @field_validator("sitekey")
    @classmethod
    def _validate_sitekey(cls, v: str) -> str:
        try:
            uuid.UUID(v)
        except (ValueError, AttributeError):
            raise ValueError("sitekey must be a UUID string")
        return v

    @field_validator("siteurl")
    @classmethod
    def _validate_siteurl(cls, v: str) -> str:
        if not isinstance(v, str) or not v.lower().startswith(("http://", "https://")):
            raise ValueError("siteurl must be an absolute http(s) URL")
        return v


class SolveResponse(BaseModel):
    success: bool = Field(description="Whether the challenge was passed.")
    token: Optional[str] = Field(
        default=None,
        description="The hCaptcha response token (h-captcha-response / passcode). "
        "Submit this to the target site.",
    )
    expiration: Optional[int] = Field(
        default=None, description="Token lifetime in seconds, when reported."
    )
    elapsed_ms: int = Field(description="End-to-end solve time in milliseconds.")
    error: Optional[str] = Field(default=None, description="Failure reason, if any.")
    raw: Optional[Dict[str, Any]] = Field(
        default=None, description="Raw CaptchaResponse payload from the agent."
    )


class StatsResponse(BaseModel):
    active: int = Field(description="Solves currently running.")
    queued: int = Field(description="Requests waiting for a slot.")
    max_concurrent: int = Field(description="Configured concurrency limit.")
    max_queue: int = Field(description="Configured queue depth.")
    capacity_used: float = Field(description="active / max_concurrent (0..1+).")


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
