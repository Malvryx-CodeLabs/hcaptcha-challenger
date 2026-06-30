# -*- coding: utf-8 -*-
"""
Production-grade hCaptcha solver API.

Deploy the service, then POST a `sitekey` + `siteurl` to `/v1/solve`; the service
drives a headless browser via :class:`hcaptcha_challenger.AgentV` and returns the
hCaptcha token. Concurrency is bounded and excess requests are queued so the
service stays within the resources of a small machine (default tuning targets
Linux, 4 GB RAM, 2 CPU).
"""
from hcaptcha_challenger.api.config import ApiSettings

__all__ = ["ApiSettings"]
