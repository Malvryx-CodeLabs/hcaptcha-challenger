# -*- coding: utf-8 -*-
"""Offline tests for the solver API (no browser, no LLM provider)."""
import asyncio

import pytest
from fastapi.testclient import TestClient

from hcaptcha_challenger.api.app import create_app
from hcaptcha_challenger.api.config import ApiSettings
from hcaptcha_challenger.api.queue import QueueFull, QueueTimeout, SolveManager

VALID_SITEKEY = "a5f74b19-9e45-40e0-b45d-47ff91b7a6c2"
VALID_SITEURL = "https://example.com/login"


def _settings(**over):
    base = dict(
        REQUIRE_AUTH=True,
        API_KEYS=["secret"],
        MAX_CONCURRENT_SOLVES=1,
        MAX_QUEUE=1,
        RATE_LIMIT_RPM=0,  # disable rate limit unless a test sets it
    )
    base.update(over)
    return ApiSettings(**base)


async def _ok_solver(sitekey, siteurl, rqdata):
    return {
        "success": True,
        "token": "P1_token_abc",
        "expiration": 120,
        "error": None,
        "raw": {"pass": True},
    }


def _client(settings=None, solve_func=_ok_solver):
    app = create_app(settings or _settings(), solve_func=solve_func)
    return TestClient(app)


class TestAuth:
    def test_missing_key_rejected(self):
        with _client() as c:
            r = c.post("/v1/solve", json={"sitekey": VALID_SITEKEY, "siteurl": VALID_SITEURL})
            assert r.status_code == 401

    def test_wrong_key_rejected(self):
        with _client() as c:
            r = c.post(
                "/v1/solve",
                json={"sitekey": VALID_SITEKEY, "siteurl": VALID_SITEURL},
                headers={"Authorization": "Bearer nope"},
            )
            assert r.status_code == 401

    def test_valid_key_accepted(self):
        with _client() as c:
            r = c.post(
                "/v1/solve",
                json={"sitekey": VALID_SITEKEY, "siteurl": VALID_SITEURL},
                headers={"Authorization": "Bearer secret"},
            )
            assert r.status_code == 200
            body = r.json()
            assert body["success"] is True
            assert body["token"] == "P1_token_abc"
            assert "elapsed_ms" in body

    def test_x_api_key_header(self):
        with _client() as c:
            r = c.post(
                "/v1/solve",
                json={"sitekey": VALID_SITEKEY, "siteurl": VALID_SITEURL},
                headers={"X-API-Key": "secret"},
            )
            assert r.status_code == 200

    def test_fail_closed_when_no_keys(self):
        with pytest.raises(RuntimeError, match="REQUIRE_AUTH"):
            create_app(_settings(API_KEYS=[]), solve_func=_ok_solver)


class TestValidation:
    def test_bad_sitekey(self):
        with _client() as c:
            r = c.post(
                "/v1/solve",
                json={"sitekey": "not-a-uuid", "siteurl": VALID_SITEURL},
                headers={"Authorization": "Bearer secret"},
            )
            assert r.status_code == 422

    def test_bad_siteurl(self):
        with _client() as c:
            r = c.post(
                "/v1/solve",
                json={"sitekey": VALID_SITEKEY, "siteurl": "ftp://x"},
                headers={"Authorization": "Bearer secret"},
            )
            assert r.status_code == 422


class TestHealth:
    def test_healthz_no_auth(self):
        with _client() as c:
            assert c.get("/healthz").json() == {"status": "ok"}

    def test_readyz(self):
        with _client() as c:
            r = c.get("/readyz")
            assert r.status_code == 200
            assert r.json()["status"] == "ready"

    def test_stats_requires_auth(self):
        with _client() as c:
            assert c.get("/v1/stats").status_code == 401
            r = c.get("/v1/stats", headers={"Authorization": "Bearer secret"})
            assert r.status_code == 200
            assert r.json()["max_concurrent"] == 1


class TestRateLimit:
    def test_rate_limit_trips(self):
        with _client(_settings(RATE_LIMIT_RPM=1)) as c:
            h = {"Authorization": "Bearer secret"}
            body = {"sitekey": VALID_SITEKEY, "siteurl": VALID_SITEURL}
            assert c.post("/v1/solve", json=body, headers=h).status_code == 200
            assert c.post("/v1/solve", json=body, headers=h).status_code == 429


class TestQueueManager:
    """Unit-test the admission/queue logic directly."""

    def test_queue_full_rejects(self):
        async def run():
            mgr = SolveManager(max_concurrent=1, max_queue=0)
            gate = asyncio.Event()

            async def slow():
                await gate.wait()
                return "done"

            # occupy the single slot
            task = asyncio.create_task(
                mgr.submit(slow, wait_timeout=5, solve_timeout=5)
            )
            await asyncio.sleep(0.05)
            # no room (concurrent=1, queue=0) -> QueueFull
            with pytest.raises(QueueFull):
                await mgr.submit(slow, wait_timeout=5, solve_timeout=5)
            gate.set()
            assert await task == "done"

        asyncio.run(run())

    def test_queue_timeout(self):
        async def run():
            mgr = SolveManager(max_concurrent=1, max_queue=5)
            gate = asyncio.Event()

            async def slow():
                await gate.wait()
                return "done"

            task = asyncio.create_task(mgr.submit(slow, wait_timeout=5, solve_timeout=5))
            await asyncio.sleep(0.05)
            # admitted into queue but never gets a slot in time
            with pytest.raises(QueueTimeout):
                await mgr.submit(slow, wait_timeout=0.1, solve_timeout=5)
            gate.set()
            assert await task == "done"

        asyncio.run(run())

    def test_solve_timeout(self):
        async def run():
            mgr = SolveManager(max_concurrent=1, max_queue=0)

            async def forever():
                await asyncio.sleep(10)

            with pytest.raises(asyncio.TimeoutError):
                await mgr.submit(forever, wait_timeout=5, solve_timeout=0.1)
            # slot is released after timeout
            assert mgr.active == 0

        asyncio.run(run())

    def test_stats_shape(self):
        mgr = SolveManager(max_concurrent=2, max_queue=4)
        s = mgr.stats()
        assert s == {
            "active": 0,
            "queued": 0,
            "max_concurrent": 2,
            "max_queue": 4,
            "capacity_used": 0.0,
        }
