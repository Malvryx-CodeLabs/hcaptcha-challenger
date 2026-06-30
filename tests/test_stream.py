# -*- coding: utf-8 -*-
"""Tests for the debug live-stream feature."""
import asyncio

from fastapi.testclient import TestClient

from hcaptcha_challenger.api.app import create_app
from hcaptcha_challenger.api.config import ApiSettings
from hcaptcha_challenger.api.stream import StreamHub, create_stream_app


def _jpeg(b: bytes) -> bool:
    return b[:3] == b"\xff\xd8\xff"


def test_hub_starts_with_idle_placeholder():
    hub = StreamHub(fps=6, max_width=640, quality=60)
    assert _jpeg(hub.snapshot)
    assert hub.status == "No Task"


def test_hub_active_then_push_then_idle():
    hub = StreamHub(max_width=640)
    idle = hub.snapshot

    hub.set_active("Solving abcd1234… · example.com")
    assert hub.status.startswith("Solving")
    assert _jpeg(hub.snapshot)  # banner frame is a real jpeg

    hub.push(b"\xff\xd8\xff-live-frame")
    assert hub.snapshot == b"\xff\xd8\xff-live-frame"

    hub.set_idle()
    assert hub.status == "No Task"
    assert hub.snapshot == idle


def test_push_ignores_empty():
    hub = StreamHub(max_width=640)
    before = hub.snapshot
    hub.push(b"")
    assert hub.snapshot == before


def test_mjpeg_yields_multipart_chunk():
    hub = StreamHub(fps=30, max_width=640)

    async def first_chunk():
        agen = hub.mjpeg()
        chunk = await agen.__anext__()
        await agen.aclose()
        return chunk

    chunk = asyncio.run(first_chunk())
    assert chunk.startswith(b"--frame")
    assert b"Content-Type: image/jpeg" in chunk
    assert hub.snapshot in chunk


def test_stream_app_endpoints():
    hub = StreamHub(max_width=640)
    app = create_stream_app(hub)
    with TestClient(app) as c:
        r = c.get("/snapshot.jpg")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/jpeg"
        assert _jpeg(r.content)

        r = c.get("/")
        assert r.status_code == 200
        assert "stream.mjpeg" in r.text

        r = c.get("/healthz")
        assert r.json()["status"] == "ok"


def test_stream_forces_single_concurrency():
    settings = ApiSettings(API_KEYS=["k"], MAX_CONCURRENT_SOLVES=8, STREAM_ENABLED=True)
    hub = StreamHub()

    async def fake(sk, su, rq):
        return {"success": True, "token": "t", "expiration": None, "error": None, "raw": None}

    app = create_app(settings, solve_func=fake, stream_hub=hub)
    assert app.state.manager.max_concurrent == 1
