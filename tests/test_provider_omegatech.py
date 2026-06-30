# -*- coding: utf-8 -*-
"""Tests for the Omegatech provider and its wiring."""
import asyncio

import pytest

from hcaptcha_challenger.models import ImageBinaryChallenge


class _FakeResp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.headers = {"content-type": "application/json"}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError("err", request=None, response=None)


class _FakeClient:
    """Fake httpx.AsyncClient covering upload (post), complete (get), delete (request)."""

    def __init__(self, calls, answer):
        self._calls = calls
        self._answer = answer

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, files=None, data=None, headers=None, **k):
        name = files["file"][0] if files else None
        self._calls.append(("POST", url, name))
        return _FakeResp(200, {"id": "om_1", "cdnUrl": "https://cdn/last.png", "deleteToken": "d1"})

    async def get(self, url, params=None, **k):
        self._calls.append(("GET", url, params))
        return _FakeResp(200, {"success": True, "answer": self._answer})

    async def request(self, method, url, headers=None, **k):
        self._calls.append((method, url, None))
        return _FakeResp(200, {"success": True})


def _patch(monkeypatch, calls, answer):
    import hcaptcha_challenger.tools.internal.providers.omegatech as om

    monkeypatch.setattr(om.httpx, "AsyncClient", lambda *a, **k: _FakeClient(calls, answer))
    return om


def test_uploads_last_image_sends_url_and_deletes(monkeypatch, tmp_path):
    calls = []
    answer = '{"challenge_prompt": "x", "coordinates": [{"box_2d": [0,1]}]}'
    om = _patch(monkeypatch, calls, answer)

    a = tmp_path / "clean.png"
    b = tmp_path / "grid.png"
    a.write_bytes(b"\x89PNGclean")
    b.write_bytes(b"\x89PNGgrid")

    prov = om.OmegatechProvider(model="Gpt-4-mini")
    out = asyncio.run(
        prov.generate_with_images(images=[a, b], response_schema=ImageBinaryChallenge)
    )

    assert isinstance(out, ImageBinaryChallenge)
    assert out.coordinates[0].box_2d == [0, 1]

    # Uploaded the LAST (grid) image, not the clean one.
    uploads = [c for c in calls if c[0] == "POST"]
    assert uploads and uploads[0][2] == "grid.png"

    # GET went to base/model with the CDN url as imageUrl.
    gets = [c for c in calls if c[0] == "GET"]
    assert gets[0][1].endswith("/Gpt-4-mini")
    assert gets[0][2]["imageUrl"] == "https://cdn/last.png"

    # Deleted afterwards.
    assert any(c[0] == "DELETE" for c in calls)


def test_parses_answer_with_json_fence(monkeypatch, tmp_path):
    calls = []
    answer = 'Sure!\n```json\n{"challenge_prompt": "y", "coordinates": [{"box_2d": [2,2]}]}\n```'
    om = _patch(monkeypatch, calls, answer)

    img = tmp_path / "g.png"
    img.write_bytes(b"x")
    prov = om.OmegatechProvider()
    out = asyncio.run(prov.generate_with_images(images=[img], response_schema=ImageBinaryChallenge))
    assert out.coordinates[0].box_2d == [2, 2]


def test_empty_answer_raises(monkeypatch, tmp_path):
    calls = []
    om = _patch(monkeypatch, calls, "")
    img = tmp_path / "g.png"
    img.write_bytes(b"x")
    prov = om.OmegatechProvider()
    with pytest.raises(Exception):
        asyncio.run(prov.generate_with_images(images=[img], response_schema=ImageBinaryChallenge))


def test_config_requires_no_key_and_fills_models(monkeypatch):
    from hcaptcha_challenger.agent.challenger import AgentConfig

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    cfg = AgentConfig(LLM_PROVIDER="omegatech")
    assert cfg.IMAGE_CLASSIFIER_MODEL == "Gpt-4-mini"
    assert cfg.SPATIAL_POINT_REASONER_MODEL == "Gpt-4-mini"
    assert cfg.OMEGATECH_BASE_URL.endswith("/api/ai")


def test_base_dispatch_creates_provider():
    from hcaptcha_challenger.tools.internal.base import Reasoner
    from hcaptcha_challenger.tools.internal.providers.omegatech import OmegatechProvider

    class _Dummy(Reasoner):
        async def __call__(self):  # pragma: no cover
            ...

    r = _Dummy("", "Gpt-4-mini", provider_type="omegatech")
    assert isinstance(r._provider, OmegatechProvider)
