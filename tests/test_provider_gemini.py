# -*- coding: utf-8 -*-
"""Tests for Gemini multi-key + multi-model rotation."""
import asyncio

import pytest
from pydantic import BaseModel

import hcaptcha_challenger.tools.internal.providers.gemini as gem


class Ans(BaseModel):
    color: str


class _Err(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


class _Resp:
    def __init__(self, parsed=None, text=None):
        self.parsed = parsed
        self.text = text


def _install_fake(monkeypatch, behavior):
    """behavior(model, key) -> _Resp or raises. Returns the call log."""
    log = []

    class _Models:
        def __init__(self, key):
            self.key = key

        async def generate_content(self, *, model, contents, config):
            log.append((model, self.key))
            return behavior(model, self.key)

    class _Aio:
        def __init__(self, key):
            self.models = _Models(key)

    class _Client:
        def __init__(self, *, api_key, http_options=None):
            self.aio = _Aio(api_key)

    monkeypatch.setattr(gem.genai, "Client", _Client)
    return log


def test_single_key_single_model(monkeypatch):
    log = _install_fake(monkeypatch, lambda m, k: _Resp({"color": "red"}))
    p = gem.GeminiProvider(api_key="k1", model="mA")
    out = asyncio.run(p.generate_with_images(images=[], response_schema=Ans))
    assert out.color == "red"
    assert log == [("mA", "k1")]


def test_rotates_keys_then_model_on_rate_limit(monkeypatch):
    def behavior(model, key):
        if model == "mA":
            raise _Err(429, "RESOURCE_EXHAUSTED")
        return _Resp({"color": "green"})

    log = _install_fake(monkeypatch, behavior)
    p = gem.GeminiProvider(api_key="k1,k2", model="mA,mB")
    out = asyncio.run(p.generate_with_images(images=[], response_schema=Ans))
    assert out.color == "green"
    # both keys tried on mA, then mB succeeds
    assert log == [("mA", "k1"), ("mA", "k2"), ("mB", "k1")]


def test_unavailable_model_skips_immediately(monkeypatch):
    def behavior(model, key):
        if model == "mBad":
            raise _Err(404, "model not found")
        return _Resp({"color": "blue"})

    log = _install_fake(monkeypatch, behavior)
    p = gem.GeminiProvider(api_key="k1,k2", model="mBad,mGood")
    out = asyncio.run(p.generate_with_images(images=[], response_schema=Ans))
    assert out.color == "blue"
    # did NOT waste both keys on the unavailable model
    assert log == [("mBad", "k1"), ("mGood", "k2")]


def test_other_error_not_masked(monkeypatch):
    log = _install_fake(monkeypatch, lambda m, k: (_ for _ in ()).throw(_Err(401, "bad key")))
    p = gem.GeminiProvider(api_key="k1,k2", model="mA,mB")
    with pytest.raises(_Err):
        asyncio.run(p.generate_with_images(images=[], response_schema=Ans))
    # 401 is 'other' -> surfaced on the first attempt, no rotation
    assert log == [("mA", "k1")]


def test_all_exhausted_raises_last(monkeypatch):
    log = _install_fake(monkeypatch, lambda m, k: (_ for _ in ()).throw(_Err(429, "quota")))
    p = gem.GeminiProvider(api_key="k1,k2", model="mA,mB")
    with pytest.raises(_Err):
        asyncio.run(p.generate_with_images(images=[], response_schema=Ans))
    # every key on every model attempted
    assert log == [("mA", "k1"), ("mA", "k2"), ("mB", "k1"), ("mB", "k2")]


def test_normalize_keys_and_models():
    p = gem.GeminiProvider(api_key=" k1 , k2 ,", model="mA, mB ,")
    assert p._keys == ["k1", "k2"]
    assert p._models == ["mA", "mB"]


def test_config_gemini_models_fills_all_tasks(monkeypatch):
    from hcaptcha_challenger.agent.challenger import AgentConfig

    chain = "gemini-3.5-flash,gemini-3-flash,gemini-2.5-flash"
    cfg = AgentConfig(LLM_PROVIDER="gemini", GEMINI_API_KEY="k1,k2", GEMINI_MODELS=chain)
    assert cfg.IMAGE_CLASSIFIER_MODEL == chain
    assert cfg.SPATIAL_POINT_REASONER_MODEL == chain
    assert cfg.SPATIAL_PATH_REASONER_MODEL == chain
    assert cfg.CHALLENGE_CLASSIFIER_MODEL == chain
