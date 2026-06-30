# -*- coding: utf-8 -*-
"""Offline tests for the Groq provider and provider selection (no network)."""
import asyncio
import base64
import json
from pathlib import Path

import pytest

from hcaptcha_challenger.models import (
    BoundingBoxCoordinate,
    ImageBinaryChallenge,
    LLMProvider,
)
from hcaptcha_challenger.tools.internal.providers.groq import (
    GroqProvider,
    extract_first_json_block,
)


class TestCoordinateRepair:
    """Weaker OpenAI-compatible models merge [row, col] into one token."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            (["00"], [0, 0]),
            ([10], [1, 0]),
            ([22], [2, 2]),
            (["01"], [0, 1]),
            ("0,0", [0, 0]),
            (["2,1"], [2, 1]),
            ([0, 2], [0, 2]),  # already well-formed -> untouched
            ([1, 1], [1, 1]),
        ],
    )
    def test_repair(self, raw, expected):
        assert BoundingBoxCoordinate(box_2d=raw).box_2d == expected

    def test_full_binary_payload_with_merged_coords(self):
        """The exact failure from the live run: all 9 cells merged."""
        merged = [["00"], [10], [20], ["01"], [11], [21], ["02"], [12], [22]]
        challenge = ImageBinaryChallenge(
            challenge_prompt="p", coordinates=[{"box_2d": c} for c in merged]
        )
        assert [c.box_2d for c in challenge.coordinates] == [
            [0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2], [1, 2], [2, 2]
        ]


@pytest.fixture
def png_path(tmp_path: Path) -> Path:
    """A tiny valid 1x1 PNG on disk."""
    # 1x1 transparent PNG
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M8AAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    )
    p = tmp_path / "tiny.png"
    p.write_bytes(png_bytes)
    return p


class TestExtractJsonBlock:
    def test_fenced_block(self):
        text = 'prefix\n```json\n{"a": 1}\n```\nsuffix'
        assert extract_first_json_block(text) == {"a": 1}

    def test_bare_object(self):
        assert extract_first_json_block('noise {"b": 2} tail') == {"b": 2}

    def test_no_json(self):
        assert extract_first_json_block("no json here") is None


class TestGroqEncoding:
    def test_encode_image_data_uri(self, png_path: Path):
        provider = GroqProvider(api_key="gsk_test", model="m")
        uri = provider._encode_image(png_path)
        assert uri.startswith("data:image/png;base64,")
        # round-trips back to the original bytes
        b64 = uri.split(",", 1)[1]
        assert base64.b64decode(b64) == png_path.read_bytes()

    def test_image_too_large_rejected(self, tmp_path: Path):
        provider = GroqProvider(api_key="gsk_test", model="m")
        big = tmp_path / "big.png"
        big.write_bytes(b"\x00" * (5 * 1024 * 1024))  # >4MB
        with pytest.raises(ValueError, match="4MB"):
            provider._encode_image(big)

    def test_too_many_images_rejected(self, png_path: Path):
        provider = GroqProvider(api_key="gsk_test", model="m")
        with pytest.raises(ValueError, match="at most"):
            provider._build_image_parts([png_path] * 6)

    def test_build_messages_shape(self, png_path: Path):
        provider = GroqProvider(api_key="gsk_test", model="m")
        parts = provider._build_image_parts([png_path])
        messages = provider._build_messages(
            image_parts=parts, user_prompt="solve it", description="system rules"
        )
        assert messages[0] == {"role": "system", "content": "system rules"}
        user = messages[1]
        assert user["role"] == "user"
        assert user["content"][0]["type"] == "image_url"
        assert user["content"][-1] == {"type": "text", "text": "solve it"}


class TestGroqParse:
    def test_parse_plain_json_content(self):
        data = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"challenge_prompt": "p", "coordinates": [{"box_2d": [0, 0]}]}
                        )
                    }
                }
            ]
        }
        result = GroqProvider._parse(data, ImageBinaryChallenge)
        assert isinstance(result, ImageBinaryChallenge)
        assert result.coordinates[0].box_2d == [0, 0]

    def test_parse_fenced_content(self):
        content = '```json\n{"challenge_prompt": "p", "coordinates": []}\n```'
        data = {"choices": [{"message": {"content": content}}]}
        result = GroqProvider._parse(data, ImageBinaryChallenge)
        assert result.coordinates == []

    def test_parse_empty_raises(self):
        data = {"choices": [{"message": {"content": ""}}]}
        with pytest.raises(ValueError, match="Empty content"):
            GroqProvider._parse(data, ImageBinaryChallenge)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeAsyncClient:
    """Minimal async httpx.AsyncClient stand-in that records POSTs."""

    last_url = None
    last_json = None
    response_payload = {}

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, **kwargs):
        type(self).last_url = url
        type(self).last_json = kwargs.get("json")
        return _FakeResponse(type(self).response_payload)


class TestAikitTokenRefresh:
    def test_refresh_updates_token_and_expiry(self, monkeypatch):
        import hcaptcha_challenger.tools.internal.providers.aikit as aikit

        _FakeAsyncClient.response_payload = {
            "expires_at": 1782755670,
            "access_token": "H4sIAAAA_NEW_TOKEN",
        }
        monkeypatch.setattr(aikit.httpx, "AsyncClient", _FakeAsyncClient)

        p = aikit.AikitProvider(api_key="H4sIAAAA_OLD", model="qwen-max-latest")
        ok = asyncio.run(p._refresh_token())
        assert ok is True
        assert p._api_key == "H4sIAAAA_NEW_TOKEN"
        assert p._expires_at == 1782755670
        # refresh sends the *current* token to /v1/refresh
        assert _FakeAsyncClient.last_url.endswith("/v1/refresh")
        assert _FakeAsyncClient.last_json == {"token": "H4sIAAAA_OLD"}

    def test_ensure_fresh_refreshes_when_expired(self, monkeypatch):
        import hcaptcha_challenger.tools.internal.providers.aikit as aikit

        p = aikit.AikitProvider(api_key="t", model="qwen-max-latest", expires_at=1)  # epoch -> expired
        called = {"n": 0}

        async def fake_refresh():
            called["n"] += 1
            return True

        monkeypatch.setattr(p, "_refresh_token", fake_refresh)
        asyncio.run(p._ensure_fresh())
        assert called["n"] == 1

    def test_ensure_fresh_skips_when_no_expiry(self, monkeypatch):
        import hcaptcha_challenger.tools.internal.providers.aikit as aikit

        p = aikit.AikitProvider(api_key="t", model="qwen-max-latest")  # expires_at=None
        called = {"n": 0}

        async def fake_refresh():
            called["n"] += 1
            return True

        monkeypatch.setattr(p, "_refresh_token", fake_refresh)
        asyncio.run(p._ensure_fresh())
        assert called["n"] == 0

    def test_base_url_default(self):
        import hcaptcha_challenger.tools.internal.providers.aikit as aikit

        p = aikit.AikitProvider(api_key="t", model="qwen-max-latest")
        assert p._base_url == "https://qwen.aikit.club/v1"


class TestAikitConfig:
    def test_requires_token(self, monkeypatch):
        from hcaptcha_challenger.agent.challenger import AgentConfig

        monkeypatch.delenv("AIKIT_API_KEY", raising=False)
        monkeypatch.delenv("AIKIT_TOKEN", raising=False)
        with pytest.raises(ValueError, match="AIKIT_API_KEY"):
            AgentConfig(LLM_PROVIDER="aikit", AIKIT_API_KEY="")

    def test_defaults_to_vision_qwen_model(self, monkeypatch):
        from hcaptcha_challenger.agent.challenger import AgentConfig

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        cfg = AgentConfig(LLM_PROVIDER="aikit", AIKIT_API_KEY="H4sIAAAA_x")
        assert cfg.IMAGE_CLASSIFIER_MODEL == "qwen-max-latest"
        assert cfg.SPATIAL_PATH_REASONER_MODEL == "qwen-max-latest"
        assert cfg.AIKIT_BASE_URL == "https://qwen.aikit.club/v1"
        assert cfg.AIKIT_AUTO_REFRESH is True


class TestAgentConfigProviderSelection:
    def test_groq_requires_key(self, monkeypatch):
        from hcaptcha_challenger.agent.challenger import AgentConfig

        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            AgentConfig(LLM_PROVIDER="groq", GROQ_API_KEY="")

    def test_groq_swaps_default_models(self, monkeypatch):
        from hcaptcha_challenger.agent.challenger import AgentConfig

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        cfg = AgentConfig(LLM_PROVIDER="groq", GROQ_API_KEY="gsk_test")
        assert cfg.LLM_PROVIDER == LLMProvider.GROQ
        assert "llama-4" in cfg.IMAGE_CLASSIFIER_MODEL
        assert "llama-4" in cfg.SPATIAL_PATH_REASONER_MODEL
        assert "llama-4" in cfg.CHALLENGE_CLASSIFIER_MODEL
        # No Gemini key needed when using Groq
        assert cfg.GEMINI_API_KEY.get_secret_value() == ""

    def test_groq_respects_explicit_model(self, monkeypatch):
        from hcaptcha_challenger.agent.challenger import AgentConfig

        cfg = AgentConfig(
            LLM_PROVIDER="groq",
            GROQ_API_KEY="gsk_test",
            IMAGE_CLASSIFIER_MODEL="meta-llama/llama-4-scout-17b-16e-instruct",
        )
        assert cfg.IMAGE_CLASSIFIER_MODEL == "meta-llama/llama-4-scout-17b-16e-instruct"

    def test_gemini_still_requires_key(self, monkeypatch):
        from hcaptcha_challenger.agent.challenger import AgentConfig

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            AgentConfig(LLM_PROVIDER="gemini", GEMINI_API_KEY="")

    def test_gemini_base_url_accepted(self):
        from hcaptcha_challenger.agent.challenger import AgentConfig

        cfg = AgentConfig(
            LLM_PROVIDER="gemini",
            GEMINI_API_KEY="x",
            GEMINI_BASE_URL="https://my-proxy.example.com",
        )
        assert cfg.GEMINI_BASE_URL == "https://my-proxy.example.com"


class TestOpenAICompatibleConfig:
    def test_requires_key_and_base_url(self, monkeypatch):
        from hcaptcha_challenger.agent.challenger import AgentConfig

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            AgentConfig(LLM_PROVIDER="openai", OPENAI_API_KEY="", OPENAI_MODEL="qwen-vl-max")
        with pytest.raises(ValueError, match="OPENAI_BASE_URL"):
            AgentConfig(
                LLM_PROVIDER="openai",
                OPENAI_API_KEY="sk-x",
                OPENAI_BASE_URL="",
                OPENAI_MODEL="qwen-vl-max",
            )

    def test_rejects_leftover_gemini_models(self):
        from hcaptcha_challenger.agent.challenger import AgentConfig

        with pytest.raises(ValueError, match="Gemini\n?.*defaults|Gemini defaults"):
            AgentConfig(
                LLM_PROVIDER="openai",
                OPENAI_API_KEY="sk-x",
                OPENAI_BASE_URL="https://host/v1",
                # no OPENAI_MODEL and no per-field override -> should error
            )

    def test_openai_model_fills_all_fields(self, monkeypatch):
        from hcaptcha_challenger.agent.challenger import AgentConfig

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        cfg = AgentConfig(
            LLM_PROVIDER="openai",
            OPENAI_API_KEY="sk-x",
            OPENAI_BASE_URL="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            OPENAI_MODEL="qwen-vl-max",
        )
        assert cfg.IMAGE_CLASSIFIER_MODEL == "qwen-vl-max"
        assert cfg.SPATIAL_POINT_REASONER_MODEL == "qwen-vl-max"
        assert cfg.SPATIAL_PATH_REASONER_MODEL == "qwen-vl-max"
        assert cfg.CHALLENGE_CLASSIFIER_MODEL == "qwen-vl-max"
