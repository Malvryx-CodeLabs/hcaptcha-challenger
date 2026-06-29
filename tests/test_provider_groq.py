# -*- coding: utf-8 -*-
"""Offline tests for the Groq provider and provider selection (no network)."""
import base64
import json
from pathlib import Path

import pytest

from hcaptcha_challenger.models import ImageBinaryChallenge, LLMProvider
from hcaptcha_challenger.tools.internal.providers.groq import (
    GroqProvider,
    extract_first_json_block,
)


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
