"""Tests for ModelReference and ProviderRouter.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import pytest

from openagent.core.presets import PRESETS
from openagent.core.router import ModelReference, ProviderRouter
from openagent.providers.anthropic import AnthropicProvider
from openagent.providers.custom import CustomJsonPathProvider
from openagent.providers.gemini import GeminiProvider
from openagent.providers.ollama import OllamaProvider
from openagent.providers.openai_compat import OpenAICompatProvider


def test_model_reference_parse_simple() -> None:
    ref = ModelReference.parse("gpt-4o")
    assert ref.provider_hint is None
    assert ref.model_name == "gpt-4o"
    assert ref.raw == "gpt-4o"


def test_model_reference_parse_slash_prefix() -> None:
    ref = ModelReference.parse("anthropic/claude-3-5-sonnet")
    assert ref.provider_hint == "anthropic"
    assert ref.model_name == "claude-3-5-sonnet"
    assert ref.raw == "anthropic/claude-3-5-sonnet"


def test_model_reference_parse_colon_prefix() -> None:
    ref = ModelReference.parse("ollama:llama3.2")
    assert ref.provider_hint == "ollama"
    assert ref.model_name == "llama3.2"
    assert ref.raw == "ollama:llama3.2"


def test_model_reference_parse_nested_slash() -> None:
    ref = ModelReference.parse("openrouter/anthropic/claude-sonnet-4.5")
    assert ref.provider_hint == "openrouter"
    assert ref.model_name == "anthropic/claude-sonnet-4.5"
    assert ref.raw == "openrouter/anthropic/claude-sonnet-4.5"


def test_model_reference_parse_tagged_slash() -> None:
    ref = ModelReference.parse("ollama/llama3.2:latest")
    assert ref.provider_hint == "ollama"
    assert ref.model_name == "llama3.2:latest"
    assert ref.raw == "ollama/llama3.2:latest"

    # Router should route this to OllamaProvider, not OpenAI fallback
    router = ProviderRouter()
    provider = router.resolve("ollama/llama3.2:latest")
    assert isinstance(provider, OllamaProvider)
    assert provider.model == "llama3.2:latest"


def test_router_resolve_preset_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-mock-anthropic")
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-key")

    router = ProviderRouter()

    # Exact preset name
    p_openai = router.resolve("openai")
    assert isinstance(p_openai, OpenAICompatProvider)
    assert p_openai.model == PRESETS["openai"].default_model

    # Anthropic
    p_ant = router.resolve("anthropic")
    assert isinstance(p_ant, AnthropicProvider)
    assert p_ant.model == PRESETS["anthropic"].default_model


def test_router_resolve_model_inference() -> None:
    router = ProviderRouter()

    # Model name starting with claude -> Anthropic
    p_claude = router.resolve("claude-3-5-sonnet", api_key="test-key")
    assert isinstance(p_claude, AnthropicProvider)
    assert p_claude.model == "claude-3-5-sonnet"

    # Model name starting with gemini -> Gemini
    p_gemini = router.resolve("gemini-1.5-pro", api_key="test-key")
    assert isinstance(p_gemini, GeminiProvider)
    assert p_gemini.model == "gemini-1.5-pro"

    # Model name starting with gpt -> OpenAI
    p_gpt = router.resolve("gpt-4o", api_key="test-key")
    assert isinstance(p_gpt, OpenAICompatProvider)
    assert p_gpt.model == "gpt-4o"

    # DeepSeek preset default / name
    p_deepseek = router.resolve("deepseek-chat", api_key="test-key")
    assert isinstance(p_deepseek, OpenAICompatProvider)
    assert p_deepseek.model == "deepseek-chat"
    assert "deepseek.com" in p_deepseek.base_url


def test_router_resolve_provider_hints() -> None:
    router = ProviderRouter()

    # ollama hint
    p_ollama = router.resolve("ollama/llama3.2")
    assert isinstance(p_ollama, OllamaProvider)
    assert p_ollama.model == "llama3.2"

    # custom hint
    p_custom = router.resolve("custom/my-model", base_url="http://localhost:9000")
    assert isinstance(p_custom, CustomJsonPathProvider)
    assert p_custom.model == "my-model"

    # groq hint
    p_groq = router.resolve("groq/llama-3.3-70b-versatile", api_key="mock-groq")
    assert isinstance(p_groq, OpenAICompatProvider)
    assert p_groq.model == "llama-3.3-70b-versatile"
    assert "groq.com" in p_groq.base_url


def test_router_env_api_key_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-anthropic-key")
    monkeypatch.setenv("GEMINI_API_KEY", "env-gemini-key")
    monkeypatch.setenv("GROQ_API_KEY", "env-groq-key")

    router = ProviderRouter()

    p1 = router.resolve("gpt-4o")
    assert isinstance(p1, OpenAICompatProvider)
    assert p1.transport.extra_headers.get("Authorization") == "Bearer env-openai-key"

    p2 = router.resolve("claude-3-5-sonnet")
    assert isinstance(p2, AnthropicProvider)
    assert p2._transport.extra_headers.get("x-api-key") == "env-anthropic-key"

    p3 = router.resolve("gemini-2.0-flash")
    assert isinstance(p3, GeminiProvider)
    assert p3._transport.extra_headers.get("x-goog-api-key") == "env-gemini-key"

    p4 = router.resolve("groq/llama-3.3-70b-versatile")
    assert isinstance(p4, OpenAICompatProvider)
    assert p4.transport.extra_headers.get("Authorization") == "Bearer env-groq-key"


def test_router_custom_base_url_and_fallback() -> None:
    router = ProviderRouter()

    # Base URL override on preset
    p = router.resolve("gpt-4o", base_url="https://my-proxy.com/v1", api_key="key")
    assert isinstance(p, OpenAICompatProvider)
    assert p.base_url == "https://my-proxy.com/v1"

    # Unrecognized model without hint -> OpenAICompat fallback
    p_fallback = router.resolve("unrecognized-exotic-model-123", api_key="test")
    assert isinstance(p_fallback, OpenAICompatProvider)
    assert p_fallback.model == "unrecognized-exotic-model-123"
    assert p_fallback.base_url == "https://api.openai.com/v1"
