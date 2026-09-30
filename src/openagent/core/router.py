"""Provider router and model reference parsing.

Resolves model names, aliases, and provider-prefixed references (e.g.
`gpt-4o`, `claude-3-5-sonnet`, `gemini-1.5-pro`, `ollama/llama3.2`, `custom/my-model`)
into configured ChatProvider instances.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .presets import ProviderPreset, all_presets
from .provider import ChatProvider


@dataclass(slots=True, frozen=True)
class ModelReference:
    """Parsed model specification containing an optional provider hint and model name."""

    provider_hint: str | None
    model_name: str
    raw: str

    @classmethod
    def parse(cls, ref: str) -> ModelReference:
        raw = ref.strip()
        # Prioritize '/' so namespaces like 'ollama/llama3.2:latest' or 'openrouter/anthropic/claude'
        # are correctly split into provider_hint='ollama' and model_name='llama3.2:latest'
        if "/" in raw:
            hint, model = raw.split("/", 1)
            return cls(provider_hint=hint.strip() or None, model_name=model.strip(), raw=raw)
        if ":" in raw:
            hint, model = raw.split(":", 1)
            return cls(provider_hint=hint.strip() or None, model_name=model.strip(), raw=raw)
        return cls(provider_hint=None, model_name=raw, raw=raw)


class ProviderRouter:
    """Routes model references to instantiated ChatProvider implementations."""

    def __init__(self, presets: Mapping[str, ProviderPreset] | None = None) -> None:
        self._presets = dict(presets if presets is not None else all_presets())

    def _find_preset(self, name: str) -> ProviderPreset | None:
        key = name.strip().lower().replace("_", "-")
        if key in self._presets:
            return self._presets[key]
        for preset in self._presets.values():
            if key in preset.aliases:
                return preset
        for candidate in (key.rstrip("s"), f"{key}s"):
            if candidate in self._presets:
                return self._presets[candidate]
        return None

    def resolve(
        self,
        model_ref: str | ModelReference,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        extra_headers: dict[str, str] | Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> ChatProvider:
        """Resolve a model name or reference into a ChatProvider instance."""
        ref = ModelReference.parse(model_ref) if isinstance(model_ref, str) else model_ref

        # 1. If provider hint is explicitly 'custom'
        if ref.provider_hint == "custom":
            from ..providers.custom import CustomJsonPathProvider

            return CustomJsonPathProvider(
                base_url=base_url or "http://localhost:8000",
                model=ref.model_name,
                api_key=api_key or os.environ.get("CUSTOM_API_KEY"),
                headers=extra_headers,
                **kwargs,
            )

        # 2. If provider hint matches a known preset
        if ref.provider_hint:
            preset = self._find_preset(ref.provider_hint)
            if preset is not None:
                return self._instantiate_preset(
                    preset,
                    model=ref.model_name,
                    api_key=api_key,
                    base_url=base_url,
                    extra_headers=extra_headers,
                    **kwargs,
                )
            if ref.provider_hint == "ollama":
                from ..providers.ollama import OllamaProvider

                return OllamaProvider(
                    model=ref.model_name,
                    base_url=base_url or "http://localhost:11434",
                    api_key=api_key,
                    extra_headers=extra_headers,
                    **kwargs,
                )
            if ref.provider_hint == "gemini":
                from ..providers.gemini import GeminiProvider

                key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                return GeminiProvider(
                    model=ref.model_name,
                    base_url=base_url or "https://generativelanguage.googleapis.com/v1beta",
                    api_key=key,
                    extra_headers=extra_headers,
                    **kwargs,
                )
            if ref.provider_hint == "anthropic":
                from ..providers.anthropic import AnthropicProvider

                key = api_key or os.environ.get("ANTHROPIC_API_KEY")
                return AnthropicProvider(
                    model=ref.model_name,
                    base_url=base_url or "https://api.anthropic.com",
                    api_key=key,
                    extra_headers=extra_headers,
                    **kwargs,
                )
            # Unrecognized provider hint -> default to OpenAICompatProvider
            from ..providers.openai_compat import OpenAICompatProvider

            env_name = f"{ref.provider_hint.upper().replace('-', '_')}_API_KEY"
            env_key = os.environ.get(env_name)
            key = api_key or env_key or os.environ.get("OPENAI_API_KEY")
            return OpenAICompatProvider(
                base_url=base_url or "https://api.openai.com/v1",
                model=ref.model_name,
                api_key=key,
                provider_name=ref.provider_hint,
                headers=extra_headers,
                **kwargs,
            )

        # 3. No provider hint: check if model_name matches a preset name / alias directly
        preset = self._find_preset(ref.model_name)
        if preset is not None:
            model = preset.default_model or ref.model_name
            return self._instantiate_preset(
                preset,
                model=model,
                api_key=api_key,
                base_url=base_url,
                extra_headers=extra_headers,
                **kwargs,
            )

        # 4. Check if any preset has default_model == ref.model_name
        model_lower = ref.model_name.lower()
        for p in self._presets.values():
            if p.default_model and p.default_model.lower() == model_lower:
                return self._instantiate_preset(
                    p,
                    model=ref.model_name,
                    api_key=api_key,
                    base_url=base_url,
                    extra_headers=extra_headers,
                    **kwargs,
                )

        # 5. Model prefix heuristics
        if model_lower.startswith("claude"):
            from ..providers.anthropic import AnthropicProvider

            ant_preset = self._find_preset("anthropic")
            base = base_url or (ant_preset.base_url if ant_preset else "https://api.anthropic.com")
            key = api_key or (ant_preset.auth.resolve_key() if ant_preset else os.environ.get("ANTHROPIC_API_KEY"))
            return AnthropicProvider(
                model=ref.model_name,
                base_url=base,
                api_key=key,
                extra_headers=extra_headers,
                **kwargs,
            )

        if model_lower.startswith("gemini"):
            from ..providers.gemini import GeminiProvider

            gem_preset = self._find_preset("gemini")
            base = base_url or (
                gem_preset.base_url if gem_preset else "https://generativelanguage.googleapis.com/v1beta"
            )
            key = api_key or (
                gem_preset.auth.resolve_key()
                if gem_preset
                else (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
            )
            return GeminiProvider(
                model=ref.model_name,
                base_url=base,
                api_key=key,
                extra_headers=extra_headers,
                **kwargs,
            )

        if model_lower.startswith(("gpt-", "o1", "o3", "o4", "chatgpt")):
            from ..providers.openai_compat import OpenAICompatProvider

            openai_preset = self._find_preset("openai")
            base = base_url or (openai_preset.base_url if openai_preset else "https://api.openai.com/v1")
            key = api_key or (
                openai_preset.auth.resolve_key() if openai_preset else os.environ.get("OPENAI_API_KEY")
            )
            return OpenAICompatProvider(
                base_url=base,
                model=ref.model_name,
                api_key=key,
                provider_name="openai",
                headers=extra_headers,
                **kwargs,
            )

        if model_lower.startswith("deepseek"):
            from ..providers.openai_compat import OpenAICompatProvider

            ds_preset = self._find_preset("deepseek")
            base = base_url or (ds_preset.base_url if ds_preset else "https://api.deepseek.com/v1")
            key = api_key or (
                ds_preset.auth.resolve_key() if ds_preset else os.environ.get("DEEPSEEK_API_KEY")
            )
            return OpenAICompatProvider(
                base_url=base,
                model=ref.model_name,
                api_key=key,
                provider_name="deepseek",
                headers=extra_headers,
                **kwargs,
            )

        # 6. Fallback to OpenAICompatProvider
        from ..providers.openai_compat import OpenAICompatProvider

        key = api_key or os.environ.get("OPENAI_API_KEY")
        return OpenAICompatProvider(
            base_url=base_url or "https://api.openai.com/v1",
            model=ref.model_name,
            api_key=key,
            headers=extra_headers,
            **kwargs,
        )

    def _instantiate_preset(
        self,
        preset: ProviderPreset,
        *,
        model: str,
        api_key: str | None,
        base_url: str | None,
        extra_headers: dict[str, str] | Mapping[str, str] | None,
        **kwargs: Any,
    ) -> ChatProvider:
        key = preset.auth.resolve_key(api_key)
        effective_base_url = base_url or preset.base_url
        merged_headers = {**preset.headers, **dict(extra_headers or {})}

        if preset.kind == "anthropic" or preset.name == "anthropic":
            from ..providers.anthropic import AnthropicProvider

            return AnthropicProvider(
                model=model,
                base_url=effective_base_url,
                api_key=key,
                extra_headers=merged_headers,
                context_window=preset.context_window,
                **kwargs,
            )

        if preset.kind == "gemini" or preset.name == "gemini":
            from ..providers.gemini import GeminiProvider

            return GeminiProvider(
                model=model,
                base_url=effective_base_url,
                api_key=key,
                extra_headers=merged_headers,
                context_window=preset.context_window,
                **kwargs,
            )

        if preset.kind == "ollama" or preset.name == "ollama":
            from ..providers.ollama import OllamaProvider

            return OllamaProvider(
                model=model,
                base_url=effective_base_url,
                api_key=key,
                extra_headers=merged_headers,
                context_window=preset.context_window,
                **kwargs,
            )

        # Default to OpenAICompatProvider
        from ..providers.openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(
            base_url=effective_base_url,
            model=model,
            api_key=key,
            provider_name=preset.name,
            headers=merged_headers,
            context_window=preset.context_window,
            **kwargs,
        )
