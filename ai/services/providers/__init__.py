"""Provider registry: maps a settings name to an LLMProvider class + defaults.

Adding a provider later is a one-line registration (plus its class) here —
nothing else in the application changes. The `openai_compatible` alias lets
the same class serve any OpenAI-compatible endpoint with different defaults.

Current providers:
    - ``openai``          OpenAI Chat Completions (native).
    - ``openai_compatible`` Any OpenAI-compatible endpoint (explicit alias).
    - ``groq``            Groq (https://api.groq.com/openai/v1) — OpenAI-compatible
                          API hosting open models (e.g. llama-3.3-70b-versatile).
    - ``gemini``          Google Gemini via its OpenAI-compatible endpoint.
    - ``anthropic``       Claude via the native Anthropic Messages API
                          (tool calling supported).
"""

from ..exceptions import LLMConfigurationError
from .anthropic import AnthropicProvider
from .base import LLMProvider
from .openai_compatible import OpenAICompatibleProvider

PROVIDER_REGISTRY = {
    "openai": OpenAICompatibleProvider,
    "openai_compatible": OpenAICompatibleProvider,
    "groq": OpenAICompatibleProvider,
    "gemini": OpenAICompatibleProvider,
    "anthropic": AnthropicProvider,
}

# Vendor defaults; everything is overridable through env/settings.
PROVIDER_DEFAULTS = {
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "openai_compatible": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile"},
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-2.0-flash",
    },
    "anthropic": {"base_url": "https://api.anthropic.com/v1", "model": "claude-sonnet-4-5-20250929"},
}


def get_provider_class(name: str) -> type[LLMProvider]:
    """Resolve a provider name (from settings) to a provider class."""
    try:
        return PROVIDER_REGISTRY[name.strip().lower()]
    except KeyError:
        available = ", ".join(sorted(PROVIDER_REGISTRY))
        raise LLMConfigurationError(
            f"Unknown LLM provider '{name}'. Available providers: {available}."
        )


def get_provider_defaults(name: str) -> dict:
    """Per-vendor defaults (base URL / model)."""
    return dict(PROVIDER_DEFAULTS.get(name.strip().lower(), {}))


def available_providers() -> tuple[str, ...]:
    return tuple(sorted(PROVIDER_REGISTRY))


__all__ = [
    "LLMProvider",
    "OpenAICompatibleProvider",
    "AnthropicProvider",
    "PROVIDER_REGISTRY",
    "PROVIDER_DEFAULTS",
    "get_provider_class",
    "get_provider_defaults",
    "available_providers",
]
