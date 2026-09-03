"""Provider registry: maps a settings name to an LLMProvider subclass.

Adding a provider later is a one-line registration here — nothing else in
the application changes.
"""

from ..exceptions import LLMConfigurationError
from .base import LLMProvider
from .openai_compatible import OpenAICompatibleProvider

# Aliases keep config names short while allowing explicit "openai-compatible".
PROVIDER_REGISTRY = {
    "openai": OpenAICompatibleProvider,
    "openai_compatible": OpenAICompatibleProvider,
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


def available_providers() -> tuple[str, ...]:
    return tuple(sorted(PROVIDER_REGISTRY))


__all__ = ["LLMProvider", "OpenAICompatibleProvider", "PROVIDER_REGISTRY",
           "get_provider_class", "available_providers"]
