"""LLM router: the only place that talks to providers.

Callers (views, services, future features) use `LLMRouter.chat()` — never a
provider class directly. This iteration intentionally implements a single
active provider (the first registered provider that is configured). The
design is list-based with explicit extension points, so multi-provider
priority ordering, fallback, rate-limit gating and circuit breaking later
slot into `_select_provider()` without touching callers.
"""

from typing import Optional

from .exceptions import LLMProviderUnavailableError
from .providers.base import LLMProvider
from .types import ChatRequest, ChatResponse


class LLMRouter:
    """Routes ChatRequests to a configured LLM provider."""

    def __init__(self, providers: Optional[list[LLMProvider]] = None):
        self._providers: list[LLMProvider] = []
        if providers:
            self.register_many(providers)

    def register(self, provider: LLMProvider) -> None:
        if not isinstance(provider, LLMProvider):
            raise TypeError("router.register() expects an LLMProvider instance.")
        self._providers.append(provider)

    def register_many(self, providers) -> None:
        for provider in providers:
            self.register(provider)

    @property
    def providers(self) -> tuple[LLMProvider, ...]:
        return tuple(self._providers)

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Send a normalized request to the active provider."""
        provider = self._select_provider()
        return provider.chat(request)

    def health(self) -> list[dict]:
        """Status of every registered provider (no API calls, no secrets)."""
        return [provider.health() for provider in self._providers]

    # ── Selection logic ────────────────────────────────────────────────────
    # This step: first configured provider wins.
    # Later steps (reserved, not implemented yet) will extend this method:
    #   - sort by ProviderConfig.priority (priority queue)
    #   - skip providers flagged unhealthy by circuit breaker
    #   - enforce ProviderConfig.rate_limit_per_minute
    #   - fall back to the next provider on LLMProviderError / rate limits,
    #     with retry + exponential backoff

    def _select_provider(self) -> LLMProvider:
        for provider in self._providers:
            if provider.is_configured():
                return provider
        raise LLMProviderUnavailableError(
            "No LLM provider is configured. Set AI_OPENAI_API_KEY in your "
            "environment (.env file) and try again."
        )
