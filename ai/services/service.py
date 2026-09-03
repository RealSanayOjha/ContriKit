"""Public AI service gateway.

This is the single entry point the rest of ContribKit calls to get an LLM
response:

    user message / feature
        → AIService.chat(messages, ...)
        → LLMRouter
        → LLMProvider
        → ChatResponse

Providers/keys live server-side only; nothing is ever passed to templates or
JavaScript. Views stay thin and provider-agnostic.
"""

from django.conf import settings

from .exceptions import AIServiceError
from .providers import get_provider_class
from .providers.base import LLMProvider
from .router import LLMRouter
from .types import ChatMessage, ChatRequest, ChatResponse, ProviderConfig

# Module-level cache: the router is built once per process from settings.
_ai_service: "AIService | None" = None

OPENAI_COMPATIBLE_DEFAULT_BASE_URL = "https://api.openai.com/v1"
OPENAI_COMPATIBLE_DEFAULT_MODEL = "gpt-4o-mini"


class AIService:
    """Provider-agnostic AI facade used by the rest of the application."""

    def __init__(self, router: LLMRouter | None = None):
        self.router = router or LLMRouter()

    def chat(
        self,
        messages,
        *,
        model=None,
        temperature=None,
        max_tokens=None,
        timeout=None,
    ) -> ChatResponse:
        """Send messages (list[ChatMessage]) to the LLM router.

        Args are optional overrides; provider/config defaults fill the rest.
        """
        self._validate_messages(messages)
        request = ChatRequest(
            messages=tuple(messages),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return self.router.chat(request)

    def is_configured(self) -> bool:
        """True when at least one registered provider has credentials."""
        return any(provider.is_configured() for provider in self.router.providers)

    def health(self) -> dict:
        """Diagnostics for monitoring/health checks (no secrets, no API call)."""
        providers = self.router.health()
        configured = [p["provider"] for p in providers if p.get("configured")]
        return {
            "status": "ok" if configured else "not_configured",
            "service": "contribkit-ai",
            "active_provider": configured[0] if configured else None,
            "providers": providers,
            # Flags describing which extension points are active (all reserved
            # for later steps — this iteration is single-provider only).
            "extensions": {
                "multi_provider_priority": False,
                "fallback": False,
                "retry_backoff": False,
                "rate_limit_tracking": False,
                "circuit_breaker": False,
                "priority_queue": False,
                "usage_tracking": True,
            },
        }

    @staticmethod
    def _validate_messages(messages) -> None:
        if not messages:
            raise AIServiceError("At least one message is required.")
        for message in messages:
            if not isinstance(message, ChatMessage):
                raise TypeError("messages must be ChatMessage instances.")
            if message.role not in ("system", "user", "assistant"):
                raise ValueError(f"Unsupported message role: {message.role!r}.")


def build_default_router() -> LLMRouter:
    """Build the router from Django settings (single provider for now)."""
    provider_name = getattr(settings, "AI_PROVIDER", "openai").strip().lower()
    provider_cls = get_provider_class(provider_name)

    config = ProviderConfig(
        name=provider_name,
        api_key=getattr(settings, "AI_OPENAI_API_KEY", ""),
        base_url=getattr(settings, "AI_OPENAI_BASE_URL", OPENAI_COMPATIBLE_DEFAULT_BASE_URL),
        model=getattr(settings, "AI_OPENAI_MODEL", OPENAI_COMPATIBLE_DEFAULT_MODEL),
        timeout=getattr(settings, "AI_OPENAI_TIMEOUT", 60),
        max_tokens=getattr(settings, "AI_OPENAI_MAX_TOKENS", 1024),
        temperature=getattr(settings, "AI_OPENAI_TEMPERATURE", 0.7),
    )
    return LLMRouter([provider_cls(config)])


def get_ai_service() -> AIService:
    """Return the process-wide AI service (built once from settings)."""
    global _ai_service
    if _ai_service is None:
        _ai_service = AIService(build_default_router())
    return _ai_service


def ai_chat(messages, **kwargs) -> ChatResponse:
    """Convenience function: ai_chat([ChatMessage(...), ...], model=...)."""
    return get_ai_service().chat(messages, **kwargs)
