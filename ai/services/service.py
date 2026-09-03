"""Public AI service gateway.

This is the single entry point the rest of ContribKit calls to get an LLM
response:

    user message / feature
        → AIService.chat(messages, tools=..., user=..., page_path=...)
        → LLMRouter
        → LLMProvider
        → (tool calls executed via ToolRegistry, looped)
        → ChatResponse

Providers/keys live server-side only; nothing is ever passed to templates or
JavaScript. Views stay thin and provider-agnostic.

When a ToolRegistry is supplied, the service runs the provider's native tool
loop: the LLM may call registered tools (search_issues, get_template, ...),
results are fed back as real data, and the loop terminates with a final
text answer. Tool execution never raises into the caller.
"""

import json
import logging

from django.conf import settings

from .exceptions import AIServiceError
from .prompts import build_system_prompt
from .providers import get_provider_class
from .providers.base import LLMProvider
from .router import LLMRouter
from .tools import ToolContext, ToolRegistry, build_default_tool_registry
from .types import ChatMessage, ChatRequest, ChatResponse, ProviderConfig

logger = logging.getLogger(__name__)

# Module-level cache: the router is built once per process from settings.
_ai_service: "AIService | None" = None

OPENAI_COMPATIBLE_DEFAULT_BASE_URL = "https://api.openai.com/v1"
OPENAI_COMPATIBLE_DEFAULT_MODEL = "gpt-4o-mini"

MAX_TOOL_ROUNDS = 4


class AIService:
    """Provider-agnostic AI facade used by the rest of the application."""

    def __init__(self, router: LLMRouter | None = None, tools: ToolRegistry | None = None):
        self.router = router or LLMRouter()
        self.tools = tools

    def chat(
        self,
        messages,
        *,
        model=None,
        temperature=None,
        max_tokens=None,
        timeout=None,
        tools: ToolRegistry | None = None,
        user=None,
        page_path=None,
        issue_id=None,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
        tool_events: list | None = None,
    ) -> ChatResponse:
        """Send messages through the LLM router, executing tools when asked.

        ``tools`` (a ToolRegistry) enables the tool loop; ``user``/``page_path``/
        ``issue_id`` build the server-controlled ToolContext tools receive.
        ``tool_events`` (an optional list) receives ``{"name", "arguments",
        "result"}`` for every executed tool so callers can build UI source
        links without re-running tools or re-querying the database.
        """
        self._validate_messages(messages)
        registry = tools if tools is not None else self.tools
        context = ToolContext(user=user, page_path=page_path, issue_id=issue_id)
        working = list(messages)
        specs = registry.specs() if registry else ()

        rounds = max_tool_rounds if registry else 1
        for _ in range(rounds):
            request = self._build_request(
                working, specs, model=model, temperature=temperature,
                max_tokens=max_tokens, timeout=timeout,
            )
            response = self.router.chat(request)
            if not response.tool_calls or not registry:
                return response

            # Ask the provider again with the tool results appended.
            working.append(
                ChatMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )
            for call in response.tool_calls:
                result = registry.execute(call.name, call.arguments, context)
                if tool_events is not None:
                    tool_events.append(
                        {
                            "name": call.name,
                            "arguments": call.arguments,
                            "result": result,
                        }
                    )
                working.append(
                    ChatMessage(
                        role="tool",
                        content=json.dumps(result, ensure_ascii=False, default=str),
                        tool_call_id=call.id,
                    )
                )

        # Safety net: force a final answer without more tool calls.
        request = self._build_request(
            working, (), model=model, temperature=temperature,
            max_tokens=max_tokens, timeout=timeout,
        )
        return self.router.chat(request)

    def is_configured(self) -> bool:
        """True when at least one registered provider has credentials."""
        return any(provider.is_configured() for provider in self.router.providers)

    def health(self) -> dict:
        """Diagnostics for monitoring/health checks (no secrets, no API call)."""
        providers = self.router.health()
        configured = [p["provider"] for p in providers if p.get("configured")]
        tools_enabled = self.tools is not None
        return {
            "status": "ok" if configured else "not_configured",
            "service": "contribkit-ai",
            "active_provider": configured[0] if configured else None,
            "providers": providers,
            "tools": {
                "enabled": tools_enabled,
                "count": len(self.tools.names()) if tools_enabled else 0,
                "names": self.tools.names() if tools_enabled else [],
            },
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
    def _build_request(
        messages, specs, *, model=None, temperature=None, max_tokens=None, timeout=None
    ) -> ChatRequest:
        return ChatRequest(
            messages=tuple(messages),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            tools=tuple(specs),
        )

    @staticmethod
    def _validate_messages(messages) -> None:
        if not messages:
            raise AIServiceError("At least one message is required.")
        for message in messages:
            if not isinstance(message, ChatMessage):
                raise TypeError("messages must be ChatMessage instances.")
            if message.role not in ("system", "user", "assistant", "tool"):
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
        _ai_service = AIService(
            router=build_default_router(),
            tools=build_default_tool_registry(),
        )
    return _ai_service


def ai_chat(messages, **kwargs) -> ChatResponse:
    """Convenience function: ai_chat([ChatMessage(...), ...], user=..., ...)."""
    return get_ai_service().chat(messages, **kwargs)


def ai_assistant_chat(
    message: str,
    *,
    user=None,
    page_path=None,
    issue_id=None,
    history: list[ChatMessage] | None = None,
    **kwargs,
) -> ChatResponse:
    """One-turn assistant call: system prompt + user/page context + tools.

    This is what the future chat endpoint will use: it builds the ContribKit
    system prompt, appends any conversation history, adds the user message,
    and lets the model use the controlled tool registry.
    """
    system_prompt = build_system_prompt(user=user, page_path=page_path)
    messages = [ChatMessage(role="system", content=system_prompt)]
    if history:
        messages.extend(history)
    messages.append(ChatMessage(role="user", content=message))
    return get_ai_service().chat(
        messages,
        user=user,
        page_path=page_path,
        issue_id=issue_id,
        **kwargs,
    )
