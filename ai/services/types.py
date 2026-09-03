"""Shared, provider-agnostic data types for the AI service layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# Roles accepted by chat completion APIs.
VALID_ROLES = ("system", "user", "assistant")

OPENAI_COMPATIBLE_DEFAULT_BASE_URL = "https://api.openai.com/v1"
OPENAI_COMPATIBLE_DEFAULT_MODEL = "gpt-4o-mini"


@dataclass(frozen=True)
class ChatMessage:
    """One message inside a conversation. `role` is system/user/assistant."""

    role: str
    content: str


@dataclass(frozen=True)
class UsageStats:
    """Token usage reported by a provider (when available)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_mapping(cls, data: Optional[dict]) -> Optional["UsageStats"]:
        """Build UsageStats from a provider payload; returns None if absent."""
        if not data:
            return None
        return cls(
            prompt_tokens=int(data.get("prompt_tokens") or 0),
            completion_tokens=int(data.get("completion_tokens") or 0),
            total_tokens=int(data.get("total_tokens") or 0),
        )


@dataclass(frozen=True)
class ChatRequest:
    """A normalized request the router hands to a provider."""

    messages: tuple[ChatMessage, ...]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    # Reserved for later: provider hints, metadata, request ids for usage tracking.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatResponse:
    """A normalized provider response returned to the caller."""

    content: str
    provider: str
    model: str
    usage: Optional[UsageStats] = None
    raw: Optional[dict] = None


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider instance (built from env vars).

    Fields marked "reserved" are read by the upcoming multi-provider router
    (priorities, rate limits, fallback) but are not used in this iteration.
    """

    name: str = "openai"
    api_key: str = ""
    base_url: str = OPENAI_COMPATIBLE_DEFAULT_BASE_URL
    model: str = OPENAI_COMPATIBLE_DEFAULT_MODEL
    timeout: int = 60
    max_tokens: int = 1024
    temperature: float = 0.7
    enabled: bool = True
    # Reserved for multi-provider routing (later step).
    priority: int = 100
    rate_limit_per_minute: Optional[int] = None
