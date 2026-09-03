"""AI service layer public API.

Usage from anywhere in the project:

    from ai.services import ai_chat, ChatMessage

    response = ai_chat([ChatMessage(role="user", content="Help me...")])
    print(response.content)  # provider-agnostic ChatResponse
"""

from .exceptions import (
    AIServiceError,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMInvalidResponseError,
    LLMProviderError,
    LLMProviderUnavailableError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from .router import LLMRouter
from .service import AIService, ai_chat, get_ai_service
from .types import ChatMessage, ChatRequest, ChatResponse, ProviderConfig, UsageStats

__all__ = [
    "AIService",
    "LLMRouter",
    "ai_chat",
    "get_ai_service",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ProviderConfig",
    "UsageStats",
    "AIServiceError",
    "LLMConfigurationError",
    "LLMProviderError",
    "LLMTimeoutError",
    "LLMAuthenticationError",
    "LLMRateLimitError",
    "LLMProviderUnavailableError",
    "LLMInvalidResponseError",
]
