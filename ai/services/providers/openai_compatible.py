"""OpenAI-compatible Chat Completions provider.

Talks to any endpoint implementing `POST {base_url}/chat/completions` with a
Bearer token (OpenAI, Azure OpenAI-style gateways, Groq, OpenRouter, local
Ollama/LM Studio, etc.). Using plain `requests` keeps the provider dependency
free: `requests` is already a project dependency (used for GitHub API), so no
new package is required and the code stays portable to PythonAnywhere.
"""

import logging

import requests

from ..exceptions import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMInvalidResponseError,
    LLMProviderError,
    LLMProviderUnavailableError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from ..types import ChatRequest, ChatResponse, UsageStats
from .base import LLMProvider

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    """Chat Completions provider for OpenAI and compatible endpoints."""

    name = "openai"
    display_name = "OpenAI (Chat Completions)"

    def is_configured(self) -> bool:
        return bool(self.config.api_key)

    def chat(self, request: ChatRequest) -> ChatResponse:
        if not self.is_configured():
            raise LLMConfigurationError(
                f"Provider '{self.name}' is not configured (missing API key)."
            )

        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        payload = self._build_payload(request)
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        timeout = request.timeout or self.config.timeout

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        except requests.exceptions.Timeout as exc:
            raise LLMTimeoutError(
                f"Provider '{self.name}' timed out after {timeout}s.",
                provider=self.name,
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise LLMProviderUnavailableError(
                f"Provider '{self.name}' is unreachable: {exc}",
            ) from exc

        self._raise_for_status(response)
        return self._parse_response(response)

    # ── Internals ──────────────────────────────────────────────────────────

    def _build_payload(self, request: ChatRequest) -> dict:
        payload = {
            "model": request.model or self.config.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
            "stream": False,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        elif self.config.temperature is not None:
            payload["temperature"] = self.config.temperature

        max_tokens = request.max_tokens if request.max_tokens is not None else self.config.max_tokens
        if max_tokens:
            payload["max_tokens"] = max_tokens
        return payload

    def _raise_for_status(self, response: requests.Response) -> None:
        status = response.status_code
        provider = self.name
        if status in (401, 403):
            raise LLMAuthenticationError(
                f"Provider '{provider}' rejected the API key (HTTP {status}). "
                "Check AI_OPENAI_API_KEY.",
                status_code=status,
                provider=provider,
            )
        if status == 429:
            raise LLMRateLimitError(
                f"Provider '{provider}' rate limit hit (HTTP 429). Try again later.",
                status_code=status,
                provider=provider,
            )
        if status >= 500:
            raise LLMProviderError(
                f"Provider '{provider}' server error (HTTP {status}).",
                status_code=status,
                provider=provider,
            )
        if status >= 400:
            raise LLMProviderError(
                f"Provider '{provider}' request failed (HTTP {status}): "
                f"{response.text[:300]}",
                status_code=status,
                provider=provider,
            )

    def _parse_response(self, response: requests.Response) -> ChatResponse:
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMInvalidResponseError(
                f"Provider '{self.name}' returned a non-JSON response."
            ) from exc

        try:
            message = data["choices"][0]["message"]
            content = message.get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LLMInvalidResponseError(
                f"Provider '{self.name}' returned an unexpected payload shape."
            ) from exc

        if not content.strip():
            raise LLMInvalidResponseError(
                f"Provider '{self.name}' returned empty content."
            )

        usage = UsageStats.from_mapping(data.get("usage"))
        logger.info(
            "LLM response received from %s (model=%s, total_tokens=%s)",
            self.name,
            data.get("model") or self.config.model,
            usage.total_tokens if usage else "unknown",
        )
        return ChatResponse(
            content=content.strip(),
            provider=self.name,
            model=data.get("model") or self.config.model,
            usage=usage,
            raw=data,
        )
