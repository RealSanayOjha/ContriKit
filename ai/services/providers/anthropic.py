"""Anthropic (Claude) Messages API provider.

Implements the native Anthropic Messages API with tool calling:

    POST {base_url}/messages
    headers: x-api-key, anthropic-version: 2023-06-01

Like the OpenAI-compatible provider it uses ``requests`` (already a project
dependency) and normalizes everything into the shared ChatRequest/ChatResponse
types, so the rest of the application never knows which vendor answered.
"""

import logging

import requests

from ..exceptions import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMInvalidResponseError,
    LLMProviderError,
    LLMProviderServerError,
    LLMProviderUnavailableError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from ..types import ChatRequest, ChatResponse, ToolCall, UsageStats
from .base import LLMProvider

logger = logging.getLogger(__name__)

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_BASE_URL = "https://api.anthropic.com/v1"


class AnthropicProvider(LLMProvider):
    """Messages API provider for Claude models."""

    name = "anthropic"
    display_name = "Anthropic (Claude Messages API)"

    def is_configured(self) -> bool:
        return bool(self.config.api_key)

    def chat(self, request: ChatRequest) -> ChatResponse:
        if not self.is_configured():
            raise LLMConfigurationError(
                f"Provider '{self.name}' is not configured (missing API key)."
            )

        url = f"{self.config.base_url.rstrip('/')}/messages"
        payload = self._build_payload(request)
        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
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

    # ── Serialization ──────────────────────────────────────────────────────

    def _build_payload(self, request: ChatRequest) -> dict:
        system_parts, provider_messages, pending_tool_results = [], [], []

        for message in request.messages:
            if message.role == "system":
                if message.content:
                    system_parts.append(message.content)
                continue

            if message.role == "tool":
                pending_tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id or "",
                        "content": message.content or "",
                    }
                )
                continue

            if pending_tool_results:
                provider_messages.append({"role": "user", "content": pending_tool_results})
                pending_tool_results = []

            if message.role == "user":
                provider_messages.append({"role": "user", "content": message.content})
            elif message.role == "assistant":
                blocks = []
                if message.content:
                    blocks.append({"type": "text", "text": message.content})
                for call in message.tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.arguments,
                        }
                    )
                provider_messages.append(
                    {"role": "assistant", "content": blocks if blocks else []}
                )

        if pending_tool_results:
            provider_messages.append({"role": "user", "content": pending_tool_results})

        payload = {
            "model": request.model or self.config.model,
            "max_tokens": request.max_tokens or self.config.max_tokens,
            "messages": provider_messages,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        elif self.config.temperature is not None:
            payload["temperature"] = self.config.temperature

        if request.tools:
            payload["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                }
                for tool in request.tools
            ]
        return payload

    # ── Errors / parsing ───────────────────────────────────────────────────

    def _raise_for_status(self, response: requests.Response) -> None:
        status = response.status_code
        provider = self.name
        if status in (401, 403):
            raise LLMAuthenticationError(
                f"Provider '{provider}' rejected the API key (HTTP {status}). "
                "Check AI_ANTHROPIC_API_KEY.",
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
            raise LLMProviderServerError(
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

        blocks = data.get("content") or []
        if not isinstance(blocks, list):
            raise LLMInvalidResponseError(
                f"Provider '{self.name}' returned an unexpected payload shape."
            )

        text_parts, tool_calls = [], []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(block.get("text") or "")
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=str(block.get("id") or ""),
                        name=str(block.get("name") or ""),
                        arguments=dict(block.get("input") or {}),
                    )
                )

        content = "".join(text_parts).strip()
        if not content and not tool_calls:
            raise LLMInvalidResponseError(
                f"Provider '{self.name}' returned empty content with no tool calls."
            )

        usage = UsageStats.from_mapping(
            {
                "prompt_tokens": data.get("usage", {}).get("input_tokens"),
                "completion_tokens": data.get("usage", {}).get("output_tokens"),
                "total_tokens": None,
            }
        )
        total = None
        if usage and (usage.prompt_tokens or usage.completion_tokens):
            total = usage.prompt_tokens + usage.completion_tokens
            usage = UsageStats(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=total,
            )

        stop_reason = data.get("stop_reason") or "end_turn"
        logger.info(
            "LLM response received from %s (model=%s, stop=%s, tool_calls=%d)",
            self.name,
            data.get("model") or self.config.model,
            stop_reason,
            len(tool_calls),
        )
        return ChatResponse(
            content=content,
            provider=self.name,
            model=data.get("model") or self.config.model,
            usage=usage,
            raw=data,
            tool_calls=tuple(tool_calls),
            finish_reason=str(stop_reason),
        )
