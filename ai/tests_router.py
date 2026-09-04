"""Tests for multi-provider routing and the Anthropic (Claude) provider.

No real LLM calls: HTTP is mocked and fake providers simulate failures to
exercise priority ordering, fallback, retries/backoff, rate limiting, and the
circuit breaker. Settings wiring is tested with override_settings.
"""

import json
from unittest import mock

import requests
from django.test import SimpleTestCase, override_settings

from ai.services import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMProviderError,
    LLMProviderUnavailableError,
    LLMRateLimitError,
    LLMRouter,
    LLMTimeoutError,
    ProviderConfig,
    ToolCall,
    UsageStats,
)
from ai.services import service as service_module
from ai.services.providers import (
    PROVIDER_REGISTRY,
    available_providers,
    get_provider_class,
    get_provider_defaults,
)
from ai.services.providers.anthropic import AnthropicProvider
from ai.services.providers.base import LLMProvider
from ai.services.providers.openai_compatible import OpenAICompatibleProvider


class FakeProvider(LLMProvider):
    """Configurable in-memory provider used to test router behavior."""

    def __init__(self, name="fake", configured=True, error=None, priority=100, rate_limit=None):
        super().__init__(
            ProviderConfig(
                name=name, model="fake-model", priority=priority,
                rate_limit_per_minute=rate_limit,
            )
        )
        self._configured = configured
        self.error = error
        self.calls = 0

    def is_configured(self):
        return self._configured

    def chat(self, request):
        self.calls += 1
        if self.error:
            raise self.error
        return ChatResponse(
            content=f"reply from {self.name}",
            provider=self.name,
            model="fake-model",
            usage=UsageStats(total_tokens=1),
        )


def request():
    return ChatRequest(messages=(ChatMessage(role="user", content="hi"),))


def fake_anthropic_response(data, status_code=200):
    response = mock.Mock(status_code=status_code)
    response.json.return_value = data
    response.text = "body"
    return response


# ── Registry / vendor defaults ────────────────────────────────────────────

class ProviderRegistryTests(SimpleTestCase):
    def test_all_providers_registered(self):
        self.assertIn("openai", PROVIDER_REGISTRY)
        self.assertIn("groq", PROVIDER_REGISTRY)
        self.assertIn("grok", PROVIDER_REGISTRY)
        self.assertIn("xai", PROVIDER_REGISTRY)
        self.assertIn("gemini", PROVIDER_REGISTRY)
        self.assertIn("anthropic", PROVIDER_REGISTRY)

    def test_grok_uses_openai_compatible_class(self):
        self.assertIs(get_provider_class("grok"), OpenAICompatibleProvider)

    def test_groq_uses_openai_compatible_class(self):
        # Groq is a separate vendor from xAI Grok but shares the
        # OpenAI-compatible wire protocol (different endpoint/key/model).
        self.assertIs(get_provider_class("groq"), OpenAICompatibleProvider)

    def test_anthropic_uses_native_class(self):
        self.assertIs(get_provider_class("anthropic"), AnthropicProvider)

    def test_vendor_defaults_present(self):
        defaults = get_provider_defaults("grok")
        self.assertIn("https://api.x.ai", defaults["base_url"])
        self.assertNotIn("https://api.openai.com", defaults["base_url"])
        self.assertIn("claude", get_provider_defaults("anthropic")["model"])

    def test_groq_defaults_point_at_groq_not_xai(self):
        # Regression: a Groq key must never be sent to api.x.ai. The groq
        # defaults must resolve to Groq's endpoint/model, distinct from grok.
        groq_defaults = get_provider_defaults("groq")
        grok_defaults = get_provider_defaults("grok")
        self.assertIn("https://api.groq.com/openai/v1", groq_defaults["base_url"])
        self.assertNotIn("api.x.ai", groq_defaults["base_url"])
        self.assertNotEqual(groq_defaults["base_url"], grok_defaults["base_url"])
        self.assertNotEqual(groq_defaults["model"], grok_defaults["model"])

    def test_available_providers_sorted(self):
        names = available_providers()
        self.assertIn("anthropic", names)
        self.assertEqual(names, tuple(sorted(names)))


# ── Anthropic provider ─────────────────────────────────────────────────────

class AnthropicProviderTests(SimpleTestCase):
    def setUp(self):
        self.provider = AnthropicProvider(
            ProviderConfig(
                name="anthropic",
                api_key="sk-ant-test",
                base_url="https://api.anthropic.test/v1",
                model="claude-test",
                timeout=5,
                max_tokens=256,
                temperature=0.4,
            )
        )

    @mock.patch("ai.services.providers.anthropic.requests.post")
    def test_builds_native_payload(self, mock_post):
        mock_post.return_value = fake_anthropic_response(
            {
                "model": "claude-test",
                "content": [{"type": "text", "text": "Hello from Claude!"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        )
        self.provider.chat(
            ChatRequest(
                messages=(
                    ChatMessage(role="system", content="be helpful"),
                    ChatMessage(role="user", content="hello"),
                )
            )
        )
        payload = mock_post.call_args.kwargs["json"]
        headers = mock_post.call_args.kwargs["headers"]
        self.assertEqual(headers["x-api-key"], "sk-ant-test")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertEqual(payload["model"], "claude-test")
        self.assertEqual(payload["max_tokens"], 256)
        self.assertEqual(payload["system"], "be helpful")
        self.assertEqual(payload["messages"][0], {"role": "user", "content": "hello"})

    @mock.patch("ai.services.providers.anthropic.requests.post")
    def test_parses_text_and_tool_use(self, mock_post):
        mock_post.return_value = fake_anthropic_response(
            {
                "model": "claude-test",
                "content": [
                    {"type": "text", "text": "Let me search."},
                    {"type": "tool_use", "id": "toolu_1", "name": "search_issues", "input": {"q": "python"}},
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 8, "output_tokens": 4},
            }
        )
        response = self.provider.chat(
            ChatRequest(messages=(ChatMessage(role="user", content="find"),))
        )
        self.assertEqual(response.content, "Let me search.")
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0].name, "search_issues")
        self.assertEqual(response.tool_calls[0].arguments, {"q": "python"})
        self.assertEqual(response.finish_reason, "tool_use")
        self.assertEqual(response.usage.total_tokens, 12)

    @mock.patch("ai.services.providers.anthropic.requests.post")
    def test_round_trips_tool_result_message(self, mock_post):
        mock_post.return_value = fake_anthropic_response(
            {"model": "m", "content": [{"type": "text", "text": "done"}],
             "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}}
        )
        self.provider.chat(
            ChatRequest(
                messages=(
                    ChatMessage(
                        role="assistant",
                        content="",
                        tool_calls=(
                            ToolCall(id="toolu_9", name="search_issues", arguments={"q": "x"}),
                        ),
                    ),
                    ChatMessage(role="tool", content='{"results": []}', tool_call_id="toolu_9"),
                )
            )
        )
        payload = mock_post.call_args.kwargs["json"]
        # Tool results are converted to a user message with tool_result blocks.
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertEqual(payload["messages"][1]["content"][0]["type"], "tool_result")
        self.assertEqual(payload["messages"][1]["content"][0]["tool_use_id"], "toolu_9")
        self.assertEqual(payload["messages"][0]["content"][0]["type"], "tool_use")

    @mock.patch("ai.services.providers.anthropic.requests.post")
    def test_error_mapping(self, mock_post):
        cases = [
            (401, LLMAuthenticationError),
            (429, LLMRateLimitError),
            (500, LLMProviderError),  # now LLMProviderServerError subclass
        ]
        for status, expected in cases:
            mock_post.return_value = fake_anthropic_response({}, status_code=status)
            with self.assertRaises(expected):
                self.provider.chat(
                    ChatRequest(messages=(ChatMessage(role="user", content="hi"),))
                )

    @mock.patch("ai.services.providers.anthropic.requests.post")
    def test_timeout_maps_to_typed_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("slow")
        with self.assertRaises(LLMTimeoutError):
            self.provider.chat(ChatRequest(messages=(ChatMessage(role="user", content="hi"),)))

    @mock.patch("ai.services.providers.anthropic.requests.post")
    def test_empty_response_raises_invalid(self, mock_post):
        mock_post.return_value = fake_anthropic_response(
            {"model": "m", "content": [], "stop_reason": "end_turn", "usage": {}}
        )
        with self.assertRaises(Exception):
            self.provider.chat(ChatRequest(messages=(ChatMessage(role="user", content="hi"),)))

    def test_unconfigured_raises(self):
        provider = AnthropicProvider(ProviderConfig(name="anthropic", api_key=""))
        self.assertFalse(provider.is_configured())
        with self.assertRaises(LLMConfigurationError):
            provider.chat(ChatRequest(messages=(ChatMessage(role="user", content="hi"),)))


# ── Router: priority, fallback, retry, rate limit, circuit breaker ────────

class RouterPriorityTests(SimpleTestCase):
    def test_lower_priority_number_tried_first(self):
        secondary = FakeProvider(name="secondary", priority=50)
        primary = FakeProvider(name="primary", priority=10)
        router = LLMRouter([secondary, primary])
        response = router.chat(request())
        self.assertEqual(response.content, "reply from primary")
        self.assertEqual(primary.calls, 1)
        self.assertEqual(secondary.calls, 0)

    def test_unconfigured_priority_provider_skipped(self):
        unconfigured = FakeProvider(name="unconfigured", configured=False, priority=1)
        configured = FakeProvider(name="configured", priority=50)
        router = LLMRouter([unconfigured, configured])
        response = router.chat(request())
        self.assertEqual(response.provider, "configured")

    def test_health_includes_priority_and_status(self):
        router = LLMRouter([FakeProvider(name="p", priority=7)])
        health = router.health()[0]
        self.assertEqual(health["priority"], 7)
        self.assertEqual(health["status"], "healthy")


class RouterFallbackTests(SimpleTestCase):
    def test_falls_back_on_timeout(self):
        primary = FakeProvider(name="primary", error=LLMTimeoutError("slow", provider="primary"))
        secondary = FakeProvider(name="secondary")
        router = LLMRouter([primary, secondary], retries=0)
        response = router.chat(request())
        self.assertEqual(response.provider, "secondary")

    def test_falls_back_on_rate_limit(self):
        primary = FakeProvider(name="primary", error=LLMRateLimitError("429", provider="primary"))
        secondary = FakeProvider(name="secondary")
        router = LLMRouter([primary, secondary], retries=0)
        self.assertEqual(router.chat(request()).provider, "secondary")

    def test_falls_back_on_auth_error(self):
        primary = FakeProvider(name="primary", error=LLMAuthenticationError("bad key", provider="primary"))
        secondary = FakeProvider(name="secondary")
        router = LLMRouter([primary, secondary], retries=0)
        self.assertEqual(router.chat(request()).provider, "secondary")

    def test_all_fail_raises_unavailable(self):
        primary = FakeProvider(name="primary", error=LLMTimeoutError("slow"))
        secondary = FakeProvider(name="secondary", error=LLMProviderUnavailableError("down"))
        router = LLMRouter([primary, secondary], retries=0)
        with self.assertRaises(LLMProviderUnavailableError):
            router.chat(request())

    def test_fallback_disabled_raises_original_error(self):
        primary = FakeProvider(name="primary", error=LLMTimeoutError("slow"))
        router = LLMRouter([primary], fallback=False, retries=0)
        with self.assertRaises(LLMTimeoutError):
            router.chat(request())

    def test_deterministic_4xx_not_retried_or_failed_over(self):
        primary = FakeProvider(name="primary", error=LLMProviderError("bad request", status_code=400))
        router = LLMRouter([primary], retries=5)
        with self.assertRaises(LLMProviderError):
            router.chat(request())
        self.assertEqual(primary.calls, 1)


class RouterRetryTests(SimpleTestCase):
    def test_transient_error_retried_then_succeeds(self):
        provider = FakeProvider(name="p")

        def flaky(request):
            provider.calls += 1
            if provider.calls < 3:
                raise LLMProviderUnavailableError("flaky")
            return ChatResponse(content="ok", provider="p", model="m")

        provider.chat = flaky
        router = LLMRouter([provider], retries=2, retry_backoff=0)
        self.assertEqual(router.chat(request()).content, "ok")
        self.assertEqual(provider.calls, 3)

    def test_retries_exhausted_raises(self):
        provider = FakeProvider(name="p", error=LLMTimeoutError("slow"))
        router = LLMRouter([provider], retries=2, retry_backoff=0)
        with self.assertRaises(LLMProviderUnavailableError):
            router.chat(request())
        self.assertEqual(provider.calls, 3)  # 1 initial + 2 retries


class RouterRateLimitTests(SimpleTestCase):
    def test_rate_limit_skips_provider(self):
        provider = FakeProvider(name="limited", rate_limit=1)
        router = LLMRouter([provider], retries=0)
        router.chat(request())  # first attempt consumes the single slot
        with self.assertRaises(LLMProviderUnavailableError):
            router.chat(request())  # second attempt is blocked
        self.assertEqual(provider.calls, 1)


class RouterCircuitBreakerTests(SimpleTestCase):
    def test_breaker_opens_after_failures(self):
        provider = FakeProvider(name="p", error=LLMProviderUnavailableError("down"))
        router = LLMRouter([provider], retries=0, circuit_failure_threshold=2,
                           circuit_reset_seconds=60)
        with self.assertRaises(LLMProviderUnavailableError):
            router.chat(request())
        with self.assertRaises(LLMProviderUnavailableError):
            router.chat(request())
        # Breaker now open: the third request should not even call the provider.
        provider.error = None
        with self.assertRaises(LLMProviderUnavailableError):
            router.chat(request())
        self.assertEqual(provider.calls, 2)

    def test_breaker_resets_after_window(self):
        provider = FakeProvider(name="p", error=LLMProviderUnavailableError("down"))
        router = LLMRouter([provider], retries=0, circuit_failure_threshold=1,
                           circuit_reset_seconds=0)
        with self.assertRaises(LLMProviderUnavailableError):
            router.chat(request())
        provider.error = None
        self.assertEqual(router.chat(request()).provider, "p")


# ── Settings wiring: multiple providers from env ─────────────────────────

class SettingsMultiProviderTests(SimpleTestCase):
    def tearDown(self):
        service_module._ai_service = None

    @override_settings(
        AI_PROVIDERS="openai, anthropic, grok",
        AI_OPENAI_API_KEY="sk-openai",
        AI_ANTHROPIC_API_KEY="sk-ant",
        AI_GROK_API_KEY="xai-key",
        AI_OPENAI_PRIORITY=5,
        AI_ANTHROPIC_PRIORITY=1,  # anthropic should win despite list order
        AI_GROK_MODEL="grok-custom",
    )
    def test_builds_multiple_providers_with_priority(self):
        router = service_module.build_default_router()
        names = [p.name for p in router.providers]
        self.assertEqual(names, ["openai", "anthropic", "grok"])
        self.assertTrue(router.fallback)
        self.assertEqual(router.retries, 2)

        ordered = router._ordered_providers()
        self.assertEqual(ordered[0].name, "anthropic")  # priority 1
        self.assertEqual(ordered[0].config.api_key, "sk-ant")
        self.assertEqual(ordered[1].name, "openai")
        self.assertEqual(ordered[2].name, "grok")
        grok = ordered[2]
        self.assertEqual(grok.config.model, "grok-custom")
        self.assertIn("api.x.ai", grok.config.base_url)

    @override_settings(AI_PROVIDERS="", AI_PROVIDER="grok", AI_GROK_API_KEY="key")
    def test_legacy_single_provider_still_works(self):
        router = service_module.build_default_router()
        self.assertEqual([p.name for p in router.providers], ["grok"])
        self.assertTrue(router.providers[0].is_configured())

    @override_settings(
        AI_PROVIDERS="groq,grok",
        AI_GROQ_API_KEY="gsk-test",
        AI_GROK_API_KEY="xai-test",
        AI_GROQ_MODEL="llama-3.3-70b-versatile",
        AI_GROK_MODEL="grok-3-mini",
    )
    def test_groq_and_grok_use_distinct_keys_endpoints_and_models(self):
        # Regression for the Groq-vs-Grok mix-up: each vendor must read its
        # own key setting and keep its own endpoint/model. A gsk_ key in the
        # grok slot (or vice versa) always 401s, so they must never be shared.
        from ai.services.service import PROVIDER_KEY_SETTINGS

        self.assertEqual(PROVIDER_KEY_SETTINGS["groq"], "AI_GROQ_API_KEY")
        self.assertEqual(PROVIDER_KEY_SETTINGS["grok"], "AI_GROK_API_KEY")
        self.assertNotEqual(
            PROVIDER_KEY_SETTINGS["groq"], PROVIDER_KEY_SETTINGS["grok"]
        )

        router = service_module.build_default_router()
        by_name = {p.name: p for p in router.providers}
        self.assertIn("groq", by_name)
        self.assertIn("grok", by_name)
        groq, grok = by_name["groq"], by_name["grok"]
        self.assertEqual(groq.config.api_key, "gsk-test")
        self.assertEqual(grok.config.api_key, "xai-test")
        self.assertIn("api.groq.com", groq.config.base_url)
        self.assertIn("api.x.ai", grok.config.base_url)
        self.assertNotEqual(groq.config.base_url, grok.config.base_url)

    @override_settings(
        AI_PROVIDERS="groq",
        AI_GROQ_API_KEY="gsk-test",
    )
    def test_groq_only_configures_router(self):
        router = service_module.build_default_router()
        self.assertEqual([p.name for p in router.providers], ["groq"])
        self.assertTrue(router.providers[0].is_configured())
        self.assertIn("api.groq.com", router.providers[0].config.base_url)
        health = router.health()[0]
        self.assertEqual(health["provider"], "groq")
        self.assertEqual(health["status"], "healthy")

    @override_settings(
        AI_PROVIDERS="groq,grok",
        AI_GROQ_API_KEY="",
        AI_GROK_API_KEY="gsk-wrong-slot",
    )
    def test_groq_key_in_grok_slot_does_not_configure_groq(self):
        # Documents the original bug: a Groq gsk_ key placed in AI_GROK_API_KEY
        # leaves the groq provider unconfigured and points the grok provider
        # at api.x.ai, where that key can never authenticate.
        router = service_module.build_default_router()
        by_name = {p.name: p for p in router.providers}
        self.assertFalse(by_name["groq"].is_configured())
        self.assertTrue(by_name["grok"].is_configured())
        self.assertIn("api.x.ai", by_name["grok"].config.base_url)

    @override_settings(
        AI_PROVIDERS="openai,annotated_provider",
        AI_OPENAI_API_KEY="key",
    )
    def test_unknown_provider_skipped_without_breaking_others(self):
        router = service_module.build_default_router()
        self.assertEqual([p.name for p in router.providers], ["openai"])

    @override_settings(
        AI_PROVIDERS="openai,anthropic",
        AI_OPENAI_API_KEY="",
        AI_ANTHROPIC_API_KEY="sk-ant",
        AI_PROVIDER_RETRIES=3,
        AI_PROVIDER_RETRY_BACKOFF=0.1,
    )
    def test_service_uses_first_configured_provider(self):
        service = service_module.get_ai_service()
        self.assertTrue(service.is_configured())
        self.assertEqual(service.router.retries, 3)
        health = service.health()
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["active_provider"], "anthropic")
        self.assertTrue(health["extensions"]["multi_provider_priority"])
        self.assertTrue(health["extensions"]["fallback"])
        self.assertTrue(health["extensions"]["circuit_breaker"])
