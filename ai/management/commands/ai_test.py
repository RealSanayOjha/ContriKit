"""CLI smoke test for the AI service layer.

Sends one message through the full pipeline (AIService → LLMRouter →
LLMProvider) without any UI or database. Useful for verifying credentials
before building the chat feature.

Usage:
    python manage.py ai_test "What is a good first issue for a beginner?"
    python manage.py ai_test "Explain how git rebase works" --model gpt-4o-mini
"""

from django.core.management.base import BaseCommand, CommandError

from ai.services import AIServiceError, ChatMessage, get_ai_service

DEFAULT_TEST_MESSAGE = (
    "You are testing the ContribKit AI service. Reply with a short, "
    "friendly confirmation that you are operational."
)


class Command(BaseCommand):
    help = "Send a single test message to the configured LLM provider (no UI/DB required)."

    def add_arguments(self, parser):
        parser.add_argument(
            "message",
            nargs="?",
            default=None,
            help="Message to send. Defaults to a connectivity test prompt.",
        )
        parser.add_argument(
            "--model",
            default=None,
            help="Override the model (defaults to the AI_OPENAI_MODEL setting).",
        )
        parser.add_argument(
            "--temperature",
            type=float,
            default=None,
            help="Override temperature (defaults to AI_OPENAI_TEMPERATURE).",
        )
        parser.add_argument(
            "--max-tokens",
            type=int,
            default=None,
            help="Override max response tokens (defaults to AI_OPENAI_MAX_TOKENS).",
        )

    def handle(self, *args, **options):
        service = get_ai_service()

        if not service.is_configured():
            raise CommandError(
                "AI is not configured: no LLM provider has credentials. "
                "Set at least one provider key in your environment / .env file first "
                "(e.g. AI_GROQ_API_KEY for Groq, AI_OPENAI_API_KEY for OpenAI, "
                "AI_ANTHROPIC_API_KEY)."
            )

        message = options["message"] or DEFAULT_TEST_MESSAGE
        messages = [
            ChatMessage(role="system", content="You are ContribKit's AI assistant."),
            ChatMessage(role="user", content=message),
        ]

        self.stdout.write(self.style.WARNING("Sending test message to LLM provider..."))
        try:
            response = service.chat(
                messages,
                model=options["model"],
                temperature=options["temperature"],
                max_tokens=options["max-tokens"],
            )
        except AIServiceError as exc:
            raise CommandError(f"AI request failed: {exc}")

        self.stdout.write(self.style.SUCCESS("✓ AI request completed"))
        self.stdout.write(f"Provider : {response.provider}")
        self.stdout.write(f"Model    : {response.model}")
        if response.usage:
            self.stdout.write(
                "Usage    : prompt={} completion={} total={}".format(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    response.usage.total_tokens,
                )
            )
        self.stdout.write("")
        self.stdout.write(response.content)
