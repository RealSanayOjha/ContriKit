"""Django system checks for the AI service configuration.

Why this exists: when Groq has no credentials the router short-circuits
*before* any HTTP call, so ``POST /ai/chat/`` answers
``503 {"code": "ai_unavailable"}`` and nothing is ever sent to Groq.
From the outside that is indistinguishable from a dead feature ("the
widget does nothing, there is no request to any API"), so the reason is
reported at startup by ``runserver`` / ``manage.py check`` instead of
only at request time.
"""

from django.conf import settings
from django.core.checks import Warning, register


def _key_setting(name: str) -> str:
    from ai.services.service import PROVIDER_KEY_SETTINGS

    return PROVIDER_KEY_SETTINGS.get(name, "")


@register()
def ai_provider_credentials(app_configs, **kwargs):
    """Warn when Groq has no API key."""
    from ai.services.service import _provider_names

    names = _provider_names()
    known = [name for name in names if _key_setting(name)]
    configured = [name for name in known if getattr(settings, _key_setting(name), "")]
    if configured:
        return []

    key_names = sorted({_key_setting(name) for name in known if _key_setting(name)}) or [
        "AI_GROQ_API_KEY",
    ]
    return [
        Warning(
            "The AI assistant has no Groq credentials, so /ai/chat/ will "
            "answer 503 'not configured' without calling Groq.",
            hint=(
                "Set at least one of these in .env (see .env.example): "
                + ", ".join(key_names)
                + ". Then verify with `python manage.py ai_test`."
            ),
            id="ai.W001",
        )
    ]
