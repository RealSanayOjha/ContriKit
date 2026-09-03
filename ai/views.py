"""Thin views for the AI service layer.

The chatbot UI and the chat endpoint come in a later step. This view exists
so the service layer can be verified without a frontend and without touching
the database.
"""

from django.http import JsonResponse
from django.views.decorators.http import require_GET

from .services import get_ai_service


@require_GET
def ai_health_view(request):
    """Report AI service status.

    Safe for unauthenticated monitoring: it never calls the LLM API and
    returns only provider names, booleans, and model names — no secrets.
    """
    return JsonResponse(get_ai_service().health())
