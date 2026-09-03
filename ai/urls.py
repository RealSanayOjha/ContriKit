from django.urls import path

from . import views

urlpatterns = [
    path("health/", views.ai_health_view, name="ai_health"),
]
