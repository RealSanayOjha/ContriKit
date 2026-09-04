import os
from pathlib import Path
from decouple import config

BASE_DIR = Path(__file__).resolve().parent.parent.parent

SECRET_KEY = config('SECRET_KEY', default='django-insecure-dev-key-contribkit-2026')

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    # Third-party
    'axes',
    'social_django',

    # Custom apps
    'accounts',
    'repos',
    'issues',
    'templates_app',
    'cheatsheet',
    'core',
    'ai',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    # Social auth (Google OAuth) — handles OAuth errors/cancellation gracefully
    'social_django.middleware.SocialAuthExceptionMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    # Axes rate limiting middleware
    'axes.middleware.AxesMiddleware',
    'core.csp_middleware.CSPMiddleware',
]

ROOT_URLCONF = 'contribkit.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                # Publishes `{{ csrf_token }}` to every template. base.html
                # renders it into <meta name="csrf-token"> so the same-origin
                # fetch() calls (AI chat, issue bookmark) can read a token
                # without depending on document.cookie. This is one of
                # Django's default context processors; without it the
                # variable renders as an empty string.
                'django.template.context_processors.csrf',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'core.context_processors.role_context',
                # Axes context processor for template usage
                'social_django.context_processors.backends',
                'social_django.context_processors.login_redirect',
            ],
        },
    },
]

WSGI_APPLICATION = 'contribkit.wsgi.application'

# Authentication backends — axes must be first
AUTHENTICATION_BACKENDS = [
    'axes.backends.AxesStandaloneBackend',
    'social_core.backends.google.GoogleOAuth2',
    'django.contrib.auth.backends.ModelBackend',
]

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
        'OPTIONS': {'min_length': 8},
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

AUTH_USER_MODEL = 'accounts.User'

LOGIN_REDIRECT_URL = '/dashboard/'
LOGOUT_REDIRECT_URL = '/'
LOGIN_URL = '/accounts/login/'

# Session security
SESSION_COOKIE_AGE = 1209600  # 2 weeks in seconds
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Lax'
SESSION_EXPIRE_AT_BROWSER_CLOSE = False  # Set to True for higher security

# CSRF security
CSRF_COOKIE_HTTPONLY = False  # Must be False for JavaScript to read CSRF token
CSRF_COOKIE_SAMESITE = 'Lax'

# Security
SECURE_REFERRER_POLICY = 'same-origin'
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'

# Axes rate limiting configuration
AXES_ENABLED = config('AXES_ENABLED', default=True, cast=bool)
AXES_FAILURE_LIMIT = 5  # Lock after 5 failed attempts
AXES_COOLOFF_TIME = 0.5  # 30 minutes cooloff in hours
AXES_RESET_ON_SUCCESS = True  # Reset counter on successful login
# Axes v8 defaults to locking by username+IP combination

# Data upload limits
DATA_UPLOAD_MAX_MEMORY_SIZE = 5242880  # 5 MB
FILE_UPLOAD_MAX_MEMORY_SIZE = 5242880  # 5 MB

# Axes lockout template and URL
AXES_LOCKOUT_TEMPLATE = 'registration/locked_out.html'
AXES_LOCKOUT_URL = None  # Use template, not redirect

# ---------------------------------------------------------------------------
# Google OAuth 2.0 (python-social-auth / social-auth-app-django)
# Credentials are read from environment variables only — never committed.
# ---------------------------------------------------------------------------
SOCIAL_AUTH_JSONFIELD_ENABLED = True  # Store extra_data in a JSONField (works on SQLite)

SOCIAL_AUTH_GOOGLE_OAUTH2_KEY = config('GOOGLE_OAUTH2_CLIENT_ID', default='')
SOCIAL_AUTH_GOOGLE_OAUTH2_SECRET = config('GOOGLE_OAUTH2_CLIENT_SECRET', default='')

SOCIAL_AUTH_GOOGLE_OAUTH2_SCOPE = [
    'openid',
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/userinfo.profile',
]

# Optional GitHub Personal Access Token (PAT) for higher rate limits
# (5000/hr instead of 60/hr). Used server-side by repos.github_api and the
# AI GitHub tools; never exposed to templates or JavaScript.
GITHUB_PAT = config('GITHUB_PAT', default='')

# Where social auth should land / start / go on failure.
SOCIAL_AUTH_LOGIN_URL = '/accounts/login/'
SOCIAL_AUTH_LOGIN_REDIRECT_URL = '/dashboard/'
SOCIAL_AUTH_LOGIN_ERROR_URL = '/accounts/login/'

# Never re-raise social-auth exceptions (even when DEBUG=True). Instead, the
# SocialAuthExceptionMiddleware shows a user-friendly message and redirects back
# to the login page. This keeps cancellation / invalid callback / auth failures
# graceful in both local development and production.
SOCIAL_AUTH_RAISE_EXCEPTIONS = False

# Hosts a form submission may end up on, used to build the CSP `form-action`
# directive in core/csp_middleware.py.
#
# "Sign in with Google" POSTs a same-origin form to /login/google-oauth2/, which
# 302-redirects to Google's consent screen. Chrome/Chromium and Safari check
# `form-action` across the whole redirect chain of a form submission (Firefox
# does not), so without the Google host listed here the browser blocks the
# redirect: the button looks dead and the server log only shows a clean
# `"POST /login/google-oauth2/" 302 0`.
CSP_FORM_ACTION_EXTRA = [
    'https://accounts.google.com',
]

# Custom pipeline: add associate_by_email (before create_user) so a Google
# login whose verified email already belongs to an existing account is linked
# to that account instead of creating a duplicate user. Google verifies email
# addresses, so this association is safe.
SOCIAL_AUTH_PIPELINE = (
    'social_core.pipeline.social_auth.social_details',
    'social_core.pipeline.social_auth.social_uid',
    'social_core.pipeline.social_auth.auth_allowed',
    'social_core.pipeline.social_auth.social_user',
    'social_core.pipeline.user.get_username',
    'social_core.pipeline.social_auth.associate_by_email',
    'social_core.pipeline.user.create_user',
    'social_core.pipeline.social_auth.associate_user',
    'social_core.pipeline.social_auth.load_extra_data',
    'social_core.pipeline.user.user_details',
)

# ---------------------------------------------------------------------------
# AI Contribution Assistant (backend service layer)
#
# The `ai` app talks to LLM providers from the server only; API keys are read
# from the environment (.env) and are never exposed to templates or JavaScript.
# One OpenAI-compatible provider is wired now; multi-provider prioritization,
# fallback, rate limiting, and queueing are added in later steps.
# ---------------------------------------------------------------------------
# Provider list, in priority order (first key with credentials wins, then
# fallback). Accepts any of: openai, openai_compatible, groq, gemini,
# anthropic. Backward compatible: falls back to AI_PROVIDER when unset.
AI_PROVIDERS = config('AI_PROVIDERS', default='')
AI_PROVIDER = config('AI_PROVIDER', default='openai')

# Multi-provider router behavior
AI_PROVIDER_FALLBACK = config('AI_PROVIDER_FALLBACK', default=True, cast=bool)
AI_PROVIDER_RETRIES = config('AI_PROVIDER_RETRIES', default=2, cast=int)
AI_PROVIDER_RETRY_BACKOFF = config('AI_PROVIDER_RETRY_BACKOFF', default=0.5, cast=float)
AI_PROVIDER_CIRCUIT_BREAKER = config('AI_PROVIDER_CIRCUIT_BREAKER', default=True, cast=bool)
AI_PROVIDER_CIRCUIT_FAILURE_THRESHOLD = config('AI_PROVIDER_CIRCUIT_FAILURE_THRESHOLD', default=3, cast=int)
AI_PROVIDER_CIRCUIT_RESET_SECONDS = config('AI_PROVIDER_CIRCUIT_RESET_SECONDS', default=60, cast=int)

# ── OpenAI (native / OpenAI-compatible endpoints) ─────────────────────────
AI_OPENAI_API_KEY = config('AI_OPENAI_API_KEY', default='')
AI_OPENAI_BASE_URL = config('AI_OPENAI_BASE_URL', default='https://api.openai.com/v1')
AI_OPENAI_MODEL = config('AI_OPENAI_MODEL', default='gpt-4o-mini')
AI_OPENAI_TIMEOUT = config('AI_OPENAI_TIMEOUT', default=60, cast=int)
AI_OPENAI_MAX_TOKENS = config('AI_OPENAI_MAX_TOKENS', default=1024, cast=int)
AI_OPENAI_TEMPERATURE = config('AI_OPENAI_TEMPERATURE', default=0.7, cast=float)
AI_OPENAI_PRIORITY = config('AI_OPENAI_PRIORITY', default=10, cast=int)
AI_OPENAI_RATE_LIMIT = config('AI_OPENAI_RATE_LIMIT', default=0, cast=int)

# ── Groq (OpenAI-compatible API) ──────────────────────────────────────────
# Groq hosts open models (Llama, Qwen, ...) at api.groq.com. Keys start with
# gsk_ and the default model is llama-3.3-70b-versatile.
AI_GROQ_API_KEY = config('AI_GROQ_API_KEY', default='')
AI_GROQ_BASE_URL = config('AI_GROQ_BASE_URL', default='https://api.groq.com/openai/v1')
AI_GROQ_MODEL = config('AI_GROQ_MODEL', default='llama-3.3-70b-versatile')
AI_GROQ_TIMEOUT = config('AI_GROQ_TIMEOUT', default=60, cast=int)
AI_GROQ_MAX_TOKENS = config('AI_GROQ_MAX_TOKENS', default=1024, cast=int)
AI_GROQ_TEMPERATURE = config('AI_GROQ_TEMPERATURE', default=0.7, cast=float)
AI_GROQ_PRIORITY = config('AI_GROQ_PRIORITY', default=15, cast=int)
AI_GROQ_RATE_LIMIT = config('AI_GROQ_RATE_LIMIT', default=0, cast=int)


# ── Google Gemini (OpenAI-compatible endpoint) ────────────────────────────
AI_GEMINI_API_KEY = config('AI_GEMINI_API_KEY', default='')
AI_GEMINI_BASE_URL = config('AI_GEMINI_BASE_URL', default='https://generativelanguage.googleapis.com/v1beta/openai')
AI_GEMINI_MODEL = config('AI_GEMINI_MODEL', default='gemini-2.0-flash')
AI_GEMINI_TIMEOUT = config('AI_GEMINI_TIMEOUT', default=60, cast=int)
AI_GEMINI_MAX_TOKENS = config('AI_GEMINI_MAX_TOKENS', default=1024, cast=int)
AI_GEMINI_TEMPERATURE = config('AI_GEMINI_TEMPERATURE', default=0.7, cast=float)
AI_GEMINI_PRIORITY = config('AI_GEMINI_PRIORITY', default=30, cast=int)
AI_GEMINI_RATE_LIMIT = config('AI_GEMINI_RATE_LIMIT', default=0, cast=int)

# ── Anthropic Claude (native Messages API) ────────────────────────────────
AI_ANTHROPIC_API_KEY = config('AI_ANTHROPIC_API_KEY', default='')
AI_ANTHROPIC_BASE_URL = config('AI_ANTHROPIC_BASE_URL', default='https://api.anthropic.com/v1')
AI_ANTHROPIC_MODEL = config('AI_ANTHROPIC_MODEL', default='claude-sonnet-4-5-20250929')
AI_ANTHROPIC_TIMEOUT = config('AI_ANTHROPIC_TIMEOUT', default=60, cast=int)
AI_ANTHROPIC_MAX_TOKENS = config('AI_ANTHROPIC_MAX_TOKENS', default=1024, cast=int)
AI_ANTHROPIC_TEMPERATURE = config('AI_ANTHROPIC_TEMPERATURE', default=0.7, cast=float)
AI_ANTHROPIC_PRIORITY = config('AI_ANTHROPIC_PRIORITY', default=40, cast=int)
AI_ANTHROPIC_RATE_LIMIT = config('AI_ANTHROPIC_RATE_LIMIT', default=0, cast=int)

# AI chat endpoint (works with the existing Django session — no DB changes)
AI_CHAT_RATE_LIMIT = config('AI_CHAT_RATE_LIMIT', default=10, cast=int)
AI_CHAT_RATE_WINDOW = config('AI_CHAT_RATE_WINDOW', default=60, cast=int)
AI_CHAT_MAX_MESSAGE_LENGTH = config('AI_CHAT_MAX_MESSAGE_LENGTH', default=2000, cast=int)
AI_CHAT_HISTORY_LIMIT = config('AI_CHAT_HISTORY_LIMIT', default=12, cast=int)
