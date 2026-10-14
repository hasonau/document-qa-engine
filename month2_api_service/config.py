"""Configurable limits for Days 28-29. Defaults match the decisions specified up front."""
from __future__ import annotations

import os


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _str_env(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


# Upload
MAX_UPLOAD_BYTES = _int_env("MAX_UPLOAD_BYTES", 10 * 1024 * 1024)  # 10 MB
MAX_PDF_PAGES = _int_env("MAX_PDF_PAGES", 100)
MAX_DOCS_PER_SESSION = _int_env("MAX_DOCS_PER_SESSION", 10)
MIN_EXTRACTABLE_CHARS = _int_env("MIN_EXTRACTABLE_CHARS", 200)
SAFE_FILENAME_MAX_LEN = _int_env("SAFE_FILENAME_MAX_LEN", 100)

# Query
MAX_QUERY_CHARS = _int_env("MAX_QUERY_CHARS", 1000)

# Rate limits (per session, sliding 60s window)
QUERY_RATE_LIMIT_PER_MINUTE = _int_env("QUERY_RATE_LIMIT_PER_MINUTE", 20)
UPLOAD_RATE_LIMIT_PER_MINUTE = _int_env("UPLOAD_RATE_LIMIT_PER_MINUTE", 5)
RATE_LIMIT_WINDOW_SECONDS = _int_env("RATE_LIMIT_WINDOW_SECONDS", 60)

# CORS — comma-separated origins
DEFAULT_ALLOWED_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"


def allowed_origins() -> list[str]:
    raw = _str_env("ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS)
    return [part.strip() for part in raw.split(",") if part.strip()]
