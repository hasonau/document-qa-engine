"""Pure helpers: filename sanitizer, validators, in-memory rate limiter."""
from __future__ import annotations

import re
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import PurePosixPath, PureWindowsPath


_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(filename: str | None, max_len: int = 100) -> str:
    """Reduce a client filename to a safe base name (no path parts)."""
    if not filename or not str(filename).strip():
        return "document.pdf"

    # Drop any directory components from POSIX or Windows-style paths.
    name = PureWindowsPath(str(filename)).name
    name = PurePosixPath(name).name
    name = name.replace("\x00", "")

    stem_ext = name
    safe = _SAFE_CHARS.sub("_", stem_ext).strip("._")
    if not safe:
        safe = "document.pdf"

    # Preserve a trailing .pdf when the client claimed one; otherwise append.
    lower = safe.lower()
    if not lower.endswith(".pdf"):
        # If there is another extension, still force .pdf for our store names.
        if "." in safe:
            safe = safe.rsplit(".", 1)[0] + ".pdf"
        else:
            safe = safe + ".pdf"

    if len(safe) > max_len:
        # Keep .pdf suffix when truncating.
        suffix = ".pdf"
        keep = max_len - len(suffix)
        if keep < 1:
            return (safe[:max_len]) or "d.pdf"
        safe = safe[:keep].rstrip("._") + suffix
        if safe == ".pdf" or safe == "pdf":
            safe = "document.pdf"

    return safe


def is_pdf_magic(data: bytes) -> bool:
    return len(data) >= 5 and data[:5] == b"%PDF-"


def validate_query_text(query: str, max_chars: int) -> str | None:
    """Return an error message if invalid, else None. Caller maps to status codes."""
    if query is None or not str(query).strip():
        return "empty"
    if len(query) > max_chars:
        return "too_long"
    return None


class InMemoryRateLimiter:
    """Sliding-window limiter. Resets on process restart; per instance only."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def check(
        self,
        key: str,
        limit: int,
        window_seconds: float,
        *,
        now: float | None = None,
    ) -> tuple[bool, int]:
        """Return (allowed, retry_after_seconds). retry_after is 0 when allowed."""
        if limit <= 0:
            return True, 0
        ts = time.monotonic() if now is None else now
        with self._lock:
            q = self._events[key]
            cutoff = ts - window_seconds
            while q and q[0] <= cutoff:
                q.popleft()
            if len(q) >= limit:
                retry_after = max(1, int(window_seconds - (ts - q[0])) + 1)
                return False, retry_after
            q.append(ts)
            return True, 0

    def reset(self) -> None:
        with self._lock:
            self._events.clear()


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def is_llm_upstream_failure(exc: BaseException) -> bool:
    """True when the failure is (or wraps) a Groq / language-model outage."""
    try:
        from groq import APIConnectionError, APIError, APIStatusError, APITimeoutError, InternalServerError, RateLimitError
    except ImportError:  # pragma: no cover
        APIError = ()  # type: ignore
        APIConnectionError = ()  # type: ignore
        APIStatusError = ()  # type: ignore
        APITimeoutError = ()  # type: ignore
        InternalServerError = ()  # type: ignore
        RateLimitError = ()  # type: ignore

    llm_types = (
        APIError,
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )

    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, llm_types):
            return True
        msg = str(cur).lower()
        # answer_question wraps Groq errors as RuntimeError("ask: ...")
        if msg.startswith("ask:") or "groq" in msg:
            # Prefer typed causes; string match is a fallback for wrapped errors.
            if isinstance(cur, RuntimeError) or any(
                isinstance(cur, t) for t in llm_types if t
            ):
                cause = cur.__cause__ or cur.__context__
                if cause is not None and isinstance(cause, llm_types):
                    return True
                if msg.startswith("ask:"):
                    return True
        cur = cur.__cause__ or cur.__context__
    return False
