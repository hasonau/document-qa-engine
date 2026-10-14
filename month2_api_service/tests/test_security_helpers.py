"""Unit tests for filename sanitizer, validators, and rate limiter (stdlib unittest)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Repo root on sys.path when run as: python -m unittest ...
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from month2_api_service.services.security import (
    InMemoryRateLimiter,
    is_llm_upstream_failure,
    is_pdf_magic,
    sanitize_filename,
    validate_query_text,
)


class TestSanitizeFilename(unittest.TestCase):
    def test_strips_path_parts(self):
        self.assertEqual(sanitize_filename("../etc/passwd.pdf"), "passwd.pdf")
        self.assertEqual(
            sanitize_filename("..\\..\\windows\\evil.pdf"), "evil.pdf"
        )

    def test_unsafe_chars(self):
        out = sanitize_filename('my file (1)#$.PDF')
        self.assertTrue(out.lower().endswith(".pdf"))
        self.assertNotIn(" ", out)
        self.assertNotIn("#", out)
        self.assertNotIn("$", out)

    def test_empty_and_none(self):
        self.assertEqual(sanitize_filename(""), "document.pdf")
        self.assertEqual(sanitize_filename(None), "document.pdf")

    def test_long_name_truncated(self):
        long_name = "a" * 200 + ".pdf"
        out = sanitize_filename(long_name, max_len=40)
        self.assertLessEqual(len(out), 40)
        self.assertTrue(out.endswith(".pdf"))


class TestValidators(unittest.TestCase):
    def test_pdf_magic(self):
        self.assertTrue(is_pdf_magic(b"%PDF-1.4 rest"))
        self.assertFalse(is_pdf_magic(b"not a pdf"))
        self.assertFalse(is_pdf_magic(b"%PD"))

    def test_query_validation(self):
        self.assertEqual(validate_query_text("  ", 1000), "empty")
        self.assertEqual(validate_query_text("", 1000), "empty")
        self.assertEqual(validate_query_text("x" * 1001, 1000), "too_long")
        self.assertIsNone(validate_query_text("hello", 1000))


class TestRateLimiter(unittest.TestCase):
    def test_allows_then_blocks(self):
        lim = InMemoryRateLimiter()
        now = 1000.0
        for _ in range(3):
            ok, retry = lim.check("k", 3, 60, now=now)
            self.assertTrue(ok)
            self.assertEqual(retry, 0)
            now += 0.01
        ok, retry = lim.check("k", 3, 60, now=now)
        self.assertFalse(ok)
        self.assertGreaterEqual(retry, 1)

    def test_window_resets(self):
        lim = InMemoryRateLimiter()
        t0 = 1000.0
        for i in range(2):
            ok, _ = lim.check("k2", 2, 60, now=t0 + i)
            self.assertTrue(ok)
        ok, _ = lim.check("k2", 2, 60, now=t0 + 2)
        self.assertFalse(ok)
        ok, _ = lim.check("k2", 2, 60, now=t0 + 61)
        self.assertTrue(ok)


class TestLlmFailureDetection(unittest.TestCase):
    def test_runtime_ask_prefix(self):
        self.assertTrue(is_llm_upstream_failure(RuntimeError("ask: timeout")))

    def test_plain_value_error(self):
        self.assertFalse(is_llm_upstream_failure(ValueError("bad input")))


if __name__ == "__main__":
    unittest.main()
