"""Tests for the text cleaning helpers."""

from __future__ import annotations

from ingest.text_cleaner import clean_text, slugify, split_into_chunks


def test_clean_text_collapses_whitespace():
    assert clean_text("hello    \n\n  world  ") == "hello world"


def test_clean_text_removes_control_chars():
    assert "\x00" not in clean_text("a\x00b")


def test_clean_text_empty():
    assert clean_text("") == ""


def test_split_short_text():
    chunks = split_into_chunks("short text", max_chars=100)
    assert len(chunks) == 1


def test_split_long_text():
    text = "sentence. " * 200
    chunks = split_into_chunks(text, max_chars=100, overlap=10)
    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)


def test_slugify():
    assert slugify("Hello, World!") == "hello-world"
    assert slugify("a  b") == "a-b"
    assert slugify("!!!") == "document"
