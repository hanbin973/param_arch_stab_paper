#!/usr/bin/env python3
"""Shared BibTeX helpers for loading, sorting, and writing bibliography files."""

from __future__ import annotations

import re
from pathlib import Path

import bibtexparser
from bibtexparser.bwriter import BibTexWriter


def load_bibliography(path: Path):
    """Load a BibTeX file, or return an empty database if the file is missing."""
    if not path.exists() or path.stat().st_size == 0:
        return bibtexparser.loads("")
    return bibtexparser.loads(path.read_text(encoding="utf-8"))


def _normalize_sort_text(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", text.lower())


def _first_author_token(entry: dict) -> tuple[bool, str]:
    raw_author = str(entry.get("author") or entry.get("editor") or "").strip()
    if raw_author:
        first = re.split(r"\s+and\s+", raw_author, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        return False, first

    fallback = str(entry.get("title") or entry.get("ID") or "").strip()
    return True, fallback


def author_sort_key(entry: dict) -> tuple[str, str, str, str, str, str]:
    """Return a sort key that orders entries by the first author alphabetically."""
    missing_author, first = _first_author_token(entry)
    surname = ""
    given = ""
    if "," in first:
        surname, given = [part.strip() for part in first.split(",", 1)]
    else:
        parts = first.split()
        if parts:
            surname = parts[-1]
            given = " ".join(parts[:-1])

    year = str(entry.get("year", "")).strip().lower()
    title = str(entry.get("title", "")).strip().lower()
    key = str(entry.get("ID", "")).strip().lower()
    return (
        "1" if missing_author else "0",
        _normalize_sort_text(surname),
        _normalize_sort_text(given),
        year,
        title,
        key,
    )


def sort_bibliography(db):
    """Sort entries in-place by author key."""
    db.entries = sorted(db.entries, key=author_sort_key)
    return db


def write_sorted_bibliography(path: Path, db=None) -> bool:
    """Write a bibliography file sorted by author order.

    Returns True if a non-empty bibliography was written.
    """
    if db is None:
        db = load_bibliography(path)

    if not getattr(db, "entries", None) and not getattr(db, "comments", None) and not getattr(db, "preambles", None):
        return False

    sort_bibliography(db)
    writer = BibTexWriter()
    writer.order_entries_by = None
    text = writer.write(db).rstrip()
    if not text:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    return True
