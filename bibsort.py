#!/usr/bin/env python3
"""Shared BibTeX helpers for loading, sorting, and writing bibliography files."""

from __future__ import annotations

import re
from pathlib import Path

import bibtexparser
from bibtexparser.bwriter import BibTexWriter

MONTH_ALIASES = {
    "jan": "1",
    "january": "1",
    "feb": "2",
    "february": "2",
    "mar": "3",
    "march": "3",
    "apr": "4",
    "april": "4",
    "may": "5",
    "jun": "6",
    "june": "6",
    "jul": "7",
    "july": "7",
    "aug": "8",
    "august": "8",
    "sep": "9",
    "sept": "9",
    "september": "9",
    "oct": "10",
    "october": "10",
    "nov": "11",
    "november": "11",
    "dec": "12",
    "december": "12",
}

MONTH_FIELD_PATTERN = re.compile(
    r'(?i)(\bmonth\s*=\s*)(\{[^{}]*\}|"[^"]*"|[^,\n}]+)'
)


def load_bibliography(path: Path):
    """Load a BibTeX file, or return an empty database if the file is missing."""
    if not path.exists() or path.stat().st_size == 0:
        return bibtexparser.loads("")
    return parse_bibtex_text(path.read_text(encoding="utf-8"))


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


def _month_token(value: str) -> str:
    return re.sub(r"[^a-z]", "", value.lower())


def _normalize_month_value(value: str) -> str:
    """Convert recognized BibTeX month values to integer month strings."""
    cleaned = str(value).strip()
    if not cleaned:
        return cleaned

    # Accept legacy names, abbreviations, and quoted month macros.
    token = _month_token(cleaned)
    if token in MONTH_ALIASES:
        return MONTH_ALIASES[token]

    if cleaned.isdigit():
        month_number = int(cleaned)
        if 1 <= month_number <= 12:
            return str(month_number)

    return cleaned


def normalize_month_fields(db):
    """Normalize recognized month fields across all bibliography entries."""
    for entry in getattr(db, "entries", []):
        normalize_month_entry(entry)
    return db


def normalize_month_entry(entry: dict):
    """Normalize the month field for a single BibTeX entry in place."""
    month = entry.get("month")
    if month is not None:
        entry["month"] = _normalize_month_value(month)
    return entry


def _normalize_month_field_match(match: re.Match[str]) -> str:
    prefix, raw_value = match.groups()
    stripped = raw_value.strip()

    if (
        (stripped.startswith("{") and stripped.endswith("}"))
        or (stripped.startswith('"') and stripped.endswith('"'))
    ):
        normalized = _normalize_month_value(stripped[1:-1])
    else:
        normalized = _normalize_month_value(stripped)

    return f"{prefix}{{{normalized}}}"


def normalize_months_in_bibtex_text(text: str) -> str:
    """Normalize month fields in raw BibTeX text before parsing."""
    return MONTH_FIELD_PATTERN.sub(_normalize_month_field_match, text)


def parse_bibtex_text(text: str):
    """Parse BibTeX text after normalizing month fields."""
    db = bibtexparser.loads(normalize_months_in_bibtex_text(text))
    return normalize_month_fields(db)


def write_sorted_bibliography(path: Path, db=None) -> bool:
    """Write a bibliography file sorted by author order.

    Returns True if a non-empty bibliography was written.
    """
    if db is None:
        db = load_bibliography(path)

    if not getattr(db, "entries", None) and not getattr(db, "comments", None) and not getattr(db, "preambles", None):
        return False

    normalize_month_fields(db)
    sort_bibliography(db)
    writer = BibTexWriter()
    writer.order_entries_by = None
    text = writer.write(db).rstrip()
    if not text:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    return True
