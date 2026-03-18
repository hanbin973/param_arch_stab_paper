#!/usr/bin/env python3
"""Add ISBN-derived BibTeX entries to the current bibliography.

Usage:
    python isbnadd.py ISBN [ISBN ...]
    python isbnadd.py --bib references.bib ISBN [ISBN ...]

Accepted ISBN forms include bare ISBN-10/ISBN-13 values, hyphenated strings,
``isbn:...`` prefixes, and text containing an ISBN. The script normalizes the
input, looks up book metadata through public APIs, converts the result to a
BibTeX ``@book`` entry, and appends it to the target bibliography unless an
entry with the same BibTeX key or ISBN already exists.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import bibtexparser
import requests

OPEN_LIBRARY_URL = "https://openlibrary.org/api/books"
GOOGLE_BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"

ISBN_TOKEN = re.compile(r"(?i)(?:97[89][- ]?)?(?:\d[- ]?){9}[\dX]")
ISBN10_TOKEN = re.compile(r"(?i)(?:\d[- ]?){9}[\dX]")


@dataclass
class BookMetadata:
    """Normalized book metadata used to build a BibTeX entry."""

    title: str
    authors: list[str]
    publisher: str | None
    year: str | None
    isbn: str
    source_url: str | None = None


def clean_isbn(raw: str) -> str:
    """Extract and validate the first ISBN-like token from free-form text."""
    text = " ".join(raw.strip().split())
    candidates = []
    candidates.extend(match.group(0) for match in ISBN_TOKEN.finditer(text))
    candidates.extend(match.group(0) for match in ISBN10_TOKEN.finditer(text))

    seen: set[str] = set()
    for candidate in candidates:
        normalized = re.sub(r"[^0-9Xx]", "", candidate).upper()
        if normalized in seen:
            continue
        seen.add(normalized)
        if is_valid_isbn10(normalized) or is_valid_isbn13(normalized):
            return normalized

    raise ValueError(f"Could not parse a valid ISBN from: {raw!r}")


def is_valid_isbn10(value: str) -> bool:
    """Return True if *value* is a valid ISBN-10."""
    if len(value) != 10 or not re.fullmatch(r"\d{9}[\dX]", value):
        return False
    total = 0
    for idx, char in enumerate(value[:9]):
        total += (10 - idx) * int(char)
    check = 10 if value[-1] == "X" else int(value[-1])
    total += check
    return total % 11 == 0


def is_valid_isbn13(value: str) -> bool:
    """Return True if *value* is a valid ISBN-13."""
    if len(value) != 13 or not value.isdigit():
        return False
    total = 0
    for idx, char in enumerate(value):
        digit = int(char)
        total += digit if idx % 2 == 0 else 3 * digit
    return total % 10 == 0


def isbn10_to_isbn13(value: str) -> str:
    """Convert a valid ISBN-10 to ISBN-13."""
    if not is_valid_isbn10(value):
        raise ValueError(f"Invalid ISBN-10: {value}")
    core = "978" + value[:9]
    total = 0
    for idx, char in enumerate(core):
        digit = int(char)
        total += digit if idx % 2 == 0 else 3 * digit
    check = (10 - (total % 10)) % 10
    return core + str(check)


def isbn13_to_isbn10(value: str) -> str | None:
    """Convert a 978-prefixed ISBN-13 to ISBN-10, if possible."""
    if not is_valid_isbn13(value) or not value.startswith("978"):
        return None
    core = value[3:12]
    total = 0
    for idx, char in enumerate(core, start=1):
        total += (11 - idx) * int(char)
    check = (11 - (total % 11)) % 11
    return core + ("X" if check == 10 else str(check))


def isbn_variants(raw: str) -> list[str]:
    """Return normalized ISBN variants used for lookup and duplicate checks."""
    isbn = clean_isbn(raw)
    variants: list[str] = []
    if is_valid_isbn13(isbn):
        variants.append(isbn)
        isbn10 = isbn13_to_isbn10(isbn)
        if isbn10:
            variants.append(isbn10)
    elif is_valid_isbn10(isbn):
        variants.append(isbn)
        variants.append(isbn10_to_isbn13(isbn))
    else:
        raise ValueError(f"Unsupported ISBN format: {raw!r}")

    deduped: list[str] = []
    seen: set[str] = set()
    for item in variants:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def load_bibliography(path: Path):
    """Load an existing BibTeX file, or return an empty database if missing."""
    if not path.exists() or path.stat().st_size == 0:
        return bibtexparser.loads("")
    return bibtexparser.loads(path.read_text(encoding="utf-8"))


def existing_keys_and_isbns(db) -> tuple[set[str], set[str]]:
    """Collect existing BibTeX IDs and normalized ISBN values from a bibliography."""
    keys: set[str] = set()
    isbns: set[str] = set()
    for entry in getattr(db, "entries", []):
        entry_id = entry.get("ID")
        if entry_id:
            keys.add(entry_id.strip())

        for isbn_field in ("isbn", "ISBN"):
            isbn_value = entry.get(isbn_field)
            if not isbn_value:
                continue
            for variant in isbn_variants(str(isbn_value)):
                isbns.add(variant)
            break

    return keys, isbns


def http_get(url: str, params: dict[str, str]) -> requests.Response:
    """Fetch a URL with a short timeout and a book-friendly user agent."""
    return requests.get(
        url,
        params=params,
        timeout=20,
        headers={"User-Agent": "bibadd/1.0 (+https://doi.org/; ISBN lookup)"},
    )


def parse_open_library(item: dict, isbn: str) -> BookMetadata | None:
    """Parse Open Library API data into normalized book metadata."""
    if not item:
        return None

    title = item.get("title")
    if not title:
        return None

    authors = [author.get("name", "").strip() for author in item.get("authors", []) if author.get("name")]
    publisher = None
    publishers = item.get("publishers") or []
    if publishers:
        publisher = publishers[0].get("name")

    year = None
    publish_date = item.get("publish_date")
    if publish_date:
        match = re.search(r"(\d{4})", str(publish_date))
        if match:
            year = match.group(1)

    return BookMetadata(
        title=str(title).strip(),
        authors=[author for author in authors if author],
        publisher=publisher.strip() if isinstance(publisher, str) and publisher.strip() else None,
        year=year,
        isbn=isbn,
        source_url=item.get("url"),
    )


def parse_google_books(item: dict, isbn: str) -> BookMetadata | None:
    """Parse Google Books API data into normalized book metadata."""
    info = item.get("volumeInfo", {})
    title = info.get("title")
    if not title:
        return None

    authors = [author.strip() for author in info.get("authors", []) if author and author.strip()]
    publisher = info.get("publisher")
    published_date = info.get("publishedDate")
    year = None
    if published_date:
        match = re.search(r"(\d{4})", str(published_date))
        if match:
            year = match.group(1)

    source_url = info.get("infoLink")
    return BookMetadata(
        title=str(title).strip(),
        authors=authors,
        publisher=publisher.strip() if isinstance(publisher, str) and publisher.strip() else None,
        year=year,
        isbn=isbn,
        source_url=source_url,
    )


def fetch_book_metadata(raw_isbn: str) -> BookMetadata:
    """Lookup book metadata for an ISBN using Open Library, then Google Books."""
    variants = isbn_variants(raw_isbn)
    last_error: Exception | None = None

    for isbn in variants:
        params = {"bibkeys": f"ISBN:{isbn}", "format": "json", "jscmd": "data"}
        try:
            response = http_get(OPEN_LIBRARY_URL, params)
            response.raise_for_status()
            payload = response.json()
            item = payload.get(f"ISBN:{isbn}")
            metadata = parse_open_library(item, isbn)
            if metadata:
                return metadata
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    for isbn in variants:
        params = {"q": f"isbn:{isbn}"}
        try:
            response = http_get(GOOGLE_BOOKS_URL, params)
            response.raise_for_status()
            payload = response.json()
            items = payload.get("items") or []
            if items:
                metadata = parse_google_books(items[0], isbn)
                if metadata:
                    return metadata
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    if last_error is None:
        raise RuntimeError(f"No metadata found for ISBN: {raw_isbn}")
    raise RuntimeError(f"No metadata found for ISBN: {raw_isbn}") from last_error


def slugify(text: str) -> str:
    """Create a short BibTeX-safe slug from arbitrary text."""
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    if not words:
        return "book"
    return "".join(words[:4])


def author_surname(name: str) -> str:
    """Extract a compact surname from an author string."""
    if "," in name:
        return re.sub(r"[^A-Za-z0-9]+", "", name.split(",", 1)[0]).lower() or "anon"
    parts = name.split()
    if not parts:
        return "anon"
    return re.sub(r"[^A-Za-z0-9]+", "", parts[-1]).lower() or "anon"


def bib_key(metadata: BookMetadata) -> str:
    """Generate a stable BibTeX key from the first author, year, and title."""
    first_author = author_surname(metadata.authors[0]) if metadata.authors else "anon"
    year = metadata.year or "n.d."
    title_bits = slugify(metadata.title)
    return f"{first_author}{year}{title_bits}"


def escape_bibtex(value: str) -> str:
    """Escape a string for inclusion in a BibTeX field."""
    return (
        value.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


def format_authors(authors: Iterable[str]) -> str:
    """Format authors in BibTeX's ``and``-separated style."""
    cleaned = [a.strip() for a in authors if a and a.strip()]
    return " and ".join(cleaned)


def build_book_entry(metadata: BookMetadata) -> tuple[str, dict[str, str]]:
    """Build a BibTeX ``@book`` entry and its parsed field dictionary."""
    key = bib_key(metadata)
    fields: dict[str, str] = {
        "title": metadata.title,
        "isbn": metadata.isbn,
    }
    author_field = format_authors(metadata.authors)
    if author_field:
        fields["author"] = author_field
    if metadata.publisher:
        fields["publisher"] = metadata.publisher
    if metadata.year:
        fields["year"] = metadata.year
    if metadata.source_url:
        fields["url"] = metadata.source_url

    lines = [f"@book{{{key},"]
    for field in ("author", "title", "publisher", "year", "isbn", "url"):
        if field in fields:
            lines.append(f"  {field} = {{{escape_bibtex(fields[field])}}},")
    lines.append("}")
    return "\n".join(lines), {"ID": key, **fields}


def add_isbn_to_bib(raw_isbn: str, bib_path: Path) -> bool:
    """Add one ISBN-derived BibTeX entry to a bibliography.

    Usage:
        add_isbn_to_bib("9780131103627", Path("references.bib"))
        add_isbn_to_bib("isbn:9780131103627", Path("references.bib"))

    The input may be a bare ISBN, a hyphenated ISBN, a ``isbn:``-prefixed
    string, or free-form text containing an ISBN. If the generated BibTeX key
    or any normalized ISBN already exists in ``bib_path``, the entry is skipped.
    Returns ``True`` when an entry is added and ``False`` when it is skipped.
    """
    metadata = fetch_book_metadata(raw_isbn)
    raw_bibtex, new_entry = build_book_entry(metadata)

    db = load_bibliography(bib_path)
    existing_keys, existing_isbns = existing_keys_and_isbns(db)
    new_key = new_entry.get("ID", "").strip()
    new_isbns = set(isbn_variants(metadata.isbn))

    if new_key in existing_keys or bool(new_isbns & existing_isbns):
        print(
            f"Duplicate detected, skipping {raw_isbn!r} "
            f"(key={new_key!r}, isbn={metadata.isbn})",
            file=sys.stderr,
        )
        return False

    bib_path.parent.mkdir(parents=True, exist_ok=True)
    needs_separator = bib_path.exists() and bib_path.stat().st_size > 0
    with bib_path.open("a", encoding="utf-8") as handle:
        if needs_separator:
            handle.write("\n\n")
        handle.write(raw_bibtex.rstrip())
        handle.write("\n")

    print(f"Added {new_key} to {bib_path}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch ISBN BibTeX entries and append them to a bibliography file."
    )
    parser.add_argument(
        "isbn",
        nargs="+",
        help="ISBN, ISBN URL, or free-form text containing an ISBN to add.",
    )
    parser.add_argument(
        "--bib",
        default="references.bib",
        help="Target BibTeX file to update (default: references.bib).",
    )
    return parser.parse_args()


def main() -> int:
    """Command-line entry point for adding one or more ISBN-sourced entries."""
    args = parse_args()
    bib_path = Path(args.bib)
    for raw_isbn in args.isbn:
        try:
            add_isbn_to_bib(raw_isbn, bib_path)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to add {raw_isbn!r}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
