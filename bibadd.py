#!/usr/bin/env python3
"""Add DOI-sourced BibTeX entries to the current bibliography.

Usage:
    python bibadd.py DOI [DOI ...]
    python bibadd.py --bib references.bib DOI [DOI ...]
    python bibadd.py --abstract DOI

Accepted DOI forms include bare DOIs, ``doi:...`` strings, and DOI URLs such
as ``https://doi.org/...`` or ``http://dx.doi.org/...``. The script fetches the
BibTeX entry through the installed ``doi2bib`` package, adds it to the target
bibliography file unless an entry with the same BibTeX key or DOI is already
present, and then rewrites the file in author alphabetical order.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import bibtexparser
from doi2bib.crossref import get_bib_from_doi

from bibsort import load_bibliography, write_sorted_bibliography

DOI_PATTERN = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.IGNORECASE)


def extract_doi(raw: str) -> str:
    """Extract a DOI from a raw DOI string, URL, or ``doi:``-prefixed input."""
    text = " ".join(raw.strip().split())
    match = DOI_PATTERN.search(text)
    if not match:
        raise ValueError(f"Could not parse a DOI from: {raw!r}")
    doi = match.group(1).rstrip(").,;]}>\"'")
    return doi


def existing_keys_and_dois(db) -> tuple[set[str], set[str]]:
    """Collect existing BibTeX IDs and normalized DOI values from a bibliography."""
    keys: set[str] = set()
    dois: set[str] = set()
    for entry in getattr(db, "entries", []):
        entry_id = entry.get("ID")
        if entry_id:
            keys.add(entry_id.strip())

        for doi_field in ("doi", "DOI"):
            doi_value = entry.get(doi_field)
            if not doi_value:
                continue
            try:
                dois.add(extract_doi(doi_value).lower())
            except ValueError:
                dois.add(str(doi_value).strip().lower())
            break

    return keys, dois


def fetch_bibtex(raw_doi: str, abstract: bool = False) -> dict:
    """Fetch BibTeX for a DOI through ``doi2bib`` and return the parsed entry."""
    doi = extract_doi(raw_doi)
    found, bibtex = get_bib_from_doi(doi, add_abstract=abstract)
    if not found or not bibtex.strip():
        raise RuntimeError(f"No BibTeX entry found for DOI: {raw_doi}")

    db = bibtexparser.loads(bibtex)
    if not getattr(db, "entries", []):
        raise RuntimeError(f"doi2bib returned no parsable entries for DOI: {raw_doi}")

    return db.entries[0]


def add_doi_to_bib(raw_doi: str, bib_path: Path, abstract: bool = False) -> bool:
    """Add one DOI-derived BibTeX entry to a bibliography.

    Usage:
        add_doi_to_bib("10.1038/nature12373", Path("references.bib"))
        add_doi_to_bib("https://doi.org/10.1038/nature12373", Path("references.bib"), abstract=True)

    The DOI may be a bare DOI, a ``doi:``-prefixed string, or a DOI URL.
    If the fetched entry has the same BibTeX key or DOI as an existing entry in
    ``bib_path``, the function reports the duplicate and skips appending it.
    Returns ``True`` when an entry is added and ``False`` when it is skipped.
    """
    new_entry = fetch_bibtex(raw_doi, abstract=abstract)
    new_key = (new_entry.get("ID") or "").strip()
    new_doi = ""
    for field in ("doi", "DOI"):
        if new_entry.get(field):
            try:
                new_doi = extract_doi(str(new_entry[field])).lower()
            except ValueError:
                new_doi = str(new_entry[field]).strip().lower()
            break

    db = load_bibliography(bib_path)
    existing_keys, existing_dois = existing_keys_and_dois(db)

    if (new_key and new_key in existing_keys) or (new_doi and new_doi in existing_dois):
        print(
            f"Duplicate detected, skipping {raw_doi!r} "
            f"(key={new_key!r}, doi={new_doi or 'n/a'})",
            file=sys.stderr,
        )
        return False

    db.entries.append(new_entry)
    write_sorted_bibliography(bib_path, db)
    print(f"Added {new_key or raw_doi} to {bib_path}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch DOI BibTeX entries and append them to a bibliography file."
    )
    parser.add_argument(
        "doi",
        nargs="+",
        help="DOI, DOI URL, or doi:-prefixed string to add.",
    )
    parser.add_argument(
        "--bib",
        default="references.bib",
        help="Target BibTeX file to update (default: references.bib).",
    )
    parser.add_argument(
        "--abstract",
        action="store_true",
        help="Ask doi2bib to include an abstract when available.",
    )
    return parser.parse_args()


def main() -> int:
    """Command-line entry point for adding one or more DOI-sourced entries."""
    args = parse_args()
    bib_path = Path(args.bib)
    for raw_doi in args.doi:
        try:
            add_doi_to_bib(raw_doi, bib_path, abstract=args.abstract)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to add {raw_doi!r}: {exc}", file=sys.stderr)
            return 1
    write_sorted_bibliography(bib_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
