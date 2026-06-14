"""Human-readable author-year citations (e.g. 'Powell et al., 2020') built from a
retrieved chunk's stored metadata. Shared by the shell and the web UI so the
source displays read the same. Falls back to the [PMID/DOI] id when no author is
stored (e.g. PDF imports without resolved metadata)."""

from __future__ import annotations


def _surname(first_author: str) -> str:
    """Pull the surname from an NCBI-style 'Surname II' author string.

    'Powell A' -> 'Powell'; 'van der Berg AB' -> 'van der Berg'; 'Alexander AS' ->
    'Alexander'. A trailing all-caps initials token is dropped."""
    parts = first_author.strip().split()
    if len(parts) > 1 and parts[-1].isupper() and len(parts[-1]) <= 3:
        parts = parts[:-1]
    return " ".join(parts)


def citation(md: dict) -> str:
    """Author-year citation for a chunk's metadata, e.g. 'Powell et al., 2020'.

    One author -> 'Powell, 2020'; multiple -> 'Powell et al., 2020'. Falls back to
    the bracketed identifier ('[PMID]') when no author surname is available."""
    surname = _surname(md.get("first_author", "") or "")
    year = str(md.get("year", "") or "").strip()
    if not surname:
        return f"[{md.get('pmid', '?')}]"
    multi = "," in (md.get("authors", "") or "")  # comma-joined list => >1 author
    name = f"{surname} et al." if multi else surname
    return f"{name}, {year}" if year else name
