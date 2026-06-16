"""Zotero source — import papers from the local library (offline).

Zotero keeps the whole library in a single SQLite database (`zotero.sqlite`) in
its data directory, and stores each attachment as a file under `storage/<key>/`.
We read it directly, so import works fully offline — no API key / web sync. Unlike
Mendeley's FTS shadow tables, Zotero uses a normalised relational schema:

  items(itemID, itemTypeID, key, …) — one row per library entry (a top-level
      reference *or* a child attachment/note).
  itemData / itemDataValues / fields — field values. `title`, `abstractNote`,
      `publicationTitle` (journal), `date`, `DOI`, `extra` are read by name.
  itemCreators / creators — authors, as separate first/last names ordered by
      `orderIndex`; `fieldMode = 1` marks a single-field name (an institution).
  itemAttachments(itemID, parentItemID, contentType, path) — a `path` of
      `storage:foo.pdf` resolves to `storage/<attachment key>/foo.pdf`.
  deletedItems — items in the trash, excluded from search.

The PMID isn't a first-class Zotero field, so we mine it from the `extra` blob
(CSL "PMID: 12345678" convention); items imported from Crossref carry only a DOI,
so we resolve that DOI to its PMID via PubMed esearch (best-effort — falls back to
a local key offline). Full text reuses Zotero's own extraction cache
(`storage/<key>/.zotero-ft-cache`) when present, else reads the PDF.

Builds the same `Article` as the other sources, reusing pdf.py's Results-section
isolation. Dedup key (the system-wide primary key): PMID when present or resolvable
from the DOI (dedups with /index), bioRxiv DOI for 10.1101 preprints (dedups with
/bindex), else a `zotero-<item id>` key.
"""

from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path

from scirag.config import pipeline_cfg
from scirag.sources.pdf import (
    _extract_doi,
    _strip_back_matter,
    extract_results_section,
    extract_text_from_pdf,
)
from scirag.sources.pubmed import Article

# PMID inside the `extra` blob, CSL convention: "PMID: 12345678".
_PMID_RE = re.compile(r"\bPMID:\s*(\d{4,8})\b", re.IGNORECASE)
# DOI fallback if the dedicated DOI field is empty (e.g. "DOI: 10.…" in extra).
_DOI_RE = re.compile(r"10\.\d{4,9}/\S+")

# Item types that are children of a reference, not references themselves.
_NON_REFERENCE_TYPES = ("attachment", "note", "annotation")


def _cfg() -> dict:
    """sources.zotero block from pipeline.yaml ({} if absent)."""
    try:
        return (pipeline_cfg().get("sources", {}) or {}).get("zotero", {}) or {}
    except Exception:
        return {}


def _data_dir() -> Path:
    """Zotero's default data directory (~/Zotero on every OS)."""
    return Path.home() / "Zotero"


def db_path() -> Path | None:
    """Locate the Zotero SQLite database, or None if not found.

    Resolution order: ZOTERO_DB_PATH env var -> sources.zotero.db_path in
    pipeline.yaml -> the default data directory (~/Zotero/zotero.sqlite).
    """
    override = os.getenv("ZOTERO_DB_PATH") or _cfg().get("db_path")
    if override:
        p = Path(str(override)).expanduser()
        return p if p.exists() else None
    p = _data_dir() / "zotero.sqlite"
    return p if p.exists() else None


def _storage_dir() -> Path | None:
    """Directory of attached files (storage/<key>/…), or None if not found.

    Resolution order: sources.zotero.storage_path in pipeline.yaml -> derived
    next to the resolved DB (<dir>/zotero.sqlite -> <dir>/storage) so a db_path
    override fixes both -> the default data directory.
    """
    cfg_path = _cfg().get("storage_path")
    if cfg_path:
        d = Path(str(cfg_path)).expanduser()
        return d if d.is_dir() else None
    db = db_path()
    if db is not None:
        derived = db.parent / "storage"
        if derived.is_dir():
            return derived
    d = _data_dir() / "storage"
    return d if d.is_dir() else None


def _connect(path: Path) -> sqlite3.Connection:
    """Open the library read-only (Zotero holds a SQLite lock while running)."""
    return sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)


def _to_ncbi(first: str, last: str) -> str:
    """Render ("Andrew S.", "Alexander") as NCBI-style "Alexander AS".

    Matches the author format used by the PubMed/bioRxiv sources so cite.citation()
    and _authors_short() read the same across all sources. A single-field creator
    (institution) has no first name, so it passes through unchanged.
    """
    last = (last or "").strip()
    first = (first or "").strip()
    if not first:
        return last
    initials = "".join(p[0] for p in first.split() if p[:1].isalpha())
    return f"{last} {initials}" if initials else last


def _fields(con: sqlite3.Connection, item_id: str) -> dict[str, str]:
    """All named field values for one item, keyed by Zotero field name."""
    rows = con.execute(
        "SELECT f.fieldName, idv.value FROM itemData id "
        "JOIN itemDataValues idv ON id.valueID = idv.valueID "
        "JOIN fields f ON id.fieldID = f.fieldID "
        "WHERE id.itemID = ?",
        (item_id,),
    ).fetchall()
    return {name: str(val) for name, val in rows}


def _authors(con: sqlite3.Connection, item_id: str) -> list[str]:
    """Authors for one item, NCBI-formatted, in Zotero's stored order."""
    rows = con.execute(
        "SELECT c.firstName, c.lastName FROM itemCreators ic "
        "JOIN creators c ON ic.creatorID = c.creatorID "
        "WHERE ic.itemID = ? ORDER BY ic.orderIndex",
        (item_id,),
    ).fetchall()
    return [_to_ncbi(first or "", last or "") for first, last in rows]


def _attachment(con: sqlite3.Connection, item_id: str) -> tuple[str, Path | None]:
    """(attachment key, PDF path) for an item's first PDF child, else ("", None)."""
    rows = con.execute(
        "SELECT i.key, ia.path FROM itemAttachments ia "
        "JOIN items i ON ia.itemID = i.itemID "
        "WHERE ia.parentItemID = ? AND ia.contentType = 'application/pdf'",
        (item_id,),
    ).fetchall()
    storage = _storage_dir()
    for key, path in rows:
        path = path or ""
        if path.startswith("storage:") and storage is not None:
            pdf = storage / key / path[len("storage:") :]
            if pdf.exists():
                return key, pdf
        elif path:  # linked file: an absolute path on disk
            pdf = Path(path.replace("attachments:", "")).expanduser()
            if pdf.exists():
                return key, pdf
    return "", None


def _full_text(con: sqlite3.Connection, item_id: str) -> str:
    """Extracted full text for an item: Zotero's ft-cache if present, else the PDF."""
    key, pdf = _attachment(con, item_id)
    if key:
        storage = _storage_dir()
        cache = storage / key / ".zotero-ft-cache" if storage else None
        if cache and cache.exists():
            try:
                return cache.read_text(errors="ignore")
            except OSError:
                pass
    if pdf is not None:
        try:
            return extract_text_from_pdf(pdf)
        except Exception:
            return ""
    return ""


def _resolve_pmid(doi: str, title: str) -> str:
    """Map a Zotero item to its PubMed PMID via DOI, then title, or '' if none.

    Mirrors pdf.py's resolution: a Crossref DOI resolves exactly; failing that (a
    library entry with only a title), a title search is accepted only on a confident
    word-overlap match, guarding against esearch returning an unrelated top hit.
    Best-effort — returns '' when offline, or when the item isn't in PubMed at all
    (e.g. an unpublished manuscript or a preprint), leaving a local zotero-<id> key.
    """
    from scirag.sources import pubmed
    from scirag.sources.pdf import _titles_match

    if doi:
        try:
            pmids = pubmed.search(f"{doi}[doi]", retmax=1)
        except Exception:
            pmids = []
        if pmids:
            return pmids[0]
    if title:
        try:
            pmids = pubmed.search(f"{title}[title]", retmax=1)
            arts = pubmed.fetch(pmids) if pmids else []
        except Exception:
            arts = []
        if arts and _titles_match(title, arts[0].title):
            return arts[0].pmid
    return ""


def _build_article(con: sqlite3.Connection, item_id: str) -> Article:
    fields = _fields(con, item_id)
    extra = fields.get("extra", "")

    doi = (fields.get("DOI", "") or "").strip().rstrip(".,;)")
    if not doi:
        m = _DOI_RE.search(extra)
        if m:
            doi = m.group(0).rstrip(".,;)")
    m = _PMID_RE.search(extra)
    pmid = m.group(1) if m else ""

    text = _full_text(con, item_id)

    # A record with no DOI in its metadata (e.g. a preprint added with just a title)
    # may still print its DOI on the attached PDF's first page — bioRxiv stamps a
    # "bioRxiv preprint doi: …" watermark there. _extract_doi normalizes it.
    if not doi and text:
        doi = _extract_doi(text)

    # Zotero items imported from Crossref carry a DOI but no PMID; manual entries may
    # have only a title. Resolve to the canonical PMID (by DOI, else title) so the
    # record keys/dedups with /index and shows the right id — best-effort, falling
    # back to a local key when offline, a preprint, or otherwise not in PubMed.
    if not pmid and not doi.startswith("10.1101/"):
        pmid = _resolve_pmid(doi, fields.get("title", ""))

    if pmid:
        key, source = pmid, "pubmed"
    elif doi.startswith("10.1101/"):
        key, source = doi, "biorxiv"
    else:
        key, source = f"zotero-{item_id}", "zotero"

    # Prefer the isolated Results section; when isolation fails (Zotero's PDF text
    # often loses "Results" as a clean heading) fall back to the whole body minus
    # references rather than the abstract — mirrors the Mendeley import.
    full_text, kind = "", "results"
    if text:
        full_text = extract_results_section(text)
        if not full_text and len(text) > 1000:
            body = _strip_back_matter(text)
            if len(body) > 500:
                full_text, kind = body, "fulltext"

    year = ""
    m = re.search(r"\d{4}", fields.get("date", ""))
    if m:
        year = m.group(0)
    # No date field (common for title-only entries) — a bioRxiv preprint DOI embeds
    # the posting date (10.1101/YYYY.MM.DD.NNNNNN), so the year isn't left blank.
    if not year and source == "biorxiv":
        m = re.match(r"10\.\d{4,9}/(\d{4})\.", key)
        if m:
            year = m.group(1)

    return Article(
        pmid=key,
        title=fields.get("title", ""),
        abstract=fields.get("abstractNote", ""),
        journal=fields.get("publicationTitle", ""),
        year=year,
        authors=_authors(con, item_id),
        doi=doi,
        full_text=full_text,
        full_text_kind=kind,
        source=source,
    )


def _searchable(con: sqlite3.Connection) -> dict[str, dict]:
    """Per-reference search index: {itemID: {text, year}} for top-level items.

    Loads title/abstract/date (bulk) and author names (bulk) for every reference
    that isn't an attachment/note/annotation and isn't trashed. Filtering happens
    in Python — Zotero's normalised schema has no FTS table to query directly.
    """
    refs = con.execute(
        "SELECT i.itemID FROM items i "
        "JOIN itemTypes it ON i.itemTypeID = it.itemTypeID "
        f"WHERE it.typeName NOT IN ({','.join('?' * len(_NON_REFERENCE_TYPES))}) "
        "AND i.itemID NOT IN (SELECT itemID FROM deletedItems)",
        _NON_REFERENCE_TYPES,
    ).fetchall()
    index: dict[str, dict] = {str(r[0]): {"text": "", "year": ""} for r in refs}

    rows = con.execute(
        "SELECT id.itemID, f.fieldName, idv.value FROM itemData id "
        "JOIN itemDataValues idv ON id.valueID = idv.valueID "
        "JOIN fields f ON id.fieldID = f.fieldID "
        "WHERE f.fieldName IN ('title', 'abstractNote', 'date')"
    ).fetchall()
    for item_id, name, value in rows:
        rec = index.get(str(item_id))
        if rec is None:
            continue
        if name == "date":
            m = re.search(r"\d{4}", str(value))
            if m:
                rec["year"] = m.group(0)
        else:
            rec["text"] += " " + str(value)

    rows = con.execute(
        "SELECT ic.itemID, c.firstName, c.lastName FROM itemCreators ic "
        "JOIN creators c ON ic.creatorID = c.creatorID"
    ).fetchall()
    for item_id, first, last in rows:
        rec = index.get(str(item_id))
        if rec is not None:
            rec["text"] += f" {first or ''} {last or ''}"

    return index


def search(query: str, retmax: int = 25) -> list[str]:
    """Return Zotero item ids matching `query` (title/abstract/authors).

    Every word must appear (AND) somewhere in a reference's searchable text. Empty
    query returns the most recent `retmax` references. Newest first by year.
    """
    path = db_path()
    if path is None:
        return []
    con = _connect(path)
    try:
        index = _searchable(con)
    finally:
        con.close()

    words = [w.lower() for w in query.split()]
    items = [
        (item_id, rec)
        for item_id, rec in index.items()
        if all(w in rec["text"].lower() for w in words)
    ]
    items.sort(key=lambda kv: (kv[1]["year"], kv[0]), reverse=True)
    return [item_id for item_id, _ in items[:retmax]]


def fetch(item_ids: list[str]) -> list[Article]:
    """Build Article records for the given Zotero item ids (order preserved)."""
    if not item_ids:
        return []
    path = db_path()
    if path is None:
        return []
    con = _connect(path)
    try:
        return [_build_article(con, item_id) for item_id in item_ids]
    finally:
        con.close()


def search_and_fetch(query: str, retmax: int = 25) -> list[Article]:
    """Search the local Zotero library and return the matching Articles."""
    return fetch(search(query, retmax=retmax))
