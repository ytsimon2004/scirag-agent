"""Mendeley Reference Manager source — import papers from the local library.

The modern Mendeley Reference Manager (the Electron app, *not* the old SQLite
"Mendeley Desktop") keeps the whole library in a local SQLite database and stores
each attachment as a PDF on disk. We read it directly, so import works fully
offline — no OAuth / REST API. The database holds only FTS5 virtual tables; the
shadow `*_fts_content` tables carry the original column values:

  documents_fts(id, title, authors, source, year, abstract, citation_key, tags,
                identifiers)
      -> *_content columns c0..c8 in that order. `source` is the publication
         venue (journal); `identifiers` is a space-joined blob with the DOI and
         PMID. Authors are space-joined full names with no per-author delimiter.
  files_fts(id, document_id, fulltext)
      -> c0 = file id (= userfiles/<file id>.pdf), c1 = document id, c2 = the
         PDF text Mendeley already extracted.

Builds the same `Article` as the other sources, reusing pdf.py's Results-section
isolation. Dedup key (the system-wide primary key): PMID when present (dedups with
/index), bioRxiv DOI for 10.1101 preprints (dedups with /bindex), else a
`mendeley-<doc id>` key.
"""

from __future__ import annotations

import os
import re
import sqlite3
import sys
from pathlib import Path

from scirag.config import pipeline_cfg
from scirag.sources.pdf import (
    _strip_back_matter,
    extract_results_section,
    extract_text_from_pdf,
)
from scirag.sources.article import Article

# DOI inside the identifiers blob (publisher DOIs have no spaces).
_DOI_RE = re.compile(r"10\.\d{4,9}/\S+")

# A name token that is an initial, e.g. "S.", "M.G.", "AB", "J-P" — used to
# segment the space-joined "First [Initials] Last" author strings.
_INITIAL_RE = re.compile(r"[A-Z](?:[.\-]?[A-Z])*\.?$")

_APP_DIR = "Mendeley Reference Manager"


def _platform_base() -> Path:
    """Mendeley Reference Manager's per-OS data directory (Electron userData).

    macOS: ~/Library/Application Support/…; Windows: %APPDATA%\\…; Linux:
    $XDG_CONFIG_HOME/… (else ~/.config/…).
    """
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / _APP_DIR
    if os.name == "nt":
        appdata = os.getenv("APPDATA")
        return (Path(appdata) if appdata else home / "AppData" / "Roaming") / _APP_DIR
    xdg = os.getenv("XDG_CONFIG_HOME")
    return (Path(xdg) if xdg else home / ".config") / _APP_DIR


def _cfg() -> dict:
    """sources.mendeley block from pipeline.yaml ({} if absent)."""
    try:
        return (pipeline_cfg().get("sources", {}) or {}).get("mendeley", {}) or {}
    except Exception:
        return {}


def db_path() -> Path | None:
    """Locate the Mendeley SQLite database, or None if not found.

    Resolution order: MENDELEY_DB_PATH env var -> sources.mendeley.db_path in
    pipeline.yaml -> auto-detect the per-OS install (newest `*.db`).
    """
    override = os.getenv("MENDELEY_DB_PATH") or _cfg().get("db_path")
    if override:
        p = Path(str(override)).expanduser()
        return p if p.exists() else None
    dbs = sorted(
        (_platform_base() / "mrm" / "databases").glob("*.db"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return dbs[0] if dbs else None


def _userfiles_dir() -> Path | None:
    """Directory of attached PDFs (userfiles/<file id>.pdf), or None if not found.

    Resolution order: sources.mendeley.userfiles_path in pipeline.yaml -> derived
    next to the resolved DB (<base>/mrm/databases/x.db -> <base>/userfiles) so a
    db_path override fixes both -> the per-OS default.
    """
    cfg_path = _cfg().get("userfiles_path")
    if cfg_path:
        d = Path(str(cfg_path)).expanduser()
        return d if d.is_dir() else None
    db = db_path()
    if db is not None:
        derived = db.parent.parent.parent / "userfiles"
        if derived.is_dir():
            return derived
    d = _platform_base() / "userfiles"
    return d if d.is_dir() else None


def _connect(path: Path) -> sqlite3.Connection:
    """Open the library read-only (never touch a live DB / its WAL)."""
    return sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)


def _is_initial(tok: str) -> bool:
    return bool(_INITIAL_RE.fullmatch(tok))


def _split_authors(raw: str) -> list[str]:
    """Segment a space-joined "First [Initials] Last …" string into authors.

    Mendeley's FTS store keeps authors as one flat string with no delimiter, so we
    segment heuristically: an author is a given name, any initials, then the first
    non-initial token (the surname); the next token starts a new author. Handles
    the common Western "First M. Last" form; multi-word surnames are rare and may
    split imperfectly (affects display only, not retrieval).
    """
    authors: list[str] = []
    cur: list[str] = []
    have_surname = False
    for tok in raw.split():
        if not cur:
            cur, have_surname = [tok], False
        elif _is_initial(tok):
            cur.append(tok)
        elif not have_surname:
            cur.append(tok)
            have_surname = True
        else:
            authors.append(" ".join(cur))
            cur, have_surname = [tok], False
    if cur:
        authors.append(" ".join(cur))
    return authors


def _to_ncbi(name: str) -> str:
    """Render "Andrew S. Alexander" as NCBI-style "Alexander AS" (surname + initials).

    Matches the author format used by the PubMed/bioRxiv sources so cite.citation()
    and _authors_short() read the same across all sources.
    """
    parts = name.split()
    if len(parts) < 2:
        return name
    surname = parts[-1]
    initials = "".join(p[0] for p in parts[:-1] if p[:1].isalpha())
    return f"{surname} {initials}" if initials else surname


# Max plausible PubMed PMID — comfortably above the current high-water mark
# (~40M in 2026), used to reject 8-digit Scopus IDs that share the PMID's width.
_PMID_MAX = 40_000_000


def _issn_checksum_ok(token: str) -> bool:
    """True if an 8-char token is a checksum-valid ISSN.

    The identifiers blob carries the journal ISSN as a bare 8-digit token that
    collides with the PMID's width; an ISSN's mod-11 check digit lets us tell them
    apart so the ISSN is never mistaken for a PMID (e.g. Nat Neurosci's 1546-1726).
    """
    if len(token) != 8 or not token[:7].isdigit():
        return False
    if not (token[7].isdigit() or token[7] in "Xx"):
        return False
    total = sum(int(d) * (8 - i) for i, d in enumerate(token[:7]))
    check = (11 - total % 11) % 11
    return token[7].upper() == ("X" if check == 10 else str(check))


def _parse_ids(blob: str) -> tuple[str, str]:
    """Pull (doi, pmid) from the unlabeled, space-joined identifiers blob.

    Mendeley joins ISSN, DOI, PMID, Scopus IDs and DOI fragments in an order that
    varies per record, so position is unreliable. The DOI is matched by regex; the
    PMID is the sole bare integer that's plausibly a PMID — 4–8 digits, ≤ the PMID
    high-water mark — once checksum-valid ISSNs, DOI fragments (e.g. "00959" → "959")
    and ≥9-digit / out-of-range Scopus IDs are dropped. Returns ("", "") for absent
    components; pmid is "" when zero OR more than one candidate survives (ambiguous),
    leaving the caller to resolve it from the DOI via PubMed.
    """
    doi = ""
    m = _DOI_RE.search(blob)
    if m:
        doi = m.group(0).rstrip(".,;)")
    candidates: list[str] = []
    for t in blob.split():
        if not (t.isdigit() and 4 <= len(t) <= 8):
            continue
        if doi and t in doi:  # a fragment of the DOI, not a separate identifier
            continue
        if _issn_checksum_ok(t):  # the journal ISSN, not a PMID
            continue
        if not 1 <= int(t) <= _PMID_MAX:  # an out-of-range Scopus id
            continue
        candidates.append(t)
    pmid = candidates[0] if len(candidates) == 1 else ""
    return doi, pmid


def _pmid_from_doi(doi: str) -> str:
    """Resolve a publisher DOI to its PMID via PubMed esearch, or "" if no match.

    The offline fallback for records whose identifiers blob yields no confident
    PMID. Best-effort: returns "" on any failure (offline, timeout, no hit) so the
    caller falls back to a bioRxiv-DOI or `mendeley-<id>` key.
    """
    if not doi:
        return ""
    try:
        from scirag.sources import pubmed

        hits = pubmed.search(f"{doi}[doi]", retmax=1)
    except Exception:
        return ""
    return hits[0] if hits else ""


def _row_to_article(row: tuple, fulltext: str, file_id: str) -> Article:
    doc_id, title, authors_raw, journal, year_raw, abstract, ids_blob = row

    doi, pmid = _parse_ids(ids_blob or "")
    # bioRxiv preprints stay keyed by their DOI (dedups with /bindex) and aren't in
    # PubMed, so skip the lookup. Otherwise, when the blob gave no confident PMID,
    # resolve it from the DOI via PubMed so the record dedups with /index.
    is_biorxiv = doi.startswith("10.1101/")
    if not pmid and doi and not is_biorxiv:
        pmid = _pmid_from_doi(doi)
    if pmid:
        key, source = pmid, "pubmed"
    elif is_biorxiv:
        key, source = doi, "biorxiv"
    else:
        key, source = f"mendeley-{doc_id}", "mendeley"

    # Full text: prefer Mendeley's extracted text; else read the PDF from disk.
    text = fulltext or ""
    if not text and file_id:
        uf = _userfiles_dir()
        pdf = uf / f"{file_id}.pdf" if uf else None
        if pdf and pdf.exists():
            try:
                text = extract_text_from_pdf(pdf)
            except Exception:
                text = ""

    # Prefer the isolated Results section. Mendeley's PDF text extraction often
    # doesn't preserve "Results" as a clean heading (two-column layouts, merged
    # page headers), so when isolation fails fall back to the whole body (minus
    # references) rather than the abstract — the user wants the paper's content.
    full_text, kind = "", "results"
    if text:
        full_text = extract_results_section(text)
        if not full_text and len(text) > 1000:
            body = _strip_back_matter(text)
            if len(body) > 500:
                full_text, kind = body, "fulltext"

    authors = [_to_ncbi(a) for a in _split_authors(authors_raw or "")]
    year = str(year_raw or "").split(".")[0]

    return Article(
        pmid=key,
        title=title or "",
        abstract=abstract or "",
        journal=journal or "",
        year=year,
        authors=authors,
        doi=doi,
        full_text=full_text,
        full_text_kind=kind,
        source=source,
    )


def search(query: str, retmax: int = 25) -> list[str]:
    """Return Mendeley document ids matching `query` (title/authors/abstract).

    Empty query returns the most recent `retmax` documents (newest first).
    """
    path = db_path()
    if path is None:
        return []
    con = _connect(path)
    try:
        words = query.split()
        if words:
            # Require every word to appear (AND), each in any of title/authors/
            # abstract — so "grid cells remapping" matches without being a phrase.
            clauses = " AND ".join("(c1 LIKE ? OR c2 LIKE ? OR c5 LIKE ?)" for _ in words)
            params: list = []
            for w in words:
                params += [f"%{w}%"] * 3
            params.append(retmax)
            rows = con.execute(
                f"SELECT c0 FROM documents_fts_content WHERE {clauses} "
                "ORDER BY CAST(c4 AS REAL) DESC LIMIT ?",
                params,
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT c0 FROM documents_fts_content ORDER BY CAST(c4 AS REAL) DESC LIMIT ?",
                (retmax,),
            ).fetchall()
        return [r[0] for r in rows]
    finally:
        con.close()


def fetch(doc_ids: list[str]) -> list[Article]:
    """Build Article records for the given Mendeley document ids (order preserved)."""
    if not doc_ids:
        return []
    path = db_path()
    if path is None:
        return []
    con = _connect(path)
    articles: list[Article] = []
    try:
        for doc_id in doc_ids:
            row = con.execute(
                "SELECT c0, c1, c2, c3, c4, c5, c8 FROM documents_fts_content WHERE c0 = ?",
                (doc_id,),
            ).fetchone()
            if row is None:
                continue
            # A document may have several attachments; take the first with text.
            files = con.execute(
                "SELECT c0, c2 FROM files_fts_content WHERE c1 = ?",
                (doc_id,),
            ).fetchall()
            file_id, fulltext = "", ""
            for fid, ftext in files:
                file_id = file_id or fid
                if ftext:
                    file_id, fulltext = fid, ftext
                    break
            articles.append(_row_to_article(row, fulltext, file_id))
    finally:
        con.close()
    return articles


def search_and_fetch(query: str, retmax: int = 25) -> list[Article]:
    """Search the local Mendeley library and return the matching Articles."""
    return fetch(search(query, retmax=retmax))
