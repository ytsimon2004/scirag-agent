"""PDF ingestion: text extraction and Results-section isolation.

Used by manual import commands (`scirag import-pdf / import-dir`) and
by the Unpaywall fallback in sources/pubmed.py.
"""

from __future__ import annotations

import hashlib
import re
import warnings
from pathlib import Path

import pypdf

from scirag.sources import pubmed
from scirag.sources.pubmed import Article

# DOI anywhere in the extracted text (publisher PDFs print it on the first page).
_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")

# Years plausible as a publication date.
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

# Lines that are page furniture, not the paper's title.
_TITLE_SKIP_RE = re.compile(
    r"^(article|letter|review|open|www\.|https?://|doi[:\s]|received|accepted|published)\b",
    re.IGNORECASE,
)

# Headings that start the Results section (and variants like "Results and Discussion")
_RESULTS_RE = re.compile(
    r"(?m)^\s*(results?(\s+and\s+(discussion|analysis|interpretation))?)\s*[:\.]?\s*$",
    re.IGNORECASE,
)

# Headings that end the Results section
_END_SECTION_RE = re.compile(
    r"(?m)^\s*(discussion|methods?|materials?\s*and\s*methods?|"
    r"experimental(\s+procedures?)?|conclusion|references?|"
    r"acknowledgements?|supplementary|bibliography)\s*[:\.]?\s*$",
    re.IGNORECASE,
)


def extract_text_from_pdf(path: Path) -> str:
    """Extract all text from a PDF, page by page."""
    reader = pypdf.PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_results_section(text: str) -> str:
    """Return the Results section from plain text, or '' if not found."""
    match = _RESULTS_RE.search(text)
    if not match:
        return ""
    start = match.end()
    end_match = _END_SECTION_RE.search(text, start)
    end = end_match.start() if end_match else len(text)
    return text[start:end].strip()


def _pmid_from_stem(stem: str) -> str:
    """Return stem if it looks like a PMID (all digits), else a short hash."""
    return stem if stem.isdigit() else f"pdf:{hashlib.md5(stem.encode()).hexdigest()[:8]}"


def _extract_doi(text: str) -> str:
    """Pull the first DOI out of the extracted text, or '' if none found."""
    m = _DOI_RE.search(text)
    if not m:
        return ""
    # Strip trailing sentence/line punctuation that the regex may have swallowed.
    return m.group(0).rstrip(".,;)")


def _extract_year(text: str) -> str:
    """Best-effort publication year from the first page, or '' (offline fallback)."""
    years = [int(y) for y in _YEAR_RE.findall(text[:3000])]
    return str(max(years)) if years else ""


def _guess_title(text: str) -> str:
    """Pick the most title-like line, skipping page furniture (offline fallback)."""
    for ln in text.splitlines():
        s = ln.strip()
        if len(s) < 15 or "://" in s or s.startswith("10."):
            continue
        if _TITLE_SKIP_RE.match(s):
            continue
        return s[:200]
    return ""


def _resolve_via_pubmed(doi: str) -> Article | None:
    """Resolve a real PubMed record from a DOI, or None if unavailable/offline."""
    try:
        pmids = pubmed.search(f"{doi}[doi]", retmax=1)
        if not pmids:
            return None
        arts = pubmed.fetch(pmids)
        return arts[0] if arts else None
    except Exception:
        return None  # NCBI unreachable / offline — caller falls back to local extraction


def load_pdf_as_article(path: Path) -> Article:
    """Load a single PDF as an Article.

    Metadata resolution, best to worst:
    1. If a DOI is found in the text, look up the real PubMed record (correct
       PMID, title, year, journal, MeSH) and graft the PDF's Results text onto it.
    2. Otherwise fall back to local extraction: PMID from filename (or a hash),
       year and title guessed from the text.

    - full_text: Results section only. Empty if no Results section detected.
    - abstract: left empty for the local fallback (not reliably parseable).
    """
    text = extract_text_from_pdf(path)
    results = extract_results_section(text)

    if not results:
        warnings.warn(
            f"{path.name}: no Results section found — article will not contribute text to the index.",
            stacklevel=2,
        )

    doi = _extract_doi(text)

    if doi:
        resolved = _resolve_via_pubmed(doi)
        if resolved is not None:
            resolved.full_text = results
            return resolved
        warnings.warn(
            f"{path.name}: DOI {doi} found but no PubMed match (or offline) — "
            "using metadata extracted from the PDF.",
            stacklevel=2,
        )

    title = _guess_title(text) or next(
        (ln.strip() for ln in text.splitlines() if ln.strip()), path.stem
    )
    return Article(
        pmid=_pmid_from_stem(path.stem),
        title=title[:200],
        abstract="",
        year=_extract_year(text),
        doi=doi,
        full_text=results,
    )


def load_pdf_directory(dir_path: Path) -> list[Article]:
    """Load all *.pdf files in dir_path as Articles. Skips files that fail."""
    articles: list[Article] = []
    pdfs = sorted(dir_path.glob("*.pdf"))
    if not pdfs:
        warnings.warn(f"No PDF files found in {dir_path}", stacklevel=2)
        return articles
    for pdf_path in pdfs:
        try:
            articles.append(load_pdf_as_article(pdf_path))
        except Exception as exc:
            warnings.warn(f"Skipping {pdf_path.name}: {exc}", stacklevel=2)
    return articles
