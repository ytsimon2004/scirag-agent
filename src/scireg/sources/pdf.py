"""PDF ingestion: text extraction and Results-section isolation.

Used by manual import commands (`scireg import-pdf / import-dir`) and
by the Unpaywall fallback in sources/pubmed.py.
"""
from __future__ import annotations

import hashlib
import re
import warnings
from pathlib import Path

import pypdf

from scireg.sources.pubmed import Article

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
    return "\n".join(
        page.extract_text() or "" for page in reader.pages
    )


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


def load_pdf_as_article(path: Path) -> Article:
    """Load a single PDF as an Article.

    - PMID: taken from filename if numeric (e.g. '12345678.pdf'), else a hash.
    - full_text: Results section only. Empty if no Results section detected.
    - title: first non-empty line of the extracted text, capped at 200 chars.
    - abstract: left empty (not available from a raw PDF).
    """
    text = extract_text_from_pdf(path)
    results = extract_results_section(text)

    if not results:
        warnings.warn(
            f"{path.name}: no Results section found — article will not contribute text to the index.",
            stacklevel=2,
        )

    title = next((ln.strip() for ln in text.splitlines() if ln.strip()), path.stem)
    return Article(
        pmid=_pmid_from_stem(path.stem),
        title=title[:200],
        abstract="",
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
