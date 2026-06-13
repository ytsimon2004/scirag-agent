"""PDF ingestion: resolve a PDF to its source record + isolate the Results section.

A PDF is only imported if it can be resolved to a real record (so the index never
holds guessed/garbage metadata). Resolution order: PMID (numeric filename) -> DOI
-> title search (all PubMed) -> bioRxiv DOI (preprints aren't in PubMed). If none
resolves, the PDF is skipped and a warning points the user at a PubMed URL.

Also used by the Unpaywall fallback in sources/pubmed.py (`extract_text_from_pdf`,
`extract_results_section`).
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from urllib.parse import quote

import pypdf

from scirag.sources import pubmed
from scirag.sources.pubmed import Article

# DOI anywhere in the extracted text (publisher PDFs print it on the first page).
_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")

# eLife prints a per-component DOI (…/eLife.NNNNN.00N); PubMed indexes the bare
# article DOI (…/eLife.NNNNN), so the trailing component must be stripped.
_ELIFE_COMPONENT_RE = re.compile(r"(10\.7554/eLife\.\d+)\.\d+$", re.IGNORECASE)

# A /Title that is only digits/spaces/dots is a typesetting artifact, not a title.
_BAD_TITLE_RE = re.compile(r"^[\d\s.]+$")

# Lines that are page furniture, not the paper's title.
_TITLE_SKIP_RE = re.compile(
    r"^(article|letter|review|open|www\.|https?://|doi[:\s]|received|accepted|published|"
    r"copyright|this is an open|creative commons|\*?for correspondence|competing interests|"
    r"funding|advance access|original article)\b",
    re.IGNORECASE,
)

# Optional section numbering before a heading: "3.", "3 ", "3.1.", "III." …
_SECTION_NUM = r"(?:\d{1,2}(?:\.\d{1,2})*\.?\s+|[IVX]{1,4}\.\s+)?"

# Optional trailing line/page number after a heading. bioRxiv/medRxiv manuscripts
# number every line, so a heading extracts as e.g. "RESULTS 83" / "DISCUSSION 399".
_LINE_NO = r"(?:\s+\d{1,4})?"

# Headings that start the Results section (and variants like "Results and Discussion")
_RESULTS_RE = re.compile(
    rf"(?m)^\s*{_SECTION_NUM}results?(\s+and\s+(discussion|analysis|interpretation))?"
    rf"\s*[:\.]?{_LINE_NO}\s*$",
    re.IGNORECASE,
)

# Back-matter heading (on its own line) marking where a review's body ends.
_BACK_MATTER_RE = re.compile(
    rf"(?im)^\s*(references|bibliography|acknowledgements?)\s*:?{_LINE_NO}\s*$"
)

# Headings that end the Results section
_END_SECTION_RE = re.compile(
    rf"(?m)^\s*{_SECTION_NUM}(discussion|methods?|materials?\s*and\s*methods?|"
    rf"experimental(\s+procedures?)?|conclusion|references?|"
    rf"acknowledgements?|supplementary|bibliography)\s*[:\.]?{_LINE_NO}\s*$",
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


def _strip_back_matter(text: str) -> str:
    """Drop the references/bibliography tail from a review's body text."""
    matches = list(_BACK_MATTER_RE.finditer(text))
    return text[: matches[-1].start()].strip() if matches else text.strip()


def _extract_doi(text: str) -> str:
    """Pull the first DOI out of the extracted text (normalized), or '' if none."""
    m = _DOI_RE.search(text)
    if not m:
        return ""
    # Strip trailing sentence/line punctuation that the regex may have swallowed.
    doi = m.group(0).rstrip(".,;)")
    # bioRxiv/medRxiv DOIs (…/YYYY.MM.DD.NNNNNN) pick up watermark text in the
    # extracted layer (e.g. "…661247doi:"); truncate to the canonical form.
    preprint = re.match(r"10\.\d{4,9}/\d{4}\.\d{2}\.\d{2}\.\d+", doi)
    if preprint:
        return preprint.group(0)
    # eLife's per-component suffix (…NNNNN.00N) isn't what PubMed indexes.
    elife = _ELIFE_COMPONENT_RE.match(doi)
    return elife.group(1) if elife else doi


def _clean_pdf_title(raw) -> str:
    """Return a usable title from a PDF's /Title metadata, or '' if missing/garbage."""
    if not isinstance(raw, str):
        return ""
    t = raw.strip()
    if len(t) < 10 or _BAD_TITLE_RE.match(t):
        return ""
    return t[:200]


def _pdf_meta_field(reader, field: str) -> str:
    """Read a /Title or /Author string from PDF document info, '' if absent."""
    info = getattr(reader, "metadata", None)
    if not info:
        return ""
    val = getattr(info, field, None)
    return val.strip() if isinstance(val, str) else ""


def _guess_title(text: str) -> str:
    """Pick the most title-like line, skipping page furniture."""
    for ln in text.splitlines():
        s = ln.strip()
        if len(s) < 15 or "://" in s or "@" in s or s.startswith("10.") or s.startswith("©"):
            continue
        if _TITLE_SKIP_RE.match(s):
            continue
        return s[:200]
    return ""


def _titles_match(a: str, b: str) -> bool:
    """True if titles `a` and `b` share enough words to be the same paper."""
    wa = set(re.findall(r"[a-z0-9]+", a.lower()))
    wb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa) >= 0.6


def _fetch_first(pmids: list[str]) -> Article | None:
    arts = pubmed.fetch(pmids) if pmids else []
    return arts[0] if arts else None


def _resolve_via_pmid(pmid: str) -> Article | None:
    """Fetch a PubMed record directly by PMID (numeric filename), or None."""
    try:
        return _fetch_first([pmid])
    except Exception:
        return None


def _resolve_via_doi(doi: str) -> Article | None:
    """Resolve a PubMed record from a DOI, or None if no match/offline."""
    try:
        return _fetch_first(pubmed.search(f"{doi}[doi]", retmax=1))
    except Exception:
        return None


def _resolve_via_title(title: str) -> Article | None:
    """Resolve a PubMed record by title search, accepting only a confident match."""
    if not title:
        return None
    try:
        art = _fetch_first(pubmed.search(f"{title}[title]", retmax=1))
    except Exception:
        return None
    if art is not None and _titles_match(title, art.title):
        return art
    return None


def _resolve_via_biorxiv(doi: str) -> Article | None:
    """Resolve a bioRxiv preprint DOI to an Article via the bioRxiv API, or None.

    Used as a fallback when a PDF doesn't resolve to PubMed — preprints aren't in
    PubMed, but the bioRxiv API knows them by DOI. The DOI lands in the Article's
    `pmid` slot with source="biorxiv", matching the /bindex path.
    """
    if not doi:
        return None
    try:
        from scirag.sources import biorxiv

        arts = biorxiv.fetch([doi])
    except Exception:
        return None
    return arts[0] if arts else None


def _warn_unresolved(path: Path, doi: str, title: str) -> None:
    if doi:
        term, detail = doi, f"DOI {doi}"
    elif title:
        term, detail = title, "no DOI found"
    else:
        term, detail = "", "no DOI or title found"
    url = (
        f"https://pubmed.ncbi.nlm.nih.gov/?term={quote(term)}"
        if term
        else "https://pubmed.ncbi.nlm.nih.gov/"
    )
    warnings.warn(
        f"{path.name}: could not resolve to a PubMed record ({detail}) — NOT imported.\n"
        f"  Check {url}\n"
        f"  then rename the file to <PMID>.pdf and re-import.",
        stacklevel=2,
    )


def load_pdf_as_article(path: Path) -> Article | None:
    """Resolve a PDF to its source record, with the PDF's Results text grafted on.

    Resolution order: PMID (numeric filename) -> DOI -> title (all PubMed) ->
    bioRxiv DOI. Returns None (and warns with a PubMed lookup URL) when none
    resolves, so unresolved PDFs are skipped rather than indexed with guessed
    metadata.
    """
    reader = pypdf.PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    results = extract_results_section(text)

    stem = path.stem
    doi = _extract_doi(text)
    candidate_title = _clean_pdf_title(_pdf_meta_field(reader, "title")) or _guess_title(text)

    resolved: Article | None = None
    if stem.isdigit():
        resolved = _resolve_via_pmid(stem)
    if resolved is None and doi:
        resolved = _resolve_via_doi(doi)
    if resolved is None and candidate_title:
        resolved = _resolve_via_title(candidate_title)
    # bioRxiv preprints aren't in PubMed — fall back to the bioRxiv API by DOI.
    if resolved is None and doi:
        resolved = _resolve_via_biorxiv(doi)

    if resolved is not None:
        resolved.full_text = results
        # Reviews have no Results section — index the whole body instead. Research
        # articles and preprints stay Results-only (abstract if none was found).
        if not results and resolved.is_review:
            body = _strip_back_matter(text)
            if body:
                resolved.full_text = body
                resolved.full_text_kind = "review"
        return resolved

    _warn_unresolved(path, doi, candidate_title)
    return None


def load_pdf_directory(dir_path: Path) -> list[Article]:
    """Load all *.pdf in dir_path as Articles. Skips files that fail to parse or
    that can't be resolved to a PubMed record (each emits a warning)."""
    articles: list[Article] = []
    pdfs = sorted(dir_path.glob("*.pdf"))
    if not pdfs:
        warnings.warn(f"No PDF files found in {dir_path}", stacklevel=2)
        return articles
    for pdf_path in pdfs:
        try:
            article = load_pdf_as_article(pdf_path)
        except Exception as exc:
            warnings.warn(f"Skipping {pdf_path.name}: {exc}", stacklevel=2)
            continue
        if article is not None:
            articles.append(article)
    return articles
