"""PubMed retrieval via NCBI E-utilities (esearch + efetch).

This is the raw data-source client. The same functions are re-exported as MCP
tools in scirag.mcp_server so agents can call them through the MCP protocol.
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field
from xml.etree import ElementTree as ET

import httpx

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
UNPAYWALL = "https://api.unpaywall.org/v2"
# Europe PMC indexes MEDLINE/PubMed (SRC:MED) with a relevance-ranked search that,
# unlike NCBI esearch, tolerates natural-language phrasing — it won't misread
# "in human" as an author. Used by search_semantic() to acquire PubMed PMIDs.
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def _get(path: str, params: dict, *, timeout: float, retries: int = 3) -> httpx.Response:
    """GET an E-utilities endpoint with backoff on NCBI's transient 5xx/429s."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = httpx.get(f"{EUTILS}/{path}", params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except httpx.HTTPStatusError as e:
            last = e
            if e.response.status_code not in (429, 500, 502, 503):
                raise
            time.sleep(2**attempt)  # 1s, 2s, 4s
    raise RuntimeError(f"NCBI E-utilities unavailable after {retries} tries: {last}")


@dataclass
class Article:
    pmid: str
    title: str
    abstract: str
    journal: str = ""
    year: str = ""
    authors: list[str] = field(default_factory=list)
    mesh_terms: list[str] = field(default_factory=list)
    pmc_id: str = ""
    doi: str = ""
    full_text: str = ""
    pub_types: list[str] = field(default_factory=list)
    # How to label full_text in metadata: "results" (Results section) or "review"
    # (whole-body text of a review article, which has no Results section).
    full_text_kind: str = "results"
    # Origin of the record: "pubmed", "biorxiv", "text", or "mendeley". For bioRxiv
    # preprints the DOI is stored in the `pmid` slot (the system-wide primary key),
    # so dedup, /show, /remove, and [id] citations work unchanged. Mendeley imports
    # keyed without a PMID/preprint-DOI use a "mendeley-<id>" pmid slot.
    source: str = "pubmed"

    @property
    def url(self) -> str:
        if self.source == "biorxiv":
            return f"https://www.biorxiv.org/content/{self.doi}"
        if self.source == "text":
            return ""
        if self.source == "mendeley":
            return f"https://doi.org/{self.doi}" if self.doi else ""
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"

    @property
    def is_review(self) -> bool:
        return any("review" in t.lower() for t in self.pub_types)

    def to_text(self) -> str:
        """Flatten to a single chunkable document. Uses full text when available.

        Authors are embedded in the text so author-name queries (e.g. "Powell")
        are retrievable by both dense and BM25 search.
        """
        body = self.full_text if self.full_text else self.abstract
        parts = [self.title]
        if self.authors:
            parts.append("Authors: " + ", ".join(self.authors))
        parts.append(body)
        return "\n\n".join(parts)

    def metadata(self) -> dict:
        return {
            "pmid": self.pmid,
            "title": self.title,
            "journal": self.journal,
            "year": self.year,
            "url": self.url,
            "doi": self.doi,
            "mesh": ", ".join(self.mesh_terms),
            "authors": ", ".join(self.authors),
            "first_author": self.authors[0] if self.authors else "",
            "text_source": self.full_text_kind if self.full_text else "abstract",
        }


def _params(**extra) -> dict:
    p = {"db": "pubmed", "retmode": "xml"}
    if key := os.getenv("NCBI_API_KEY"):
        p["api_key"] = key
    if email := os.getenv("NCBI_EMAIL"):
        p["email"] = email
        p["tool"] = "scirag"
    p.update(extra)
    return p


def search(
    query: str,
    retmax: int = 25,
    min_year: str = "",
    max_year: str = "",
) -> list[str]:
    """esearch -> list of PMIDs.

    Year range is embedded in the query string as a [pdat] filter rather than
    using the mindate/maxdate params — NCBI silently ignores those params when
    sort=relevance is active.
    """
    term = query
    if min_year or max_year:
        lo = min_year or "1000"
        hi = max_year or "3000"
        term = f'({query}) AND ("{lo}"[pdat]:"{hi}"[pdat])'
    r = _get(
        "esearch.fcgi",
        _params(term=term, retmax=retmax, sort="relevance"),
        timeout=30,
    )
    root = ET.fromstring(r.text)
    return [e.text for e in root.findall(".//IdList/Id") if e.text]


def search_semantic(
    query: str,
    retmax: int = 25,
    min_year: str = "",
    max_year: str = "",
) -> list[str]:
    """Relevance-ranked PMID search via Europe PMC (SRC:MED) — natural-language friendly.

    NCBI esearch maps each token to a field, so a plain-English question like
    "retrosplenial cortex related disorder in human" gets mangled (the trailing
    "in human" becomes an [Author] clause) and returns nothing. Europe PMC ranks
    the same words by relevance over MEDLINE/PubMed records and ignores the noise,
    so a sentence works. It returns PMIDs, which fetch() then expands exactly as
    the esearch path does — dedup, full-text, and the picker are all unchanged.
    """
    q = f"({query}) AND SRC:MED"
    if min_year or max_year:
        start = f"{min_year}-01-01" if min_year else "1000-01-01"
        end = f"{max_year}-12-31" if max_year else "3000-12-31"
        q += f" AND (FIRST_PDATE:[{start} TO {end}])"
    last: Exception | None = None
    for attempt in range(3):
        try:
            r = httpx.get(
                EPMC,
                params={
                    "query": q,
                    "format": "json",
                    "resultType": "lite",
                    "pageSize": min(retmax, 100),
                },
                timeout=30,
            )
            r.raise_for_status()
            break
        except httpx.HTTPStatusError as e:
            last = e
            if e.response.status_code not in (429, 500, 502, 503):
                raise
            time.sleep(2**attempt)
    else:
        raise RuntimeError(f"Europe PMC unavailable after 3 tries: {last}")

    seen: set[str] = set()
    pmids: list[str] = []
    for rec in r.json().get("resultList", {}).get("result", []):
        pmid = rec.get("pmid")
        if pmid and pmid not in seen:
            seen.add(pmid)
            pmids.append(pmid)
    return pmids[:retmax]


def fetch(pmids: list[str]) -> list[Article]:
    """efetch -> parsed Article records."""
    if not pmids:
        return []
    r = _get(
        "efetch.fcgi",
        _params(id=",".join(pmids), rettype="abstract"),
        timeout=60,
    )
    root = ET.fromstring(r.text)
    out: list[Article] = []
    for art in root.findall(".//PubmedArticle"):
        out.append(_parse_article(art))
    return out


def _parse_article(art: ET.Element) -> Article:
    def text(path: str, default: str = "") -> str:
        el = art.find(path)
        return el.text if el is not None and el.text else default

    pmid = text(".//PMID")
    title = text(".//ArticleTitle")
    # Abstract may have multiple labeled sections.
    parts = []
    for ab in art.findall(".//Abstract/AbstractText"):
        label = ab.get("Label")
        body = "".join(ab.itertext()).strip()
        parts.append(f"{label}: {body}" if label else body)
    abstract = "\n".join(parts)

    authors = []
    for a in art.findall(".//AuthorList/Author"):
        last = a.findtext("LastName")
        init = a.findtext("Initials")
        if last:
            authors.append(f"{last} {init}".strip())

    mesh = [m.text for m in art.findall(".//MeshHeadingList/MeshHeading/DescriptorName") if m.text]
    year = text(".//PubDate/Year") or text(".//PubDate/MedlineDate")[:4]
    journal = text(".//Journal/Title")
    pub_types = [t.text for t in art.findall(".//PublicationTypeList/PublicationType") if t.text]

    doi = ""
    for aid in art.findall(".//ArticleIdList/ArticleId"):
        if aid.get("IdType") == "doi" and aid.text:
            doi = aid.text
            break

    return Article(
        pmid=pmid,
        title=title,
        abstract=abstract,
        journal=journal,
        year=year,
        authors=authors,
        mesh_terms=mesh,
        doi=doi,
        pub_types=pub_types,
    )


def search_and_fetch(
    query: str,
    retmax: int = 25,
    min_year: str = "",
    max_year: str = "",
) -> list[Article]:
    return fetch(search(query, retmax=retmax, min_year=min_year, max_year=max_year))


def search_and_fetch_semantic(
    query: str,
    retmax: int = 25,
    min_year: str = "",
    max_year: str = "",
) -> list[Article]:
    """Like search_and_fetch, but via Europe PMC relevance search (see search_semantic)."""
    return fetch(search_semantic(query, retmax=retmax, min_year=min_year, max_year=max_year))


# ---------------------------------------------------------------------------
# PMC full-text (Results section only)
# ---------------------------------------------------------------------------


def _pmids_to_pmcids(pmids: list[str]) -> dict[str, str]:
    """Map PubMed IDs to PMC IDs via elink. Returns only PMIDs that have a PMC entry."""
    r = _get(
        "elink.fcgi",
        _params(dbfrom="pubmed", db="pmc", id=",".join(pmids), cmd="neighbor"),
        timeout=30,
    )
    root = ET.fromstring(r.text)
    mapping: dict[str, str] = {}
    for linkset in root.findall(".//LinkSet"):
        pmid_el = linkset.find(".//IdList/Id")
        if pmid_el is None or not pmid_el.text:
            continue
        pmid = pmid_el.text
        for link in linkset.findall(".//LinkSetDb/Link/Id"):
            if link.text:
                mapping[pmid] = link.text
                break
    return mapping


def _extract_results_from_jats(root: ET.Element) -> str:
    """Extract only the Results section paragraphs from a JATS XML tree."""
    for sec in root.iter():
        if sec.tag != "sec" and not sec.tag.endswith("}sec"):
            continue
        title_el = next((c for c in sec if c.tag == "title" or c.tag.endswith("}title")), None)
        if title_el is None:
            continue
        if "result" not in "".join(title_el.itertext()).lower():
            continue
        parts = [
            "".join(el.itertext()).strip()
            for el in sec.iter()
            if (el.tag == "p" or el.tag.endswith("}p"))
        ]
        parts = [p for p in parts if p]
        if parts:
            return "\n\n".join(parts)
    return ""


def _fetch_pmc_fulltext(pmc_id: str) -> str:
    """Fetch PMC JATS XML and return the Results section text. Returns '' on failure."""
    try:
        r = _get(
            "efetch.fcgi",
            {"db": "pmc", "id": pmc_id, "rettype": "xml", "retmode": "xml"},
            timeout=60,
        )
        return _extract_results_from_jats(ET.fromstring(r.text))
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Unpaywall fallback
# ---------------------------------------------------------------------------


def _unpaywall_pdf_url(doi: str) -> str:
    """Return the best open-access PDF URL for a DOI via Unpaywall, or ''."""
    if not doi:
        return ""
    email = os.getenv("NCBI_EMAIL", "scirag@example.com")
    try:
        r = httpx.get(f"{UNPAYWALL}/{doi}", params={"email": email}, timeout=20)
        r.raise_for_status()
        loc = r.json().get("best_oa_location") or {}
        return loc.get("url_for_pdf") or loc.get("url") or ""
    except Exception:
        return ""


def _download_pdf_results(url: str) -> str:
    """Download a PDF from url and return its Results section text."""
    try:
        import tempfile
        from pathlib import Path

        from scirag.sources.pdf import extract_results_section, extract_text_from_pdf

        r = httpx.get(url, timeout=60, follow_redirects=True)
        r.raise_for_status()
        # Unpaywall's "OA location" is often a publisher landing page or paywall
        # (HTML), not a real PDF. Bail before pypdf chokes on it and prints
        # "invalid pdf header" / "EOF marker not found" to the console.
        ctype = r.headers.get("content-type", "").lower()
        looks_like_pdf = "pdf" in ctype or r.content.lstrip()[:5] == b"%PDF-"
        if not looks_like_pdf:
            return ""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(r.content)
            tmp = Path(f.name)
        text = extract_text_from_pdf(tmp)
        tmp.unlink(missing_ok=True)
        return extract_results_section(text)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Public enrichment entry point
# ---------------------------------------------------------------------------


def enrich_with_fulltext(articles: list[Article]) -> None:
    """Enrich articles with Results-section text via PMC then Unpaywall.

    Mutates articles in place. Emits a warning for each article where no
    full text could be retrieved — the caller should offer a manual PDF
    import fallback in that case.
    """
    if not articles:
        return

    pmids = [a.pmid for a in articles if a.pmid]
    pmc_map = _pmids_to_pmcids(pmids)
    has_key = bool(os.getenv("NCBI_API_KEY"))
    articles_by_pmid = {a.pmid: a for a in articles}

    for pmid, pmc_id in pmc_map.items():
        article = articles_by_pmid.get(pmid)
        if article is None:
            continue
        article.pmc_id = pmc_id
        article.full_text = _fetch_pmc_fulltext(pmc_id)
        if not has_key:
            time.sleep(0.34)

    # Unpaywall fallback for articles still missing full text
    for article in articles:
        if article.full_text:
            continue
        pdf_url = _unpaywall_pdf_url(article.doi)
        if pdf_url:
            article.full_text = _download_pdf_results(pdf_url)

    # Warn for anything still empty
    missing = [a for a in articles if not a.full_text]
    if missing:
        pmids_str = ", ".join(a.pmid for a in missing)
        warnings.warn(
            f"{len(missing)} article(s) have no retrievable full text "
            f"(PMIDs: {pmids_str}). "
            "Use `scirag import-pdf` to add manually downloaded PDFs.",
            stacklevel=2,
        )
