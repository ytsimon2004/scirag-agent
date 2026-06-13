"""bioRxiv retrieval.

Mirrors the public surface of scirag.sources.pubmed so the CLI can treat bioRxiv
like a second PubMed: search() returns identifiers, fetch() returns Articles, and
enrich_with_fulltext() fills in the Results section.

Keyword search goes through **Europe PMC**, which indexes bioRxiv preprints and
offers a real relevance-ranked search endpoint (the bioRxiv API itself has none —
it only serves date-window dumps). Direct-DOI metadata and full-text JATS come
from the bioRxiv API.

Identifier model: bioRxiv preprints have no PMID, only a DOI (10.1101/…, 10.64898/…).
The DOI is stored in the Article.pmid slot — the system-wide primary key — so
indexing, dedup, /show, /remove, and [id] citations all work unchanged.
Article.source is set to "biorxiv" to drive the source-aware URL.
"""

from __future__ import annotations

import time
import warnings
from datetime import date, timedelta
from xml.etree import ElementTree as ET

import httpx

from scirag.sources.pubmed import Article, _extract_results_from_jats

API = "https://api.biorxiv.org/details/biorxiv"  # bioRxiv: details-by-DOI + jatsxml
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"  # keyword search
_POLITE_SLEEP = 0.5
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def _get(
    url: str, *, params: dict | None = None, timeout: float, retries: int = 3
) -> httpx.Response:
    """GET with backoff on transient 5xx/429s."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = httpx.get(
                url, params=params, timeout=timeout, follow_redirects=True, headers=_HEADERS
            )
            r.raise_for_status()
            return r
        except httpx.HTTPStatusError as e:
            last = e
            if e.response.status_code not in (429, 500, 502, 503):
                raise
            time.sleep(2**attempt)  # 1s, 2s, 4s
    raise RuntimeError(f"Source unavailable after {retries} tries: {last}")


# ---------------------------------------------------------------------------
# Keyword search via Europe PMC
# ---------------------------------------------------------------------------


def _epmc_query(keywords: str, days_back: int | None) -> str:
    q = f'{keywords} AND (SRC:PPR AND PUBLISHER:"bioRxiv")'
    if days_back:
        end = date.today()
        start = end - timedelta(days=days_back)
        q += f" AND (FIRST_PDATE:[{start.isoformat()} TO {end.isoformat()}])"
    return q


def _epmc_search(keywords: str, *, days_back: int | None, retmax: int) -> list[dict]:
    """Relevance-ranked Europe PMC search for bioRxiv preprints. Returns core records."""
    r = _get(
        EPMC,
        params={
            "query": _epmc_query(keywords, days_back),
            "format": "json",
            "resultType": "core",
            "pageSize": min(retmax, 100),
        },
        timeout=30,
    )
    return r.json().get("resultList", {}).get("result", [])[:retmax]


def _epmc_to_article(rec: dict) -> Article:
    authors = [a.strip() for a in (rec.get("authorString") or "").split(",") if a.strip()]
    doi = rec.get("doi", "") or ""
    return Article(
        pmid=doi,  # DOI occupies the primary-key slot for preprints
        title=rec.get("title", "") or "",
        abstract=rec.get("abstractText", "") or "",
        journal="bioRxiv",
        year=str(rec.get("pubYear", "") or "")[:4],
        authors=authors,
        doi=doi,
        source="biorxiv",
    )


def search(keywords: str, *, days_back: int | None = 180, retmax: int = 25) -> list[str]:
    """Return DOIs of bioRxiv preprints matching `keywords` (via Europe PMC)."""
    seen: set[str] = set()
    dois: list[str] = []
    for rec in _epmc_search(keywords, days_back=days_back, retmax=retmax):
        doi = rec.get("doi")
        if doi and doi not in seen:
            seen.add(doi)
            dois.append(doi)
    return dois


def search_and_fetch(
    keywords: str, *, days_back: int | None = 180, retmax: int = 25
) -> list[Article]:
    """Keyword search → Article records, in one Europe PMC round-trip."""
    seen: set[str] = set()
    out: list[Article] = []
    for rec in _epmc_search(keywords, days_back=days_back, retmax=retmax):
        doi = rec.get("doi")
        if doi and doi not in seen:
            seen.add(doi)
            out.append(_epmc_to_article(rec))
    return out


# ---------------------------------------------------------------------------
# Direct-DOI metadata via the bioRxiv API
# ---------------------------------------------------------------------------


def _record_to_article(rec: dict) -> Article:
    authors = [a.strip() for a in (rec.get("authors") or "").split(";") if a.strip()]
    doi = rec.get("doi", "")
    return Article(
        pmid=doi,
        title=rec.get("title", ""),
        abstract=rec.get("abstract", ""),
        journal="bioRxiv",
        year=(rec.get("date", "") or "")[:4],
        authors=authors,
        doi=doi,
        pub_types=[rec.get("type", "")] if rec.get("type") else [],
        source="biorxiv",
    )


def fetch(dois: list[str]) -> list[Article]:
    """Look up each DOI via the bioRxiv details endpoint (latest version)."""
    out: list[Article] = []
    for doi in dois:
        try:
            r = _get(f"{API}/{doi}", timeout=30)
            collection = r.json().get("collection", [])
        except Exception:
            continue
        if collection:
            out.append(_record_to_article(collection[-1]))
    return out


# ---------------------------------------------------------------------------
# Full text (Results section) via the bioRxiv JATS XML
# ---------------------------------------------------------------------------


def _jatsxml_url(doi: str) -> str:
    """Fetch the details record for a DOI and return its jatsxml URL, or ''."""
    try:
        r = _get(f"{API}/{doi}", timeout=30)
        collection = r.json().get("collection", [])
    except Exception:
        return ""
    return collection[-1].get("jatsxml", "") if collection else ""


def _fetch_jats_results(jats_url: str) -> str:
    """Download a bioRxiv JATS XML document and return its Results section."""
    try:
        r = _get(jats_url, timeout=60)
        return _extract_results_from_jats(ET.fromstring(r.text))
    except Exception:
        return ""


def enrich_with_fulltext(articles: list[Article]) -> None:
    """Best-effort: fill in each preprint's Results section from its bioRxiv JATS XML.

    Mutates articles in place. bioRxiv's full-text host (biorxiv.org) sits behind
    Cloudflare bot protection, so this often fails; articles then index on their
    abstract (Article.to_text falls back to abstract). Warns for any preprint with
    no retrievable full text.
    """
    if not articles:
        return

    for article in articles:
        jats_url = _jatsxml_url(article.doi)
        if jats_url:
            article.full_text = _fetch_jats_results(jats_url)
        time.sleep(_POLITE_SLEEP)

    missing = [a for a in articles if not a.full_text]
    if missing:
        dois = ", ".join(a.doi for a in missing)
        warnings.warn(
            f"{len(missing)} preprint(s) have no retrievable full text "
            f"(DOIs: {dois}) — indexing on abstract. The bioRxiv full-text host is "
            "Cloudflare-protected; download the PDF and use `scirag import-pdf` to add it.",
            stacklevel=2,
        )
