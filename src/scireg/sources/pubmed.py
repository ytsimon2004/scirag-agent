"""PubMed retrieval via NCBI E-utilities (esearch + efetch).

This is the raw data-source client. The same functions are re-exported as MCP
tools in scireg.mcp_server so agents can call them through the MCP protocol.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from xml.etree import ElementTree as ET

import httpx

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


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

    @property
    def url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"

    def to_text(self) -> str:
        """Flatten to a single chunkable document."""
        return f"{self.title}\n\n{self.abstract}"

    def metadata(self) -> dict:
        return {
            "pmid": self.pmid,
            "title": self.title,
            "journal": self.journal,
            "year": self.year,
            "url": self.url,
            "mesh": ", ".join(self.mesh_terms),
        }


def _params(**extra) -> dict:
    p = {"db": "pubmed", "retmode": "xml"}
    if key := os.getenv("NCBI_API_KEY"):
        p["api_key"] = key
    if email := os.getenv("NCBI_EMAIL"):
        p["email"] = email
        p["tool"] = "scireg"
    p.update(extra)
    return p


def search(query: str, retmax: int = 25) -> list[str]:
    """esearch -> list of PMIDs."""
    r = _get(
        "esearch.fcgi",
        _params(term=query, retmax=retmax, sort="relevance"),
        timeout=30,
    )
    root = ET.fromstring(r.text)
    return [e.text for e in root.findall(".//IdList/Id") if e.text]


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

    return Article(
        pmid=pmid,
        title=title,
        abstract=abstract,
        journal=journal,
        year=year,
        authors=authors,
        mesh_terms=mesh,
    )


def search_and_fetch(query: str, retmax: int = 25) -> list[Article]:
    return fetch(search(query, retmax=retmax))
