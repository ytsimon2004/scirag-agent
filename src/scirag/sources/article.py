"""Source-neutral record type shared by every source (pubmed, biorxiv, text,
mendeley, zotero), plus the JATS Results-section parser they share.

Lives here, not in any one source module, so the sources are peers: none has to
import the record type (or the shared JATS parser) from another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
from xml.etree import ElementTree as ET


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
    # How full_text is labelled in metadata's text_source: results = Results section,
    # fulltext = whole article body, review = review-article body, text = free-text import.
    full_text_kind: Literal["results", "review", "text", "fulltext"] = "results"
    # Origin of the record. bioRxiv preprints store their DOI in the `pmid` slot (the
    # system-wide primary key), so dedup, /show, /remove, and [id] citations work
    # unchanged; Mendeley/Zotero imports without a PMID/preprint-DOI use a
    # "mendeley-<id>"/"zotero-<id>" pmid slot.
    source: Literal["pubmed", "biorxiv", "text", "mendeley", "zotero"] = "pubmed"

    @property
    def url(self) -> str:
        if self.source == "biorxiv":
            return f"https://www.biorxiv.org/content/{self.doi}"
        if self.source == "text":
            return ""
        if self.source in ("mendeley", "zotero"):
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
            # Persist the originating source so read-back doesn't have to infer it
            # from the key's shape (origin_of), which can't distinguish a PubMed
            # journal DOI from a bioRxiv/IEEE/etc. one for non-PubMed sources.
            "source": self.source,
        }


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
