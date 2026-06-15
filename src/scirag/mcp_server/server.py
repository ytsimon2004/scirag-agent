"""MCP server exposing scirag's retrieval tools over the Model Context Protocol.

Run:  uv run python -m scirag.mcp_server.server
Requires the `mcp` extra:  uv sync --extra mcp
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from scirag.sources import pubmed

mcp = FastMCP("scirag")


@mcp.tool()
def search_pubmed(query: str, retmax: int = 25, semantic: bool = False) -> list[dict]:
    """Search PubMed and return article metadata (pmid, title, abstract, year).

    By default uses NCBI esearch (Boolean/field syntax). Set `semantic=True` to rank
    by relevance via Europe PMC instead — this tolerates natural-language questions
    that esearch mangles (e.g. it won't misread a trailing "in human" as an author),
    so prefer it when passing a plain-English query rather than Boolean terms.
    """
    fetch = pubmed.search_and_fetch_semantic if semantic else pubmed.search_and_fetch
    return [
        {
            "pmid": a.pmid,
            "title": a.title,
            "abstract": a.abstract,
            "year": a.year,
            "journal": a.journal,
            "url": a.url,
            "mesh": a.mesh_terms,
        }
        for a in fetch(query, retmax=retmax)
    ]


@mcp.tool()
def search_biorxiv(query: str, retmax: int = 25, days_back: int = 180) -> list[dict]:
    """Keyword-search recent bioRxiv preprints (title/abstract match over a date window).

    bioRxiv has no keyword API endpoint, so this scans the last `days_back` days.
    `doi` is the preprint's identifier (used in place of a PMID).
    """
    from scirag.sources import biorxiv

    return [
        {
            "doi": a.doi,
            "title": a.title,
            "abstract": a.abstract,
            "year": a.year,
            "url": a.url,
            "source": a.source,
        }
        for a in biorxiv.search_and_fetch(query, days_back=days_back, retmax=retmax)
    ]


@mcp.tool()
def ask_index(query: str) -> str:
    """Retrieve from the local index and return a cited answer."""
    from scirag.agents.pipeline import prepare_answer
    from scirag.llm.router import complete

    result = prepare_answer(query)
    return complete("synthesizer", result.messages, max_tokens=1200)


if __name__ == "__main__":
    mcp.run()
