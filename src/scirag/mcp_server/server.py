"""MCP server exposing scirag's retrieval tools over the Model Context Protocol.

Run:  uv run python -m scirag.mcp_server.server
Requires the `mcp` extra:  uv sync --extra mcp
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from scirag.sources import pubmed

mcp = FastMCP("scirag")


@mcp.tool()
def search_pubmed(query: str, retmax: int = 25) -> list[dict]:
    """Search PubMed and return article metadata (pmid, title, abstract, year)."""
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
        for a in pubmed.search_and_fetch(query, retmax=retmax)
    ]


@mcp.tool()
def ask_index(query: str) -> str:
    """Retrieve from the local index and return a cited answer."""
    from scirag.agents.synthesize import synthesize
    from scirag.neuro.entities import expand_query, extract_entities
    from scirag.retrieval.retriever import retrieve

    ents = extract_entities(query)
    nodes = retrieve(expand_query(query, ents))
    return synthesize(query, nodes)


if __name__ == "__main__":
    mcp.run()
