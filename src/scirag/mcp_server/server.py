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
    from scirag.agents.pipeline import prepare_answer
    from scirag.llm.router import complete

    result = prepare_answer(query)
    return complete("synthesizer", result.messages, max_tokens=1200)


if __name__ == "__main__":
    mcp.run()
