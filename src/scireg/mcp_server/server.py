"""MCP server exposing scireg's retrieval tools over the Model Context Protocol,
so the same data layer is reusable by LangGraph agents, Claude Desktop, etc.

Run:  uv run python -m scireg.mcp_server.server
Requires the `mcp` extra:  uv sync --extra mcp
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from scireg.sources import pubmed

mcp = FastMCP("scireg")


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
    """Run the local multi-agent RAG graph and return a cited answer."""
    from scireg.graph.state import build_graph

    return build_graph().invoke({"query": query}).get("answer", "")


if __name__ == "__main__":
    mcp.run()
