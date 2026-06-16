"""MCP server exposing scirag's retrieval tools over the Model Context Protocol.

Run:  uv run python -m scirag.mcp_server.server
Requires the `mcp` extra:  uv sync --extra mcp
"""

from __future__ import annotations

from contextlib import contextmanager

from mcp.server.fastmcp import FastMCP

from scirag.sources import pubmed

mcp = FastMCP("scirag")


@contextmanager
def _scope(project: str):
    """Scope a tool call to a named project's index (empty = the active project).

    Validates the name up front so a typo yields a clear error rather than silently
    querying an empty/wrong index. Per-call and concurrency-safe (see
    projects.using_project).
    """
    from scirag.projects import list_projects, using_project

    if not project:
        yield
        return
    if not any(p["name"] == project for p in list_projects()):
        names = ", ".join(p["name"] for p in list_projects()) or "(none)"
        raise ValueError(
            f"Unknown project {project!r}. Available: {names}. Or omit to use the active one."
        )
    with using_project(project):
        yield


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
    """Search bioRxiv preprints by relevance (natural-language queries work).

    Goes through Europe PMC relevance ranking (bioRxiv has no keyword API), so a
    plain-English question is fine. `days_back` restricts results to recently
    posted preprints. `doi` is the preprint's identifier (used in place of a PMID).
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
def list_projects() -> dict:
    """List the available indexes (projects) and which one is active.

    Each project is an isolated corpus. Pass a name to the `project` argument of
    `retrieve_chunks` / `get_record` / `index_status` to query a specific one;
    omit it to use the active project shown here. Returns {active, projects:
    [{name, description}, …]}; `active` is null when the default global index is in use.
    """
    from scirag.projects import get_active_project
    from scirag.projects import list_projects as _list

    return {
        "active": get_active_project(),
        "projects": [{"name": p["name"], "description": p.get("description", "")} for p in _list()],
    }


@mcp.tool()
def retrieve_chunks(query: str, project: str = "") -> list[dict]:
    """Retrieve the most relevant chunks from the LOCAL index — no LLM synthesis.

    Use this to ground your own answer in the user's indexed corpus: it returns the
    raw evidence (chunk text + citation + source) so you can reason over it yourself,
    rather than getting a pre-written answer. Prefer this over `ask_index` when you
    are an LLM that wants the sources. Each result has: `citation` (author-year
    marker to cite the claim with), `id` (PMID or bioRxiv DOI — the primary key),
    `title`, `text_source` (results / fulltext / review / abstract), `url`, `score`
    (cosine relevance), and `text` (the chunk). Empty if nothing is indexed yet.

    `project` selects which indexed corpus to query (see `list_projects`); omit it
    to use the active project.
    """
    from scirag.cite import citation
    from scirag.retrieval.retriever import retrieve as _retrieve

    with _scope(project):
        nodes = _retrieve(query)
    out = []
    for n in nodes:
        md = n.node.metadata
        out.append(
            {
                "citation": citation(md),
                "id": md.get("pmid", ""),
                "title": md.get("title", ""),
                "text_source": md.get("text_source", ""),
                "url": md.get("url", ""),
                "score": n.score,
                "text": n.node.get_content(),
            }
        )
    return out


@mcp.tool()
def get_record(id: str, project: str = "") -> dict | None:
    """Return the full stored text + metadata for one indexed paper, or null.

    `id` is the primary key shown by `retrieve_chunks`/`index_status` — a PMID for
    PubMed records, a bioRxiv DOI for preprints. Use after `retrieve_chunks` to read
    a source in full. Returns {id, title, year, first_author, authors, text_source,
    chunks: [text,…]}; null if no such paper is indexed.

    `project` selects which indexed corpus to look in (see `list_projects`); omit it
    to use the active project.
    """
    from scirag.ingest.index import get_article_chunks

    with _scope(project):
        art = get_article_chunks(id.strip())
    if not art:
        return None
    art["id"] = art.pop("pmid")
    return art


@mcp.tool()
def index_status(project: str = "") -> dict:
    """Summarize the local index: count + one row per indexed paper.

    Call this first to see what corpus is available before retrieving. Returns
    {count, full_text, abstract_only, articles: [{id, origin, year, first_author,
    title, text_source}, …]}.

    `project` selects which indexed corpus to summarize (see `list_projects`); omit
    it to use the active project.
    """
    from scirag.ingest.index import get_indexed_articles

    with _scope(project):
        articles = get_indexed_articles()
    full_text = sum(
        1 for a in articles if a.get("text_source") in ("results", "fulltext", "review")
    )
    return {
        "count": len(articles),
        "full_text": full_text,
        "abstract_only": len(articles) - full_text,
        "articles": [
            {
                "id": a["pmid"],
                "origin": a.get("origin", ""),
                "year": a.get("year", ""),
                "first_author": a.get("first_author", ""),
                "title": a.get("title", ""),
                "text_source": a.get("text_source", ""),
            }
            for a in sorted(articles, key=lambda x: x["year"], reverse=True)
        ],
    }


@mcp.tool()
def ask_index(query: str) -> str:
    """Retrieve from the local index and return a fully-written cited answer.

    This runs scirag's own synthesizer LLM over the retrieved chunks. If you are an
    LLM yourself and just want the evidence to reason over, use `retrieve_chunks`
    instead — it skips this extra inference step.
    """
    from scirag.agents.pipeline import prepare_answer
    from scirag.llm.router import complete

    result = prepare_answer(query)
    return complete("synthesizer", result.messages, max_tokens=1200)


if __name__ == "__main__":
    mcp.run()
