"""scireg CLI.

    scireg index "hippocampal place cells" --retmax 30   # fetch + index PubMed
    scireg ask   "How do place cells remap across environments?"
    scireg search "grid cells entorhinal"                # raw PubMed, no LLM
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.markdown import Markdown

from scireg.sources import pubmed

app = typer.Typer(add_completion=False, help="Multi-agent RAG for scientific literature.")
console = Console()


@app.command()
def search(query: str, retmax: int = 15):
    """Raw PubMed search (no LLM, no index) — sanity-check the data source."""
    arts = pubmed.search_and_fetch(query, retmax=retmax)
    for a in arts:
        console.print(f"[bold cyan]{a.pmid}[/] {a.title} [dim]({a.year}, {a.journal})[/]")
    console.print(f"\n[green]{len(arts)} articles[/]")


@app.command()
def index(query: str, retmax: int = 25):
    """Fetch PubMed results for a query and add them to the LanceDB index."""
    from scireg.ingest.index import build_index

    arts = pubmed.search_and_fetch(query, retmax=retmax)
    console.print(f"Fetched {len(arts)} articles, embedding + indexing...")
    build_index(arts)
    console.print("[green]Indexed.[/]")


@app.command()
def ask(query: str):
    """Run the full multi-agent RAG graph against the local index."""
    from scireg.graph.state import build_graph

    graph = build_graph()
    result = graph.invoke({"query": query})

    ents = result.get("entities", {})
    nonempty = {k: v for k, v in ents.items() if v}
    if nonempty:
        console.print(f"[dim]neuro entities: {nonempty}[/]")
    console.print(f"[dim]{len(result.get('nodes', []))} passages retrieved[/]\n")
    console.print(Markdown(result.get("answer", "(no answer)")))


if __name__ == "__main__":
    app()
