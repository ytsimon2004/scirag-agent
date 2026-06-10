"""scireg CLI — subcommands for scripting, or just `scireg` for the interactive shell.

    scireg                                        # interactive shell (default)
    scireg search "grid cells entorhinal"         # raw PubMed, no LLM
    scireg index  "hippocampal place cells" --retmax 30 --full-text
    scireg retrieve "place cells remapping"
    scireg ask    "How do place cells remap across environments?"
    scireg import-pdf paper.pdf
    scireg import-dir ./papers/
"""
from __future__ import annotations

import warnings
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown

from scireg.sources import pubmed

app = typer.Typer(
    invoke_without_command=True,
    add_completion=False,
    help="Multi-agent RAG for scientific literature. Run with no args for the interactive shell.",
)
console = Console()


# ---------------------------------------------------------------------------
# Typer entry point: no subcommand → shell
# ---------------------------------------------------------------------------

@app.callback()
def _main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        from scireg.shell import run_shell
        run_shell()


# ---------------------------------------------------------------------------
# Shared display helpers (used by both CLI commands and the shell)
# ---------------------------------------------------------------------------

def print_article_list(arts, existing: set, pmc_map: dict) -> None:
    for i, a in enumerate(arts, 1):
        in_db = a.pmid in existing
        tags = []
        if a.pmid in pmc_map:
            tags.append("[green]PMC✓[/]")
        if a.doi:
            tags.append("[green]DOI✓[/]")
        if not a.abstract:
            tags.append("[red]noABS[/]")
        tag_str = "  " + " ".join(tags) if tags else ""

        if in_db:
            console.print(
                f"[dim]{i:>2}. [yellow][indexed][/yellow] "
                f"[strike]{a.pmid}  {a.title[:65]}[/strike]  ({a.year}){tag_str}[/dim]"
            )
        else:
            console.print(
                f"{i:>2}. [bold cyan]{a.pmid}[/]  {a.title[:65]}  [dim]({a.year})[/]{tag_str}"
            )
        console.print(f"     [dim][link={a.url}]{a.url}[/link][/]")


def print_retrieve_results(nodes) -> None:
    if not nodes:
        console.print("[yellow]No results — have you run /index yet?[/]")
        return
    for n in nodes:
        md = n.node.metadata
        snippet = n.node.get_content()[:120].replace("\n", " ")
        url = md.get("url", "")
        pmid_display = f"[link={url}]{md.get('pmid', '?')}[/link]" if url else md.get("pmid", "?")
        src = md.get("text_source", "")
        src_tag = "[green]results[/]" if src == "results" else "[dim]abstract[/]"
        console.print(f"[bold cyan]{pmid_display}[/] {md.get('title', '')} [dim]({md.get('year', 'n.d.')})[/]  {src_tag}")
        if url:
            console.print(f"  [dim][link={url}]{url}[/link][/]")
        console.print(f"  [dim]{snippet}…[/]\n")
    console.print(f"[green]{len(nodes)} chunks retrieved[/]")


# ---------------------------------------------------------------------------
# Core logic (called by both CLI commands and the shell)
# ---------------------------------------------------------------------------

def do_search(query: str, retmax: int = 15) -> None:
    from scireg.sources.pubmed import _pmids_to_pmcids

    arts = pubmed.search_and_fetch(query, retmax=retmax)
    if not arts:
        console.print("[yellow]No results.[/]")
        return

    pmc_map = _pmids_to_pmcids([a.pmid for a in arts])
    console.print()
    for a in arts:
        pmc_tag  = "[green]PMC✓[/]" if a.pmid in pmc_map else "[dim]PMC✗[/]"
        doi_tag  = "[green]DOI✓[/]" if a.doi              else "[dim]DOI✗[/]"
        abst_tag = "[green]ABS✓[/]" if a.abstract         else "[red]ABS✗[/]"
        console.print(
            f"[bold cyan][link={a.url}]{a.pmid}[/link][/] {a.title[:70]} [dim]({a.year})[/]\n"
            f"  {pmc_tag} {doi_tag} {abst_tag}  [dim]{a.journal}[/]\n"
            f"  [dim][link={a.url}]{a.url}[/link][/]"
        )
    n_pmc = sum(1 for a in arts if a.pmid in pmc_map)
    n_doi = sum(1 for a in arts if a.doi)
    console.print(
        f"\n[green]{len(arts)} articles[/]  —  "
        f"PMC: {n_pmc}  |  DOI/Unpaywall: {n_doi}  |  abstract-only: {len(arts) - n_pmc}"
    )


def do_index(query: str, retmax: int = 25, full_text: bool = False) -> None:
    import questionary
    from scireg.ingest.index import build_index, get_indexed_pmids
    from scireg.sources.pubmed import _pmids_to_pmcids

    arts = pubmed.search_and_fetch(query, retmax=retmax)
    if not arts:
        console.print("[yellow]No results.[/]")
        return

    existing = get_indexed_pmids()
    pmc_map  = _pmids_to_pmcids([a.pmid for a in arts])

    console.print()
    print_article_list(arts, existing, pmc_map)
    console.print()

    choices = [
        questionary.Choice(
            title=f"{'[indexed] ' if a.pmid in existing else ''}"
                  f"{a.pmid}  {a.title[:60]}  ({a.year})"
                  + (f"  [{', '.join(t for t in (['PMC✓'] if a.pmid in pmc_map else []) + (['DOI✓'] if a.doi else []))}]"
                     if a.pmid in pmc_map or a.doi else ""),
            value=a,
            checked=(a.pmid not in existing),
        )
        for a in arts
    ]

    selected = questionary.checkbox(
        "Select articles to index  (space = toggle, a = all, i = invert, enter = confirm):",
        choices=choices,
    ).ask()

    if not selected:
        console.print("[yellow]Nothing selected.[/]")
        return

    console.print()

    if full_text:
        from scireg.sources.pubmed import enrich_with_fulltext
        console.print("Fetching full text (Results section)...")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            enrich_with_fulltext(selected)
        n = sum(1 for a in selected if a.full_text)
        console.print(f"Full text retrieved for [cyan]{n}[/] / {len(selected)} articles.")
        missing = [a for a in selected if not a.full_text]
        if missing:
            console.print(f"\n[yellow]{len(missing)} article(s) without full text — download manually:[/]")
            for a in missing:
                console.print(f"  [link={a.url}]{a.url}[/link]")
            console.print()

    new_arts = [a for a in selected if a.pmid not in existing]
    already  = len(selected) - len(new_arts)
    if already:
        console.print(f"[dim]Skipping {already} already-indexed article(s).[/]")
    if not new_arts:
        console.print("[green]Nothing new to index.[/]")
        return

    console.print(f"Embedding + indexing [cyan]{len(new_arts)}[/] article(s)...")
    build_index(new_arts)
    console.print("[green]Indexed.[/]")


def do_retrieve(query: str) -> None:
    from scireg.retrieval.retriever import retrieve as _retrieve
    print_retrieve_results(_retrieve(query))


def do_ask(query: str) -> None:
    from scireg.graph.state import build_graph
    graph = build_graph()
    result = graph.invoke({"query": query})
    ents = {k: v for k, v in result.get("entities", {}).items() if v}
    if ents:
        console.print(f"[dim]entities: {ents}[/]")
    console.print(f"[dim]{len(result.get('nodes', []))} passages retrieved[/]\n")
    console.print(Markdown(result.get("answer", "(no answer)")))


def do_import_pdf(path: str) -> None:
    from scireg.ingest.index import build_index
    from scireg.sources.pdf import load_pdf_as_article

    p = Path(path)
    if not p.exists():
        console.print(f"[red]File not found:[/] {path}")
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        article = load_pdf_as_article(p)
    for w in caught:
        console.print(f"[yellow]Warning:[/] {w.message}")
    console.print(f"Loaded [cyan]{p.name}[/] → PMID={article.pmid}, title={article.title[:60]}")
    console.print("Embedding + indexing...")
    build_index([article])
    console.print("[green]Indexed.[/]")


def do_import_dir(path: str) -> None:
    from scireg.ingest.index import build_index
    from scireg.sources.pdf import load_pdf_directory

    d = Path(path)
    if not d.is_dir():
        console.print(f"[red]Not a directory:[/] {path}")
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        articles = load_pdf_directory(d)
    for w in caught:
        console.print(f"[yellow]Warning:[/] {w.message}")
    if not articles:
        console.print("[yellow]No articles loaded.[/]")
        return
    console.print(f"Loaded [cyan]{len(articles)}[/] PDFs. Embedding + indexing...")
    build_index(articles)
    console.print("[green]Indexed.[/]")


def do_status() -> None:
    from scireg.ingest.index import get_indexed_pmids
    pmids = get_indexed_pmids()
    if pmids:
        console.print(f"Index: [cyan]{len(pmids)}[/] unique article(s) stored.")
    else:
        console.print("[yellow]Index is empty.[/] Run [cyan]/index <query>[/] to populate it.")


# ---------------------------------------------------------------------------
# Typer subcommands (thin wrappers — all logic lives in do_* above)
# ---------------------------------------------------------------------------

@app.command()
def search(query: str, retmax: int = 15):
    """Raw PubMed search with full-text availability indicators."""
    do_search(query, retmax)


@app.command()
def index(query: str, retmax: int = 25, full_text: bool = False):
    """Fetch, preview, select, and index PubMed articles interactively."""
    do_index(query, retmax, full_text)


@app.command()
def retrieve(query: str):
    """Query the local index and show retrieved chunks (no LLM)."""
    do_retrieve(query)


@app.command()
def ask(query: str):
    """Run the full multi-agent RAG pipeline and return a cited answer."""
    do_ask(query)


@app.command(name="import-pdf")
def import_pdf(path: str):
    """Index a single manually downloaded PDF (Results section only)."""
    do_import_pdf(path)


@app.command(name="import-dir")
def import_dir(path: str):
    """Index all PDFs in a directory (Results section only)."""
    do_import_dir(path)


if __name__ == "__main__":
    app()
