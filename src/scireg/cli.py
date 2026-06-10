"""scireg CLI.

    scireg index "hippocampal place cells" --retmax 30   # fetch + index PubMed
    scireg ask   "How do place cells remap across environments?"
    scireg search "grid cells entorhinal"                # raw PubMed, no LLM
    scireg import-pdf paper.pdf                          # index a single PDF
    scireg import-dir ./papers/                          # index a directory of PDFs
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
    """Raw PubMed search — shows availability of full-text sources before indexing."""
    from scireg.sources.pubmed import _pmids_to_pmcids

    arts = pubmed.search_and_fetch(query, retmax=retmax)
    if not arts:
        console.print("[yellow]No results.[/]")
        return

    pmc_map = _pmids_to_pmcids([a.pmid for a in arts])

    console.print()
    for a in arts:
        has_pmc = a.pmid in pmc_map
        has_doi = bool(a.doi)
        has_abstract = bool(a.abstract)

        pmc_tag  = "[green]PMC✓[/]"  if has_pmc  else "[dim]PMC✗[/]"
        doi_tag  = "[green]DOI✓[/]"  if has_doi  else "[dim]DOI✗[/]"
        abst_tag = "[green]ABS✓[/]"  if has_abstract else "[red]ABS✗[/]"

        console.print(
            f"[bold cyan][link={a.url}]{a.pmid}[/link][/] {a.title[:70]} [dim]({a.year})[/]\n"
            f"  {pmc_tag} {doi_tag} {abst_tag}  [dim]{a.journal}[/]  [dim][link={a.url}]{a.url}[/link][/]"
        )

    n_pmc  = sum(1 for a in arts if a.pmid in pmc_map)
    n_doi  = sum(1 for a in arts if a.doi)
    n_abst = sum(1 for a in arts if a.abstract)
    console.print(
        f"\n[green]{len(arts)} articles[/]  —  "
        f"PMC full-text: {n_pmc}  |  DOI (Unpaywall): {n_doi}  |  Abstract only: {n_abst - n_pmc}"
    )


def _print_article_list(arts, existing: set, pmc_map: dict) -> None:
    """Print articles as a numbered list with full URLs on separate lines."""
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


def _checkbox_label(a, existing: set, pmc_map: dict) -> str:
    """Compact single-line label for the questionary checkbox."""
    tags = []
    if a.pmid in existing:
        tags.append("indexed")
    if a.pmid in pmc_map:
        tags.append("PMC✓")
    if a.doi:
        tags.append("DOI✓")
    tag_str = f"  [{', '.join(tags)}]" if tags else ""
    return f"{a.pmid}  {a.title[:60]}  ({a.year}){tag_str}"


@app.command()
def index(query: str, retmax: int = 25, full_text: bool = False):
    """Fetch PubMed results, let you choose which to index, then embed and store."""
    import warnings
    import questionary
    from scireg.ingest.index import build_index, get_indexed_pmids
    from scireg.sources.pubmed import _pmids_to_pmcids

    arts = pubmed.search_and_fetch(query, retmax=retmax)
    if not arts:
        console.print("[yellow]No results.[/]")
        return

    existing = get_indexed_pmids()
    pmc_map = _pmids_to_pmcids([a.pmid for a in arts])

    console.print()
    _print_article_list(arts, existing, pmc_map)
    console.print()

    choices = [
        questionary.Choice(
            title=_checkbox_label(a, existing, pmc_map),
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
        console.print("[yellow]Nothing selected — exiting.[/]")
        return

    console.print()

    if full_text:
        from scireg.sources.pubmed import enrich_with_fulltext
        console.print("Fetching full text (Results section) for selected articles...")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            enrich_with_fulltext(selected)
        n = sum(1 for a in selected if a.full_text)
        console.print(f"Full text retrieved for [cyan]{n}[/] / {len(selected)} articles.")

        missing = [a for a in selected if not a.full_text]
        if missing:
            console.print(f"\n[yellow]{len(missing)} article(s) without full text — download PDF manually:[/]")
            for a in missing:
                console.print(f"  [link={a.url}]{a.url}[/link]")
            console.print()

    already = [a for a in selected if a.pmid in existing]
    new_arts = [a for a in selected if a.pmid not in existing]
    if already:
        console.print(f"[dim]Skipping {len(already)} already-indexed article(s).[/]")
    if not new_arts:
        console.print("[green]Nothing new to index.[/]")
        return

    console.print(f"Embedding + indexing [cyan]{len(new_arts)}[/] article(s)...")
    build_index(new_arts)
    console.print("[green]Indexed.[/]")


@app.command()
def retrieve(query: str):
    """Query the local index and show retrieved chunks (no LLM)."""
    from scireg.retrieval.retriever import retrieve as _retrieve

    nodes = _retrieve(query)
    if not nodes:
        console.print("[yellow]No results — have you run `scireg index` yet?[/]")
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


@app.command(name="import-pdf")
def import_pdf(path: str):
    """Index a single manually downloaded PDF (extracts Results section)."""
    from pathlib import Path

    from scireg.ingest.index import build_index
    from scireg.sources.pdf import load_pdf_as_article

    p = Path(path)
    if not p.exists():
        console.print(f"[red]File not found:[/] {path}")
        raise typer.Exit(1)

    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        article = load_pdf_as_article(p)
    for w in caught:
        console.print(f"[yellow]Warning:[/] {w.message}")

    console.print(f"Loaded [cyan]{p.name}[/] → PMID={article.pmid}, title={article.title[:60]}")
    console.print("Embedding + indexing...")
    build_index([article])
    console.print("[green]Indexed.[/]")


@app.command(name="import-dir")
def import_dir(path: str):
    """Index all PDFs in a directory (extracts Results section from each)."""
    from pathlib import Path

    from scireg.ingest.index import build_index
    from scireg.sources.pdf import load_pdf_directory

    d = Path(path)
    if not d.is_dir():
        console.print(f"[red]Not a directory:[/] {path}")
        raise typer.Exit(1)

    import warnings
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


if __name__ == "__main__":
    app()
