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
    # Brief pause so the elink call doesn't immediately follow esearch+efetch
    # and hit NCBI's 3 req/s limit (causing silent empty responses).
    if not __import__("os").getenv("NCBI_API_KEY"):
        __import__("time").sleep(0.4)
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


_LLM_AGENTS = ("synthesizer", "critic", "neuro_entity", "planner", "retriever")


def do_model(backend_key: str = "") -> None:
    """List available backends or switch all LLM agents to a new backend."""
    from rich.table import Table
    from scireg.config import active_backend_key, models_cfg, set_agent_backend

    cfg = models_cfg()
    backends = cfg["backends"]
    current = active_backend_key("synthesizer")

    if not backend_key:
        import questionary

        def _label(key: str, spec: dict) -> str:
            needs = ""
            if "anthropic" in spec["model"]:
                needs = "  [ANTHROPIC_API_KEY]"
            elif "openai" in spec["model"]:
                needs = "  [OPENAI_API_KEY]"
            active = "  ← active" if key == current else ""
            return f"{key:<20} {spec['model']}{needs}{active}"

        choices = [
            questionary.Choice(title=_label(k, v), value=k)
            for k, v in backends.items()
        ]
        # Pre-select the currently active backend
        default = next((c for c in choices if c.value == current), choices[0])

        selected = questionary.select(
            "Select LLM backend  (↑↓ to move, enter to confirm):",
            choices=choices,
            default=default,
        ).ask()

        if selected is None or selected == current:
            console.print("[dim]Unchanged.[/]")
            return
        backend_key = selected

    if backend_key not in backends:
        console.print(f"[red]Unknown backend:[/] {backend_key!r}")
        console.print(f"[dim]Available: {', '.join(backends)}[/]")
        return

    for agent in _LLM_AGENTS:
        set_agent_backend(agent, backend_key)
    spec = backends[backend_key]
    console.print(f"Switched to [cyan]{backend_key}[/]  [dim]({spec['model']})[/]")


# Conversation history for /llm — lives for the duration of the process.
_llm_history: list[dict[str, str]] = []


def do_llm(query: str, *, reset: bool = False) -> None:
    """RAG-grounded answer with visible sources and multi-turn conversation memory."""
    from rich.rule import Rule
    from scireg.agents.synthesize import SYSTEM, _format_sources
    from scireg.llm.router import complete
    from scireg.neuro.entities import expand_query, extract_entities
    from scireg.retrieval.retriever import retrieve

    global _llm_history
    if reset:
        _llm_history = []
        console.print("[dim]Conversation history cleared.[/]")
        return

    # --- Entity extraction + retrieval ---
    ents = extract_entities(query)
    expanded = expand_query(query, ents)
    nonempty = {k: v for k, v in ents.items() if v}
    if nonempty:
        console.print(f"[dim]entities: {nonempty}[/]")

    nodes = retrieve(expanded)
    if not nodes:
        console.print("[yellow]Nothing retrieved — run [/][cyan]/index <query>[/][yellow] first.[/]")
        return

    # --- Show sources ---
    console.print(Rule("[dim]Sources[/]", style="dim"))
    for n in nodes:
        md = n.node.metadata
        url  = md.get("url", "")
        src  = md.get("text_source", "")
        src_tag = "[green]results[/]" if src == "results" else "[dim]abstract[/]"
        pmid_str = f"[link={url}]{md.get('pmid','?')}[/link]" if url else md.get("pmid", "?")
        snippet = n.node.get_content()[:100].replace("\n", " ")
        console.print(
            f"  [bold cyan]{pmid_str}[/] {md.get('title','')[:60]}  "
            f"[dim]({md.get('year','n.d.')})[/]  {src_tag}"
        )
        console.print(f"  [dim]{snippet}…[/]")
        if url:
            console.print(f"  [dim][link={url}]{url}[/link][/]")
    console.print(Rule("[dim]Answer[/]", style="dim"))

    # --- Build messages with history ---
    sources_block = _format_sources(nodes)
    user_content = (
        f"Question: {query}\n\nSources:\n{sources_block}\n\n"
        "Write a concise, cited answer."
    )
    messages = [{"role": "system", "content": SYSTEM}]
    messages.extend(_llm_history)
    messages.append({"role": "user", "content": user_content})

    answer = complete("synthesizer", messages, max_tokens=1200)

    # Persist turn in history (store the bare question, not the full sources block)
    _llm_history.append({"role": "user", "content": f"Question: {query}"})
    _llm_history.append({"role": "assistant", "content": answer})

    console.print(Markdown(answer))
    console.print(f"\n[dim](conversation turn {len(_llm_history) // 2} — /llm --reset to clear)[/]")


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


def do_clear_db(force: bool = False) -> None:
    import shutil
    from scireg.projects import get_active_db_uri
    uri = get_active_db_uri()
    db_path = Path(uri) if Path(uri).is_absolute() else Path.cwd() / uri

    if not db_path.exists():
        console.print("[yellow]Index directory does not exist — nothing to clear.[/]")
        return

    from scireg.ingest.index import get_indexed_pmids
    n = len(get_indexed_pmids())

    if not force:
        import questionary
        confirmed = questionary.confirm(
            f"Delete the entire index ({n} article(s) at {db_path})? This cannot be undone."
        ).ask()
        if not confirmed:
            console.print("[dim]Cancelled.[/]")
            return

    shutil.rmtree(db_path)
    console.print(f"[green]Index cleared.[/] ({db_path})")


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
def llm(
    query: str = typer.Argument("", help="Question to ask. Omit with --reset to clear history."),
    reset: bool = typer.Option(False, "--reset", help="Clear conversation history."),
):
    """Ask a question grounded in the indexed papers, with conversation memory."""
    do_llm(query, reset=reset)


@app.command(name="import-pdf")
def import_pdf(path: str):
    """Index a single manually downloaded PDF (Results section only)."""
    do_import_pdf(path)


@app.command(name="import-dir")
def import_dir(path: str):
    """Index all PDFs in a directory (Results section only)."""
    do_import_dir(path)


@app.command()
def model(backend_key: str = typer.Argument("", help="Backend key to switch to. Omit to list.")):
    """List available LLM backends or switch the active model."""
    do_model(backend_key)


@app.command(name="clear-db")
def clear_db(force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt.")):
    """Delete the entire local index (irreversible)."""
    do_clear_db(force=force)


@app.command(name="delete-project")
def delete_project_cmd(
    name: str,
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt."),
):
    """Delete a project and its entire index (irreversible)."""
    import questionary
    from scireg.projects import delete_project, get_active_project, list_projects

    if not any(p["name"] == name for p in list_projects()):
        console.print(f"[red]Project {name!r} not found.[/]")
        raise typer.Exit(1)
    if not force:
        confirmed = questionary.confirm(
            f"Delete project '{name}' and all its indexed articles? This cannot be undone."
        ).ask()
        if not confirmed:
            console.print("[dim]Cancelled.[/]")
            return
    delete_project(name)
    console.print(f"[green]Project [cyan]{name}[/] deleted.[/]")
    if get_active_project() != name:
        pass
    else:
        console.print("[dim]Switched to default global index.[/]")


if __name__ == "__main__":
    app()
