"""scirag CLI — subcommands for scripting, or just `scirag` for the interactive shell.

scirag                                        # interactive shell (default)
scirag search "grid cells entorhinal"         # raw PubMed, no LLM
scirag index  "hippocampal place cells" --retmax 30 --full-text
scirag retrieve "place cells remapping"
scirag ask    "How do place cells remap across environments?"
scirag import-pdf paper.pdf
scirag import-dir ./papers/
"""

from __future__ import annotations

import warnings
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown

from scirag.sources import pubmed

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
        from scirag.shell import run_shell

        run_shell()


# ---------------------------------------------------------------------------
# Shared display helpers (used by both CLI commands and the shell)
# ---------------------------------------------------------------------------


def _patch_escape(question):
    """Allow Escape to cancel a questionary prompt (returns None like Ctrl+C)."""
    from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings

    extra = KeyBindings()

    @extra.add("escape")
    def _escape(event):
        event.app.exit(result=None)

    app = question.application
    app.key_bindings = merge_key_bindings([app.key_bindings, extra]) if app.key_bindings else extra
    return question


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
        console.print(
            f"[bold cyan]{pmid_display}[/] {md.get('title', '')} [dim]({md.get('year', 'n.d.')})[/]  {src_tag}"
        )
        if url:
            console.print(f"  [dim][link={url}]{url}[/link][/]")
        console.print(f"  [dim]{snippet}…[/]\n")
    console.print(f"[green]{len(nodes)} chunks retrieved[/]")


# ---------------------------------------------------------------------------
# Core logic (called by both CLI commands and the shell)
# ---------------------------------------------------------------------------


def do_search(query: str, retmax: int = 15) -> None:
    from scirag.sources.pubmed import _pmids_to_pmcids

    arts = pubmed.search_and_fetch(query, retmax=retmax)
    if not arts:
        console.print("[yellow]No results.[/]")
        return

    pmc_map = _pmids_to_pmcids([a.pmid for a in arts])
    console.print()
    for a in arts:
        pmc_tag = "[green]PMC✓[/]" if a.pmid in pmc_map else "[dim]PMC✗[/]"
        doi_tag = "[green]DOI✓[/]" if a.doi else "[dim]DOI✗[/]"
        abst_tag = "[green]ABS✓[/]" if a.abstract else "[red]ABS✗[/]"
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
    from scirag.ingest.index import build_index, get_indexed_pmids
    from scirag.sources.pubmed import _pmids_to_pmcids

    arts = pubmed.search_and_fetch(query, retmax=retmax)
    if not arts:
        console.print("[yellow]No results.[/]")
        return

    existing = get_indexed_pmids()
    # Brief pause so the elink call doesn't immediately follow esearch+efetch
    # and hit NCBI's 3 req/s limit (causing silent empty responses).
    if not __import__("os").getenv("NCBI_API_KEY"):
        __import__("time").sleep(0.4)
    pmc_map = _pmids_to_pmcids([a.pmid for a in arts])

    def _choice_title(a) -> list:
        parts = []
        if a.pmid in existing:
            parts += [("fg:ansiyellow", "[indexed] ")]
        parts += [
            ("fg:ansicyan bold", a.pmid),
            ("", f"  {a.title}  "),
            ("fg:ansibrightblack", f"({a.year})"),
        ]
        tags = []
        if a.pmid in pmc_map:
            tags.append(("fg:ansigreen", "PMC✓"))
        if a.doi:
            tags.append(("fg:ansigreen", "DOI✓"))
        for i, (style, tag) in enumerate(tags):
            parts += [("", "  [" if i == 0 else ", "), (style, tag)]
        if tags:
            parts += [("", "]")]
        parts += [("", "\n     "), ("fg:ansiblue", a.url)]
        return parts

    choices = [
        questionary.Choice(
            title=_choice_title(a),
            value=a,
            checked=(a.pmid not in existing),
        )
        for a in arts
    ]

    import questionary.prompts.common as _qpc

    _qpc.INDICATOR_SELECTED = "✔"
    _qpc.INDICATOR_UNSELECTED = " "

    selected = _patch_escape(
        questionary.checkbox(
            "Select articles to index  (space = toggle, a = all, i = invert, enter = confirm):",
            choices=choices,
        )
    ).ask()

    if not selected:
        console.print("[yellow]Nothing selected.[/]")
        return

    console.print()

    if full_text:
        from scirag.sources.pubmed import enrich_with_fulltext

        console.print("Fetching full text (Results section)...")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            enrich_with_fulltext(selected)
        n = sum(1 for a in selected if a.full_text)
        console.print(f"Full text retrieved for [cyan]{n}[/] / {len(selected)} articles.")
        missing = [a for a in selected if not a.full_text]
        if missing:
            console.print(
                f"\n[yellow]{len(missing)} article(s) without full text — download manually:[/]"
            )
            for a in missing:
                console.print(f"  [link={a.url}]{a.url}[/link]")
            console.print()

    new_arts = [a for a in selected if a.pmid not in existing]
    already = len(selected) - len(new_arts)
    if already:
        console.print(f"[dim]Skipping {already} already-indexed article(s).[/]")
    if not new_arts:
        console.print("[green]Nothing new to index.[/]")
        return

    console.print(f"Embedding + indexing [cyan]{len(new_arts)}[/] article(s)...")
    build_index(new_arts)
    console.print("[green]Indexed.[/]")


def do_retrieve(query: str) -> None:
    from scirag.retrieval.retriever import retrieve as _retrieve

    print_retrieve_results(_retrieve(query))


_LLM_AGENTS = ("synthesizer", "critic", "neuro_entity", "planner", "retriever")


def do_model(backend_key: str = "") -> None:
    """List available backends or switch all LLM agents to a new backend."""
    from scirag.config import active_backend_key, models_cfg, set_agent_backend

    cfg = models_cfg()
    backends = cfg["backends"]
    current = active_backend_key("synthesizer")

    if not backend_key:
        import questionary

        def _label(key: str, spec: dict) -> str:
            if spec["model"] == "claude-code":
                needs = "  [claude CLI · Plus subscription]"
            elif "anthropic" in spec["model"]:
                needs = "  [ANTHROPIC_API_KEY]"
            elif "openai" in spec["model"]:
                needs = "  [OPENAI_API_KEY]"
            else:
                needs = ""
            active = "  ← active" if key == current else ""
            return f"{key:<20} {spec['model']}{needs}{active}"

        choices = [questionary.Choice(title=_label(k, v), value=k) for k, v in backends.items()]
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
    from scirag.agents.synthesize import SYSTEM, _format_sources
    from scirag.llm.router import complete
    from scirag.neuro.entities import expand_query, extract_entities
    from scirag.retrieval.retriever import retrieve

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
        console.print(
            "[yellow]Nothing retrieved — run [/][cyan]/index <query>[/][yellow] first.[/]"
        )
        return

    # --- Show sources ---
    console.print(Rule("[dim]Sources[/]", style="dim"))
    for n in nodes:
        md = n.node.metadata
        url = md.get("url", "")
        src = md.get("text_source", "")
        src_tag = "[green]results[/]" if src == "results" else "[dim]abstract[/]"
        pmid_str = f"[link={url}]{md.get('pmid', '?')}[/link]" if url else md.get("pmid", "?")
        snippet = n.node.get_content()[:100].replace("\n", " ")
        console.print(
            f"  [bold cyan]{pmid_str}[/] {md.get('title', '')[:60]}  "
            f"[dim]({md.get('year', 'n.d.')})[/]  {src_tag}"
        )
        console.print(f"  [dim]{snippet}…[/]")
        if url:
            console.print(f"  [dim][link={url}]{url}[/link][/]")
    console.print(Rule("[dim]Answer[/]", style="dim"))

    # --- Build messages with history ---
    sources_block = _format_sources(nodes)
    user_content = (
        f"Question: {query}\n\nSources:\n{sources_block}\n\nWrite a concise, cited answer."
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
    from scirag.ingest.index import build_index
    from scirag.sources.pdf import load_pdf_as_article

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
    from scirag.ingest.index import build_index
    from scirag.sources.pdf import load_pdf_directory

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


_ENV_KEYS = {
    "NCBI_API_KEY": "NCBI API key — raises rate limit from 3 to 10 req/s",
    "NCBI_EMAIL": "Email sent with NCBI requests (politeness header)",
    "ANTHROPIC_API_KEY": "Required for claude-sonnet / claude-opus backends",
    "OPENAI_API_KEY": "Required for gpt backend",
}

_HOME_ENV = Path.home() / ".scirag-agent" / ".env"


def _mask(value: str) -> str:
    return value[:4] + "****" if len(value) > 4 else "****"


def _read_home_env() -> dict[str, str]:
    """Parse ~/.scirag-agent/.env into a dict."""
    if not _HOME_ENV.exists():
        return {}
    result = {}
    for line in _HOME_ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        result[k.strip()] = v.strip().strip('"').strip("'")
    return result


def _write_home_env(env: dict[str, str]) -> None:
    _HOME_ENV.parent.mkdir(parents=True, exist_ok=True)
    _HOME_ENV.write_text("\n".join(f'{k}="{v}"' for k, v in env.items()) + "\n")


def do_env(action: str = "", key: str = "", value: str = "") -> None:
    """Show, set, or unset environment variables in ~/.scirag-agent/.env."""
    import os
    from rich.table import Table

    home_env = _read_home_env()

    if not action:
        table = Table(box=None, padding=(0, 2), show_header=True, header_style="bold")
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Status", no_wrap=True)
        table.add_column("Value", style="dim", no_wrap=True)
        table.add_column("Description", style="dim")
        for k, desc in _ENV_KEYS.items():
            live = os.getenv(k, "")
            stored = home_env.get(k, "")
            if live:
                status = "[green]set[/]"
                display = _mask(live)
                if not stored:
                    display += "  [dim](local .env only)[/]"
            else:
                status = "[red]missing[/]"
                display = ""
            table.add_row(k, status, display, desc)
        console.print(table)
        console.print(f"\n[dim]Stored in: {_HOME_ENV}[/]")
        console.print("[dim]Usage: /env set <KEY> <value>   /env unset <KEY>[/]")
        return

    if action == "set":
        if not key or not value:
            console.print("[yellow]Usage:[/] /env set <KEY> <value>")
            return
        if key not in _ENV_KEYS:
            import questionary

            if not questionary.confirm(f"{key!r} is not a known key. Save anyway?").ask():
                return
        home_env[key] = value
        _write_home_env(home_env)
        os.environ[key] = value
        console.print(f"[green]Set[/] {key} = {_mask(value)}  [dim](saved to {_HOME_ENV})[/]")
        return

    if action == "unset":
        if not key:
            console.print("[yellow]Usage:[/] /env unset <KEY>")
            return
        if home_env.pop(key, None):
            _write_home_env(home_env)
            console.print(f"[green]Removed[/] {key} from {_HOME_ENV}")
        else:
            console.print(f"[yellow]{key} not in {_HOME_ENV}[/]")
        return

    console.print(f"[red]Unknown action:[/] {action!r}  (use: set / unset)")


def do_llm_ui(port: int = 8000) -> None:
    """Launch the Chainlit web UI and open it in the browser."""
    import subprocess
    import time
    import webbrowser
    from pathlib import Path

    try:
        import chainlit  # noqa: F401
    except ImportError:
        console.print("[red]chainlit not installed.[/] Run: [cyan]uv sync --extra ui[/]")
        return

    ui_path = Path(__file__).parent / "ui.py"
    url = f"http://localhost:{port}"
    console.print(f"Starting web UI at [link={url}]{url}[/link] …")

    proc = subprocess.Popen(
        [
            "uv",
            "run",
            "--extra",
            "ui",
            "chainlit",
            "run",
            str(ui_path),
            "--port",
            str(port),
            "--headless",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Poll until the server is up (max 10 s)
    import httpx

    for _ in range(20):
        time.sleep(0.5)
        try:
            httpx.get(url, timeout=0.5)
            break
        except Exception:
            pass

    webbrowser.open(url)
    console.print("[green]Web UI open.[/] Press [bold]Ctrl+C[/] to stop.")
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        console.print("\n[dim]Web UI stopped.[/]")


def do_status() -> None:
    from scirag.ingest.index import get_indexed_articles

    articles = get_indexed_articles()
    if not articles:
        console.print("[yellow]Index is empty.[/] Run [cyan]/index <query>[/] to populate it.")
        return

    console.print(f"Index: [cyan]{len(articles)}[/] unique article(s) stored.\n")
    from rich.table import Table

    table = Table(box=None, padding=(0, 1), show_header=True, header_style="bold dim")
    table.add_column("PMID", style="cyan", no_wrap=True)
    table.add_column("Year", style="dim", no_wrap=True)
    table.add_column("Title")
    for a in sorted(articles, key=lambda x: x["year"], reverse=True):
        table.add_row(a["pmid"], a["year"], a["title"])
    console.print(table)


def do_remove(pmids: list[str]) -> None:
    import questionary
    from scirag.ingest.index import get_indexed_articles, remove_articles

    articles = get_indexed_articles()
    if not articles:
        console.print("[yellow]Index is empty.[/]")
        return

    if pmids:
        known = {a["pmid"] for a in articles}
        bad = [p for p in pmids if p not in known]
        if bad:
            console.print(f"[red]Not found in index:[/] {', '.join(bad)}")
            pmids = [p for p in pmids if p in known]
        if not pmids:
            return
    else:
        import questionary.prompts.common as _qpc

        _qpc.INDICATOR_SELECTED = "✘"
        _qpc.INDICATOR_UNSELECTED = " "

        sorted_articles = sorted(articles, key=lambda x: x["year"], reverse=True)
        choices = [
            questionary.Choice(
                title=[
                    ("fg:ansicyan bold", a["pmid"]),
                    ("", f"  {a['title']}  "),
                    ("fg:ansibrightblack", f"({a['year']})"),
                    ("", "\n     "),
                    ("fg:ansiblue", f"https://pubmed.ncbi.nlm.nih.gov/{a['pmid']}/"),
                ],
                value=a["pmid"],
            )
            for a in sorted_articles
        ]
        selected = _patch_escape(
            questionary.checkbox("Select articles to remove:", choices=choices)
        ).ask()
        if not selected:
            console.print("[dim]Cancelled.[/]")
            return
        pmids = selected

    titles = {a["pmid"]: a["title"] for a in articles}
    for p in pmids:
        console.print(f"  [dim]removing[/] [cyan]{p}[/]  {titles.get(p, '')[:60]}")

    remove_articles(pmids)
    console.print(f"[green]Removed {len(pmids)} article(s) from the index.[/]")


def do_clear_db(force: bool = False) -> None:
    import shutil
    from scirag.projects import get_active_db_uri

    uri = get_active_db_uri()
    db_path = Path(uri) if Path(uri).is_absolute() else Path.cwd() / uri

    if not db_path.exists():
        console.print("[yellow]Index directory does not exist — nothing to clear.[/]")
        return

    from scirag.ingest.index import get_indexed_pmids

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


@app.command(name="llm-ui")
def llm_ui(port: int = typer.Option(8000, "--port", "-p", help="Port to listen on.")):
    """Launch the Chainlit web UI for RAG chat."""
    do_llm_ui(port)


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
    from scirag.projects import delete_project, get_active_project, list_projects

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


@app.command()
def env(
    action: str = typer.Argument("", help="Action: set / unset. Omit to list."),
    key: str = typer.Argument("", help="Environment variable name."),
    value: str = typer.Argument("", help="Value to set."),
):
    """Show, set, or unset API keys stored in ~/.scirag-agent/.env."""
    do_env(action, key, value)


if __name__ == "__main__":
    app()
