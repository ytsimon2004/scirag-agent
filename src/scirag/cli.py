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
    # Esc leads every ANSI escape sequence (arrows, etc.), so prompt_toolkit waits
    # ttimeoutlen (default 0.5s) to tell a lone Esc from a sequence. Shorten it so
    # Esc-to-cancel feels instant; raise toward 0.1–0.15 if used over laggy SSH.
    app.ttimeoutlen = 0.05
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


def _source_tag(src: str) -> str:
    """Rich markup for a chunk's text_source: results / review / abstract / text."""
    return {
        "results": "[green]results[/]",
        "review": "[cyan]review[/]",
        "abstract": "[dim]abstract[/]",
        "text": "[cyan]text[/]",
    }.get(src, "[dim]—[/]")


def _origin_tag(origin: str) -> str:
    """Rich markup for a record's origin: pubmed / biorxiv / text."""
    return {
        "pubmed": "[blue]PubMed[/]",
        "biorxiv": "[magenta]bioRxiv[/]",
        "text": "[cyan]text[/]",
    }.get(origin, "[dim]—[/]")


def _record_url(identifier: str, origin: str) -> str:
    """Public URL for an indexed record. Returns '' for free-text entries."""
    if origin == "biorxiv":
        return f"https://www.biorxiv.org/content/{identifier}"
    if origin == "text":
        return ""
    return f"https://pubmed.ncbi.nlm.nih.gov/{identifier}/"


def _authors_short(authors: list[str]) -> str:
    """Compact byline: first author … last author (the last is usually senior/
    corresponding). One or two authors are shown in full."""
    if not authors:
        return ""
    if len(authors) == 1:
        return authors[0]
    if len(authors) == 2:
        return f"{authors[0]}, {authors[1]}"
    return f"{authors[0]} … {authors[-1]}"


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
        src_tag = _source_tag(src)
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


def do_index(
    query: str,
    retmax: int = 25,
    full_text: bool = False,
    year_from: str = "",
    year_to: str = "",
) -> None:
    import questionary
    from scirag.ingest.index import build_index, get_indexed_pmids
    from scirag.sources.pubmed import _pmids_to_pmcids

    arts = pubmed.search_and_fetch(query, retmax=retmax, min_year=year_from, max_year=year_to)
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
        authors = _authors_short(a.authors)
        if authors:
            parts += [("", "\n     "), ("fg:#6c6c6c", authors)]
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


def do_bindex(
    query: str,
    retmax: int = 25,
    days_back: int = 180,
    full_text: bool = False,
    year_from: str = "",
    year_to: str = "",
) -> None:
    """Fetch, preview, select, and index bioRxiv preprints interactively."""
    import questionary
    from scirag.ingest.index import build_index, get_indexed_pmids
    from scirag.sources import biorxiv

    if year_from or year_to:
        date_hint = f"{year_from or '…'}–{year_to or '…'}"
        console.print(f"[dim]Searching bioRxiv via Europe PMC ({date_hint})…[/]")
    else:
        console.print(f"[dim]Searching bioRxiv via Europe PMC (last {days_back} days)…[/]")
    arts = biorxiv.search_and_fetch(
        query, days_back=days_back, retmax=retmax, min_year=year_from, max_year=year_to
    )
    if not arts:
        console.print("[yellow]No results.[/] Try other terms or a wider [cyan]--days-back[/].")
        return

    existing = get_indexed_pmids()  # holds DOIs for preprints, PMIDs for PubMed

    def _choice_title(a) -> list:
        parts = []
        if a.pmid in existing:
            parts += [("fg:ansiyellow", "[indexed] ")]
        parts += [
            ("fg:ansicyan bold", a.doi),
            ("", f"  {a.title}  "),
            ("fg:ansibrightblack", f"({a.year})"),
        ]
        category = a.pub_types[0] if a.pub_types else ""
        if category:
            parts += [("", "  ["), ("fg:ansimagenta", category), ("", "]")]
        authors = _authors_short(a.authors)
        if authors:
            parts += [("", "\n     "), ("fg:#6c6c6c", authors)]
        parts += [("", "\n     "), ("fg:ansiblue", a.url)]
        return parts

    choices = [
        questionary.Choice(title=_choice_title(a), value=a, checked=(a.pmid not in existing))
        for a in arts
    ]

    import questionary.prompts.common as _qpc

    _qpc.INDICATOR_SELECTED = "✔"
    _qpc.INDICATOR_UNSELECTED = " "

    selected = _patch_escape(
        questionary.checkbox(
            "Select preprints to index  (space = toggle, a = all, i = invert, enter = confirm):",
            choices=choices,
        )
    ).ask()

    if not selected:
        console.print("[yellow]Nothing selected.[/]")
        return

    console.print()

    if full_text:
        console.print("Fetching full text (Results section)...")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            biorxiv.enrich_with_fulltext(selected)
        n = sum(1 for a in selected if a.full_text)
        console.print(f"Full text retrieved for [cyan]{n}[/] / {len(selected)} preprint(s).")
        missing = [a for a in selected if not a.full_text]
        if missing:
            console.print(
                f"\n[yellow]{len(missing)} preprint(s) without full text — download manually:[/]"
            )
            for a in missing:
                console.print(f"  [link={a.url}]{a.url}[/link]")
            console.print()

    new_arts = [a for a in selected if a.pmid not in existing]
    already = len(selected) - len(new_arts)
    if already:
        console.print(f"[dim]Skipping {already} already-indexed preprint(s).[/]")
    if not new_arts:
        console.print("[green]Nothing new to index.[/]")
        return

    console.print(f"Embedding + indexing [cyan]{len(new_arts)}[/] preprint(s)...")
    build_index(new_arts)
    console.print("[green]Indexed.[/]")


def do_retrieve(query: str) -> None:
    from scirag.retrieval.retriever import retrieve as _retrieve

    print_retrieve_results(_retrieve(query))


def do_show(pmid: str) -> None:
    """Print the stored embedded text (abstract or Results) for one indexed PMID."""
    from rich.rule import Rule

    from scirag.ingest.index import get_article_chunks

    pmid = pmid.strip()
    if not pmid:
        console.print("[yellow]Usage:[/] /show <pmid>")
        return

    art = get_article_chunks(pmid)
    if not art:
        console.print(
            f"[yellow]No indexed article with PMID[/] [cyan]{pmid}[/]. "
            "Run [cyan]/status[/] to list stored PMIDs."
        )
        return

    src_tag = _source_tag(art["text_source"])
    chunks = art["chunks"]
    console.print(
        f"[bold cyan]{art['pmid']}[/] {art['title']} [dim]({art['year'] or 'n.d.'})[/]  "
        f"{src_tag} · {len(chunks)} chunk(s)"
    )
    if art["authors"]:
        console.print(f"  [dim]{art['authors']}[/]")
    for i, text in enumerate(chunks, 1):
        console.print(Rule(f"[dim]chunk {i}/{len(chunks)}[/]", style="dim"))
        console.print(text)


_LLM_AGENTS = ("synthesizer", "critic", "planner", "retriever")


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
            elif spec["model"] == "codex":
                needs = "  [codex CLI · OpenAI subscription]"
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


def do_effort(level: str = "") -> None:
    """Show or set the session reasoning effort (low/medium/high)."""
    from scirag.config import _VALID_EFFORT, get_effort, set_effort

    if not level:
        console.print(
            f"Reasoning effort: [cyan]{get_effort()}[/]  "
            f"[dim](options: {' / '.join(_VALID_EFFORT)} — higher = slower, more thorough)[/]"
        )
        return

    try:
        set_effort(level.lower())
    except ValueError:
        console.print(f"[red]Unknown effort:[/] {level!r}")
        console.print(f"[dim]Choose one of: {', '.join(_VALID_EFFORT)}[/]")
        return
    console.print(f"Reasoning effort set to [cyan]{get_effort()}[/]")


# Conversation history for /llm — lives for the duration of the process.
_llm_history: list[dict[str, str]] = []


def _source_summary(nodes) -> str:
    """One-line summary of the sources grounding the answer. Full sources with
    snippets and links are available in the web UI (/llm-ui)."""
    pmids: list[str] = []
    for n in nodes:
        p = n.node.metadata.get("pmid", "?")
        if p not in pmids:
            pmids.append(p)
    papers = "paper" if len(pmids) == 1 else "papers"
    return (
        f"[dim]▸ grounded on {len(nodes)} chunk(s) from {len(pmids)} {papers} "
        f"({', '.join(pmids)})[/]"
    )


def _answer_with_spinner(messages: list[dict[str, str]], *, max_tokens: int | None = None) -> str:
    """Run the synthesizer call behind a live 'Reasoning…' spinner that counts up
    elapsed seconds, so the long wait shows progress instead of a frozen screen.
    Returns the answer text.

    The blocking completion runs in a worker thread while the main thread refreshes
    the timer. With max_tokens left None, the router sizes the budget from the
    session effort."""
    import threading
    import time

    from rich.text import Text

    from scirag.llm.router import complete

    out: dict = {}

    def _run() -> None:
        try:
            out["answer"] = complete("synthesizer", messages, max_tokens=max_tokens)
        except BaseException as exc:  # propagate to the caller below
            out["error"] = exc

    worker = threading.Thread(target=_run, daemon=True)
    t0 = time.perf_counter()
    with console.status(Text("Reasoning… 0.0s", style="dim"), spinner="dots") as status:
        worker.start()
        while worker.is_alive():
            status.update(Text(f"Reasoning… {time.perf_counter() - t0:.1f}s", style="dim"))
            worker.join(timeout=0.1)

    if "error" in out:
        raise out["error"]
    return out["answer"]


def do_llm(query: str, *, reset: bool = False) -> None:
    """RAG-grounded answer with a one-line source summary and conversation memory.

    Full source passages (with snippets and links) live in the web UI (/llm-ui)."""
    import time

    from rich.rule import Rule
    from scirag.agents.pipeline import prepare_answer

    global _llm_history
    if reset:
        _llm_history = []
        console.print("[dim]Conversation history cleared.[/]")
        return

    result = prepare_answer(query, _llm_history)

    if result.use_rag:
        console.print(_source_summary(result.nodes))
    else:
        console.print(
            f"[dim]No relevant sources (top score {result.top_score:.2f}) — answering from general knowledge.[/]"
        )

    t0 = time.perf_counter()
    answer = _answer_with_spinner(result.messages)
    elapsed = time.perf_counter() - t0

    # Persist turn in history (store the bare question, not the full sources block)
    _llm_history.append({"role": "user", "content": f"Question: {query}"})
    _llm_history.append({"role": "assistant", "content": answer})

    from scirag.config import get_effort

    console.print(Rule("[dim]Answer[/]", style="dim"))
    console.print(Markdown(answer))
    console.print(
        f"\n[dim](effort {get_effort()} · {elapsed:.1f}s · turn {len(_llm_history) // 2} "
        "— /llm --reset to clear)[/]"
    )


def do_import_pdf(path: str) -> None:
    from scirag.ingest.index import build_index
    from scirag.sources.pdf import load_pdf_as_article

    p = Path(path).expanduser()
    if not p.exists():
        console.print(f"[red]File not found:[/] {path}")
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        article = load_pdf_as_article(p)
    for w in caught:
        console.print(f"[yellow]Warning:[/] {w.message}")
    if article is None:
        console.print(
            f"[yellow]Not imported:[/] {p.name} could not be resolved to a PubMed record."
        )
        return
    id_label = "DOI" if article.source == "biorxiv" else "PMID"
    console.print(
        f"Loaded [cyan]{p.name}[/] → {id_label}={article.pmid} "
        f"[dim]({article.source})[/], title={article.title[:55]}"
    )
    console.print("Embedding + indexing...")
    build_index([article])
    console.print("[green]Indexed.[/]")


def do_import_dir(path: str) -> None:
    from scirag.ingest.index import build_index
    from scirag.sources.pdf import load_pdf_directory

    d = Path(path).expanduser()
    if not d.is_dir():
        console.print(f"[red]Not a directory:[/] {path}")
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        articles = load_pdf_directory(d)
    for w in caught:
        console.print(f"[yellow]Warning:[/] {w.message}")
    if not articles:
        console.print("[yellow]No PDFs resolved to a PubMed record — nothing imported.[/]")
        return
    console.print(f"Resolved [cyan]{len(articles)}[/] article(s). Embedding + indexing...")
    build_index(articles)
    console.print("[green]Indexed.[/]")


def do_import(path: str) -> None:
    """Index a PDF file or every PDF in a directory — routed by what `path` is."""
    p = Path(path).expanduser()
    if p.is_dir():
        do_import_dir(str(p))
    elif p.is_file() or p.suffix.lower() == ".pdf":
        do_import_pdf(str(p))
    else:
        console.print(f"[red]No such file or directory:[/] {path}")


def do_text_index() -> None:
    """Interactively collect metadata + free text, then embed directly into the index."""
    import click
    import questionary
    from datetime import datetime

    from scirag.ingest.index import build_index, get_indexed_pmids
    from scirag.sources.pubmed import Article

    console.print("[dim]Enter metadata — press Enter to skip any field.[/]")

    title = questionary.text("Title:").ask()
    if title is None:
        console.print("[dim]Cancelled.[/]")
        return
    title = title.strip()

    raw_id = questionary.text("Identifier (blank = auto-generate):").ask()
    if raw_id is None:
        console.print("[dim]Cancelled.[/]")
        return
    raw_id = raw_id.strip()
    if not raw_id:
        identifier = f"text-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    elif not raw_id.startswith("text-"):
        identifier = f"text-{raw_id}"
    else:
        identifier = raw_id

    origin = questionary.text("Origin (journal / source):").ask()
    if origin is None:
        console.print("[dim]Cancelled.[/]")
        return
    origin = origin.strip()

    year = questionary.text("Year:").ask()
    if year is None:
        console.print("[dim]Cancelled.[/]")
        return
    year = year.strip()

    author_raw = questionary.text("Author(s) (comma-separated):").ask()
    if author_raw is None:
        console.print("[dim]Cancelled.[/]")
        return
    authors = [a.strip() for a in author_raw.split(",") if a.strip()]

    existing = get_indexed_pmids()
    if identifier in existing:
        if not questionary.confirm(
            f"Identifier {identifier!r} is already indexed. Overwrite?"
        ).ask():
            console.print("[dim]Cancelled.[/]")
            return

    console.print("[dim]Opening editor for text body (save and close to continue)…[/]")
    text_body = click.edit(require_save=True, extension=".txt")
    if not text_body or not text_body.strip():
        console.print("[yellow]No text entered — cancelled.[/]")
        return

    article = Article(
        pmid=identifier,
        title=title or identifier,
        abstract="",
        full_text=text_body.strip(),
        full_text_kind="text",
        journal=origin,
        year=year,
        authors=authors,
        source="text",
    )

    console.print(f"Embedding + indexing [cyan]{identifier}[/]…")
    build_index([article])
    console.print(f"[green]Indexed.[/]  identifier=[cyan]{identifier}[/]")


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


def print_system_info() -> None:
    """Print the system info panel (LLM, embedding, Ollama, project, index, directory)."""
    from pathlib import Path

    from rich.panel import Panel
    from rich.table import Table

    from scirag.config import active_backend_key, get_effort, models_cfg
    from scirag.ingest.index import get_indexed_pmids
    from scirag.projects import get_active_project

    emb = models_cfg()["embeddings"]["model"]
    llm_key = active_backend_key("synthesizer")
    llm_model = models_cfg()["backends"][llm_key]["model"]
    project = get_active_project() or "none (global)"
    cwd = Path.cwd()

    try:
        n_articles = len(get_indexed_pmids())
        index_str = f"{n_articles} article(s)"
    except Exception:
        index_str = "empty"

    # Ollama connectivity check
    try:
        import httpx

        base = models_cfg()["embeddings"]["api_base"]
        httpx.get(f"{base}/api/tags", timeout=1.5).raise_for_status()
        ollama = "[green]running[/]"
    except Exception:
        ollama = "[red]offline[/]"

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", no_wrap=True, min_width=11)
    grid.add_column()
    grid.add_row("llm", f"[cyan]{llm_model}[/]  [dim]· effort {get_effort()}[/]")
    grid.add_row("embedding", f"[dim]{emb}[/]")
    grid.add_row("ollama", ollama)
    grid.add_row(
        "project",
        f"[yellow]{project}[/]" if get_active_project() else f"[dim]{project}[/]",
    )
    grid.add_row("index", f"[dim]{index_str}[/]")
    grid.add_row("directory", f"[dim]{cwd}[/]")

    console.print(
        Panel(
            grid,
            title="[bold cyan]scirag-agent[/]  [dim]scientific RAG · PubMed/PMC[/]",
            border_style="dim",
            padding=(0, 1),
        )
    )


def do_status() -> None:
    from rich.table import Table

    from scirag.ingest.index import get_indexed_articles

    print_system_info()
    console.print()

    articles = get_indexed_articles()
    if not articles:
        console.print("[yellow]Index is empty.[/] Run [cyan]/index <query>[/] to populate it.")
        return

    console.print(f"Index: [cyan]{len(articles)}[/] unique article(s) stored.\n")

    table = Table(box=None, padding=(0, 1), show_header=True, header_style="bold dim")
    table.add_column("PMID / DOI", style="cyan", no_wrap=True)
    table.add_column("Origin", no_wrap=True)
    table.add_column("Year", style="dim", no_wrap=True)
    table.add_column("First author", no_wrap=True)
    table.add_column("Text", no_wrap=True)
    table.add_column("Title")
    for a in sorted(articles, key=lambda x: x["year"], reverse=True):
        source_cell = _source_tag(a.get("text_source", ""))
        origin_cell = _origin_tag(a.get("origin", "pubmed"))
        author = a.get("first_author") or "[dim]—[/]"
        url = _record_url(a["pmid"], a.get("origin", "pubmed"))
        id_cell = f"[link={url}]{a['pmid']}[/link]" if url else a["pmid"]
        table.add_row(id_cell, origin_cell, a["year"], author, source_cell, a["title"])
    console.print(table)

    n_full = sum(1 for a in articles if a.get("text_source") in ("results", "review"))
    console.print(f"\n[dim]{n_full} with full text, {len(articles) - n_full} abstract-only.[/]")


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
    from scirag.ingest.index import get_indexed_pmids

    uri = get_active_db_uri()
    db_path = Path(uri) if Path(uri).is_absolute() else Path.cwd() / uri

    if not db_path.exists():
        console.print("[yellow]Index directory does not exist — nothing to clear.[/]")
        return

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


# Reused flag help (single source — read by the shell's completer/toolbar too).
_RETMAX_HELP = "max number of results to fetch"
_DAYS_BACK_HELP = "how many days back to search bioRxiv (via Europe PMC)"
_FULL_TEXT_HELP = "also fetch + index each paper's full-text Results section (slower)"
_YEAR_FROM_HELP = "earliest publication year to include (e.g. 2018)"
_YEAR_TO_HELP = "latest publication year to include (e.g. 2024)"


@app.command()
def index(
    query: str,
    retmax: int = typer.Option(25, help=_RETMAX_HELP),
    full_text: bool = typer.Option(False, help=_FULL_TEXT_HELP),
    year_from: str = typer.Option("", help=_YEAR_FROM_HELP),
    year_to: str = typer.Option("", help=_YEAR_TO_HELP),
):
    """Fetch, preview, select, and index PubMed articles interactively."""
    do_index(query, retmax, full_text, year_from=year_from, year_to=year_to)


@app.command()
def bindex(
    query: str,
    retmax: int = typer.Option(25, help=_RETMAX_HELP),
    days_back: int = typer.Option(180, help=_DAYS_BACK_HELP),
    full_text: bool = typer.Option(False, help=_FULL_TEXT_HELP),
    year_from: str = typer.Option("", help=_YEAR_FROM_HELP),
    year_to: str = typer.Option("", help=_YEAR_TO_HELP),
):
    """Fetch, preview, select, and index bioRxiv preprints interactively."""
    do_bindex(query, retmax, days_back, full_text, year_from=year_from, year_to=year_to)


@app.command()
def retrieve(query: str):
    """Query the local index and show retrieved chunks (no LLM)."""
    do_retrieve(query)


@app.command()
def show(pmid: str):
    """Print a paper's stored abstract/results text by PMID."""
    do_show(pmid)


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


@app.command(name="import")
def import_(path: str):
    """Index a PDF file or every PDF in a directory (auto-detected)."""
    do_import(path)


@app.command(name="import-pdf")
def import_pdf(path: str):
    """Index a single manually downloaded PDF (Results section only)."""
    do_import_pdf(path)


@app.command(name="import-dir")
def import_dir(path: str):
    """Index all PDFs in a directory (Results section only)."""
    do_import_dir(path)


@app.command(name="text-index")
def text_index():
    """Index free-form text entered interactively (prompts for title, identifier, etc.)."""
    do_text_index()


@app.command()
def model(backend_key: str = typer.Argument("", help="Backend key to switch to. Omit to list.")):
    """List available LLM backends or switch the active model."""
    do_model(backend_key)


@app.command()
def effort(
    level: str = typer.Argument("", help="Reasoning effort: low/medium/high. Omit to show."),
):
    """Show or set the LLM reasoning effort (speed vs. accuracy)."""
    do_effort(level)


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
