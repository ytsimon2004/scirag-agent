"""Interactive scirag shell — launched by `scirag` with no arguments."""

from __future__ import annotations

import shlex
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# (command, args-hint, description)
_COMMANDS: list[tuple[str, str, str]] = [
    ("/search", "<query> [--retmax N]", "search PubMed, show full-text availability"),
    ("/index", "<query> [--retmax N] [--full-text]", "interactive fetch + select + index"),
    ("/retrieve", "<query>", "query local index (no LLM)"),
    ("/llm", "<question> [--reset]", "RAG answer with sources + conversation memory"),
    ("/llm-ui", "[--port N]", "open Chainlit web UI in browser"),
    ("/model", "[backend-key]", "list or switch LLM backend"),
    ("/import-pdf", "<path>", "index a single PDF (Results section only)"),
    ("/import-dir", "<path>", "index all PDFs in a directory"),
    ("/env", "[set <KEY> <val> | unset <KEY>]", "manage API keys in ~/.scirag-agent/.env"),
    ("/status", "", "show index statistics"),
    ("/remove", "[pmid ...]", "remove article(s) from the index (interactive if no args)"),
    ("/clear-db", "[--force]", "delete the active index"),
    ("/create-project", "<name> [description]", "create a new project and switch to it"),
    ("/project", "[name|--default]", "list projects or switch to one"),
    ("/delete-project", "<name> [--force]", "delete a project and its index"),
    ("/help", "", "show this help"),
    ("/clear", "", "clear the screen"),
    ("/exit", "", "exit scirag"),
]

_COMPLETER = WordCompleter([cmd for cmd, _, _ in _COMMANDS], sentence=True)


def _prompt() -> HTML:
    from scirag.projects import get_active_project

    project = get_active_project()
    if project:
        return HTML(
            f"<ansigreen><b>scirag</b></ansigreen>"
            f"<ansiwhite>[</ansiwhite><ansiyellow>{project}</ansiyellow><ansiwhite>]</ansiwhite>"
            f" <ansicyan>❯</ansicyan> "
        )
    return HTML("<ansigreen><b>scirag</b></ansigreen> <ansicyan>❯</ansicyan> ")


def _ollama_status() -> str:
    try:
        import httpx
        from scirag.config import models_cfg

        base = models_cfg()["embeddings"]["api_base"]
        httpx.get(f"{base}/api/tags", timeout=1.5).raise_for_status()
        return "[green]running[/]"
    except Exception:
        return "[red]offline[/]"


def _banner() -> None:
    from pathlib import Path
    from scirag.config import active_backend_key, models_cfg
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

    ollama = _ollama_status()

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", no_wrap=True, min_width=11)
    grid.add_column()
    grid.add_row("llm", f"[cyan]{llm_model}[/]")
    grid.add_row("embedding", f"[dim]{emb}[/]")
    grid.add_row("ollama", ollama)
    grid.add_row(
        "project",
        f"[yellow]{project}[/]" if get_active_project() else f"[dim]{project}[/]",
    )
    grid.add_row("index", f"[dim]{index_str}[/]")
    grid.add_row("directory", f"[dim]{cwd}[/]")

    console.print()
    console.print(
        Panel(
            grid,
            title="[bold cyan]scirag-agent[/]  [dim]scientific RAG · PubMed/PMC[/]",
            border_style="dim",
            padding=(0, 1),
        )
    )
    console.print("[dim]  /help for commands  ·  /exit to quit[/]")
    console.print()

    try:
        pmids = get_indexed_pmids()
        active = get_active_project()
        label = f"project [cyan]{active}[/]" if active else "global index"
        if pmids:
            console.print(f"[dim]  {label} — {len(pmids)} article(s) stored.[/]\n")
        else:
            console.print(
                f"[dim]  {label} is empty — run [/][cyan]/index <query>[/][dim] to populate.[/]\n"
            )
    except Exception:
        pass


def _handle_help() -> None:
    table = Table(box=None, padding=(0, 2), show_header=False)
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(style="white", no_wrap=True)
    table.add_column(style="dim")
    for cmd, args, desc in _COMMANDS:
        table.add_row(cmd, args, f"— {desc}")
    console.print(table)


def _parse_flags(args: list[str]) -> tuple[list[str], dict]:
    """Split positional args from --flag, --flag N, and --flag=N forms."""
    positional, flags = [], {}
    i = 0
    while i < len(args):
        if args[i].startswith("--"):
            token = args[i][2:]
            if "=" in token:  # --key=value
                key, val = token.split("=", 1)
                flags[key] = val
                i += 1
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                flags[token] = args[i + 1]  # --key value
                i += 2
            else:
                flags[token] = True  # --flag (boolean)
                i += 1
        else:
            positional.append(args[i])
            i += 1
    return positional, flags


def _dispatch(line: str) -> None:
    try:
        parts = shlex.split(line)
    except ValueError as e:
        console.print(f"[red]Parse error:[/] {e}")
        return

    if not parts:
        return

    cmd, args = parts[0].lower(), parts[1:]
    positional, flags = _parse_flags(args)
    query = " ".join(positional)

    if cmd in ("/exit", "/quit"):
        raise SystemExit(0)

    if cmd == "/help":
        _handle_help()
        return

    if cmd == "/clear":
        console.clear()
        return

    if cmd == "/env":
        from scirag.cli import do_env

        action = positional[0] if positional else ""
        key = positional[1] if len(positional) > 1 else ""
        value = " ".join(positional[2:]) if len(positional) > 2 else ""
        do_env(action, key, value)
        return

    if cmd == "/status":
        from scirag.cli import do_status

        do_status()
        return

    if cmd == "/remove":
        from scirag.cli import do_remove

        do_remove(positional)
        return

    if cmd == "/clear-db":
        from scirag.cli import do_clear_db

        do_clear_db(force="force" in flags)
        return

    if cmd == "/delete-project":
        if not positional:
            console.print("[yellow]Usage:[/] /delete-project <name> [--force]")
            return
        name = positional[0]
        from scirag.projects import delete_project, get_active_project, list_projects

        if not any(p["name"] == name for p in list_projects()):
            console.print(f"[red]Project {name!r} not found.[/]")
            return
        if not flags.get("force"):
            import questionary

            confirmed = questionary.confirm(
                f"Delete project '{name}' and all its indexed articles? This cannot be undone."
            ).ask()
            if not confirmed:
                console.print("[dim]Cancelled.[/]")
                return
        delete_project(name)
        active = get_active_project()
        console.print(f"[green]Project [cyan]{name}[/] deleted.[/]")
        if active != name:
            pass  # still on a different project
        else:
            console.print("[dim]Switched to default global index.[/]")
        return

    if cmd == "/create-project":
        if not positional:
            console.print("[yellow]Usage:[/] /create-project <name> [description]")
            return
        name = positional[0]
        desc = " ".join(positional[1:])
        from scirag.projects import create_project, set_active_project

        try:
            create_project(name, desc)
            set_active_project(name)
            console.print(f"Created project [cyan]{name}[/] and switched to it.")
        except ValueError as e:
            console.print(f"[red]Error:[/] {e}")
        return

    if cmd == "/project":
        from scirag.projects import (
            create_project,
            delete_project,
            get_active_project,
            list_projects,
            set_active_project,
        )

        if not positional and "default" not in flags:
            # List all projects
            projects = list_projects()
            active = get_active_project()
            if not projects:
                console.print(
                    "[yellow]No projects yet.[/] Use [cyan]/create-project <name>[/] to create one."
                )
            else:
                for p in projects:
                    is_active = p["name"] == active
                    marker = "●" if is_active else "○"
                    style = "bold cyan" if is_active else "dim"
                    desc = f"  [dim]{p['description']}[/]" if p.get("description") else ""
                    created = f"  [dim]created {p['created']}[/]"
                    console.print(f"[{style}]{marker} {p['name']}[/]{desc}{created}")
                if not active:
                    console.print("[dim](using default global index)[/]")
        elif "default" in flags:
            set_active_project(None)
            console.print("[dim]Switched to default global index.[/]")
        else:
            name = positional[0]
            projects = list_projects()
            if not any(p["name"] == name for p in projects):
                console.print(
                    f"[red]Project {name!r} not found.[/]  "
                    f"Use [cyan]/create-project {name}[/] to create it."
                )
                return
            set_active_project(name)
            console.print(f"Switched to project [cyan]{name}[/].")
        return

    if cmd == "/search":
        if not query:
            console.print("[yellow]Usage:[/] /search <query> [--retmax N]")
            return
        from scirag.cli import do_search

        do_search(query, retmax=int(flags.get("retmax", 15)))

    elif cmd == "/index":
        if not query:
            console.print("[yellow]Usage:[/] /index <query> [--retmax N] [--full-text]")
            return
        from scirag.cli import do_index

        do_index(
            query,
            retmax=int(flags.get("retmax", 25)),
            full_text="full-text" in flags or "full_text" in flags,
        )

    elif cmd == "/retrieve":
        if not query:
            console.print("[yellow]Usage:[/] /retrieve <query>")
            return
        from scirag.cli import do_retrieve

        do_retrieve(query)

    elif cmd == "/llm":
        if flags.get("reset"):
            from scirag.cli import do_llm

            do_llm("", reset=True)
        elif not query:
            console.print(
                "[yellow]Usage:[/] /llm <question>  [dim](or /llm --reset to clear history)[/]"
            )
        else:
            from scirag.cli import do_llm

            do_llm(query)

    elif cmd == "/llm-ui":
        from scirag.cli import do_llm_ui

        port = int(flags.get("port", 8000))
        do_llm_ui(port)

    elif cmd == "/model":
        from scirag.cli import do_model

        do_model(query)

    elif cmd == "/import-pdf":
        if not query:
            console.print("[yellow]Usage:[/] /import-pdf <path>")
            return
        from scirag.cli import do_import_pdf

        do_import_pdf(query)

    elif cmd == "/import-dir":
        if not query:
            console.print("[yellow]Usage:[/] /import-dir <path>")
            return
        from scirag.cli import do_import_dir

        do_import_dir(query)

    else:
        console.print(f"[yellow]Unknown command:[/] {cmd}   (type [cyan]/help[/])")


def run_shell() -> None:
    session: PromptSession = PromptSession(
        history=FileHistory(str(Path.home() / ".scirag_history")),
        completer=_COMPLETER,
    )

    _banner()

    while True:
        try:
            line = session.prompt(_prompt())
        except KeyboardInterrupt:
            console.print()
            continue
        except EOFError:
            console.print("[dim]Bye.[/]")
            break

        line = line.strip()
        if not line:
            continue

        if line in ("/exit", "/quit"):
            console.print("[dim]Bye.[/]")
            break

        try:
            _dispatch(line)
        except SystemExit:
            console.print("[dim]Bye.[/]")
            break
        except Exception as exc:
            console.print(f"[red]Error:[/] {exc}")
