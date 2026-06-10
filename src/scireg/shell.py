"""Interactive scireg shell — launched by `scireg` with no arguments."""
from __future__ import annotations

import shlex
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.table import Table

console = Console()

_COMMANDS: dict[str, str] = {
    "/search":     "<query> [--retmax N]              — search PubMed, show full-text availability",
    "/index":      "<query> [--retmax N] [--full-text] — interactive fetch + select + index",
    "/retrieve":   "<query>                            — query local index (no LLM)",
    "/ask":        "<query>                            — full RAG pipeline with cited answer",
    "/import-pdf": "<path>                             — index a single PDF (Results section only)",
    "/import-dir": "<path>                             — index all PDFs in a directory",
    "/status":     "                                   — show index statistics",
    "/help":       "                                   — show this help",
    "/clear":      "                                   — clear the screen",
    "/exit":       "                                   — exit scireg",
}

_COMPLETER = WordCompleter(list(_COMMANDS), sentence=True)


def _prompt() -> HTML:
    return HTML("<ansigreen><b>scireg</b></ansigreen> <ansicyan>❯</ansicyan> ")


def _banner() -> None:
    console.print()
    console.print("[bold green]scireg[/] [dim]— multi-agent RAG for scientific literature[/]")
    console.print("[dim]Type [/][cyan]/help[/][dim] for commands, [/][cyan]/exit[/][dim] to quit.[/]")
    console.print()
    # Show quick index status inline
    try:
        from scireg.ingest.index import get_indexed_pmids
        pmids = get_indexed_pmids()
        if pmids:
            console.print(f"[dim]Index ready — {len(pmids)} article(s) stored.[/]\n")
        else:
            console.print("[dim]Index empty — run [/][cyan]/index <query>[/][dim] to populate.[/]\n")
    except Exception:
        pass


def _handle_help() -> None:
    table = Table(box=None, padding=(0, 2), show_header=False)
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(style="dim")
    for cmd, desc in _COMMANDS.items():
        table.add_row(cmd, desc)
    console.print(table)


def _parse_flags(args: list[str]) -> tuple[list[str], dict]:
    """Split positional args from --flag / --flag N pairs."""
    positional, flags = [], {}
    i = 0
    while i < len(args):
        if args[i].startswith("--"):
            key = args[i][2:]
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                flags[key] = args[i + 1]
                i += 2
            else:
                flags[key] = True
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

    if cmd in ("/exit", "/quit"):
        raise SystemExit(0)

    if cmd == "/help":
        _handle_help()
        return

    if cmd == "/clear":
        console.clear()
        return

    if cmd == "/status":
        from scireg.cli import do_status
        do_status()
        return

    positional, flags = _parse_flags(args)
    query = " ".join(positional)

    if cmd == "/search":
        if not query:
            console.print("[yellow]Usage:[/] /search <query> [--retmax N]")
            return
        from scireg.cli import do_search
        do_search(query, retmax=int(flags.get("retmax", 15)))

    elif cmd == "/index":
        if not query:
            console.print("[yellow]Usage:[/] /index <query> [--retmax N] [--full-text]")
            return
        from scireg.cli import do_index
        do_index(
            query,
            retmax=int(flags.get("retmax", 25)),
            full_text="full-text" in flags or "full_text" in flags,
        )

    elif cmd == "/retrieve":
        if not query:
            console.print("[yellow]Usage:[/] /retrieve <query>")
            return
        from scireg.cli import do_retrieve
        do_retrieve(query)

    elif cmd == "/ask":
        if not query:
            console.print("[yellow]Usage:[/] /ask <query>")
            return
        from scireg.cli import do_ask
        do_ask(query)

    elif cmd == "/import-pdf":
        if not query:
            console.print("[yellow]Usage:[/] /import-pdf <path>")
            return
        from scireg.cli import do_import_pdf
        do_import_pdf(query)

    elif cmd == "/import-dir":
        if not query:
            console.print("[yellow]Usage:[/] /import-dir <path>")
            return
        from scireg.cli import do_import_dir
        do_import_dir(query)

    else:
        console.print(f"[yellow]Unknown command:[/] {cmd}   (type [cyan]/help[/])")


def run_shell() -> None:
    session: PromptSession = PromptSession(
        history=FileHistory(str(Path.home() / ".scireg_history")),
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
