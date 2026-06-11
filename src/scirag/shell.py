"""Interactive scirag shell — launched by `scirag` with no arguments."""

from __future__ import annotations

import shlex
from pathlib import Path

import os

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, PathCompleter, WordCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.table import Table

console = Console()

# (command, args-hint, description)
_COMMANDS: list[tuple[str, str, str]] = [
    ("/search", "<query> [--retmax N]", "search PubMed, show full-text availability"),
    ("/index", "<query> [--retmax N] [--full-text]", "interactive fetch + select + index"),
    ("/retrieve", "<query>", "query local index (no LLM)"),
    ("/llm", "[<question>] [--reset]", "RAG answer; bare /llm = sticky conversation mode"),
    ("/sources", "[on|off]", "expand last answer's sources, or set default (Ctrl+O toggles)"),
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


def _pdf_or_dir(path: str) -> bool:
    """Show directories (to navigate) and PDF files for /import-pdf completion."""
    return os.path.isdir(path) or path.lower().endswith(".pdf")


class _ShellCompleter(Completer):
    """Complete command names, and filesystem paths for the /import-* commands."""

    def __init__(self) -> None:
        self._commands = WordCompleter([cmd for cmd, _, _ in _COMMANDS], sentence=True)
        self._pdf_paths = PathCompleter(expanduser=True, file_filter=_pdf_or_dir)
        self._dir_paths = PathCompleter(expanduser=True, only_directories=True)
        self._path_completers = {
            "/import-pdf": self._pdf_paths,
            "/import-dir": self._dir_paths,
        }

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        stripped = text.lstrip()
        for cmd, completer in self._path_completers.items():
            prefix = cmd + " "
            if stripped.startswith(prefix):
                arg = stripped[len(prefix) :]
                sub = Document(arg, cursor_position=len(arg))
                yield from completer.get_completions(sub, complete_event)
                return
        yield from self._commands.get_completions(document, complete_event)


_COMPLETER = _ShellCompleter()


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


def _chat_prompt() -> HTML:
    from scirag.projects import get_active_project

    project = get_active_project()
    proj = (
        f"<ansiwhite>[</ansiwhite><ansiyellow>{project}</ansiyellow><ansiwhite>]</ansiwhite>"
        if project
        else ""
    )
    return HTML(
        f"<ansigreen><b>scirag</b></ansigreen>{proj} "
        f"<ansimagenta><b>LLM mode</b></ansimagenta> <ansicyan>❯</ansicyan> "
    )


def _chat_mode(session: PromptSession) -> None:
    """Sticky conversation window: every line typed is sent to /llm.

    Conversation history is kept by do_llm across turns. Leave with /back
    (returns to the command prompt) or /exit (quits scirag); /reset clears
    the running conversation.
    """
    from scirag.cli import do_llm, do_sources

    console.print(
        "[dim]LLM mode — type questions directly.  "
        "[/][cyan]/exit[/][dim] to return to the shell · [/][cyan]/reset[/][dim] to clear history · "
        "[/][cyan]/sources[/][dim] to expand sources.[/]"
    )
    while True:
        try:
            line = session.prompt(_chat_prompt(), completer=None)
        except KeyboardInterrupt:
            console.print()
            continue
        except EOFError:
            break

        line = line.strip()
        if not line:
            continue
        low = line.lower()
        if low in ("/exit", "/quit", "/q", "/back", "/b"):
            break  # leave LLM mode; /exit again at the shell quits scirag
        if low in ("/reset", "reset"):
            do_llm("", reset=True)
            continue
        if low == "/sources" or low.startswith("/sources "):
            do_sources(line[len("/sources") :].strip())
            continue
        if line.startswith("/"):
            # Keep LLM mode focused: foreign commands aren't run here.
            console.print(
                "[yellow]Not available in LLM mode.[/]  "
                "Type [cyan]/exit[/] to return to the shell, then run the command."
            )
            continue
        do_llm(line)

    console.print("[dim]Left LLM mode.[/]")


def _banner() -> None:
    from scirag.cli import print_system_info
    from scirag.ingest.index import get_indexed_pmids
    from scirag.projects import get_active_project

    console.print()
    print_system_info()
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


def _dispatch(line: str, session: PromptSession) -> None:
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
            _chat_mode(session)  # bare /llm → sticky conversation window
        else:
            from scirag.cli import do_llm

            do_llm(query)

    elif cmd == "/sources":
        from scirag.cli import do_sources

        do_sources(query)

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


def _import_dir_arg(text: str) -> str | None:
    """If `text` is an /import-* command whose path argument is an existing
    directory, return that expanded directory path; otherwise None."""
    stripped = text.lstrip()
    for cmd in ("/import-pdf", "/import-dir"):
        prefix = cmd + " "
        if stripped.startswith(prefix):
            arg = stripped[len(prefix) :]
            if not arg:
                return None
            expanded = os.path.expanduser(arg)
            return expanded if os.path.isdir(expanded) else None
    return None


def _build_key_bindings():
    """Shell-wide key bindings.

    - Ctrl+O toggles whether /llm shows source passages.
    - Right arrow descends into a completed directory for the /import-* commands
      (appends the separator and reopens completion); otherwise moves the cursor.
    """
    from prompt_toolkit.application import run_in_terminal
    from prompt_toolkit.key_binding import KeyBindings

    kb = KeyBindings()

    @kb.add("c-o")
    def _toggle_sources(event) -> None:
        from scirag.cli import toggle_show_sources

        shown = toggle_show_sources()
        state = "shown" if shown else "collapsed"
        run_in_terminal(lambda: console.print(f"[dim]Sources {state} (Ctrl+O).[/]"))

    @kb.add("right")
    def _descend_or_move(event) -> None:
        buf = event.current_buffer
        if buf.document.is_cursor_at_the_end and _import_dir_arg(buf.text) is not None:
            if not buf.text.endswith(os.sep):
                buf.insert_text(os.sep)
            buf.start_completion(select_first=False)
            return
        buf.cursor_right()

    return kb


def run_shell() -> None:
    session: PromptSession = PromptSession(
        history=FileHistory(str(Path.home() / ".scirag_history")),
        completer=_COMPLETER,
        key_bindings=_build_key_bindings(),
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
            _dispatch(line, session)
        except SystemExit:
            console.print("[dim]Bye.[/]")
            break
        except Exception as exc:
            console.print(f"[red]Error:[/] {exc}")
