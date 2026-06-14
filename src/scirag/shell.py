"""Interactive scirag shell — launched by `scirag` with no arguments."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import os

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion, PathCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML, FormattedText
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.table import Table

console = Console()

# (command, args-hint, description)
_COMMANDS: list[tuple[str, str, str]] = [
    (
        "/index",
        "<query> [--retmax N] [--full-text] [--year-from YYYY] [--year-to YYYY]",
        "fetch + select + index PubMed articles",
    ),
    (
        "/bindex",
        "<query> [--retmax N] [--days-back N] [--full-text] [--year-from YYYY] [--year-to YYYY]",
        "fetch + select + index bioRxiv preprints",
    ),
    ("/retrieve", "<query>", "query local index (no LLM)"),
    ("/show", "<pmid>", "print a paper's stored abstract/results text"),
    ("/llm", "[<question>] [--reset]", "RAG answer; bare /llm = sticky conversation mode"),
    ("/llm-ui", "[--port N]", "open Chainlit web UI in browser (click-to-expand sources)"),
    ("/model", "[backend-key]", "list or switch LLM backend"),
    ("/effort", "[low|medium|high]", "set LLM reasoning effort (speed vs. accuracy)"),
    ("/rag", "[<param> <value>]", "tune retrieval params (final_k, top_k, …); no args = picker"),
    ("/import", "<path>", "index a PDF file, or every PDF in a directory"),
    ("/text", "", "index free-form text (prompts for title, identifier, origin, year, author)"),
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


def _build_flag_help() -> dict[str, str]:
    """Per-flag help (for the completion meta + bottom toolbar), read straight from
    the Typer command definitions in scirag.cli — the same source as
    `scirag <cmd> --help`, so the two never drift. `/project --default` is a
    shell-only flag with no Typer command, so it's supplemented here.
    """
    help_map = {"--default": "switch to the default global index"}
    try:
        import typer

        from scirag.cli import app

        for command in typer.main.get_command(app).commands.values():
            for param in command.params:
                text = getattr(param, "help", None)
                if not text:
                    continue
                for opt in getattr(param, "opts", []):
                    if opt.startswith("--"):
                        help_map.setdefault(opt, text)
    except Exception:
        pass
    return help_map


# Per-flag help, shown for the flag being typed (completion meta + bottom toolbar).
_FLAG_HELP: dict[str, str] = _build_flag_help()


def _is_hidden(path: str) -> bool:
    """True for dotfiles/dotdirs (e.g. .git), which path completion should skip."""
    return os.path.basename(path.rstrip(os.sep)).startswith(".")


def _import_path(path: str) -> bool:
    """Completion filter for /import: non-hidden directories (to navigate into) and
    PDF files. Lets the user pick either a single PDF or a folder of PDFs."""
    return not _is_hidden(path) and (os.path.isdir(path) or path.lower().endswith(".pdf"))


def _visible_dir(path: str) -> bool:
    """Completion filter for directory-only completion: skip hidden dirs."""
    return not _is_hidden(path)


class _ShellCompleter(Completer):
    """Complete command names (with their args + description, like /help), the
    flags of the command being typed, and filesystem paths for /import-*."""

    def __init__(self) -> None:
        self._import_paths = PathCompleter(expanduser=True, file_filter=_import_path)
        self._dir_paths = PathCompleter(
            expanduser=True, only_directories=True, file_filter=_visible_dir
        )
        # /import takes either; /import-pdf and /import-dir remain as aliases.
        self._path_completers = {
            "/import": self._import_paths,
            "/import-pdf": self._import_paths,
            "/import-dir": self._dir_paths,
        }

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        stripped = text.lstrip()

        # Filesystem-path completion for the /import-* commands.
        for cmd, completer in self._path_completers.items():
            prefix = cmd + " "
            if stripped.startswith(prefix):
                arg = stripped[len(prefix) :]
                sub = Document(arg, cursor_position=len(arg))
                yield from completer.get_completions(sub, complete_event)
                return

        # Still typing the command token: list matching commands with args + help.
        if " " not in stripped:
            if not stripped.startswith("/"):
                return
            for cmd, args, desc in _COMMANDS:
                if cmd.startswith(stripped):
                    yield Completion(
                        cmd,
                        start_position=-len(stripped),
                        display=f"{cmd}  {args}".rstrip(),
                        display_meta=desc,
                    )
            return

        # Past the command: offer that command's flags when typing a "-…" token.
        cmd0 = stripped.split(maxsplit=1)[0]
        spec = next(((a, d) for c, a, d in _COMMANDS if c == cmd0), None)
        if spec is None:
            return
        args_hint, desc = spec
        word = document.get_word_before_cursor(WORD=True)
        if not word.startswith("-"):
            return
        for flag in re.findall(r"--[\w-]+", args_hint):
            if flag.startswith(word):
                yield Completion(
                    flag,
                    start_position=-len(word),
                    display=flag,
                    display_meta=_FLAG_HELP.get(flag, desc),
                )


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
    from scirag.cli import do_llm

    console.print(
        "[dim]LLM mode — type questions directly.  "
        "[/][cyan]/exit[/][dim] to return to the shell · [/][cyan]/reset[/][dim] to clear history.[/]"
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

    if cmd == "/index":
        if not query:
            console.print(
                "[yellow]Usage:[/] /index <query> [--retmax N] [--full-text] [--year-from YYYY] [--year-to YYYY]"
            )
            return
        from scirag.cli import do_index

        do_index(
            query,
            retmax=int(flags.get("retmax", 25)),
            full_text="full-text" in flags or "full_text" in flags,
            year_from=flags.get("year-from", flags.get("year_from", "")),
            year_to=flags.get("year-to", flags.get("year_to", "")),
        )

    elif cmd == "/bindex":
        if not query:
            console.print(
                "[yellow]Usage:[/] /bindex <query> [--retmax N] [--days-back N] [--full-text] [--year-from YYYY] [--year-to YYYY]"
            )
            return
        from scirag.cli import do_bindex

        do_bindex(
            query,
            retmax=int(flags.get("retmax", 25)),
            days_back=int(flags.get("days-back", flags.get("days_back", 180))),
            full_text="full-text" in flags or "full_text" in flags,
            year_from=flags.get("year-from", flags.get("year_from", "")),
            year_to=flags.get("year-to", flags.get("year_to", "")),
        )

    elif cmd == "/retrieve":
        if not query:
            console.print("[yellow]Usage:[/] /retrieve <query>")
            return
        from scirag.cli import do_retrieve

        do_retrieve(query)

    elif cmd == "/show":
        if not query:
            console.print("[yellow]Usage:[/] /show <pmid>")
            return
        from scirag.cli import do_show

        do_show(query)

    elif cmd == "/llm":
        if flags.get("reset"):
            from scirag.cli import do_llm

            do_llm("", reset=True)
        elif not query:
            _chat_mode(session)  # bare /llm → sticky conversation window
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

    elif cmd == "/effort":
        from scirag.cli import do_effort

        do_effort(query)

    elif cmd == "/rag":
        from scirag.cli import do_rag

        do_rag(query)

    elif cmd == "/import":
        if not query:
            console.print("[yellow]Usage:[/] /import <path>  (a PDF file or a directory of PDFs)")
            return
        from scirag.cli import do_import

        do_import(query)

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

    elif cmd == "/text":
        from scirag.cli import do_text_index

        do_text_index()

    else:
        console.print(f"[yellow]Unknown command:[/] {cmd}   (type [cyan]/help[/])")


def _import_dir_arg(text: str) -> str | None:
    """If `text` is an /import-* command whose path argument is an existing
    directory, return that expanded directory path; otherwise None."""
    stripped = text.lstrip()
    for cmd in ("/import", "/import-pdf", "/import-dir"):
        prefix = cmd + " "
        if stripped.startswith(prefix):
            arg = stripped[len(prefix) :]
            if not arg:
                return None
            expanded = os.path.expanduser(arg)
            return expanded if os.path.isdir(expanded) else None
    return None


def _bottom_toolbar() -> FormattedText:
    """Persistent usage hint shown below the prompt, updated as the user types.

    Unlike the completion popup, this never disappears — so multi-arg commands
    like `/env set <KEY> <val>` keep showing their usage past the command token.
    """
    from prompt_toolkit.application import get_app

    doc = get_app().current_buffer.document
    line = doc.text.lstrip()
    token = line.split(maxsplit=1)[0] if line else ""

    spec = next(((c, a, d) for c, a, d in _COMMANDS if c == token), None)

    # If the cursor is on a "--…" token of a known command, show that flag's help.
    word = doc.get_word_before_cursor(WORD=True)
    if spec is not None and word.startswith("-"):
        cmd, args, desc = spec
        matches = [f for f in re.findall(r"--[\w-]+", args) if f.startswith(word)]
        if len(matches) == 1:
            flag = matches[0]
            return FormattedText(
                [("bold", flag), ("fg:#888888", f"   — {_FLAG_HELP.get(flag, desc)}")]
            )
        if len(matches) > 1:
            return FormattedText([("fg:#888888", "flags:  " + "   ".join(matches))])

    if spec is not None:
        cmd, args, desc = spec
        return FormattedText(
            [("bold", cmd), ("", "  "), ("fg:ansicyan", args), ("fg:#888888", f"   — {desc}")]
        )

    if token.startswith("/"):
        matches = [c for c, _, _ in _COMMANDS if c.startswith(token)]
        if matches:
            return FormattedText([("fg:#888888", "matches:  " + "   ".join(matches[:10]))])
        return FormattedText([("fg:#888888", f"unknown command {token} — /help for the list")])

    return FormattedText([("fg:#888888", "type a command — Tab to complete · /help for the list")])


def _build_key_bindings():
    """Shell-wide key bindings.

    - Right arrow descends into a completed directory for the /import-* commands
      (appends the separator and reopens completion); otherwise moves the cursor.
    """
    from prompt_toolkit.key_binding import KeyBindings

    from prompt_toolkit.filters import completion_is_selected

    kb = KeyBindings()

    @kb.add("right")
    def _descend_or_move(event) -> None:
        buf = event.current_buffer
        if buf.document.is_cursor_at_the_end and _import_dir_arg(buf.text) is not None:
            if not buf.text.endswith(os.sep):
                buf.insert_text(os.sep)
            buf.start_completion(select_first=False)
            return
        buf.cursor_right()

    @kb.add("enter", filter=completion_is_selected)
    def _enter_accepts_completion(event) -> None:
        """Accept the highlighted completion without submitting the line."""
        buf = event.current_buffer
        buf.apply_completion(buf.complete_state.current_completion)
        buf.cancel_completion()

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
            line = session.prompt(_prompt(), bottom_toolbar=_bottom_toolbar)
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
