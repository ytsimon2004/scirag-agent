"""Chainlit web UI for scirag-agent.

Launch via the shell:  scirag ❯ /ui
Or directly:           uv run --extra ui chainlit run src/scirag/ui.py
"""

from __future__ import annotations

import os
from pathlib import Path

import chainlit as cl
from chainlit.input_widget import Select, Slider, Switch, TextInput

from scirag.agents.pipeline import prepare_answer
from scirag.config import (
    active_backend_key,
    get_effort,
    get_retrieval,
    models_cfg,
    set_agent_backend,
    set_effort,
    set_retrieval_param,
)
from scirag.cite import citation
from scirag.ingest.index import build_index, get_indexed_articles_full, get_indexed_pmids
from scirag.llm.router import complete_stream
from scirag.projects import (
    get_active_project,
    get_active_system_prompt,
    set_project_system_prompt,
)

_LLM_AGENTS = ("synthesizer", "critic", "planner", "retriever")
_EFFORTS = ("low", "medium", "high")


def _active_name() -> str | None:
    """Active project name honouring the same precedence as the index reads:
    ``SCIRAG_PROJECT`` env (set by ``scirag ui --project``) → persisted active
    project. None means the global index (which has no system prompt)."""
    env = os.environ.get("SCIRAG_PROJECT")
    return (env or None) if env is not None else get_active_project()


def _status_text() -> str:
    project = get_active_project() or "global"
    try:
        n = len(get_indexed_pmids())
        index_str = f"{n} article(s)"
    except Exception:
        index_str = "empty"
    llm_key = active_backend_key("synthesizer")
    llm_model = models_cfg()["backends"][llm_key]["model"]
    emb = models_cfg()["embeddings"]["model"]
    r = get_retrieval()
    retrieval_str = (
        f"final_k {r['final_k']} · top_k {r['top_k']} · bm25_k {r['bm25_k']} · "
        f"hybrid {'on' if r.get('hybrid') else 'off'} · rerank {'on' if r.get('rerank') else 'off'} · "
        f"threshold {r['rag_score_threshold']}"
    )
    return (
        f"**scirag-agent** · scientific RAG · PubMed/PMC\n\n"
        f"| | |\n|---|---|\n"
        f"| project | `{project}` |\n"
        f"| index | {index_str} |\n"
        f"| llm | `{llm_model}` · effort `{get_effort()}` |\n"
        f"| embedding | `{emb}` |\n"
        f"| retrieval | {retrieval_str} |\n\n"
        f"Adjust the model, reasoning effort, retrieval params, and the project's "
        f"**system prompt** in the ⚙️ settings panel. Ask a question about your indexed "
        f"papers, or:\n\n"
        f"- open the **📚 Studies** side panel to browse the papers indexed in this project\n"
        f"- **attach a PDF** (📎 button or drag-and-drop) to import a paper into this "
        f"project's index — only PDFs are accepted\n"
        f"- `/reset` — clear the conversation history"
    )


def _studies_text() -> str:
    """Markdown table of the papers indexed in the active project."""
    project = _active_name() or "global"
    try:
        arts = get_indexed_articles_full()
    except Exception:
        arts = []
    if not arts:
        return (
            f"No studies are indexed in `{project}` yet. Index papers from the shell "
            f"(`/index`, `/bindex`, `/import*`) or drag PDFs into the chat here."
        )
    arts.sort(key=lambda a: (str(a.get("year") or ""), a.get("first_author") or ""))
    rows = [f"**{len(arts)} stud{'y' if len(arts) == 1 else 'ies'} in `{project}`**", ""]
    rows.append("| citation | title | source | text | id |")
    rows.append("|---|---|---|---|---|")
    for a in arts:
        cite = citation(a)
        title = (a.get("title") or "—").replace("|", "·")[:80]
        url = a.get("url") or ""
        title_md = f"[{title}]({url})" if url else title
        rows.append(
            f"| {cite} | {title_md} | {a.get('origin', '?')} | "
            f"{a.get('text_source') or 'abstract'} | `{a.get('pmid', '?')}` |"
        )
    return "\n".join(rows)


async def _refresh_studies_sidebar() -> None:
    """Populate the docked side panel ("Studies" tab) with this project's papers.

    A persistent alternative to the `/studies` command — set on chat start and
    refreshed after an import so the panel always mirrors the index.
    """
    try:
        n = len(get_indexed_pmids())
    except Exception:
        n = 0
    await cl.ElementSidebar.set_title(f"📚 Studies · {n}")
    await cl.ElementSidebar.set_elements([cl.Text(name="studies", content=_studies_text())])


@cl.action_callback("show_studies")
async def _on_show_studies(action: cl.Action) -> None:
    """Reopen/refresh the Studies panel. Chainlit drops the sidebar's header toggle
    once you close it, so this button is the always-present way back in."""
    await _refresh_studies_sidebar()


def _collect_uploads(message: cl.Message) -> tuple[list[Path], list[str]]:
    """Split a message's attachments into importable PDFs and rejected non-PDFs.

    The UI only ingests PDFs (resolve → Results section → embed); anything else
    is returned by name so the caller can tell the user instead of silently
    dropping it. Covers both the 📎 attach button and drag-and-drop.
    """
    pdfs: list[Path] = []
    rejected: list[str] = []
    for el in message.elements or []:
        path = getattr(el, "path", None)
        if not path:
            continue
        mime = (getattr(el, "mime", "") or "").lower()
        if path.lower().endswith(".pdf") or mime == "application/pdf":
            pdfs.append(Path(path))
        else:
            rejected.append(Path(path).name)
    return pdfs, rejected


async def _ingest_pdfs(paths: list[Path]) -> None:
    """Resolve + index dropped PDFs into the active project, reporting progress.

    Reuses the same resolve→Results-section→embed path as the shell's `/import`
    (`load_pdf_as_article` + `build_index`). Unresolved PDFs are skipped and named.
    """
    from scirag.sources.pdf import load_pdf_as_article

    project = _active_name() or "global"
    async with cl.Step(name=f"Importing {len(paths)} PDF(s) → {project}", type="tool") as step:
        articles = []
        skipped: list[str] = []
        for p in paths:
            try:
                art = load_pdf_as_article(p)
            except Exception as exc:  # parse failure
                skipped.append(f"`{p.name}` — {exc}")
                continue
            if art is None:
                skipped.append(f"`{p.name}` — could not resolve to a PubMed/bioRxiv record")
            else:
                articles.append(art)
        if articles:
            build_index(articles)  # embeds via Ollama; blocking, mirrors the shell path
        step.output = f"Indexed {len(articles)} · skipped {len(skipped)}"

    # The uploaded PDF was only needed to extract + embed its text; the content
    # now lives in LanceDB, so drop the staged copy in .files/ rather than letting
    # uploads pile up under the app root.
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass

    lines = []
    if articles:
        lines.append(f"✅ Imported **{len(articles)}** paper(s) into `{project}`:")
        lines += [f"- {citation(a.metadata())} — {a.title[:80]}" for a in articles]
    if skipped:
        lines.append(f"\n⚠️ Skipped **{len(skipped)}**:")
        lines += [f"- {s}" for s in skipped]
    await cl.Message(content="\n".join(lines) or "No PDFs found in the upload.").send()
    if articles:
        await _refresh_studies_sidebar()


@cl.on_chat_start
async def on_start() -> None:
    cl.user_session.set("history", [])

    backends = list(models_cfg()["backends"].keys())
    current = active_backend_key("synthesizer")
    r = get_retrieval()

    await cl.ChatSettings(
        [
            Select(id="backend", label="LLM backend", values=backends, initial_value=current),
            Select(
                id="effort",
                label="Reasoning effort (speed vs. accuracy)",
                values=list(_EFFORTS),
                initial_value=get_effort(),
            ),
            Slider(
                id="final_k",
                label="final_k — chunks sent to the LLM",
                initial=r["final_k"],
                min=1,
                max=30,
                step=1,
            ),
            Slider(
                id="top_k",
                label="top_k — dense candidates",
                initial=r["top_k"],
                min=1,
                max=100,
                step=1,
            ),
            Slider(
                id="bm25_k",
                label="bm25_k — keyword candidates",
                initial=r["bm25_k"],
                min=1,
                max=100,
                step=1,
            ),
            Switch(
                id="hybrid", label="hybrid — dense + BM25 fusion", initial=bool(r.get("hybrid"))
            ),
            Switch(
                id="rerank",
                label="rerank — cross-encoder (needs 'rerank' extra)",
                initial=bool(r.get("rerank")),
            ),
            Slider(
                id="rag_score_threshold",
                label="rag_score_threshold — grounding gate",
                initial=r["rag_score_threshold"],
                min=0.0,
                max=1.0,
                step=0.05,
            ),
            TextInput(
                id="system_prompt",
                label="System prompt (this project; blank = built-in default)",
                initial=get_active_system_prompt(),
                multiline=True,
                placeholder="e.g. You are a careful neuroscience reviewer. Prefer primary…",
            ),
        ]
    ).send()

    await cl.Message(
        content=_status_text(),
        actions=[
            cl.Action(
                name="show_studies",
                payload={},
                label="📚 Studies",
                tooltip="Open the studies panel for this project",
            )
        ],
    ).send()
    await _refresh_studies_sidebar()


@cl.on_settings_update
async def on_settings_update(settings: dict) -> None:
    key = settings.get("backend")
    if key:
        for agent in _LLM_AGENTS:
            set_agent_backend(agent, key)

    effort = settings.get("effort")
    if effort in _EFFORTS:
        set_effort(effort)

    for p in ("final_k", "top_k", "bm25_k"):
        if settings.get(p) is not None:
            set_retrieval_param(p, int(settings[p]))
    for p in ("hybrid", "rerank"):
        if p in settings:
            set_retrieval_param(p, bool(settings[p]))
    if settings.get("rag_score_threshold") is not None:
        set_retrieval_param("rag_score_threshold", float(settings["rag_score_threshold"]))

    notes: list[str] = []
    if "system_prompt" in settings:
        new_prompt = (settings["system_prompt"] or "").strip()
        name = _active_name()
        if new_prompt != (get_active_system_prompt() or "").strip():
            if name is None:
                notes.append(
                    "⚠️ The global index has no project — system prompt not saved. "
                    "Create or switch to a project (`/create-project` in the shell) first."
                )
            else:
                set_project_system_prompt(name, new_prompt)
                notes.append(
                    f"System prompt {'cleared' if not new_prompt else 'updated'} for `{name}`."
                )

    body = "Settings updated."
    if notes:
        body += "\n\n" + "\n\n".join(notes)
    await cl.Message(content=body + "\n\n" + _status_text()).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    query = message.content.strip()

    # --- Reset command ---
    if query.lower() in ("/reset", "reset"):
        cl.user_session.set("history", [])
        await cl.Message(content="Conversation history cleared.").send()
        return

    # --- List the studies indexed in this project ---
    if query.lower() in ("/studies", "/papers"):
        await cl.Message(content=_studies_text()).send()
        return

    # --- Import any PDFs attached/dragged into the chat, then continue if a question remains ---
    pdfs, rejected = _collect_uploads(message)
    if rejected:
        names = ", ".join(f"`{n}`" for n in rejected)
        await cl.Message(
            content=(
                f"⚠️ Only **PDF** files can be imported here — skipped {len(rejected)}: {names}.\n\n"
                "Attach a paper's PDF to add it to this project's index."
            )
        ).send()
    if pdfs:
        await _ingest_pdfs(pdfs)
    # Pure upload (or an empty message): nothing left to answer.
    if not query:
        return

    history: list[dict] = cl.user_session.get("history", [])

    # --- Entity extraction + retrieval + relevance gating (shared pipeline) ---
    async with cl.Step(name="Searching index", type="retrieval") as step:
        result = prepare_answer(query, history)
        step.output = f"Retrieved {len(result.nodes)} chunk(s)"

    # --- Retrieved sources as one collapsible step (click the row to expand) ---
    if result.nodes:
        seen: set[str] = set()
        blocks: list[str] = []
        for n in result.nodes:
            md = n.node.metadata
            pmid = md.get("pmid", "?")
            if pmid in seen:
                continue
            seen.add(pmid)
            title = md.get("title", "")
            url = md.get("url", "")
            src = md.get("text_source", "abstract")
            cite = citation(md)  # 'Powell et al., 2020' (includes the year)
            snippet = n.node.get_content()[:400].replace("\n", " ")
            blocks.append(f"**{cite}** · [{title[:75]}]({url}) · `{src}`\n\n> {snippet}…")
        label = f"Sources · {len(seen)} paper(s), {len(result.nodes)} chunk(s)"
        async with cl.Step(name=label, type="tool") as src_step:
            src_step.output = "\n\n---\n\n".join(blocks)

    # --- Stream answer (messages already assembled by the shared pipeline) ---
    answer_msg = cl.Message(content="")
    await answer_msg.send()

    answer = ""
    async for token in complete_stream("synthesizer", result.messages):
        answer += token
        await answer_msg.stream_token(token)
    # Keep a reopen control near the latest turn — the welcome button scrolls away.
    answer_msg.actions = [
        cl.Action(
            name="show_studies", payload={}, label="📚 Studies", tooltip="Open the studies panel"
        )
    ]
    await answer_msg.update()

    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})
    cl.user_session.set("history", history)
