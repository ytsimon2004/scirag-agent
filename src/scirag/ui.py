"""Chainlit web UI for scirag-agent.

Launch via the shell:  scirag ❯ /llm-ui
Or directly:           uv run --extra ui chainlit run src/scirag/ui.py
"""

from __future__ import annotations

import chainlit as cl
from chainlit.input_widget import Select, Slider, Switch

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
from scirag.ingest.index import get_indexed_pmids
from scirag.llm.router import complete_stream
from scirag.projects import get_active_project

_LLM_AGENTS = ("synthesizer", "critic", "planner", "retriever")
_EFFORTS = ("low", "medium", "high")


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
        f"Adjust the model, reasoning effort, and retrieval params in the ⚙️ settings "
        f"panel. Ask a question about your indexed papers; type `/reset` to clear history."
    )


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
        ]
    ).send()

    await cl.Message(content=_status_text()).send()


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

    await cl.Message(content="Settings updated.\n\n" + _status_text()).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    query = message.content.strip()

    # --- Reset command ---
    if query.lower() in ("/reset", "reset"):
        cl.user_session.set("history", [])
        await cl.Message(content="Conversation history cleared.").send()
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
    await answer_msg.update()

    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})
    cl.user_session.set("history", history)
