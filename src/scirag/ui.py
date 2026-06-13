"""Chainlit web UI for scirag-agent.

Launch via the shell:  scirag ❯ /llm-ui
Or directly:           uv run --extra ui chainlit run src/scirag/ui.py
"""

from __future__ import annotations

import chainlit as cl
from chainlit.input_widget import Select

from scirag.agents.pipeline import prepare_answer
from scirag.config import active_backend_key, models_cfg, set_agent_backend
from scirag.ingest.index import get_indexed_pmids
from scirag.llm.router import complete_stream
from scirag.projects import get_active_project

_LLM_AGENTS = ("synthesizer", "critic", "planner", "retriever")


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
    return (
        f"**scirag-agent** · scientific RAG · PubMed/PMC\n\n"
        f"| | |\n|---|---|\n"
        f"| project | `{project}` |\n"
        f"| index | {index_str} |\n"
        f"| llm | `{llm_model}` |\n"
        f"| embedding | `{emb}` |\n\n"
        f"Ask a question about your indexed papers. "
        f"Type `/reset` to clear conversation history."
    )


@cl.on_chat_start
async def on_start() -> None:
    cl.user_session.set("history", [])

    backends = list(models_cfg()["backends"].keys())
    current = active_backend_key("synthesizer")

    await cl.ChatSettings(
        [
            Select(
                id="backend",
                label="LLM backend",
                values=backends,
                initial_value=current,
            )
        ]
    ).send()

    await cl.Message(content=_status_text()).send()


@cl.on_settings_update
async def on_settings_update(settings: dict) -> None:
    key = settings.get("backend", "")
    if not key:
        return
    for agent in _LLM_AGENTS:
        set_agent_backend(agent, key)
    model = models_cfg()["backends"][key]["model"]
    await cl.Message(content=f"Model switched to **{key}** (`{model}`)").send()


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
            year = md.get("year", "")
            url = md.get("url", "")
            src = md.get("text_source", "abstract")
            snippet = n.node.get_content()[:400].replace("\n", " ")
            blocks.append(
                f"**[{pmid}]** [{title[:75]}]({url}) *({year})* · `{src}`\n\n> {snippet}…"
            )
        label = f"Sources · {len(seen)} paper(s), {len(result.nodes)} chunk(s)"
        async with cl.Step(name=label, type="tool") as src_step:
            src_step.output = "\n\n---\n\n".join(blocks)

    # --- Stream answer (messages already assembled by the shared pipeline) ---
    answer_msg = cl.Message(content="")
    await answer_msg.send()

    answer = ""
    async for token in complete_stream("synthesizer", result.messages, max_tokens=1200):
        answer += token
        await answer_msg.stream_token(token)
    await answer_msg.update()

    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})
    cl.user_session.set("history", history)
