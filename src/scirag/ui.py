"""Chainlit web UI for scirag-agent.

Launch via the shell:  scirag ❯ /llm-ui
Or directly:           uv run --extra ui chainlit run src/scirag/ui.py
"""

from __future__ import annotations

import chainlit as cl
from chainlit.input_widget import Select

from scirag.agents.synthesize import _format_sources
from scirag.config import active_backend_key, models_cfg, set_agent_backend
from scirag.ingest.index import get_indexed_pmids
from scirag.llm.router import complete_stream
from scirag.neuro.entities import expand_query, extract_entities
from scirag.projects import get_active_project
from scirag.retrieval.retriever import retrieve

_LLM_AGENTS = ("synthesizer", "critic", "neuro_entity", "planner", "retriever")

_SYSTEM = (
    "You are scirag-agent, a scientific literature assistant. "
    "When the user asks a research question and relevant sources are provided, "
    "answer from those sources and cite every claim with its [PMID] marker. "
    "For general questions or conversation that isn't covered by the sources, "
    "answer from your own knowledge without citations. "
    "Be precise about methods, species, and brain regions when discussing science."
)


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

    # --- Retrieve (always attempt; empty index is fine) ---
    async with cl.Step(name="Searching index", type="retrieval") as step:
        ents = extract_entities(query)
        expanded = expand_query(query, ents)
        nonempty = {k: v for k, v in ents.items() if v}
        if nonempty:
            step.output = f"Entities: {nonempty}"
        nodes = retrieve(expanded) if get_indexed_pmids() else []

    # --- Show retrieved sources as an expanded step + sidebar elements ---
    seen: set[str] = set()
    elements: list[cl.Text] = []
    retrieve_lines: list[str] = []

    for n in nodes:
        md = n.node.metadata
        pmid = md.get("pmid", "?")
        title = md.get("title", "")
        year = md.get("year", "")
        url = md.get("url", "")
        src = md.get("text_source", "abstract")
        snippet = n.node.get_content()[:200].replace("\n", " ")

        # Inline retrieve display (one entry per paper)
        if pmid not in seen:
            retrieve_lines.append(
                f"**[{pmid}]** [{title[:65]}]({url}) *({year})* · `{src}`  \n> {snippet}…"
            )

        seen.add(pmid)

        # Sidebar element with full snippet
        elements.append(
            cl.Text(
                name=f"[{pmid}] {title[:55]}",
                content=(
                    f"**{title}** ({year})  \n"
                    f"Source: **{src}**  \n"
                    f"[PubMed ↗]({url})\n\n"
                    f"{n.node.get_content()[:500]}…"
                ),
                display="side",
            )
        )

    if retrieve_lines:
        await cl.Message(
            content="**Retrieved sources**\n\n" + "\n\n".join(retrieve_lines),
        ).send()

    # --- Build messages: include sources only when retrieval found something ---
    if nodes:
        sources_block = _format_sources(nodes)
        user_content = f"Question: {query}\n\nSources:\n{sources_block}\n\nAnswer concisely, citing [PMID] markers where relevant."
    else:
        user_content = query

    messages = [{"role": "system", "content": _SYSTEM}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_content})

    # --- Stream answer ---
    answer_msg = cl.Message(content="", elements=elements)
    await answer_msg.send()

    answer = ""
    async for token in complete_stream("synthesizer", messages, max_tokens=1200):
        answer += token
        await answer_msg.stream_token(token)
    await answer_msg.update()

    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})
    cl.user_session.set("history", history)
