"""Canonical RAG pipeline: entity extraction -> retrieval -> relevance gating
-> grounded-prompt assembly.

This is the single place that decides *what to send the LLM*. Every entry point
(the CLI `do_llm`, the Chainlit UI `on_message`, the MCP `ask_index`) calls
`prepare_answer()` and then owns only its own rendering and its own LLM call
(sync `complete` vs streaming `complete_stream`). Keeping the prompt-building
here is what stops the callers from drifting apart.
"""

from __future__ import annotations

from dataclasses import dataclass

from llama_index.core.schema import NodeWithScore

from scirag.agents.synthesize import SYSTEM as SYSTEM_GROUNDED
from scirag.agents.synthesize import _format_sources
from scirag.config import pipeline_cfg
from scirag.ingest.index import get_indexed_pmids
from scirag.neuro.entities import expand_query, extract_entities
from scirag.retrieval.retriever import retrieve

# Used when retrieval finds nothing relevant enough to ground the answer.
SYSTEM_GENERAL = (
    "You are scirag-agent, a scientific literature assistant. The local index "
    "had no sufficiently relevant passages for this question, so answer from "
    "your own knowledge without [PMID] citations. Be precise about methods, "
    "species, and brain regions, and make clear when a statement is general "
    "knowledge rather than grounded in the indexed sources."
)

_DEFAULT_RAG_SCORE_THRESHOLD = 0.3


@dataclass
class RagResult:
    """Everything a caller needs to render the turn and call the LLM."""

    query: str
    entities: dict[str, list[str]]
    nodes: list[NodeWithScore]  # passages actually used (empty when use_rag is False)
    use_rag: bool
    top_score: float
    messages: list[dict[str, str]]  # ready for complete() / complete_stream()


def prepare_answer(
    query: str,
    history: list[dict[str, str]] | None = None,
) -> RagResult:
    """Run extraction + retrieval + relevance gating and assemble the messages.

    When the best retrieved passage clears `retrieval.rag_score_threshold`, the
    answer is grounded: sources are embedded in the user turn and the strict
    cite-every-claim system prompt is used. Otherwise it falls back to a general
    system prompt with no sources.
    """
    cfg = pipeline_cfg()["retrieval"]
    threshold = cfg.get("rag_score_threshold", _DEFAULT_RAG_SCORE_THRESHOLD)

    entities = extract_entities(query)
    expanded = expand_query(query, entities)
    # Skip retrieval entirely on an empty index — avoids a pointless query and
    # any vector-store error when no table exists yet.
    nodes = retrieve(expanded) if get_indexed_pmids() else []

    top_score = max((n.score or 0.0 for n in nodes), default=0.0)
    use_rag = bool(nodes) and top_score >= threshold

    if use_rag:
        sources_block = _format_sources(nodes)
        user_content = (
            f"Question: {query}\n\nSources:\n{sources_block}\n\nWrite a concise, cited answer."
        )
        system = SYSTEM_GROUNDED
    else:
        nodes = []
        user_content = query
        system = SYSTEM_GENERAL

    messages = [{"role": "system", "content": system}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": user_content})

    return RagResult(
        query=query,
        entities=entities,
        nodes=nodes,
        use_rag=use_rag,
        top_score=top_score,
        messages=messages,
    )
