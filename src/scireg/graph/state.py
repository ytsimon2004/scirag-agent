"""LangGraph orchestration for the RAG pipeline.

First-cut graph (linear, with the neuro hook wired in):

    extract_entities -> retrieve -> synthesize

This is the skeleton the multi-agent system grows from: add a supervisor router
in front, and a critic/verifier loop after `synthesize` that can route back to
`retrieve` on low citation grounding.
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from llama_index.core.schema import NodeWithScore

from scireg.agents.synthesize import synthesize
from scireg.neuro.entities import expand_query, extract_entities
from scireg.retrieval.retriever import retrieve


class State(TypedDict, total=False):
    query: str
    entities: dict[str, list[str]]
    expanded_query: str
    nodes: Annotated[list[NodeWithScore], "retrieved passages"]
    answer: str


def _entities_node(state: State) -> State:
    ents = extract_entities(state["query"])
    return {"entities": ents, "expanded_query": expand_query(state["query"], ents)}


def _retrieve_node(state: State) -> State:
    return {"nodes": retrieve(state.get("expanded_query") or state["query"])}


def _synthesize_node(state: State) -> State:
    return {"answer": synthesize(state["query"], state["nodes"])}


def build_graph():
    g = StateGraph(State)
    g.add_node("extract_entities", _entities_node)
    g.add_node("retrieve", _retrieve_node)
    g.add_node("synthesize", _synthesize_node)

    g.add_edge(START, "extract_entities")
    g.add_edge("extract_entities", "retrieve")
    g.add_edge("retrieve", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()
