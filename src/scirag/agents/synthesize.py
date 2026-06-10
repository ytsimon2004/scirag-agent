"""Synthesis agent: turns retrieved passages into a grounded, cited answer.
Citations use [PMID] markers tied to the source metadata.
"""
from __future__ import annotations

from llama_index.core.schema import NodeWithScore

from scirag.llm.router import complete

SYSTEM = (
    "You are a neuroscience literature assistant. Answer ONLY from the provided "
    "sources. Cite every claim with its [PMID] marker. If the sources do not "
    "support an answer, say so explicitly. Be precise about methods, species, "
    "and brain regions."
)


def _format_sources(nodes: list[NodeWithScore]) -> str:
    blocks = []
    for n in nodes:
        md = n.node.metadata
        pmid = md.get("pmid", "?")
        title = md.get("title", "")
        blocks.append(f"[{pmid}] {title} ({md.get('year', 'n.d.')})\n{n.node.get_content()}")
    return "\n\n---\n\n".join(blocks)


def synthesize(query: str, nodes: list[NodeWithScore]) -> str:
    sources = _format_sources(nodes)
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"Question: {query}\n\nSources:\n{sources}\n\n"
            "Write a concise, cited answer.",
        },
    ]
    return complete("synthesizer", messages, max_tokens=1200)
