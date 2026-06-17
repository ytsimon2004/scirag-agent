# Architecture

scirag splits responsibilities cleanly across two frameworks and a handful of
focused modules.

## Framework division of labor

| Framework | Owns |
|-----------|------|
| **LlamaIndex** | ingestion, chunking, embeddings, the LanceDB vector store, retrievers |
| **LiteLLM** | a unified LLM router across local (Ollama) and frontier (Claude/OpenAI) backends |

## The pipeline

The current pipeline is linear and lives in {py:mod}`scirag.agents.pipeline`:

```text
extract entities  →  retrieve  →  relevance gating  →  grounded-prompt assembly
```

`prepare_answer()` is the single place that decides *what to send the LLM*. Every
entry point — the CLI, the Chainlit UI, and the MCP server — calls it and then owns
only its own rendering and its own LLM call (sync `complete` vs streaming
`complete_stream`). Keeping prompt-building here stops the callers from drifting apart.

## Retrieval

{py:func}`scirag.retrieval.retriever.retrieve` runs:

```text
dense (bge-m3)  +  BM25   →   RRF fusion   →   optional cross-encoder rerank   →   top final_k
```

The cross-encoder (`bge-reranker-v2-m3`, via sentence-transformers) only re-scores
candidates — it is lazy-loaded and cached on first use, and reordering preserves each
node's cosine score so the relevance gate still applies. See
[Configuration](configuration.md) for the knobs.

## LLM routing

**Every** LLM call goes through {py:func}`scirag.llm.router.complete` — never call
`litellm` or a provider SDK directly. Open-source and frontier models sit behind the
same call; which one an agent uses is decided in `configs/models.yaml`. The router
also maps reasoning effort to each backend's native control.

## Sources

All external data is reached through `scirag.sources` — don't make ad-hoc HTTP calls
elsewhere. Each source builds the same `Article` dataclass, keyed by a system-wide
**primary key** (PMID for PubMed, the `10.1101/…` DOI for bioRxiv) used for dedup,
`show`/`remove`, and traceability.

| Module | Source | Notes |
|--------|--------|-------|
| {py:mod}`scirag.sources.pubmed` | PubMed | keyword esearch + Europe PMC semantic ranking; full text via PMC / Unpaywall |
| {py:mod}`scirag.sources.biorxiv` | bioRxiv | Europe PMC search; full text from JATS XML (browser-impersonated fetch) |
| {py:mod}`scirag.sources.pdf` | local PDFs | resolve to a source record, isolate the Results section |
| {py:mod}`scirag.sources.mendeley` | Mendeley | offline SQLite + extracted PDF text |
| {py:mod}`scirag.sources.zotero` | Zotero | offline `zotero.sqlite` + full-text cache |

## Projects

Each project is an isolated LanceDB index (see {py:mod}`scirag.projects`). Index
access routes through `get_active_db_uri()`, resolved as:

```text
using_project() scope  →  SCIRAG_PROJECT env  →  persisted .active_project
```

The active project is shared across shell and CLI. Each project also carries an
optional `system_prompt`, resolved with the same precedence and appended to the
synthesis system prompt.

## Citations

Synthesis cites every claim with a human-readable **author-year** marker (e.g.
`(Powell et al., 2020)`), built by {py:func}`scirag.cite.citation` from source
metadata. The PMID/DOI remains the primary key in the metadata and is shown in each
source block (`[id: …]`) for traceability — it's just no longer the marker the model
emits inline.

## Roadmap

The pipeline is linear today. Planned: wire the `planner` / `retriever` / `critic`
agents; LangGraph for conditional routing (a critic/verifier loop that scores citation
grounding and routes back to retrieve on failure); a supervisor/router node; a
query-planner with MeSH expansion; and real ontology resolvers (Allen Brain Atlas,
NCBI Gene, UniProt, ChEBI, MeSH). *Done:* hybrid dense+BM25 retrieval with RRF, and
cross-encoder reranking.
```
