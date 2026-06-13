# scireg — Multi-agent RAG for scientific literature

Neuroscience-focused retrieval-augmented generation over PubMed/PMC, built with
**LlamaIndex** (indexing/retrieval) and **LiteLLM** (LLM routing). LLM backends —
open-source (Ollama: Qwen/Llama/DeepSeek) and frontier (Claude/OpenAI) — sit behind
one LiteLLM router, selectable per agent.

## Framework division of labor
- **LlamaIndex** — ingestion, chunking, embeddings, LanceDB vector store, retrievers.
- **LiteLLM** — unified LLM router across local and frontier backends.

## Environment & setup
- **uv** manages everything. Python 3.11.
- `uv sync` — install. `uv sync --extra mcp --extra eval --extra dev` for extras.
- Copy `.env.example` -> `.env`; set `NCBI_API_KEY` (raises PubMed rate limit to 10 req/s)
  and `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` only if a frontier backend is selected.
- **Ollama must be running** for local models + embeddings: `ollama serve` then
  `ollama pull qwen2.5:14b-instruct-q4_K_M && ollama pull bge-m3`.

## Hardware
- **Mac (36 GB unified)** is the primary local inference box — run Qwen2.5-14B (q4) +
  bge-m3 embeddings via Ollama. This is the default in `configs/models.yaml`.
- **Linux RTX 4060 (8 GB)** only fits 7–8B at 4-bit; best used as a remote
  embedding/reranker server. To use it, point `embeddings.api_base` (and any
  `local-*` backend `api_base`) in `models.yaml` at `http://<linux-ip>:11434`.
- To run **fully offline**, set `agents.synthesizer` and `agents.critic` to
  `local-qwen14b` in `models.yaml` (no API keys needed).

## Layout
- `src/scireg/config.py` — loads `configs/models.yaml` + `configs/pipeline.yaml` + `.env`.
- `src/scireg/llm/router.py` — `complete(agent, messages)`; the ONLY place LLMs are called.
- `src/scireg/sources/pubmed.py` — NCBI E-utilities client (`Article` dataclass).
- `src/scireg/sources/biorxiv.py` — bioRxiv source; keyword search via Europe PMC
  (the bioRxiv API has no search endpoint), direct-DOI metadata + full-text JATS via
  the bioRxiv API. Builds the same `Article` with the preprint DOI in the `pmid` slot
  and `source="biorxiv"`.
- `src/scireg/ingest/index.py` — LlamaIndex -> LanceDB (embedded, at `data/lancedb`).
- `src/scireg/retrieval/retriever.py` — hybrid dense + BM25 with RRF fusion.
- `src/scireg/neuro/entities.py` — neuro entity extraction + query expansion (extension point).
- `src/scireg/agents/synthesize.py` — cited-answer synthesis agent.
- `src/scireg/mcp_server/server.py` — exposes retrieval as MCP tools (optional extra).
- `src/scireg/cli.py` — `scireg search|index|ask`.

## Run
```
uv run scireg search  "grid cells entorhinal cortex"     # raw PubMed, no LLM
uv run scireg index   "hippocampal place cells" --retmax 30
uv run scireg bsearch "place cells remapping"            # bioRxiv preprints (last 180 days)
uv run scireg bindex  "place cells remapping" --full-text
uv run scireg ask     "How do place cells remap across environments?"
uv run python -m scireg.mcp_server.server                 # MCP server (needs --extra mcp)
```

## Conventions
- **All LLM calls go through `scireg.llm.router.complete(agent, ...)`** — never call
  `litellm`/provider SDKs directly. Add/swap models in `configs/models.yaml`, not in code.
- Agent role names in `models.yaml` (`planner`, `retriever`, `neuro_entity`,
  `synthesizer`, `critic`) are the contract between config and code.
- Synthesis must cite every claim with the `[id]` marker from source metadata —
  a PMID for PubMed records, a DOI (`10.1101/…`) for bioRxiv preprints. Both live
  in the metadata `pmid` field, which is the system-wide primary key (dedup,
  `/show`, `/remove`, citations).
- PubMed and bioRxiv are the data sources; the MCP server is how agents reach them.
  Don't bypass `sources/` with ad-hoc HTTP calls elsewhere. The bioRxiv API has no
  keyword-search endpoint, so `biorxiv.search` queries Europe PMC
  (`SRC:PPR AND PUBLISHER:"bioRxiv"`) for relevance-ranked results.

## Roadmap (multi-agent buildout)
Current pipeline is linear (`extract_entities -> retrieve -> synthesize`). Next:
LangGraph for conditional routing — critic/verifier loop that scores citation grounding
and routes back to `retrieve` on failure; supervisor/router node; query-planner
(MeSH expansion); reranker (`bge-reranker-v2-m3`); real ontology resolvers in
`neuro/` (Allen Brain Atlas, NCBI Gene, UniProt, ChEBI, MeSH).
