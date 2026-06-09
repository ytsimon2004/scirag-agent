# scireg — Multi-agent RAG for scientific literature

Neuroscience-focused retrieval-augmented generation over PubMed/PMC, built with
**LlamaIndex** (indexing/retrieval), **LangChain** (LLM/tool adapters), and
**LangGraph** (agent orchestration). LLM backends — open-source (Ollama:
Qwen/Llama/DeepSeek) and frontier (Claude/OpenAI) — sit behind one LiteLLM
router, selectable per agent.

## Framework division of labor
- **LlamaIndex** — ingestion, chunking, embeddings, LanceDB vector store, retrievers.
- **LangChain** — thin adapters (LLM wrappers, document loaders, MCP tool bindings).
- **LangGraph** — the orchestration state machine (supervisor, agent routing, critic loops).
Keep these boundaries; don't reimplement one layer's job in another.

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
- `src/scireg/ingest/index.py` — LlamaIndex -> LanceDB (embedded, at `data/lancedb`).
- `src/scireg/retrieval/retriever.py` — hybrid dense + BM25 with RRF fusion.
- `src/scireg/neuro/entities.py` — neuro entity extraction + query expansion (extension point).
- `src/scireg/agents/synthesize.py` — cited-answer synthesis agent.
- `src/scireg/graph/state.py` — LangGraph `State` + `build_graph()`.
- `src/scireg/mcp_server/server.py` — exposes retrieval as MCP tools.
- `src/scireg/cli.py` — `scireg search|index|ask`.

## Run
```
uv run scireg search "grid cells entorhinal cortex"      # raw PubMed, no LLM
uv run scireg index  "hippocampal place cells" --retmax 30
uv run scireg ask    "How do place cells remap across environments?"
uv run python -m scireg.mcp_server.server                 # MCP server (needs --extra mcp)
```

## Conventions
- **All LLM calls go through `scireg.llm.router.complete(agent, ...)`** — never call
  `litellm`/provider SDKs directly. Add/swap models in `configs/models.yaml`, not in code.
- Agent role names in `models.yaml` (`planner`, `retriever`, `neuro_entity`,
  `synthesizer`, `critic`) are the contract between config and code.
- Synthesis must cite every claim with `[PMID]` markers from source metadata.
- PubMed is the data source; the MCP server is how agents reach it. Don't bypass
  `sources/` with ad-hoc HTTP calls elsewhere.

## Roadmap (multi-agent buildout)
The current graph is linear (`extract_entities -> retrieve -> synthesize`). Next:
supervisor/router node in front; query-planner (MeSH expansion); reranker (move
`bge-reranker-v2-m3` to the 4060); **critic/verifier** loop that scores citation
grounding and routes back to `retrieve` on failure; real ontology resolvers in
`neuro/` (Allen Brain Atlas, NCBI Gene, UniProt, ChEBI, MeSH).
