# scireg

Multi-agent RAG for scientific (neuroscience) literature retrieval.

Built with **LlamaIndex** (indexing/retrieval), **LangChain** (adapters), and
**LangGraph** (orchestration). Open-source (Ollama: Qwen/Llama/DeepSeek) and
frontier (Claude/OpenAI) LLMs live behind one LiteLLM router, swappable per
agent in `configs/models.yaml`.

## Quickstart

```bash
uv sync
cp .env.example .env            # add NCBI_API_KEY, optionally ANTHROPIC_API_KEY

# local models (Mac, 36 GB):
ollama serve &
ollama pull qwen2.5:14b-instruct-q4_K_M
ollama pull bge-m3

uv run scireg search "grid cells entorhinal cortex"   # raw PubMed sanity check
uv run scireg index  "hippocampal place cells" --retmax 30
uv run scireg ask    "How do place cells remap across environments?"
```

See `CLAUDE.md` for architecture, hardware notes, and the multi-agent roadmap.
