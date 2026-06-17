# Installation

scirag-agent is managed end-to-end with [**uv**](https://docs.astral.sh/uv/) and
targets **Python 3.11+**.

## Install

```bash
git clone https://github.com/ytsimon2004/scirag-agent.git
cd scirag-agent
uv sync                 # core install (+ dev tooling)
```

`uv sync` also installs the `dev` dependency group (pytest, ruff, pre-commit) so the
formatting git hook works out of the box.

To install the project as a CLI tool on your PATH:

```bash
uv tool install .
```

The console script is `scirag`. If it is not on your PATH, prefix any command with
`uv run` (e.g. `uv run scirag`).

## Optional extras

```bash
uv sync --extra all                 # bundles the light extras: mcp, eval, ui
uv sync --extra all --extra rerank  # also the cross-encoder reranker
```

| Extra | Pulls in | Enables |
|-------|----------|---------|
| `mcp` | `mcp` | MCP server (`python -m scirag.mcp_server.server`) |
| `eval` | `ragas` | retrieval/answer evaluation |
| `ui` | `chainlit` | the web UI (`scirag ui`) |
| `rerank` | `sentence-transformers` (+ CPU `torch`) | cross-encoder reranking |

```{admonition} Why rerank is separate
:class: note

`rerank` pulls sentence-transformers and torch. torch is pinned **CPU-only** in
`[tool.uv.sources]` (the reranker only re-scores candidates — all inference is via
Ollama), keeping the install ~1 GB instead of the ~6 GB CUDA stack. Without it,
retrieval simply falls back to RRF fusion order.
```

## Ollama (local models + embeddings)

Local inference and embeddings run through [Ollama](https://ollama.com), which must
be running:

```bash
ollama serve
ollama pull qwen3:14b-q4_K_M    # default chat backend
ollama pull bge-m3              # embeddings
```

```{admonition} Hardware
:class: tip

The **Mac (36 GB unified)** is the primary local box — it runs Qwen3-14B (q4) +
bge-m3 comfortably. A smaller GPU (e.g. RTX 4060 8 GB) only fits 7–8B at 4-bit and
is best used as a remote embedding/reranker server: point `embeddings.api_base`
(and any `local-*` backend `api_base`) in `configs/models.yaml` at
`http://<host>:11434`.
```

## API keys

Keys live in `~/.scirag-agent/.env` and are managed with the `env` command:

```bash
scirag env set NCBI_API_KEY <key>        # raises the PubMed rate limit to 10 req/s
scirag env set ANTHROPIC_API_KEY <key>   # only if you select a Claude backend
scirag env set OPENAI_API_KEY <key>      # only if you select an OpenAI backend
```

To run **fully offline**, keep every agent on the default `local-qwen3-14b` backend —
no API keys required.

## Building the docs

This site is built with Sphinx. Install the `docs` extra and run `make` from the
`docs/` directory:

```bash
uv sync --extra docs
cd docs
make html        # output: docs/_build/html/index.html
make clean        # remove the build
```

`make help` lists the other targets (e.g. `make latexpdf`). The Makefile defaults
`SPHINXBUILD` to `uv run sphinx-build`, so it works without activating the venv;
override it (`make html SPHINXBUILD=sphinx-build`) if you prefer.
