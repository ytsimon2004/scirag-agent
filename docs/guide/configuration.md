# Configuration

Three things are tunable at runtime — the **model** (per agent), the **reasoning
effort**, and the **retrieval params** — and each has two surfaces with a clear
precedence.

## Resolution order

For every setting:

```
session override  →  settings.yaml default  →  shipped YAML config
```

- A **session override** is set in the shell (`/model`, `/effort`, `/rag`) and lasts
  until you exit.
- A **persistent default** is set from the CLI (`scirag model|effort|rag`) and is
  written to `~/.scirag-agent/settings.yaml`.
- The **shipped config** is `configs/models.yaml` / `configs/pipeline.yaml`.

```{admonition} Never hardcode model choices in code
:class: warning

Swap or add backends in `configs/models.yaml`, or via the CLI/shell commands above —
not in Python. Agent role names (`planner`, `retriever`, `synthesizer`, `critic`)
are the contract between config and code.
```

## Models & reasoning

The default backend for every agent is `local-qwen3-14b` (`ollama/qwen3:14b-q4_K_M`),
a **hybrid-thinking** model: by default it emits a `<think>…</think>` chain before
answering, which is the main latency cost in the pipeline.

- Disable thinking for speed by passing Ollama's `think: False` (low effort does this),
  or prefer a non-thinking backend (`local-llama4-scout`) for high-frequency agents.
- Reserve a reasoning model (`local-deepseek-r1-32b`) for where it earns its cost.

### Reasoning effort

```text
/effort low|medium|high      # session
scirag effort high           # persistent default
```

The router (`scirag.llm.router`) maps effort per backend:

| Backend type | How effort is applied |
|--------------|-----------------------|
| Ollama thinking models | toggle `think` (low = off) |
| Claude / OpenAI APIs | litellm's `reasoning_effort` |
| `claude` / `codex` CLIs | `--effort` / `model_reasoning_effort` |

Effort also scales the answer token budget. Defaults to `medium`.

## Retrieval params

```text
/rag                         # open the picker (per-param help)
/rag final_k 4               # shorthand
scirag rag final_k 12        # persistent default
```

| Param | Meaning |
|-------|---------|
| `final_k` | passages kept after reranking |
| `top_k` / `bm25_k` | dense / BM25 candidates retrieved (retrieve wide, ~30) |
| `hybrid` | enable dense + BM25 fusion |
| `rag_score_threshold` | cosine gate below which a passage can't ground an answer |
| `rerank` | enable cross-encoder reranking (needs `--extra rerank`) |

`chunk_size` / `chunk_overlap` are index-time only and intentionally excluded here.

```{admonition} Retrieve wide, rerank tight
:class: tip

Set `top_k` / `bm25_k` high (~30) and let the cross-encoder pick the best `final_k`.
This raises recall *and* precision. Reranking only reorders nodes — their cosine
`.score` is preserved, so the `rag_score_threshold` gate still works. Reranking is
**off by default**; without `--extra rerank` it degrades gracefully to RRF order.
```

## File locations

| Path | Holds |
|------|-------|
| `~/.scirag-agent/.env` | API keys |
| `~/.scirag-agent/settings.yaml` | persistent model/effort/rag defaults |
| `~/.scirag-agent/configs/` | optional per-user config overrides |
| `configs/models.yaml`, `configs/pipeline.yaml` | shipped defaults |
| `data/lancedb` | the embedded LanceDB index |
