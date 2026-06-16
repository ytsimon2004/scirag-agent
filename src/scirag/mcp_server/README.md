# scirag MCP server

Exposes scirag's retrieval over the [Model Context Protocol](https://modelcontextprotocol.io)
so any MCP client (Claude Code, Claude Desktop, Cursor, agent frameworks) can search
the literature sources and query your local index as tools.

```
uv sync --extra mcp                                   # install the optional dep
uv run python -m scirag.mcp_server.server             # run (stdio transport)
```

Ollama must be running for `retrieve_chunks` / `ask_index` (embeddings route through
it). `search_pubmed` / `search_biorxiv` / `list_projects` / `index_status` don't need it.

## Tools

| Tool | What it does |
|---|---|
| `search_pubmed(query, retmax=25, semantic=False)` | Search PubMed. `semantic=True` → Europe PMC relevance ranking (tolerates natural-language queries). Returns source metadata, **not** your index. |
| `search_biorxiv(query, retmax=25, days_back=180)` | Search recent bioRxiv preprints by relevance (Europe PMC). |
| `list_projects()` | List indexed corpora (projects) + which is active. |
| `retrieve_chunks(query, project="")` | **The main tool.** Retrieve grounded chunks from the local index — *no LLM synthesis*. Returns evidence (`citation`, `id`, `title`, `text_source`, `url`, `score`, `text`) for the caller to reason over itself. |
| `get_record(id, project="")` | Full stored text + metadata for one paper (by PMID / bioRxiv DOI). |
| `index_status(project="")` | Corpus summary: counts + one row per indexed paper. |
| `ask_index(query)` | Retrieve **and** synthesize a cited answer with scirag's own LLM. Prefer `retrieve_chunks` if you're an LLM that just wants the sources — `ask_index` adds an extra inference step. |

The `project` argument scopes a call to a specific indexed corpus (see `list_projects`);
omit it to use the active project. Scoping is per-call and concurrency-safe — it never
rewrites the shared active-project state, so it won't disturb a running `scirag` shell.

## Connecting a client

The server speaks **stdio**, so clients launch it as a subprocess. The launch spec
(adjust the path to your checkout):

```
command: uv
args:    run --directory /path/to/scirag-agent python -m scirag.mcp_server.server
```

### Claude Code

```bash
claude mcp add scirag -- uv run --directory /path/to/scirag-agent python -m scirag.mcp_server.server
```

### Claude Desktop / Cursor / VS Code (`claude_desktop_config.json`, `.cursor/mcp.json`, …)

```json
{
  "mcpServers": {
    "scirag": {
      "command": "uv",
      "args": [
        "run", "--directory", "/path/to/scirag-agent",
        "python", "-m", "scirag.mcp_server.server"
      ]
    }
  }
}
```

### Python (MCP SDK — scriptable, same stdio protocol)

Results come back as JSON in `result.content` (one item per element for a
list-returning tool like `retrieve_chunks`; a single item for a dict-returning tool
like `index_status`). `json.loads` each `.text`:

```python
import asyncio
import json
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

PARAMS = StdioServerParameters(
    command="uv",
    args=["run", "--directory", "/path/to/scirag-agent",
          "python", "-m", "scirag.mcp_server.server"],
)

async def main():
    async with stdio_client(PARAMS) as (read, write), ClientSession(read, write) as s:
        await s.initialize()
        print([t.name for t in (await s.list_tools()).tools])

        # ground a question in a specific project (list tool → one item per chunk)
        res = await s.call_tool("retrieve_chunks",
                                {"query": "place cells remapping", "project": "human-rsc"})
        chunks = [json.loads(c.text) for c in res.content]
        for ch in chunks:
            print(ch["citation"], round(ch["score"], 3), ch["title"][:60])

        # dict tool → a single JSON item
        status = json.loads((await s.call_tool("index_status", {})).content[0].text)
        print(status["count"], "papers indexed")

asyncio.run(main())
```

(LLM clients like Claude Desktop read this text content automatically — the manual
`json.loads` is only needed when scripting against the raw SDK.)

## Inspecting / debugging

Launch the server under the official MCP Inspector (web UI to call each tool and see
the raw JSON-RPC) — needs Node for `npx`:

```
uv run mcp dev src/scirag/mcp_server/server.py
```

## HTTP instead of stdio

The default `mcp.run()` is stdio (right for local, single-user — each client spawns its
own process). To run one long-lived server multiple clients reach over the network,
switch the entrypoint to `mcp.run(transport="streamable-http")` and point clients at the
URL instead of a launch command.
```
