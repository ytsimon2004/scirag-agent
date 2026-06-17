# scirag-agent

```{image} _static/logo.png
:alt: scirag-agent logo
:width: 360px
:align: center
```

**Multi-agent RAG for scientific literature** — a neuroscience-focused, interactive
shell for building a *local, curated literature index* and asking grounded, cited
questions against it.

Sources: PubMed, bioRxiv, local PDFs, Mendeley/Zotero libraries, or any free-form
text. LLM-agnostic — runs fully offline via [Ollama](https://ollama.com), or routes
to frontier models (Claude / OpenAI) through a single config switch.

Built with **LlamaIndex** (indexing/retrieval) and **LiteLLM** (LLM routing).

```{admonition} Why scirag?
:class: tip

Every claim in an answer is cited to a specific passage (`[PMID]` / `[DOI]`), the
index stays on your machine, and you choose exactly which papers are in scope.
Retrieval prioritises the **Results section** — what a paper actually found — not
just its abstract.
```

## Where to go next

::::{grid} 2
:gutter: 3

:::{grid-item-card} Installation
:link: guide/installation
:link-type: doc

Install with `uv`, set up Ollama, and configure the optional extras.
:::

:::{grid-item-card} Quickstart
:link: guide/quickstart
:link-type: doc

Create a project, index a few papers, and ask your first grounded question.
:::

:::{grid-item-card} Shell commands
:link: guide/shell-commands
:link-type: doc

The interactive REPL — every `/command` and the small scriptable CLI.
:::

:::{grid-item-card} Configuration
:link: guide/configuration
:link-type: doc

Models, reasoning effort, retrieval params, and the CLI/session override layers.
:::

:::{grid-item-card} Architecture
:link: guide/architecture
:link-type: doc

How the pipeline, retriever, sources, and LLM router fit together.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc

Auto-generated reference for the `scirag` package.
:::
::::

```{toctree}
:hidden:
:caption: Guide

guide/installation
guide/quickstart
guide/shell-commands
guide/configuration
guide/architecture
```

```{toctree}
:hidden:
:caption: Reference

api/index
```
