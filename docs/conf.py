"""Sphinx configuration for the scirag-agent documentation.

Markdown source (via MyST), API reference auto-generated from docstrings with
autodoc + napoleon. Heavy third-party deps are mocked so Read the Docs can build
without an Ollama server, a CUDA stack, or the full retrieval/LLM dependency tree.
"""

from __future__ import annotations

import importlib.metadata

# -- Project information -----------------------------------------------------
project = "scirag-agent"
author = "Yu-Ting Wei"
copyright = "2026, Yu-Ting Wei"

try:
    release = importlib.metadata.version("scirag-agent")
except importlib.metadata.PackageNotFoundError:  # not installed (e.g. local docs-only build)
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",  # Markdown source
    "sphinx.ext.autodoc",  # pull docstrings
    "sphinx.ext.autosummary",  # module/object summary tables
    "sphinx.ext.napoleon",  # Google/NumPy-style docstrings
    "sphinx.ext.viewcode",  # [source] links
    "sphinx.ext.intersphinx",  # cross-link to Python/stdlib docs
    "sphinx_copybutton",  # copy button on code blocks
    "sphinx_design",  # grid / card directives used on the landing page
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- MyST -------------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",  # ::: fenced directives
    "deflist",
    "linkify",  # bare URLs -> links
    "substitution",
]
myst_heading_anchors = 3  # auto header anchors so cross-page links resolve

# -- Autodoc ----------------------------------------------------------------
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# The package imports an LLM/retrieval stack that isn't needed just to read
# docstrings — mock it so RTD (and any docs-only environment) builds cleanly.
autodoc_mock_imports = [
    "litellm",
    "llama_index",
    "lancedb",
    "pandas",
    "rank_bm25",
    "httpx",
    "curl_cffi",
    "Bio",
    "pypdf",
    "sentence_transformers",
    "torch",
    "chainlit",
    "mcp",
    "ragas",
]

# -- Intersphinx ------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# -- HTML output ------------------------------------------------------------
html_theme = "furo"
html_title = "scirag-agent"
html_static_path = ["_static"]
html_logo = "_static/logo.png"
html_favicon = "_static/logo.png"
html_theme_options = {
    "source_repository": "https://github.com/ytsimon2004/scirag-agent",
    "source_branch": "main",
    "source_directory": "docs/",
}
