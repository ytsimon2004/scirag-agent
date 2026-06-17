# Minimal makefile for Sphinx documentation.
# Defaults SPHINXBUILD to "uv run sphinx-build" so `make html` works without a
# pre-activated venv; override with e.g. `make html SPHINXBUILD=sphinx-build`.

SPHINXBUILD   ?= uv run sphinx-build
SPHINXOPTS    ?=
SOURCEDIR     = .
BUILDDIR      = _build

# "make help" / catch-all forwards to sphinx-build -M (html, clean, latexpdf, …).
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
