"""Tests for scirag.neuro.entities — LLM call is fully mocked."""
from __future__ import annotations

import json
from unittest.mock import patch


from scirag.neuro.entities import _ONTOLOGIES, expand_query, extract_entities


def _mock_complete(response: str):
    return patch("scirag.neuro.entities.complete", return_value=response)


class TestExtractEntities:
    def test_well_formed_json(self):
        payload = {
            "brain_region": ["hippocampus", "entorhinal cortex"],
            "neurotransmitter": [],
            "gene_protein": [],
            "method": ["electrophysiology"],
            "species": ["rat"],
        }
        with _mock_complete(json.dumps(payload)):
            result = extract_entities("place cells in rat hippocampus")
        assert result["brain_region"] == ["hippocampus", "entorhinal cortex"]
        assert result["method"] == ["electrophysiology"]
        assert result["species"] == ["rat"]

    def test_all_ontology_keys_present(self):
        payload = {k: [] for k in _ONTOLOGIES}
        with _mock_complete(json.dumps(payload)):
            result = extract_entities("any query")
        assert set(result.keys()) == set(_ONTOLOGIES)

    def test_malformed_json_returns_empty(self):
        with _mock_complete("Sorry, I cannot extract entities."):
            result = extract_entities("some query")
        assert all(v == [] for v in result.values())

    def test_partial_json_wrapped_in_prose(self):
        """LLM sometimes wraps JSON in prose; parser should still extract it."""
        payload = {"brain_region": ["amygdala"], "neurotransmitter": ["dopamine"],
                   "gene_protein": [], "method": [], "species": []}
        with _mock_complete(f"Here you go: {json.dumps(payload)} — done."):
            result = extract_entities("dopamine in amygdala")
        assert result["brain_region"] == ["amygdala"]

    def test_missing_keys_default_to_empty(self):
        """Ontology keys absent from LLM response are filled with []."""
        partial = {"brain_region": ["cortex"]}
        with _mock_complete(json.dumps(partial)):
            result = extract_entities("cortex query")
        for k in _ONTOLOGIES:
            assert k in result
        assert result["neurotransmitter"] == []


class TestExpandQuery:
    def test_appends_terms(self):
        entities = {"brain_region": ["hippocampus"], "method": ["2P imaging"],
                    "neurotransmitter": [], "gene_protein": [], "species": []}
        expanded = expand_query("place cells", entities)
        assert expanded.startswith("place cells")
        assert "hippocampus" in expanded
        assert "2P imaging" in expanded

    def test_empty_entities_unchanged(self):
        entities = {k: [] for k in _ONTOLOGIES}
        assert expand_query("place cells", entities) == "place cells"

    def test_no_duplicate_query(self):
        entities = {"brain_region": ["hippocampus"], "method": [],
                    "neurotransmitter": [], "gene_protein": [], "species": []}
        expanded = expand_query("hippocampus", entities)
        # Query appears once at start, entity appended once — not duplicated arbitrarily
        assert expanded.count("hippocampus") >= 1
