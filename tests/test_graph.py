"""Tests for scireg.graph.state — node functions tested in isolation."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from scireg.graph.state import State, _entities_node, _retrieve_node, _synthesize_node


def _mock_node(node_id: str = "n1") -> MagicMock:
    nws = MagicMock()
    nws.node.node_id = node_id
    return nws


class TestEntitiesNode:
    def test_populates_entities_and_expanded_query(self):
        entities = {"brain_region": ["hippocampus"], "method": [], "neurotransmitter": [],
                    "gene_protein": [], "species": []}
        with patch("scireg.graph.state.extract_entities", return_value=entities), \
             patch("scireg.graph.state.expand_query", return_value="hippocampus place cells"):
            result = _entities_node({"query": "place cells"})
        assert result["entities"] == entities
        assert result["expanded_query"] == "hippocampus place cells"


class TestRetrieveNode:
    def test_uses_expanded_query_when_present(self):
        node = _mock_node()
        with patch("scireg.graph.state.retrieve", return_value=[node]) as mock_retrieve:
            result = _retrieve_node({"query": "original", "expanded_query": "expanded"})
        mock_retrieve.assert_called_once_with("expanded")
        assert result["nodes"] == [node]

    def test_falls_back_to_query_when_no_expanded(self):
        node = _mock_node()
        with patch("scireg.graph.state.retrieve", return_value=[node]) as mock_retrieve:
            result = _retrieve_node({"query": "original"})
        mock_retrieve.assert_called_once_with("original")


class TestSynthesizeNode:
    def test_calls_synthesize_with_query_and_nodes(self):
        nodes = [_mock_node()]
        with patch("scireg.graph.state.synthesize", return_value="Answer [1].") as mock_synth:
            result = _synthesize_node({"query": "question", "nodes": nodes})
        mock_synth.assert_called_once_with("question", nodes)
        assert result["answer"] == "Answer [1]."


class TestBuildGraph:
    def test_graph_compiles(self):
        """build_graph() should not raise during compilation."""
        from scireg.graph.state import build_graph
        graph = build_graph()
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        from scireg.graph.state import build_graph
        graph = build_graph()
        node_names = set(graph.nodes)
        assert {"extract_entities", "retrieve", "synthesize"}.issubset(node_names)
