"""Tests for scirag.retrieval.retriever — index and pipeline config are mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


from scirag.retrieval.retriever import _rrf


def _make_node(node_id: str, score: float = 1.0) -> MagicMock:
    nws = MagicMock()
    nws.node.node_id = node_id
    nws.score = score
    return nws


class TestRRF:
    def test_single_ranking_order_preserved(self):
        nodes = [_make_node(str(i)) for i in range(5)]
        result = _rrf([nodes])
        ids = [n.node.node_id for n in result]
        assert ids == [str(i) for i in range(5)]

    def test_two_identical_rankings(self):
        nodes = [_make_node(str(i)) for i in range(3)]
        result = _rrf([nodes, nodes])
        # Same order expected; top node gets highest fused score
        assert result[0].node.node_id == "0"

    def test_fusion_promotes_overlap(self):
        # node "A" appears in both lists at rank 0; "B" only in first at rank 1
        a, b, c = _make_node("A"), _make_node("B"), _make_node("C")
        result = _rrf([[a, b], [a, c]])
        ids = [n.node.node_id for n in result]
        assert ids[0] == "A"

    def test_empty_rankings(self):
        assert _rrf([]) == []

    def test_empty_inner_list(self):
        assert _rrf([[]]) == []

    def test_deduplication(self):
        a = _make_node("A")
        b = _make_node("B")
        result = _rrf([[a, b], [b, a]])
        ids = [n.node.node_id for n in result]
        assert len(ids) == len(set(ids))

    def test_k_parameter_affects_scores(self):
        """Higher k dampens rank differences; order may change vs small k."""
        nodes = [_make_node(str(i)) for i in range(4)]
        result_low_k = _rrf([nodes], k=1)
        result_high_k = _rrf([nodes], k=1000)
        # Both should return all nodes — just verifying no crash
        assert len(result_low_k) == 4
        assert len(result_high_k) == 4


class TestRetrieve:
    """Integration-style test: patch away index and config, verify routing."""

    def _cfg(self, hybrid: bool = False):
        return {"top_k": 5, "bm25_k": 5, "final_k": 3, "hybrid": hybrid}

    @patch("scirag.retrieval.retriever.get_retrieval")
    @patch("scirag.retrieval.retriever.load_index")
    def test_dense_only(self, mock_load_index, mock_get_retrieval):
        mock_get_retrieval.return_value = self._cfg(hybrid=False)
        nodes = [_make_node(str(i)) for i in range(5)]
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = nodes
        mock_load_index.return_value.as_retriever.return_value = mock_retriever

        from scirag.retrieval.retriever import retrieve

        result = retrieve("place cells")
        assert len(result) == 3  # final_k=3

    @patch("scirag.retrieval.retriever.get_retrieval")
    @patch("scirag.retrieval.retriever.load_index")
    def test_hybrid_falls_back_when_bm25_unavailable(self, mock_load_index, mock_get_retrieval):
        mock_get_retrieval.return_value = self._cfg(hybrid=True)
        nodes = [_make_node(str(i)) for i in range(5)]
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = nodes
        index = MagicMock()
        index.as_retriever.return_value = mock_retriever
        # BM25Retriever import will fail in test env — retriever must degrade gracefully
        mock_load_index.return_value = index

        with patch.dict("sys.modules", {"llama_index.retrievers.bm25": None}):
            from scirag.retrieval.retriever import retrieve

            result = retrieve("grid cells")
        assert len(result) == 3

    @patch("scirag.retrieval.retriever._get_reranker")
    @patch("scirag.retrieval.retriever.get_retrieval")
    @patch("scirag.retrieval.retriever.load_index")
    def test_rerank_reorders_and_truncates(
        self, mock_load_index, mock_get_retrieval, mock_get_reranker
    ):
        cfg = self._cfg(hybrid=False)
        cfg.update(final_k=2, rerank=True, rerank_model="x")
        mock_get_retrieval.return_value = cfg

        nodes = [_make_node(str(i), score=0.9) for i in range(5)]
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = nodes
        mock_load_index.return_value.as_retriever.return_value = mock_retriever

        # Reranker scores ascending, so node "4" is most relevant, then "3".
        fake = MagicMock()
        fake.predict.return_value = [0.0, 1.0, 2.0, 3.0, 4.0]
        mock_get_reranker.return_value = fake

        from scirag.retrieval.retriever import retrieve

        result = retrieve("place cells")
        assert [r.node.node_id for r in result] == ["4", "3"]  # reranked order, final_k=2
        assert result[0].score == 0.9  # cosine score preserved for the grounding gate

    @patch("scirag.retrieval.retriever.get_retrieval")
    @patch("scirag.retrieval.retriever.load_index")
    def test_rerank_falls_back_when_unavailable(self, mock_load_index, mock_get_retrieval):
        cfg = self._cfg(hybrid=False)
        cfg.update(final_k=2, rerank=True, rerank_model="x")
        mock_get_retrieval.return_value = cfg

        nodes = [_make_node(str(i)) for i in range(5)]
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = nodes
        mock_load_index.return_value.as_retriever.return_value = mock_retriever

        # sentence_transformers absent → _get_reranker raises → keep RRF/dense order.
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            from scirag.retrieval.retriever import retrieve

            result = retrieve("grid cells")
        assert [r.node.node_id for r in result] == ["0", "1"]  # original order, final_k=2
