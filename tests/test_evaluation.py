import pytest

from ragbench.evaluation import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k():
    retrieved = ["D1", "D2", "D3"]
    relevant = ["D1", "D3"]

    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)


def test_recall_at_k():
    retrieved = ["D1", "D2", "D3"]
    relevant = ["D1", "D3", "D4"]

    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)


def test_reciprocal_rank():
    retrieved = ["D1", "D2", "D3"]
    relevant = ["D2", "D3"]

    assert reciprocal_rank(retrieved, relevant) == pytest.approx(0.5)


def test_reciprocal_rank_when_relevant_document_is_first():
    retrieved = ["D3", "D2", "D1"]
    relevant = ["D3"]

    assert reciprocal_rank(retrieved, relevant) == 1.0


def test_empty_relevant_documents():
    retrieved = ["D1", "D2", "D3"]
    relevant = []

    assert precision_at_k(retrieved, relevant, 3) is None
    assert recall_at_k(retrieved, relevant, 3) is None
    assert reciprocal_rank(retrieved, relevant) is None


def test_precision_rejects_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k(["D1"], ["D1"], 0)