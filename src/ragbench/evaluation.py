
import json
from pathlib import Path

from ragbench.retrieval import TfidfRetriever
from ragbench.bm25_retrieval import BM25Retriever
from ragbench.hybrid_retrieval import HybridRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]

def recall_at_k(retrieved_ids, relevant_ids, k):
    """Fraction of relevant documents found in the top k."""
    relevant = set(relevant_ids)

    if not relevant:
        return None

    retrieved = set(retrieved_ids[:k])
    return len(retrieved & relevant) / len(relevant)


def precision_at_k(retrieved_ids, relevant_ids, k):
    """Fraction of the top-k results that are relevant."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    relevant = set(relevant_ids)

    if not relevant:
        return None

    retrieved = retrieved_ids[:k]

    if not retrieved:
        return 0.0

    return sum(
        doc_id in relevant for doc_id in retrieved
    ) / len(retrieved)


def reciprocal_rank(retrieved_ids, relevant_ids):
    """Reciprocal rank of the first relevant result."""
    relevant = set(relevant_ids)

    if not relevant:
        return None

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank

    return 0.0


def run_evaluation(k=3):
    tfidf_retriever = TfidfRetriever()
    bm25_retriever = BM25Retriever()
    hybrid_retriever = HybridRetriever()

    with open(
        PROJECT_ROOT / "data" / "eval_cases.json",
        encoding="utf-8",
    ) as file:
        cases = json.load(file)

    results = []

    for case in cases:
        question = case["question"]
        relevant_ids = case["relevant_ids"]

        # Retrieve documents using all three methods
        tfidf_results = tfidf_retriever.search(question, k=k)
        bm25_results = bm25_retriever.search(question, k=k)
        hybrid_results = hybrid_retriever.search(question, k=k)

        # Extract retrieved document IDs
        tfidf_ids = [item["id"] for item in tfidf_results]
        bm25_ids = [item["id"] for item in bm25_results]
        hybrid_ids = [item["id"] for item in hybrid_results]

        # Calculate evaluation metrics
        results.append({
            "question": question,
            "relevant_ids": relevant_ids,

            "tfidf_retrieved_ids": tfidf_ids,
            "bm25_retrieved_ids": bm25_ids,
            "hybrid_retrieved_ids": hybrid_ids,

            "tfidf_precision_at_k": precision_at_k(
                tfidf_ids, relevant_ids, k
            ),
            "bm25_precision_at_k": precision_at_k(
                bm25_ids, relevant_ids, k
            ),
            "hybrid_precision_at_k": precision_at_k(
                hybrid_ids, relevant_ids, k
            ),

            "tfidf_recall_at_k": recall_at_k(
                tfidf_ids, relevant_ids, k
            ),
            "bm25_recall_at_k": recall_at_k(
                bm25_ids, relevant_ids, k
            ),
            "hybrid_recall_at_k": recall_at_k(
                hybrid_ids, relevant_ids, k
            ),

            "tfidf_reciprocal_rank": reciprocal_rank(
                tfidf_ids, relevant_ids
            ),
            "bm25_reciprocal_rank": reciprocal_rank(
                bm25_ids, relevant_ids
            ),
            "hybrid_reciprocal_rank": reciprocal_rank(
                hybrid_ids, relevant_ids
            ),
        })

    # Exclude questions without relevance labels from metric averages
    valid_results = [
        row for row in results
        if row["relevant_ids"]
    ]

    # Calculate the mean of a metric across valid questions
    def mean_metric(metric_name):
        values = [
            row[metric_name]
            for row in valid_results
            if row[metric_name] is not None
        ]

        return sum(values) / len(values) if values else 0.0

    # Generate final evaluation report
    return {
        "total_questions": len(results),
        "evaluated_questions": len(valid_results),
        "excluded_questions": len(results) - len(valid_results),

        f"tfidf_mean_precision_at_{k}": mean_metric(
            "tfidf_precision_at_k"
        ),
        f"bm25_mean_precision_at_{k}": mean_metric(
            "bm25_precision_at_k"
        ),
        f"hybrid_mean_precision_at_{k}": mean_metric(
            "hybrid_precision_at_k"
        ),

        f"tfidf_mean_recall_at_{k}": mean_metric(
            "tfidf_recall_at_k"
        ),
        f"bm25_mean_recall_at_{k}": mean_metric(
            "bm25_recall_at_k"
        ),
        f"hybrid_mean_recall_at_{k}": mean_metric(
            "hybrid_recall_at_k"
        ),

        "tfidf_mrr": mean_metric(
            "tfidf_reciprocal_rank"
        ),
        "bm25_mrr": mean_metric(
            "bm25_reciprocal_rank"
        ),
        "hybrid_mrr": mean_metric(
            "hybrid_reciprocal_rank"
        ),

        "results": results,
    }


if __name__ == "__main__":
    report = run_evaluation(k=3)
    print(json.dumps(report, indent=2))

    output_path = PROJECT_ROOT / "reports" / "baseline.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_path}")
