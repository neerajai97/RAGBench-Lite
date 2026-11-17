
from ragbench.evaluation import run_evaluation


def compare_top_k(values=(1, 3, 5)):
    comparisons = []

    for k in values:
        report = run_evaluation(k=k)

        comparisons.append({
            "k": k,
            "evaluated_questions": report["evaluated_questions"],
            f"hybrid_precision_at_{k}": report[
                f"hybrid_mean_precision_at_{k}"
            ],
            f"hybrid_recall_at_{k}": report[
                f"hybrid_mean_recall_at_{k}"
            ],
            "hybrid_mrr": report["hybrid_mrr"],
        })

    return comparisons


if __name__ == "__main__":
    import json

    results = compare_top_k()

    print(json.dumps(results, indent=2))
