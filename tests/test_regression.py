
import json
from pathlib import Path

from ragbench.evaluation import run_evaluation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = PROJECT_ROOT / "reports" / "baseline.json"

ALLOWED_DROP = 0.02


def test_hybrid_retrieval_quality():
    # Load the fixed reference benchmark
    with open(BASELINE_PATH, encoding="utf-8") as file:
        baseline = json.load(file)

    # Run the current benchmark
    current = run_evaluation(k=3)

    assert current["evaluated_questions"] == baseline["evaluated_questions"]

    metrics = [
        "hybrid_mean_precision_at_3",
        "hybrid_mean_recall_at_3",
        "hybrid_mrr",
    ]

    for metric in metrics:
        assert current[metric] >= baseline[metric] - ALLOWED_DROP, (
            f"Retrieval regression detected in {metric}. "
            f"Baseline: {baseline[metric]:.4f}, "
            f"Current: {current[metric]:.4f}"
        )
