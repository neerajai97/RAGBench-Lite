
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[2]


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by",
    "do", "for", "from", "how", "in", "is", "it",
    "of", "on", "or", "that", "the", "this", "to",
    "we", "were", "what", "when", "where", "which",
    "who", "why", "with"
}


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [token for token in tokens if token not in STOP_WORDS]


class BM25Retriever:
    """Retrieve documents using BM25 lexical ranking."""

    def __init__(self, documents_path=None):
        path = documents_path or (
            PROJECT_ROOT / "data" / "documents.json"
        )

        with open(path, "r", encoding="utf-8") as file:
            self.documents = json.load(file)

        tokenized_documents = [
            tokenize(document["text"])
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

    def search(self, question: str, k: int = 3):
        """Return the top-k documents ranked by BM25."""
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if k < 1:
            raise ValueError("k must be at least 1.")

        scores = self.bm25.get_scores(tokenize(question))

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:k]

        return [
            {
                "id": self.documents[index]["id"],
                "text": self.documents[index]["text"],
                "score": float(scores[index]),
            }
            for index in ranked_indices
        ]
