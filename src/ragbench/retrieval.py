
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Project root: ragbench-lite/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TfidfRetriever:
    """Retrieve relevant documents using TF-IDF similarity."""

    def __init__(self, documents_path=None):
        path = documents_path or (
            PROJECT_ROOT / "data" / "documents.json"
        )

        with open(path, "r", encoding="utf-8") as file:
            self.documents = json.load(file)

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            stop_words="english",
        )

        texts = [document["text"] for document in self.documents]
        self.document_matrix = self.vectorizer.fit_transform(texts)

    def search(self, question: str, k: int = 3):
        """Return the top-k documents ranked by similarity."""

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if k < 1:
            raise ValueError("k must be at least 1.")

        query_vector = self.vectorizer.transform([question])

        scores = cosine_similarity(
            query_vector,
            self.document_matrix,
        ).ravel()

        ranked_indices = scores.argsort()[::-1][:k]

        return [
            {
                "id": self.documents[index]["id"],
                "text": self.documents[index]["text"],
                "score": float(scores[index]),
            }
            for index in ranked_indices
        ]
