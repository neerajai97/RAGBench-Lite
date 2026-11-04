from ragbench.retrieval import TfidfRetriever
from ragbench.bm25_retrieval import BM25Retriever

def reciprocal_rank_fusion(
    tfidf_results,
    bm25_results,
    k=60
):
    scores = {}

    for results in [tfidf_results, bm25_results]:
        for rank, result in enumerate(results, start=1):
            doc_id = result["id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

    return scores


class HybridRetriever:
    def __init__(self):
        self.tfidf_retriever = TfidfRetriever()
        self.bm25_retriever = BM25Retriever()

    def search(self, question: str, k: int = 3):
        tfidf_results = self.tfidf_retriever.search(question, k=k)
        bm25_results = self.bm25_retriever.search(question, k=k)

        fused_scores = reciprocal_rank_fusion(
            tfidf_results,
            bm25_results
        )

        documents = {
            result["id"]: result
            for result in tfidf_results + bm25_results
        }

        ranked_ids = sorted(
            fused_scores,
            key=lambda doc_id: fused_scores[doc_id],
            reverse=True
        )[:k]

        return [
            {
                "id": doc_id,
                "text": documents[doc_id]["text"],
                "score": fused_scores[doc_id]
            }
            for doc_id in ranked_ids
        ]