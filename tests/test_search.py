from ragbench.bm25_retrieval import BM25Retriever
from ragbench.hybrid_retrieval import HybridRetriever

def test_bm25_retrieves_relevant_document():
    retriever = BM25Retriever()

    question = "How do we measure how many relevant documents were found?"
    relevant_ids = {"D3", "D14"}

    results = retriever.search(question, k=3)
    retrieved_ids = [item["id"] for item in results]

    assert any(doc_id in relevant_ids for doc_id in retrieved_ids)

def test_bm25_relevant_document_ranks_first():
    retriever = BM25Retriever()

    question = "How do we measure how many relevant documents were found?"
    relevant_ids = {"D3", "D14"}

    results = retriever.search(question, k=3)
    first_result_id = results[0]["id"]

    assert first_result_id in relevant_ids

def test_hybrid_retrieves_relevant_documents():
    retriever = HybridRetriever()

    question = "How do we measure how many relevant documents were found?"
    relevant_ids = {"D3", "D14"}

    results = retriever.search(question, k=3)
    retrieved_ids = [item["id"] for item in results]

    assert any(doc_id in relevant_ids for doc_id in retrieved_ids)

if __name__ == "__main__":
    test_bm25_retrieves_relevant_document()
    
    print("BM25 retrieval test passed!")