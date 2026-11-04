
# RAGBench Lite

**An open-source framework for evaluating, comparing, and optimizing Retrieval-Augmented Generation (RAG) retrieval systems.**

RAGBench Lite is a lightweight retrieval evaluation project designed to benchmark different retrieval strategies, measure retrieval quality, identify performance issues, and experiment with retrieval optimization techniques.

The project currently implements TF-IDF, BM25, and Hybrid Retrieval using Reciprocal Rank Fusion (RRF).

## Features

- TF-IDF-based retrieval
- BM25 lexical retrieval
- Hybrid retrieval using Reciprocal Rank Fusion
- Precision@K evaluation
- Recall@K evaluation
- Mean Reciprocal Rank (MRR)
- JSON-based benchmark reports
- Automated retrieval testing with pytest

## Architecture

```text
Knowledge Base
      |
      v
Retrieval Engine
  |     |      |
TF-IDF BM25  Hybrid
  |     |      |
  v     v      v
Evaluation Engine
      |
      v
Benchmark Reports
```

## Project Structure

```text
ragbench-lite/
│
├── data/
│   ├── documents.json
│   └── eval_cases.json
│
├── reports/
│   ├── baseline.json
│   └── baseline_reference.json
│
├── src/
│   └── ragbench/
│       ├── retrieval.py
│       ├── bm25_retrieval.py
│       ├── hybrid_retrieval.py
│       └── evaluation.py
│
├── tests/
├── test_search.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ragbench-lite.git
cd ragbench-lite
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Benchmark

On Windows PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\src"
python .\src\ragbench\evaluation.py
```

The benchmark report is saved to:

```text
reports/baseline.json
```

## Evaluation Metrics

| Metric      | Description                                                |
| ----------- | ---------------------------------------------------------- |
| Precision@K | Fraction of retrieved documents that are relevant          |
| Recall@K    | Fraction of known relevant documents retrieved             |
| MRR         | Measures the ranking position of the first relevant result |

## Initial Benchmark Results

The current benchmark uses 30 evaluation questions, of which 29 have relevance labels.

| Metric      |           TF-IDF |   BM25 |           Hybrid |
| ----------- | ---------------: | -----: | ---------------: |
| Precision@3 |           0.3793 | 0.3908 | **0.4023** |
| Recall@3    |           0.8391 | 0.8563 | **0.8678** |
| MRR         | **0.8908** | 0.8506 |           0.8793 |

Hybrid retrieval currently achieves the highest Precision@3 and Recall@3 among the three approaches.

These are initial results on a small, manually labelled dataset, not a general claim that hybrid retrieval always performs better.

## Roadmap

- [X] TF-IDF retrieval
- [X] BM25 retrieval
- [X] Hybrid retrieval with RRF
- [X] Retrieval evaluation metrics
- [X] Initial benchmark reporting
- [ ] Automated retrieval regression testing
- [ ] Semantic retrieval using embeddings
- [ ] Retrieval optimization and ablation studies
- [ ] LLM integration and grounded generation
- [ ] Answer quality evaluation
- [ ] FastAPI integration
- [ ] Reporting interface
- [ ] Documentation and release

## Technology Stack

- Python
- Scikit-learn
- NumPy
- rank-bm25
- pytest

## License

To be added.
