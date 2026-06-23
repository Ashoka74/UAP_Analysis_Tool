"""RAG search service — Cohere ``rerank`` over selected columns of a dataset,
mirroring the "Dataset RAG (Cohere)" mode of the Streamlit rag_search.py page.

Each row's selected columns are serialized to a JSON document, reranked against
the natural-language query, and returned in relevance order with scores.
"""
from __future__ import annotations

import json
from typing import Any

# Cohere caps documents per rerank request; we chunk above this and merge.
_RERANK_MODEL = "rerank-english-v3.0"


def rerank(df, columns: list[str], question: str, cohere_key: str,
           top_n: int = 50) -> dict:
    import cohere
    import pandas as pd  # noqa: F401  (ensures pandas present for callers)

    if not columns:
        raise ValueError("Select at least one column to search.")
    if not question.strip():
        raise ValueError("Enter a question to search for.")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {', '.join(missing)}")

    documents = [
        json.dumps(doc, default=str)
        for doc in df[columns].to_dict("records")
    ]
    if not documents:
        raise ValueError("No rows available to search.")

    co = cohere.Client(api_key=cohere_key)
    n = min(top_n, len(documents))
    results = co.rerank(
        model=_RERANK_MODEL,
        query=question,
        documents=documents,
        top_n=n,
        return_documents=False,
    )
    ranked_indices = [r.index for r in results.results]
    ranked_scores = [float(r.relevance_score) for r in results.results]

    out = df.iloc[ranked_indices].copy()
    out.insert(0, "relevance_score", [round(s, 4) for s in ranked_scores])
    out.insert(0, "rank", range(1, len(ranked_indices) + 1))
    return {"df": out, "n_results": len(ranked_indices), "searched_columns": columns}
