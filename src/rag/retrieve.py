"""
retrieve.py — Query the heliophysics corpus via cosine similarity search.

Uses ChromaDB's built-in query method with nomic-embed-text embeddings.
Enforces a cosine similarity floor (COSINE_FLOOR from config) — chunks
below this threshold are excluded.

Usage:
    from src.rag.retrieve import retrieve

    results = retrieve("What is a CME?", k=3)
    for r in results:
        print(f"  [{r['cosine_score']:.3f}] {r['text'][:100]}...")
"""

import logging
from typing import Any

import chromadb

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PATH,
    COSINE_FLOOR,
    RETRIEVAL_TOP_K,
)

logger = logging.getLogger(__name__)


def retrieve(query: str, k: int | None = None) -> list[dict[str, Any]]:
    """
    Retrieve top-k relevant chunks for a query.

    Args:
        query: natural-language question
        k: number of results (default: RETRIEVAL_TOP_K from config)

    Returns:
        List of {text, cosine_score, source_url, source_label} dicts,
        sorted by descending similarity. Empty list if no chunk passes
        the cosine floor.
    """
    if k is None:
        k = RETRIEVAL_TOP_K

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        collection = client.get_collection(CHROMA_COLLECTION_NAME)
    except (ValueError, Exception) as e:
        logger.error("Collection '%s' not found: %s", CHROMA_COLLECTION_NAME, e)
        return []

    results = collection.query(query_texts=[query], n_results=k)

    if not results or not results.get("documents") or not results["documents"][0]:
        return []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(documents)
    distances = results.get("distances", [[1.0] * len(documents)])[0]

    output = []
    for i, doc in enumerate(documents):
        # ChromaDB returns distance (L2 for Ollama embeddings).
        # Convert to approximate cosine similarity: 1 - distance
        # (ChromaDB normalises embeddings, so L2 ≈ 2(1 - cos_sim))
        distance = distances[i] if i < len(distances) else 1.0
        cosine_score = max(0.0, 1.0 - (distance / 2.0))

        if cosine_score < COSINE_FLOOR:
            continue

        meta = metadatas[i] if i < len(metadatas) else {}
        output.append({
            "text": doc,
            "cosine_score": round(cosine_score, 4),
            "source_url": meta.get("source_url", ""),
            "source_label": meta.get("source_label", ""),
        })

    # Re-sort by cosine score descending
    output.sort(key=lambda x: x["cosine_score"], reverse=True)
    return output[:k]


def run_controlled_queries() -> dict[str, list[dict[str, Any]]]:
    """
    Run the 4 controlled queries from SC4 and return results.

    Returns dict mapping query text to list of retrieval results.
    """
    from config import CONTROLLED_QUERIES

    results = {}
    for query in CONTROLLED_QUERIES:
        logger.info("Query: %s", query)
        chunks = retrieve(query, k=3)
        results[query] = chunks
        for i, chunk in enumerate(chunks):
            logger.info(
                "  [%d] score=%.3f source=%s text=%.80s...",
                i + 1,
                chunk["cosine_score"],
                chunk["source_label"],
                chunk["text"],
            )
        if not chunks:
            logger.warning("  No chunks passed cosine floor (%.2f).", COSINE_FLOOR)
    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    run_controlled_queries()
