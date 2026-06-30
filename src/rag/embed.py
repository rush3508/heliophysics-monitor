"""
embed.py — Embed corpus chunks into ChromaDB using nomic-embed-text via Ollama.

Loads chunks from docs/corpus/chunks.json, embeds each one using
nomic-embed-text through ChromaDB's OllamaEmbeddingFunction, and
persists the vector store to docs/corpus/chroma/.

After embedding, unloads nomic-embed-text (keep_alive=0) to free RAM
for llama3.2 in later stages.

Usage:
    python -m src.rag.embed
"""

import json
import logging
import subprocess

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from config import CHROMA_COLLECTION_NAME, CHROMA_PATH, CHUNKS_FILE, OLLAMA_EMBED_MODEL

logger = logging.getLogger(__name__)


def embed_corpus(chunks_file: str | None = None) -> chromadb.Collection:
    """
    Embed all chunks and persist to ChromaDB.

    Args:
        chunks_file: path to chunks.json. Defaults to config.CHUNKS_FILE.

    Returns:
        The ChromaDB collection (persisted to disk).
    """
    if chunks_file is None:
        chunks_file = str(CHUNKS_FILE)

    # Load chunks
    with open(chunks_file) as f:
        chunks = json.load(f)
    logger.info("Loaded %d chunks from %s", len(chunks), chunks_file)

    # Build ChromaDB with Ollama embedding
    embedding_fn = OllamaEmbeddingFunction(
        model_name=OLLAMA_EMBED_MODEL,
        url="http://localhost:11434/api/embeddings",
    )

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # Delete existing collection if present (fresh rebuild)
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
        logger.info("Deleted existing collection '%s'.", CHROMA_COLLECTION_NAME)
    except (ValueError, Exception):
        pass

    collection = client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"description": "NASA heliophysics terminology corpus"},
    )

    # Prepare data for batch add
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "source_url": c["source_url"],
            "source_label": c["source_label"],
            "chunk_index": c["chunk_index"],
        }
        for c in chunks
    ]

    # Add in batches to avoid overloading the embedding endpoint
    batch_size = 20
    for i in range(0, len(documents), batch_size):
        batch_end = min(i + batch_size, len(documents))
        logger.info(
            "Embedding batch %d–%d of %d...", i + 1, batch_end, len(documents)
        )
        collection.add(
            ids=ids[i:batch_end],
            documents=documents[i:batch_end],
            metadatas=metadatas[i:batch_end],
        )

    logger.info(
        "Collection '%s' created with %d documents.",
        CHROMA_COLLECTION_NAME,
        collection.count(),
    )

    # Unload nomic-embed-text to free RAM
    _unload_embed_model()

    return collection


def _unload_embed_model() -> None:
    """Release nomic-embed-text from Ollama memory via keep_alive=0."""
    try:
        import requests
        requests.post(
            "http://localhost:11434/api/generate",
            json={"model": OLLAMA_EMBED_MODEL, "keep_alive": 0, "prompt": ""},
            timeout=5,
        )
        logger.info("Unloaded %s from Ollama memory.", OLLAMA_EMBED_MODEL)
    except Exception as e:
        logger.warning("Could not unload %s: %s", OLLAMA_EMBED_MODEL, e)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    collection = embed_corpus()
    print(f"\nCollection: {collection.name}, count: {collection.count()}")
    print("Embed complete. nomic-embed-text unloaded.")
