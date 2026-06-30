1|"""
2|embed.py — Embed corpus chunks into ChromaDB using nomic-embed-text via Ollama.
3|
4|Loads chunks from docs/corpus/chunks.json, embeds each one using
5|nomic-embed-text through ChromaDB's OllamaEmbeddingFunction, and
6|persists the vector store to docs/corpus/chroma/.
7|
8|After embedding, unloads nomic-embed-text (keep_alive=0) to free RAM
9|for llama3.2 in later stages.
10|
11|Usage:
12|    python -m src.rag.embed
13|"""
14|
15|import json
16|import logging
17|import subprocess
18|
19|import chromadb
20|from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
21|
22|from config import CHROMA_COLLECTION_NAME, CHROMA_PATH, CHUNKS_FILE, OLLAMA_EMBED_MODEL
23|
24|logger = logging.getLogger(__name__)
25|
26|
27|def embed_corpus(chunks_file: str | None = None) -> chromadb.Collection:
28|    """
29|    Embed all chunks and persist to ChromaDB.
30|
31|    Args:
32|        chunks_file: path to chunks.json. Defaults to config.CHUNKS_FILE.
33|
34|    Returns:
35|        The ChromaDB collection (persisted to disk).
36|    """
37|    if chunks_file is None:
38|        chunks_file = str(CHUNKS_FILE)
39|
40|    # Load chunks
41|    with open(chunks_file) as f:
42|        chunks = json.load(f)
43|    logger.info("Loaded %d chunks from %s", len(chunks), chunks_file)
44|
45|    # Build ChromaDB with Ollama embedding
46|    embedding_fn = OllamaEmbeddingFunction(
47|        model_name=OLLAMA_EMBED_MODEL,
48|        url="http://localhost:11434/api/embeddings",
49|    )
50|
51|    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
52|
53|    # Delete existing collection if present (fresh rebuild)
54|    try:
55|        client.delete_collection(CHROMA_COLLECTION_NAME)
56|        logger.info("Deleted existing collection '%s'.", CHROMA_COLLECTION_NAME)
57|    except (ValueError, Exception):
58|        pass
59|
60|    collection = client.create_collection(
61|        name=CHROMA_COLLECTION_NAME,
62|        embedding_function=embedding_fn,
63|        metadata={"description": "NASA heliophysics terminology corpus"},
64|    )
65|
66|    # Prepare data for batch add
67|    ids = [f"chunk_{i}" for i in range(len(chunks))]
68|    documents = [c["text"] for c in chunks]
69|    metadatas = [
70|        {
71|            "source_url": c["source_url"],
72|            "source_label": c["source_label"],
73|            "chunk_index": c["chunk_index"],
74|        }
75|        for c in chunks
76|    ]
77|
78|    # Add in batches to avoid overloading the embedding endpoint
79|    batch_size = 20
80|    for i in range(0, len(documents), batch_size):
81|        batch_end = min(i + batch_size, len(documents))
82|        logger.info(
83|            "Embedding batch %d–%d of %d...", i + 1, batch_end, len(documents)
84|        )
85|        collection.add(
86|            ids=ids[i:batch_end],
87|            documents=documents[i:batch_end],
88|            metadatas=metadatas[i:batch_end],
89|        )
90|
91|    logger.info(
92|        "Collection '%s' created with %d documents.",
93|        CHROMA_COLLECTION_NAME,
94|        collection.count(),
95|    )
96|
97|    # Unload nomic-embed-text to free RAM
98|    _unload_embed_model()
99|
100|    return collection
101|
102|
103|def _unload_embed_model() -> None:
104|    """Release nomic-embed-text from Ollama memory via keep_alive=0."""
105|    try:
106|        import requests
107|        requests.post(
108|            "http://localhost:11434/api/generate",
109|            json={"model": OLLAMA_EMBED_MODEL, "keep_alive": 0, "prompt": ""},
110|            timeout=5,
111|        )
112|        logger.info("Unloaded %s from Ollama memory.", OLLAMA_EMBED_MODEL)
113|    except Exception as e:
114|        logger.warning("Could not unload %s: %s", OLLAMA_EMBED_MODEL, e)
115|
116|
117|if __name__ == "__main__":
118|    logging.basicConfig(
119|        level=logging.INFO,
120|        format="%(asctime)s [%(levelname)s] %(message)s",
121|    )
122|    collection = embed_corpus()
123|    print(f"\nCollection: {collection.name}, count: {collection.count()}")
124|    print("Embed complete. nomic-embed-text unloaded.")
125|