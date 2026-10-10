# src/ingestion/embedder.py

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import TextNode
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
import numpy as np
from typing import List, Dict
from tqdm import tqdm
import os
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import Config


def get_embed_model():
    """
    Initialize the Ollama embedding model.
    nomic-embed-text produces 768-dim embeddings —
    good quality, runs fully local, no API key needed.
    """
    return OllamaEmbedding(
        model_name=Config.EMBED_MODEL,
        base_url=Config.OLLAMA_BASE_URL
    )


def build_faiss_index(chunks: List[Dict]):
    """
    Embed all chunks and store in a FAISS index.

    What is FAISS?
    Facebook AI Similarity Search — an extremely fast library
    for finding nearest neighbors in vector space. When you
    query with an embedding, it returns the most similar
    chunk embeddings in milliseconds even over millions of vectors.

    What is IndexFlatL2?
    The simplest FAISS index — exact nearest neighbor search
    using L2 (Euclidean) distance. No approximation, fully accurate.
    Fine for our dataset size. For millions of vectors you'd
    switch to IndexIVFFlat (approximate but faster).

    Flow:
    chunk text → embedding model → 768-dim float vector
    → stored in FAISS index with its position as the ID
    """
    embed_model = get_embed_model()

    print(f"Embedding {len(chunks)} chunks...")
    print("This will take a few minutes on CPU — grab a coffee.")

    # build FAISS index
    dimension = Config.EMBED_DIM  # 768 for nomic-embed-text
    faiss_index = faiss.IndexFlatL2(dimension)

    # we need to keep chunk texts alongside the index
    # because FAISS only stores vectors, not the original text
    texts = []
    metadatas = []
    embeddings = []

    for chunk in tqdm(chunks):
        text = chunk["text"]
        embedding = embed_model.get_text_embedding(text)
        embeddings.append(embedding)
        texts.append(text)
        metadatas.append(chunk["metadata"])

    # convert to numpy array — FAISS requires float32
    embeddings_array = np.array(embeddings, dtype=np.float32)
    faiss_index.add(embeddings_array)

    print(f"FAISS index built with {faiss_index.ntotal} vectors")
    return faiss_index, texts, metadatas


def save_index(faiss_index, texts, metadatas):
    """
    Save FAISS index + texts + metadata to disk.
    Critical for evening workflow — you embed once, reload every session.
    Embedding 1000 chunks takes ~5 mins. Loading from disk takes 2 seconds.
    """
    import json
    os.makedirs(Config.FAISS_INDEX_PATH, exist_ok=True)

    # save FAISS binary index
    faiss.write_index(
        faiss_index,
        os.path.join(Config.FAISS_INDEX_PATH, "index.bin")
    )

    # save texts and metadata alongside
    with open(os.path.join(Config.FAISS_INDEX_PATH, "texts.json"), "w") as f:
        json.dump({"texts": texts, "metadatas": metadatas}, f)

    print(f"Index saved to {Config.FAISS_INDEX_PATH}")


def load_index():
    """
    Load FAISS index from disk.
    Call this at the start of every session instead of re-embedding.
    """
    import json
    index_path = os.path.join(Config.FAISS_INDEX_PATH, "index.bin")
    texts_path = os.path.join(Config.FAISS_INDEX_PATH, "texts.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            "FAISS index not found. Run embedder.py first."
        )

    faiss_index = faiss.read_index(index_path)

    with open(texts_path) as f:
        data = json.load(f)

    print(f"Loaded FAISS index with {faiss_index.ntotal} vectors")
    return faiss_index, data["texts"], data["metadatas"]


def retrieve(query: str, faiss_index, texts, metadatas, top_k=Config.TOP_K):
    """
    Given a query string, find the top_k most relevant chunks.

    Steps:
    1. Embed the query using the same model used for chunks
    2. Search FAISS for nearest neighbors
    3. Return the corresponding texts + metadata

    This is the core retrieval operation — everything else
    (HyDE, RAPTOR, GraphRAG) is built on top of this.
    """
    embed_model = get_embed_model()
    query_embedding = embed_model.get_text_embedding(query)
    query_array = np.array([query_embedding], dtype=np.float32)

    # D = distances, I = indices of nearest neighbors
    D, I = faiss_index.search(query_array, top_k)

    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx == -1:   # FAISS returns -1 if not enough vectors
            continue
        results.append({
            "text": texts[idx],
            "metadata": metadatas[idx],
            "distance": float(dist)   # lower = more similar
        })

    return results


if __name__ == "__main__":
    from chunker import load_chunks

    chunks = load_chunks()
    faiss_index, texts, metadatas = build_faiss_index(chunks)
    save_index(faiss_index, texts, metadatas)

    # quick retrieval test
    print("\nTesting retrieval...")
    results = retrieve(
        "Who was the first person to walk on the moon?",
        faiss_index, texts, metadatas
    )
    for i, r in enumerate(results):
        print(f"\nResult {i+1} (distance: {r['distance']:.4f}):")
        print(f"  {r['text'][:200]}...")