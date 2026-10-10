# src/retrieval/raptor.py

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import numpy as np
from typing import List, Dict
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from config import Config


def get_llm():
    return Ollama(
        model=Config.LLM_MODEL,
        base_url=Config.OLLAMA_BASE_URL,
        request_timeout=300.0
    )


def get_embed_model():
    return OllamaEmbedding(
        model_name=Config.EMBED_MODEL,
        base_url=Config.OLLAMA_BASE_URL
    )


def cluster_chunks(chunks: List[Dict], n_clusters: int = 5) -> List[List[Dict]]:
    """
    Group chunks into clusters by embedding similarity.
    
    RAPTOR's core idea: instead of retrieving individual chunks,
    first group related chunks together, then summarize each group.
    This gives you a higher-level understanding of the corpus.

    We use simple kmeans-style clustering on embeddings.
    Full RAPTOR uses UMAP + Gaussian Mixture — we simplify to
    numpy kmeans to avoid extra dependencies.
    """
    embed_model = get_embed_model()

    print(f"  Embedding {len(chunks)} chunks for clustering...")
    embeddings = []
    for chunk in chunks:
        emb = embed_model.get_text_embedding(chunk["text"][:300])
        embeddings.append(emb)

    embeddings = np.array(embeddings, dtype=np.float32)

    # simple kmeans clustering in numpy
    n_clusters = min(n_clusters, len(chunks))
    # random init centroids
    rng = np.random.default_rng(42)
    centroid_indices = rng.choice(len(embeddings), n_clusters, replace=False)
    centroids = embeddings[centroid_indices]

    for _ in range(10):  # 10 iterations is enough
        # assign each chunk to nearest centroid
        diffs = embeddings[:, None, :] - centroids[None, :, :]
        distances = np.linalg.norm(diffs, axis=2)
        assignments = np.argmin(distances, axis=1)

        # update centroids
        new_centroids = np.array([
            embeddings[assignments == k].mean(axis=0)
            if (assignments == k).any()
            else centroids[k]
            for k in range(n_clusters)
        ])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    # group chunks by cluster
    clusters = [[] for _ in range(n_clusters)]
    for i, chunk in enumerate(chunks):
        clusters[assignments[i]].append(chunk)

    # remove empty clusters
    clusters = [c for c in clusters if c]
    print(f"  Created {len(clusters)} clusters")
    return clusters


def summarize_cluster(cluster: List[Dict]) -> Dict:
    """
    Summarize a cluster of related chunks into one summary node.
    
    This is the RAPTOR 'tree building' step.
    Instead of storing raw chunks, we store summaries that
    capture the theme of a group of related documents.
    
    When retrieved, these summaries give the LLM a
    higher-level view — useful for thematic questions like
    'What patterns exist in X?' rather than specific fact lookups.
    """
    llm = get_llm()

    # combine cluster texts
    combined = "\n\n".join([c["text"][:300] for c in cluster])

    prompt = f"""You are a summarization assistant.
Read the following passages and write a concise 2-3 sentence summary
that captures the main theme or topic shared across these passages.
Write only the summary, nothing else.

Passages:
{combined[:2000]}

Summary:"""

    response = llm.complete(prompt)
    summary_text = str(response).strip()

    return {
        "text": summary_text,
        "metadata": {
            "source": "raptor_summary",
            "cluster_size": len(cluster),
            "original_sources": [c["metadata"].get("title", "") for c in cluster[:3]]
        },
        "distance": 0.0,
        "is_summary": True
    }


def build_raptor_summaries(
    chunks_path: str = Config.CHUNKS_PATH,
    save_path: str = "data/raptor_summaries.json",
    n_clusters: int = 5
) -> List[Dict]:
    """
    Build RAPTOR summary nodes from all chunks.
    Run once, save to disk — same pattern as FAISS index.
    
    Full RAPTOR builds a recursive tree (summaries of summaries).
    We implement one level — still captures thematic structure
    with much less complexity.
    """
    print("Loading chunks...")
    with open(chunks_path) as f:
        chunks = json.load(f)

    # sample chunks for speed — clustering all 2991 would be slow
    sample_size = min(200, len(chunks))
    sampled = chunks[:sample_size]
    print(f"Using {sample_size} chunks for RAPTOR tree")

    # cluster
    clusters = cluster_chunks(sampled, n_clusters=n_clusters)

    # summarize each cluster
    summaries = []
    for i, cluster in enumerate(clusters):
        print(f"  Summarizing cluster {i+1}/{len(clusters)} ({len(cluster)} chunks)...")
        summary = summarize_cluster(cluster)
        summaries.append(summary)
        print(f"  Summary: {summary['text'][:100]}...")

    # save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"\nSaved {len(summaries)} RAPTOR summaries to {save_path}")
    return summaries


def load_raptor_summaries(
    save_path: str = "data/raptor_summaries.json"
) -> List[Dict]:
    if not os.path.exists(save_path):
        raise FileNotFoundError(
            f"RAPTOR summaries not found at {save_path}. Run build_raptor_summaries() first."
        )
    with open(save_path) as f:
        summaries = json.load(f)
    print(f"Loaded {len(summaries)} RAPTOR summaries")
    return summaries


def raptor_retrieve(
    query: str,
    faiss_index,
    texts: List[str],
    metadatas: List[Dict],
    summaries: List[Dict] = None
) -> Dict:
    """
    RAPTOR retrieval pipeline:
    1. Get FAISS chunks (specific facts)
    2. Add RAPTOR summary nodes (thematic context)
    3. Combine — LLM gets both specific and broad context
    
    This is most useful for thematic queries like:
    'What patterns exist in X?' or 'How did Y influence Z?'
    where a single chunk won't have the full picture.
    """
    from src.ingestion.embedder import retrieve

    if summaries is None:
        summaries = load_raptor_summaries()

    # get specific chunks from FAISS
    faiss_chunks = retrieve(query, faiss_index, texts, metadatas, top_k=4)

    # add all summary nodes as additional context
    # (in full RAPTOR you'd embed summaries and retrieve top-k)
    # simplified: just include all summaries since we only have 5
    combined = faiss_chunks + summaries

    # deduplicate
    seen = set()
    unique = []
    for chunk in combined:
        key = chunk["text"][:80]
        if key not in seen:
            seen.add(key)
            unique.append(chunk)

    return {
        "chunks": unique[:Config.TOP_K + 3],
        "source": "raptor",
        "faiss_count": len(faiss_chunks),
        "summary_count": len(summaries)
    }


if __name__ == "__main__":
    from src.ingestion.embedder import load_index
    from src.generation.generator import generate_answer

    # Step 1 — build summaries (run once)
    print("Building RAPTOR summaries...")
    summaries = build_raptor_summaries(n_clusters=5)

    # Step 2 — test retrieval
    print("\nLoading index...")
    faiss_index, texts, metadatas = load_index()

    test_queries = [
        "What patterns exist in the careers of pioneering female aviators?",
        "Who was the first woman in space?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print('='*60)

        result = raptor_retrieve(query, faiss_index, texts, metadatas, summaries)
        print(f"Chunks: {result['faiss_count']} FAISS + {result['summary_count']} summaries")

        answer = generate_answer(query, result["chunks"])
        print(f"Answer: {answer['answer']}")