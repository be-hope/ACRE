# src/retrieval/hyde.py

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import numpy as np
from typing import List, Dict
from config import Config, llm_call_with_retry


def get_llm():
    return Ollama(
        model=Config.LLM_MODEL,
        base_url=Config.OLLAMA_BASE_URL,
        request_timeout=120.0
    )


def get_embed_model():
    return OllamaEmbedding(
        model_name=Config.EMBED_MODEL,
        base_url=Config.OLLAMA_BASE_URL
    )


def generate_hypothetical_document(query: str) -> str:
    """
    Step 1 of HyDE.

    Ask the LLM to write a fake answer to the question.
    We don't care if this answer is correct — we just need
    it to be written in the style of a real answer.

    Why? Because:
    - Question: "Who directed Inception?"
    - Hypothetical: "Inception was directed by Christopher Nolan,
      who also directed The Dark Knight trilogy..."
    
    The hypothetical sounds like a Wikipedia passage.
    Embedding it finds chunks that also sound like Wikipedia passages.
    Embedding the raw question finds chunks that sound like questions.
    """
    llm = get_llm()

    prompt = f"""Write a short factual passage (2-3 sentences) that 
directly answers the following question. 
Write it as if it were an excerpt from an encyclopedia or Wikipedia.
Do not say 'I' or refer to yourself. Just write the passage.

Question: {query}

Passage:"""

    response = llm_call_with_retry(llm, prompt)
    hypothetical = str(response).strip()
    return hypothetical


def hyde_retrieve(
    query: str,
    faiss_index,
    texts: List[str],
    metadatas: List[Dict],
    top_k: int = Config.TOP_K
) -> Dict:
    """
    Full HyDE retrieval pipeline.

    Steps:
    1. Generate hypothetical document from query
    2. Embed the hypothetical (not the query)
    3. Search FAISS with that embedding
    4. Return retrieved chunks + the hypothetical for inspection

    Returns dict with:
    - chunks: retrieved results (same format as normal retrieve())
    - hypothetical: the fake answer we generated
    """
    embed_model = get_embed_model()

    # Step 1 — generate hypothetical answer
    print(f"  Generating hypothetical document...")
    hypothetical = generate_hypothetical_document(query)
    print(f"  Hypothetical: {hypothetical[:100]}...")

    # Step 2 — embed the hypothetical instead of the query
    hyp_embedding = embed_model.get_text_embedding(hypothetical)
    hyp_array = np.array([hyp_embedding], dtype=np.float32)

    # Step 3 — search FAISS
    D, I = faiss_index.search(hyp_array, top_k)

    # Step 4 — collect results
    chunks = []
    for dist, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        chunks.append({
            "text": texts[idx],
            "metadata": metadatas[idx],
            "distance": float(dist)
        })

    return {
        "chunks": chunks,
        "hypothetical": hypothetical
    }


if __name__ == "__main__":
    from src.ingestion.embedder import load_index, retrieve
    from src.generation.generator import generate_answer

    print("Loading index...")
    faiss_index, texts, metadatas = load_index()

    # these are deliberately hard/ambiguous queries
    # phrasing is far from how answers appear in Wikipedia
    hard_queries = [
        # answer is a person's name but question asks about an achievement
        "Which Soviet citizen was the pioneering female space traveler?",
        
        # indirect phrasing — answer is a place but question asks about an event
        "Where did the allied forces conduct the largest amphibious assault in history?",
        
        # abstract phrasing — answer buried in technical text
        "What technology allows computers to understand human language?",
    ]

    for query in hard_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print('='*60)

        # normal retrieval
        normal_chunks = retrieve(query, faiss_index, texts, metadatas)
        normal_result = generate_answer(query, normal_chunks)

        # hyde retrieval
        hyde_result = hyde_retrieve(query, faiss_index, texts, metadatas)
        hyde_answer = generate_answer(query, hyde_result["chunks"])

        print(f"\nNORMAL answer:  {normal_result['answer'][:150]}")
        print(f"Normal top chunk: {normal_chunks[0]['text'][:100]}...")

        print(f"\nHyDE hypothetical: {hyde_result['hypothetical'][:150]}...")
        print(f"HyDE answer:    {hyde_answer['answer'][:150]}")
        print(f"HyDE top chunk: {hyde_result['chunks'][0]['text'][:100]}...")

        # did they retrieve different chunks?
        normal_top = normal_chunks[0]['text'][:50]
        hyde_top = hyde_result['chunks'][0]['text'][:50]
        different = "YES - HyDE found different chunks!" if normal_top != hyde_top else "NO - same chunks retrieved"
        print(f"\nDifferent retrieval? {different}")