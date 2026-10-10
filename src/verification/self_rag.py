# src/verification/self_rag.py

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from llama_index.llms.ollama import Ollama
from typing import List, Dict
from config import Config, llm_call_with_retry


def get_llm():
    return Ollama(
        model=Config.LLM_MODEL,
        base_url=Config.OLLAMA_BASE_URL,
        request_timeout=120.0
    )


def grade_chunk(query: str, chunk: Dict) -> Dict:
    """
    Grade a single retrieved chunk for relevance.

    This is the core of Self-RAG — before sending chunks
    to the LLM for generation, we ask the LLM itself:
    "Is this chunk actually useful for answering the query?"

    Why this helps:
    FAISS retrieves by vector similarity — it finds chunks
    that are semantically close to the query. But semantically
    close doesn't always mean useful for answering.

    Example:
    Query: "Who founded Apple?"
    FAISS might retrieve a chunk about Apple's products,
    Apple's revenue, Steve Jobs' death — all semantically
    related but none directly answering the question.
    Self-RAG filters these out.

    Grades:
    - RELEVANT: chunk directly helps answer the query
    - PARTIAL: chunk has some useful info but not complete
    - IRRELEVANT: chunk does not help answer the query
    """
    llm = get_llm()

    prompt = f"""You are a relevance grading system.
Given a query and a document chunk, grade how useful the chunk is for answering the query.

Reply with ONLY one word from these options:
RELEVANT - chunk directly contains information needed to answer the query
PARTIAL - chunk contains some related information but not a complete answer
IRRELEVANT - chunk does not help answer the query at all

Query: {query}

Chunk: {chunk['text'][:500]}

Grade:"""

    response = llm_call_with_retry(llm, prompt)
    raw = str(response).strip().upper()

    if "RELEVANT" in raw and "IRRELEVANT" not in raw:
        grade = "RELEVANT"
    elif "PARTIAL" in raw:
        grade = "PARTIAL"
    else:
        grade = "IRRELEVANT"

    return {
        "text": chunk["text"],
        "metadata": chunk["metadata"],
        "distance": chunk.get("distance", 0),
        "grade": grade,
        "kept": grade in ["RELEVANT", "PARTIAL"]
    }


def grade_all_chunks(query: str, chunks: List[Dict]) -> List[Dict]:
    """
    Grade all retrieved chunks and filter out irrelevant ones.

    Process:
    1. Grade each chunk individually
    2. Keep RELEVANT and PARTIAL chunks
    3. If nothing passes — fall back to top 2 by distance
       (better to have something than nothing)
    """
    print(f"  Grading {len(chunks)} chunks...")
    graded = []

    for i, chunk in enumerate(chunks):
        result = grade_chunk(query, chunk)
        graded.append(result)
        print(f"    Chunk {i+1}: {result['grade']} — {chunk['text'][:60]}...")

    # filter to kept chunks
    kept = [c for c in graded if c["kept"]]

    # fallback — if everything was filtered, keep top 2
    if not kept:
        print("  No chunks passed grading — falling back to top 2")
        kept = graded[:2]
        for c in kept:
            c["kept"] = True

    print(f"  Kept {len(kept)}/{len(chunks)} chunks after grading")
    return kept


def self_rag_retrieve(
    query: str,
    faiss_index,
    texts: List[str],
    metadatas: List[Dict],
) -> Dict:
    """
    Full Self-RAG retrieval pipeline:
    1. Retrieve top_k chunks via FAISS
    2. Grade each chunk for relevance
    3. Filter to only relevant/partial chunks
    4. Return filtered chunks for generation

    The key insight: we retrieve MORE chunks than needed (top_k)
    then filter down to only the useful ones.
    This gives the grader enough to work with while keeping
    generation focused on high quality context.
    """
    from src.ingestion.embedder import retrieve

    # retrieve more chunks than usual — grader will filter
    chunks = retrieve(
        query, faiss_index, texts, metadatas,
        top_k=Config.TOP_K + 3   # get 8 chunks, filter to best
    )

    graded_chunks = grade_all_chunks(query, chunks)

    return {
        "chunks": graded_chunks,
        "total_retrieved": len(chunks),
        "total_kept": len(graded_chunks),
        "grades": [c["grade"] for c in graded_chunks]
    }


if __name__ == "__main__":
    from src.ingestion.embedder import load_index
    from src.generation.generator import generate_answer

    print("Loading index...")
    faiss_index, texts, metadatas = load_index()

    test_queries = [
        "Who was the first woman in space?",
        "What government position was held by the woman who portrayed Corliss Archer?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print('='*60)

        result = self_rag_retrieve(query, faiss_index, texts, metadatas)

        print(f"\nRetrieval summary:")
        print(f"  Retrieved: {result['total_retrieved']} chunks")
        print(f"  Kept:      {result['total_kept']} chunks")
        print(f"  Grades:    {result['grades']}")

        answer = generate_answer(query, result["chunks"])
        print(f"\nFinal answer: {answer['answer']}")