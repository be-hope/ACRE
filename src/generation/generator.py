# src/generation/generator.py

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from llama_index.llms.ollama import Ollama
from typing import List, Dict

from config import Config, llm_call_with_retry



def get_llm():
    return Ollama(
        model=Config.LLM_MODEL,
        base_url=Config.OLLAMA_BASE_URL,
        request_timeout=300.0
    )


def build_prompt(query: str, retrieved_chunks: List[Dict]) -> str:
    """
    This is the core of RAG.
    We give the LLM retrieved context + question.
    'Answer only from context' forces it to work with
    what we retrieved, not its training knowledge.
    This is what makes RAG measurable and trustworthy.
    """
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        context_parts.append(f"[Document {i+1}]\n{chunk['text']}")

    context = "\n\n".join(context_parts)

    prompt = f"""You are a precise question-answering assistant.
Use ONLY the context provided below to answer the question.
If the answer is not in the context, say "I cannot find the answer in the provided context."
Do not use any outside knowledge.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

    return prompt


def generate_answer(query: str, retrieved_chunks: List[Dict]) -> Dict:
    """
    Full RAG generation:
    1. Build prompt with retrieved context
    2. Call LLM
    3. Return answer + metadata for evaluation later
    """
    llm = get_llm()
    prompt = build_prompt(query, retrieved_chunks)
    response = llm_call_with_retry(llm, prompt)
    answer = str(response).strip()

    return {
        "query": query,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "num_chunks_used": len(retrieved_chunks)
    }


if __name__ == "__main__":
    from src.ingestion.embedder import load_index, retrieve

    print("Loading index...")
    faiss_index, texts, metadatas = load_index()

    test_queries = [
        "Who was the first woman in space?",
        "What is the capital of France?",  # not in dataset, tests fallback
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print('='*60)

        chunks = retrieve(query, faiss_index, texts, metadatas)
        result = generate_answer(query, chunks)

        print(f"ANSWER: {result['answer']}")
        print(f"\nSOURCE CHUNKS USED: {result['num_chunks_used']}")
        for i, chunk in enumerate(result['retrieved_chunks']):
            print(f"  Chunk {i+1}: {chunk['text'][:100]}...")