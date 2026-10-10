# src/routing/router.py

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from llama_index.llms.ollama import Ollama
from typing import Dict
from config import Config


def get_llm():
    return Ollama(
        model=Config.LLM_MODEL,
        base_url=Config.OLLAMA_BASE_URL,
        request_timeout=120.0
    )


def classify_query(query: str) -> Dict:
    """
    Classify a query into one of three routing buckets.

    SIMPLE — factual, single hop, one document has the answer
    Examples:
      "When was Einstein born?"
      "Who directed Inception?"
      "What is the capital of Japan?"

    RELATIONAL — multi-hop, needs connecting facts across documents
    Examples:
      "What city was the director of Inception born in?"
      "Which university did the first woman in space attend?"
      "What award did the author of Harry Potter win?"

    THEMATIC — broad, conceptual, needs summarized understanding
    Examples:
      "What are the main themes in Shakespeare's tragedies?"
      "How did the cold war influence space exploration?"
      "What patterns exist in Nobel Prize winners backgrounds?"

    Why does this matter?
    - SIMPLE queries waste compute if sent through HyDE + GraphRAG
    - RELATIONAL queries need graph traversal to connect entities
    - THEMATIC queries need RAPTOR summaries for broad understanding
    - Routing saves time and improves answer quality per query type
    """
    llm = get_llm()


    prompt = f"""You are a query classification system for a RAG pipeline.
Classify the following query into exactly one of these three categories:

SIMPLE - Single fact lookup. Answer comes from one document.
  Examples:
  - "Who was the first woman in space?" → SIMPLE
  - "When was the Eiffel Tower built?" → SIMPLE
  - "What is the capital of France?" → SIMPLE

RELATIONAL - Requires connecting two or more facts across different documents.
  Signal words: "who starred in", "directed by", "founded by", "born in", "worked at"
  combined with another question about that person/thing.
  Examples:
  - "What position was held by the woman who starred in Juno?" → RELATIONAL
  - "Which university did the iPhone company founder attend?" → RELATIONAL
  - "What award did the author of Harry Potter win?" → RELATIONAL

THEMATIC - Broad conceptual question needing patterns across many documents.
  Signal words: "patterns", "influence", "impact", "themes", "how did X affect Y"
  Examples:
  - "What patterns exist in pioneering female scientists careers?" → THEMATIC
  - "How did World War II influence technological development?" → THEMATIC
  - "What are common traits among Nobel Prize winners?" → THEMATIC

Rules:
- Reply with ONLY one word: SIMPLE, RELATIONAL, or THEMATIC
- If the query asks about a person/thing CONNECTED to another person/thing → RELATIONAL
- If the query asks about broad influence, patterns, or impact → THEMATIC
- Only use SIMPLE for direct single-fact lookups

Query: {query}

Category:"""



    response = llm_call_with_retry(llm, prompt)
    raw = str(response).strip().upper()

    # clean up in case LLM adds extra text
    if "RELATIONAL" in raw:
        category = "RELATIONAL"
    elif "THEMATIC" in raw:
        category = "THEMATIC"
    else:
        category = "SIMPLE"  # default to simple if unclear

    return {
        "query": query,
        "category": category,
        "raw_response": raw
    }


def route_query(query: str) -> Dict:
    """
    Full routing decision with pipeline recommendation.

    Returns:
    - category: SIMPLE / RELATIONAL / THEMATIC
    - pipeline: which retrieval pipeline to use
    - reason: human readable explanation
    """
    result = classify_query(query)
    category = result["category"]

    # routing table
    if category == "SIMPLE":
        pipeline = "baseline"
        reason = "Single fact lookup — standard FAISS retrieval is sufficient"
    elif category == "RELATIONAL":
        pipeline = "hyde_graph"
        reason = "Multi-hop query — needs HyDE + GraphRAG to connect entities"
    else:  # THEMATIC
        pipeline = "raptor"
        reason = "Broad conceptual query — needs RAPTOR summaries"

    return {
        "query": query,
        "category": category,
        "pipeline": pipeline,
        "reason": reason
    }


if __name__ == "__main__":
    test_queries = [
        # should be SIMPLE
        "Who was the first woman in space?",
        "When was the Eiffel Tower built?",

        # should be RELATIONAL
        "What government position was held by the woman who starred in the movie Juno?",
        "Which university did the founder of the company that makes iPhone attend?",

        # should be THEMATIC
        "What patterns exist in the careers of pioneering female scientists?",
        "How did World War II influence technological development?",
    ]

    print("INTENT ROUTER TEST")
    print("="*60)

    correct = 0
    expected = ["SIMPLE", "SIMPLE", "RELATIONAL", "RELATIONAL", "THEMATIC", "THEMATIC"]

    for query, exp in zip(test_queries, expected):
        result = route_query(query)
        match = "✅" if result["category"] == exp else "❌"
        print(f"\n{match} Query: {query[:60]}...")
        print(f"   Expected:  {exp}")
        print(f"   Got:       {result['category']}")
        print(f"   Pipeline:  {result['pipeline']}")

    print("\n" + "="*60)