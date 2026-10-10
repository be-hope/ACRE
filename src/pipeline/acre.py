# src/pipeline/acre.py

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pickle
from typing import Dict
from config import Config
from src.ingestion.embedder import load_index, retrieve
from src.routing.router import route_query
from src.retrieval.hyde import hyde_retrieve
from src.retrieval.graph_rag import graph_rag_retrieve
from src.retrieval.raptor import raptor_retrieve, load_raptor_summaries
from src.verification.self_rag import grade_all_chunks
from src.generation.generator import generate_answer


class ACRE:
    """
    Adaptive Cognitive Retrieval Engine.

    Full pipeline:
    Query → Intent Router → [retrieval path] → Self-RAG gate → LLM → Answer

    Routing logic:
    SIMPLE    → FAISS baseline
    RELATIONAL → HyDE + GraphRAG
    THEMATIC   → HyDE + RAPTOR
    All paths  → Self-RAG verification → LLM generation
    """

    def __init__(self):
        print("Initializing ACRE...")

        # load FAISS index
        self.faiss_index, self.texts, self.metadatas = load_index()

        # load graph
        print("Loading graph...")
        with open(Config.GRAPH_PATH, "rb") as f:
            self.G = pickle.load(f)

        # load RAPTOR summaries
        print("Loading RAPTOR summaries...")
        self.raptor_summaries = load_raptor_summaries()

        print("ACRE ready.\n")

    def query(self, question: str, verbose: bool = True) -> Dict:
        """
        Full ACRE pipeline for a single query.
        Returns answer + metadata about routing and retrieval decisions.
        """
        if verbose:
            print(f"Query: {question}")
            print("-" * 50)

        # Step 1 — Intent Routing
        routing = route_query(question)
        category = routing["category"]
        if verbose:
            print(f"Route: {category} → {routing['pipeline']}")

        # Step 2 — Retrieval based on route
        if category == "SIMPLE":
            chunks = retrieve(
                question, self.faiss_index, self.texts, self.metadatas
            )
            retrieval_source = "faiss"

        elif category == "RELATIONAL":
            # HyDE to improve query embedding + GraphRAG for entity linking
            hyde_result = hyde_retrieve(
                question, self.faiss_index, self.texts, self.metadatas
            )
            graph_result = graph_rag_retrieve(
                question, self.faiss_index, self.texts, self.metadatas, self.G
            )
            # merge both — graph first for entity context, hyde for semantic
            seen = set()
            chunks = []
            for chunk in graph_result["chunks"] + hyde_result["chunks"]:
                key = chunk["text"][:80]
                if key not in seen:
                    seen.add(key)
                    chunks.append(chunk)
            chunks = chunks[:Config.TOP_K + 2]
            retrieval_source = "hyde+graphrag"

        else:  # THEMATIC
            # HyDE + RAPTOR summaries for broad thematic context
            hyde_result = hyde_retrieve(
                question, self.faiss_index, self.texts, self.metadatas
            )
            raptor_result = raptor_retrieve(
                question, self.faiss_index, self.texts,
                self.metadatas, self.raptor_summaries
            )
            seen = set()
            chunks = []
            for chunk in raptor_result["chunks"] + hyde_result["chunks"]:
                key = chunk["text"][:80]
                if key not in seen:
                    seen.add(key)
                    chunks.append(chunk)
            chunks = chunks[:Config.TOP_K + 3]
            retrieval_source = "hyde+raptor"

        if verbose:
            print(f"Retrieved: {len(chunks)} chunks via {retrieval_source}")

        # Step 3 — Self-RAG verification gate
        if verbose:
            print("Self-RAG grading...")
        graded_chunks = grade_all_chunks(question, chunks)

        if verbose:
            kept = len(graded_chunks)
            print(f"Kept: {kept}/{len(chunks)} chunks after grading")

        # Step 4 — LLM generation
        result = generate_answer(question, graded_chunks)

        if verbose:
            print(f"\nAnswer: {result['answer']}")

        return {
            "question": question,
            "answer": result["answer"],
            "route": category,
            "pipeline": routing["pipeline"],
            "retrieval_source": retrieval_source,
            "chunks_retrieved": len(chunks),
            "chunks_after_selfrag": len(graded_chunks),
            "graded_chunks": graded_chunks,
        }


if __name__ == "__main__":
    acre = ACRE()

    test_queries = [
        # SIMPLE
        "Who was the first woman in space?",
        # RELATIONAL
        "What government position was held by the woman who portrayed Corliss Archer?",
        # THEMATIC
        "What patterns exist in the careers of pioneering female aviators?",
    ]

    print("\n" + "="*60)
    print("ACRE — FULL PIPELINE TEST")
    print("="*60)

    for query in test_queries:
        print(f"\n{'='*60}")
        result = acre.query(query, verbose=True)
        print(f"Route taken: {result['route']} → {result['retrieval_source']}")
        print(f"Chunks: {result['chunks_retrieved']} retrieved → "
              f"{result['chunks_after_selfrag']} after Self-RAG")