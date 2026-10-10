# src/retrieval/graph_rag.py

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import spacy
import networkx as nx
from typing import List, Dict

from config import Config
from src.retrieval.graph_builder import load_graph

nlp = spacy.load("en_core_web_sm")


def extract_query_entities(query: str) -> List[str]:
    doc = nlp(query)

    keep_types = {"PERSON", "ORG", "GPE", "WORK_OF_ART", "EVENT"}

    entities = []
    for ent in doc.ents:
        if ent.label_ in keep_types:
            entities.append(ent.text.strip())

    if not entities:
        entities = [chunk.text.strip() for chunk in doc.noun_chunks][:3]

    return entities


def fuzzy_match_nodes(G: nx.Graph, query_ent: str) -> List[str]:
    """
    Only match nodes where the overlap is meaningful.
    """
    matched = []

    q = query_ent.lower()

    for node in G.nodes():
        n = str(node).lower()

        if len(n) >= 4 and (q in n or n in q):
            matched.append(node)

    return matched[:5]


def find_graph_neighbors(
    G: nx.Graph,
    query_entities: List[str],
    hops: int = 3,
    top_n: int = 10
) -> List[Dict]:
    """
    Traverse graph starting from all matched nodes.
    """

    collected_chunks = []
    visited_nodes = set()

    for query_ent in query_entities:

        seed_nodes = fuzzy_match_nodes(G, query_ent)

        if not seed_nodes:
            print(f"  Graph: no match found for '{query_ent}'")
            continue

        print(f"  Graph: '{query_ent}' matched to nodes: {seed_nodes[:5]}")

        frontier = set(seed_nodes)

        for hop in range(hops):

            next_frontier = set()

            for node in frontier:

                if node in visited_nodes:
                    continue

                visited_nodes.add(node)

                node_data = G.nodes[node]

                for chunk_text in node_data.get("chunks", []):

                    collected_chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "source": "graphrag",
                            "entity": node,
                            "hop": hop
                        },
                        "distance": 0.0,
                        "graph_entity": node
                    })

                for neighbor in G.neighbors(node):
                    if neighbor not in visited_nodes:
                        next_frontier.add(neighbor)

            frontier = next_frontier

    # Deduplicate chunks
    seen_texts = set()
    unique_chunks = []

    for chunk in collected_chunks:

        key = chunk["text"][:100]

        if key not in seen_texts:
            seen_texts.add(key)
            unique_chunks.append(chunk)

    return unique_chunks[:top_n]


def graph_rag_retrieve(
    query: str,
    faiss_index,
    texts: List[str],
    metadatas: List[Dict],
    G: nx.Graph = None
) -> Dict:

    from src.ingestion.embedder import retrieve

    if G is None:
        G = load_graph()

    query_entities = extract_query_entities(query)

    print(f"  Query entities: {query_entities}")

    # Only use graph for short entity names
    clean_entities = [
        e for e in query_entities
        if len(e.split()) <= 3 and len(e) >= 4
    ]

    graph_chunks = []

    if clean_entities:
        graph_chunks = find_graph_neighbors(
            G,
            clean_entities,
            hops=2,
            top_n=5
        )

    # Always retrieve FAISS chunks
    faiss_chunks = retrieve(
        query,
        faiss_index,
        texts,
        metadatas,
        top_k=Config.TOP_K
    )

    if not graph_chunks:

        print("  No clean graph results — using FAISS only")

        return {
            "chunks": faiss_chunks,
            "source": "faiss_fallback",
            "query_entities": query_entities
        }

    # Merge FAISS + GraphRAG results
    seen = set()
    combined = []

    for chunk in faiss_chunks + graph_chunks:

        key = chunk["text"][:80]

        if key not in seen:
            seen.add(key)
            combined.append(chunk)

    print(
        f"  Combined {len(faiss_chunks)} FAISS + "
        f"{len(graph_chunks)} graph chunks"
    )

    return {
        "chunks": combined[: Config.TOP_K + 2],
        "source": "graphrag",
        "query_entities": query_entities,
        "graph_chunk_count": len(graph_chunks)
    }


if __name__ == "__main__":

    from src.ingestion.embedder import load_index
    from src.generation.generator import generate_answer

    print("Loading index and graph...")

    faiss_index, texts, metadatas = load_index()
    G = load_graph()

    test_queries = [
        "What government position was held by the woman who portrayed Corliss Archer?",
        "Who was the first woman in space?",
        "What science fantasy young adult series is told in first person?",
    ]

    for query in test_queries:

        print("\n" + "=" * 60)
        print(f"QUERY: {query}")
        print("=" * 60)

        result = graph_rag_retrieve(
            query,
            faiss_index,
            texts,
            metadatas,
            G
        )

        print(f"\nSource: {result['source']}")
        print(f"Entities found: {result['query_entities']}")
        print(f"Chunks retrieved: {len(result['chunks'])}")

        print("Chunk entities:")

        for c in result["chunks"]:
            print(
                f"  [{c['metadata'].get('entity', 'faiss')}] "
                f"{c['text'][:80]}..."
            )

        answer = generate_answer(query, result["chunks"])

        print(f"\nAnswer: {answer['answer']}")