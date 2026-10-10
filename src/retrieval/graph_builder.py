# src/retrieval/graph_builder.py

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import spacy
import networkx as nx
from tqdm import tqdm
from config import Config


# load spaCy model — used for entity extraction
nlp = spacy.load("en_core_web_sm")


def extract_entities(text: str):
    """
    Extract named entities from a chunk using spaCy.

    spaCy identifies things like:
    - PERSON: Shirley Temple Black, Neil Armstrong
    - ORG: NASA, Apple, United Nations
    - GPE: France, New York, Soviet Union
    - WORK_OF_ART: Vostok 6, Harry Potter
    - DATE, EVENT, etc.

    We only keep PERSON, ORG, GPE, WORK_OF_ART —
    these are the entity types that form meaningful
    relationships in our dataset.
    """
    doc = nlp(text[:1000])  # limit to 1000 chars for speed

    keep_types = {"PERSON", "ORG", "GPE", "WORK_OF_ART", "EVENT"}
    entities = []

    for ent in doc.ents:
        if ent.label_ in keep_types:
            entities.append({
                "text": ent.text.strip(),
                "label": ent.label_
            })

    # deduplicate
    seen = set()
    unique = []
    for e in entities:
        if e["text"].lower() not in seen:
            seen.add(e["text"].lower())
            unique.append(e)

    return unique


def build_graph(chunks_path: str = Config.CHUNKS_PATH) -> nx.Graph:
    """
    Build a NetworkX graph from all chunks.

    Graph structure:
    - Nodes = entities (people, places, orgs)
    - Edges = co-occurrence in same chunk
      (if two entities appear in same chunk, they're connected)

    Why co-occurrence?
    If "Shirley Temple Black" and "Corliss Archer" appear
    in the same chunk, they're likely related. The edge
    between them lets us traverse from one to the other.

    Node attributes:
    - entity_type: PERSON, ORG, GPE etc.
    - chunks: list of chunk texts containing this entity
    - frequency: how many chunks mention this entity

    Edge attributes:
    - weight: how many chunks both entities co-occur in
    - chunks: the actual chunk texts where they co-occur
    """
    print("Loading chunks...")
    with open(chunks_path) as f:
        chunks = json.load(f)

    G = nx.Graph()

    print(f"Building graph from {len(chunks)} chunks...")
    for chunk in tqdm(chunks):
        text = chunk["text"]
        entities = extract_entities(text)

        # add each entity as a node
        for ent in entities:
            if G.has_node(ent["text"]):
                # update existing node
                G.nodes[ent["text"]]["frequency"] += 1
                G.nodes[ent["text"]]["chunks"].append(text)
            else:
                # create new node
                G.add_node(
                    ent["text"],
                    entity_type=ent["label"],
                    frequency=1,
                    chunks=[text]
                )

        # add edges between all entity pairs in this chunk
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                e1 = entities[i]["text"]
                e2 = entities[j]["text"]

                if G.has_edge(e1, e2):
                    # strengthen existing edge
                    G[e1][e2]["weight"] += 1
                    G[e1][e2]["chunks"].append(text)
                else:
                    # create new edge
                    G.add_edge(
                        e1, e2,
                        weight=1,
                        chunks=[text]
                    )

    print(f"Graph built:")
    print(f"  Nodes (entities): {G.number_of_nodes()}")
    print(f"  Edges (relations): {G.number_of_edges()}")
    return G


def save_graph(G: nx.Graph, path: str = Config.GRAPH_PATH):
    """Save graph to disk — build once, load every session."""
    import pickle
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f)
    print(f"Graph saved to {path}")


def load_graph(path: str = Config.GRAPH_PATH) -> nx.Graph:
    """Load graph from disk."""
    import pickle
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Graph not found at {path}. Run graph_builder.py first."
        )
    with open(path, "rb") as f:
        G = pickle.load(f)
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

if __name__ == "__main__":
    G = build_graph()
    save_graph(G)

    # inspect sample nodes
    print("\nSample entities in graph:")
    nodes = list(G.nodes(data=True))[:10]
    for name, data in nodes:
        print(f"  {name} ({data['entity_type']}) — freq: {data['frequency']}")

    # inspect sample edges
    print("\nSample relationships:")
    edges = list(G.edges(data=True))[:5]
    for e1, e2, data in edges:
        print(f"  {e1} ←→ {e2} (co-occurs {data['weight']} times)")