# src/ingestion/chunker.py

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from typing import List, Dict
import json
import os
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import Config


def build_documents(records: List[Dict], source_name: str) -> List[Document]:
    """
    Convert raw records into LlamaIndex Document objects.

    Why Document objects?
    LlamaIndex works with Documents as its standard unit.
    Metadata you attach here travels with every chunk — so
    when you retrieve a chunk later, you always know which
    question it came from, what dataset, what title.
    """
    documents = []

    for idx, record in enumerate(records):

        if "passages" in record:
            # HotpotQA format
            for passage in record["passages"]:
                doc = Document(
                    text=passage["text"],
                    metadata={
                        "source": source_name,
                        "record_id": idx,
                        "title": passage.get("title", ""),
                        "question": record["question"],
                        "answer": record["answer"],
                    }
                )
                documents.append(doc)

        elif "evidence" in record:
            # MultiHop-RAG format
            for ev in record.get("evidence", []):
                text = ev if isinstance(ev, str) else ev.get("text", "")
                if not text.strip():
                    continue
                doc = Document(
                    text=text,
                    metadata={
                        "source": source_name,
                        "record_id": idx,
                        "question": record["question"],
                        "answer": record["answer"],
                        "type": record.get("type", "unknown")
                    }
                )
                documents.append(doc)

    print(f"Built {len(documents)} documents from {len(records)} records")
    return documents


def chunk_documents(documents: List[Document]) -> List:
    """
    Split documents into chunks using SentenceSplitter.

    Why SentenceSplitter?
    - Respects sentence boundaries, no mid-sentence cuts
    - chunk_overlap means context isn't lost at chunk edges
    - Each node inherits parent document metadata

    Think of it like this: a 2000-word Wikipedia article
    becomes ~4 chunks of 512 tokens each, with 64 tokens
    of overlap between consecutive chunks.
    """
    splitter = SentenceSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
    )

    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Created {len(nodes)} chunks from {len(documents)} documents")
    return nodes


def save_chunks(nodes: List, path: str = Config.CHUNKS_PATH):
    """
    Save chunks to disk as JSON so we never re-chunk during dev.
    Chunking is fast but embedding is slow — we want to separate them.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    serialized = []
    for node in nodes:
        serialized.append({
            "text": node.text,
            "metadata": node.metadata,
            "node_id": node.node_id
        })

    with open(path, "w") as f:
        json.dump(serialized, f, indent=2)

    print(f"Saved {len(serialized)} chunks to {path}")


def load_chunks(path: str = Config.CHUNKS_PATH) -> List[Dict]:
    """Load previously saved chunks from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Chunks not found at {path}. Run chunker first."
        )
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    from data_loader import load_local

    # use 300 records for dev — fast to process, enough to test
    print("Loading HotpotQA...")
    records = load_local("hotpotqa")[:300]
    docs = build_documents(records, source_name="hotpotqa")
    nodes = chunk_documents(docs)
    save_chunks(nodes)

    # inspect a sample
    print("\nSample chunk:")
    print(f"  Text: {nodes[0].text[:200]}...")
    print(f"  Metadata: {nodes[0].metadata}")