# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # ── LLM (Ollama - free, local) ────────────────────────
    OLLAMA_BASE_URL = "http://localhost:11434"
    LLM_MODEL = "llama3.2:3b"
    EMBED_MODEL = "nomic-embed-text"
    EMBED_DIM = 768

    # ── Chunking ──────────────────────────────────────────
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 64

    # ── Retrieval ─────────────────────────────────────────
    FAISS_INDEX_PATH = "data/faiss_index"
    TOP_K = 5

    # ── Graph ─────────────────────────────────────────────
    GRAPH_PATH = "data/graph.pkl"

    # ── MongoDB ───────────────────────────────────────────
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB = "acre"
    MONGO_COLLECTION = "documents"

    # ── Paths ─────────────────────────────────────────────
    RAW_DATA_DIR = "data/raw"
    PROCESSED_DATA_DIR = "data/processed"
    CHUNKS_PATH = "data/processed/chunks.json"

    # ── Evaluation ────────────────────────────────────────
    EVAL_SAMPLE_SIZE = 100