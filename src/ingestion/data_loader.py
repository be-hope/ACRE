# src/ingestion/data_loader.py


from datasets import load_dataset
import json
import os
from tqdm import tqdm
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import Config

def download_multihop_rag(save_dir: str = Config.RAW_DATA_DIR):
    """
    MultiHop-RAG dataset — each question needs facts from 2+ documents.
    We save it locally so we never re-download during dev.
    """
    print("Downloading MultiHop-RAG...")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "multihop_rag.json")

    # skip download if already exists
    if os.path.exists(save_path):
        print(f"Already exists at {save_path}, skipping download.")
        return load_local("multihop_rag")

    dataset = load_dataset("yixuantt/MultiHopRAG","MultiHopRAG", split="train")

    records = []
    for item in tqdm(dataset):
        records.append({
            "question": item["query"],
            "answer": item["answer"],
            "evidence": item.get("evidence", []),
            "type": item.get("query_type", "unknown")
        })

    with open(save_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Saved {len(records)} records to {save_path}")
    return records


def download_hotpotqa(save_dir: str = Config.RAW_DATA_DIR):
    """
    HotpotQA — Wikipedia multi-hop QA.
    We use the validation split since test labels are hidden.
    """
    print("Downloading HotpotQA...")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "hotpotqa.json")

    if os.path.exists(save_path):
        print(f"Already exists at {save_path}, skipping download.")
        return load_local("hotpotqa")

    dataset = load_dataset("hotpot_qa", "distractor", split="validation")

    records = []
    for item in tqdm(dataset):
        passages = []
        for title, sentences in zip(
            item["context"]["title"],
            item["context"]["sentences"]
        ):
            passages.append({
                "title": title,
                "text": " ".join(sentences)
            })

        records.append({
            "question": item["question"],
            "answer": item["answer"],
            "passages": passages,
            "type": item["type"],
            "level": item["level"]
        })

    with open(save_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Saved {len(records)} records to {save_path}")
    return records


def load_local(dataset_name: str):
    """Load already-downloaded dataset from disk."""
    path = os.path.join(Config.RAW_DATA_DIR, f"{dataset_name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run download first."
        )
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    download_multihop_rag()
    download_hotpotqa()
    print("\nAll datasets ready.")