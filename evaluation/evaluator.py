# evaluation/simple_evaluator.py

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import pandas as pd
from config import Config
from src.ingestion.embedder import load_index, retrieve
from src.generation.generator import generate_answer


def exact_match(prediction: str, ground_truth: str) -> float:
    """Does the answer contain the ground truth?"""
    return float(ground_truth.lower().strip() in prediction.lower().strip())


def token_overlap(prediction: str, ground_truth: str) -> float:
    """
    F1 score based on word overlap.
    Same metric used in SQuAD benchmark.
    No LLM needed — pure string matching.
    """
    pred_tokens = set(prediction.lower().split())
    gt_tokens = set(ground_truth.lower().split())

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = pred_tokens & gt_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def context_hit(ground_truth: str, chunks: list) -> float:
    """
    Was the answer present in any retrieved chunk?
    1 = ground truth found in context
    0 = ground truth missing from context
    """
    combined = " ".join([c["text"].lower() for c in chunks])
    return float(ground_truth.lower().strip() in combined)


def run_simple_eval(num_samples: int = 5, label: str = "baseline", pipeline: str = "normal"):
    """
    Evaluate pipeline with zero network calls.
    Fast, reliable, works offline.
    """
    print(f"Loading data and index...")

    with open(os.path.join(Config.RAW_DATA_DIR, "hotpotqa.json")) as f:
        raw_data = json.load(f)

    faiss_index, texts, metadatas = load_index()
    
# pre-load graph if needed (avoid reloading per query)
    G = None
    if pipeline == "graph_rag":
        import pickle
        with open(Config.GRAPH_PATH, 'rb') as f:
            G = pickle.load(f)
        print("Graph loaded.")
        # use bridge questions — true multi-hop
    questions = [r for r in raw_data if r.get("type") == "bridge"][:num_samples]
    raptor_summaries = None
    if pipeline == "raptor":
        from src.retrieval.raptor import load_raptor_summaries
        raptor_summaries = load_raptor_summaries()
        print("RAPTOR summaries loaded.")
    results = []
    for i, record in enumerate(questions):
        query = record["question"]
        ground_truth = record["answer"]

        print(f"  [{i+1}/{num_samples}] {query[:60]}...")

        if pipeline == "hyde":
            from src.retrieval.hyde import hyde_retrieve
            hyde_result = hyde_retrieve(query, faiss_index, texts, metadatas)
            chunks = hyde_result["chunks"]
        elif pipeline == "router":
            from src.routing.router import route_query
            routing = route_query(query)
            print(f"    Routed to: {routing['pipeline']} ({routing['category']})")
            if routing["pipeline"] == "baseline":
                chunks = retrieve(query, faiss_index, texts, metadatas)
            else:
                # HyDE for relational/thematic until we build GraphRAG + RAPTOR
                from src.retrieval.hyde import hyde_retrieve
                hyde_result = hyde_retrieve(query, faiss_index, texts, metadatas)
                chunks = hyde_result["chunks"]
        elif pipeline == "self_rag":
            from src.verification.self_rag import self_rag_retrieve
            result = self_rag_retrieve(query, faiss_index, texts, metadatas)
            chunks = result["chunks"]


        elif pipeline == "raptor":
            from src.retrieval.raptor import raptor_retrieve, load_raptor_summaries
            if 'raptor_summaries' not in dir():
                    raptor_summaries = load_raptor_summaries()
            result = raptor_retrieve(query, faiss_index, texts, metadatas, raptor_summaries)
            chunks = result["chunks"]

        elif pipeline == "acre":
            from src.pipeline.acre import ACRE
            if 'acre_system' not in locals():
                acre_system = ACRE()
            acre_result = acre_system.query(query, verbose=False)
            chunks = acre_result["graded_chunks"]
            result = {"answer": acre_result["answer"]}
            answer = acre_result["answer"]
        elif pipeline == "graph_rag":
            from src.retrieval.graph_rag import graph_rag_retrieve
            result = graph_rag_retrieve(query, faiss_index, texts, metadatas, G)
            chunks = result["chunks"]
        else:
            chunks = retrieve(query, faiss_index, texts, metadatas)

        result = generate_answer(query, chunks)
        answer = result["answer"]

        results.append({
            "question": query,
            "ground_truth": ground_truth,
            "answer": answer,
            "exact_match": exact_match(answer, ground_truth),
            "f1_score": token_overlap(answer, ground_truth),
            "context_hit": context_hit(ground_truth, chunks),
        })

    df = pd.DataFrame(results)

    # summary
    print(f"\n{'='*50}")
    print(f"SCORES — {label.upper()} ({num_samples} samples)")
    print('='*50)
    print(f"  Exact Match:   {df['exact_match'].mean():.4f}")
    print(f"  F1 Score:      {df['f1_score'].mean():.4f}")
    print(f"  Context Hit:   {df['context_hit'].mean():.4f}")
    print('='*50)
    print("Scale: 0.0 (worst) → 1.0 (best)")

    # save
    os.makedirs("evaluation/results", exist_ok=True)
    df.to_csv(f"evaluation/results/{label}.csv", index=False)
    print(f"Saved to evaluation/results/{label}.csv")

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", default="normal", choices=["normal", "hyde","router", "self_rag", "graph_rag", "raptor", "acre"])
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args()

    run_simple_eval(
        num_samples=args.samples,
        label=args.label,
        pipeline=args.pipeline
    )