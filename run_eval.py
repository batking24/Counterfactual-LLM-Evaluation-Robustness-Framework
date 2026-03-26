import json
import os
import argparse
from app.rag_pipeline import RAGPipeline
from app.evaluator import Evaluator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_jsonl(filepath):
    data = []
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            for line in f:
                data.append(json.loads(line))
    return data

def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return []

def run_evaluation(dataset_path: str):
    print("\n--- Initializing RAG Pipeline ---")
    pipeline = RAGPipeline()
    
    # Load corpus
    corpus_path = os.path.join(BASE_DIR, "data", "corpus", "knowledge_base.json")
    corpus = load_json(corpus_path)
    if corpus:
        print(f"Adding {len(corpus)} documents to FAISS index...")
        pipeline.add_documents(corpus)
    else:
        print("No corpus found. Please run scripts/generate_datasets.py and create corpus first.")
        return
        
    print("\n--- Initializing Evaluator ---")
    evaluator = Evaluator(pipeline)
    
    dataset = load_jsonl(dataset_path)
    if not dataset:
        print(f"No queries found at {dataset_path}. Exiting.")
        return
        
    print(f"\nRunning Evaluation on {len(dataset)} queries from {os.path.basename(dataset_path)}...")
    results = evaluator.evaluate(dataset)
    
    print("\n=== Evaluation Results ===")
    for res in results:
        print(f"Q: {res['query']}")
        print(f"A: {res['answer']}")
        print(f"Grounding Score: {res['grounding_score']} (Supported: {res['is_supported']})")
        print(f"Consistency Score: {res['consistency_score']}")
        print(f"Hit Rate: {res['hit_rate']}")
        print("-" * 50)
        
    print("\nEvaluation complete! Detailed metrics logged to outputs/logs/eval_logs.db")

def main():
    parser = argparse.ArgumentParser(description="Run RAG Evaluation framework.")
    parser.add_argument("--dataset", type=str, default="data/eval_queries.jsonl", help="Path to jsonl dataset.")
    args = parser.parse_args()
    
    dataset_filepath = os.path.join(BASE_DIR, args.dataset)
    run_evaluation(dataset_filepath)

if __name__ == "__main__":
    main()
