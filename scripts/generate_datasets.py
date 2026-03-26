import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
EVAL_QUERIES_FILE = os.path.join(DATA_DIR, "eval_queries.jsonl")
COUNTERFACTUAL_FILE = os.path.join(DATA_DIR, "counterfactual_queries.jsonl")

def write_jsonl(filepath, data):
    with open(filepath, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def generate_datasets():
    print("Generating Standard QA Evaluation Dataset...")
    
    # 1. Base dataset
    base_queries = [
        {
            "id": "q1",
            "query": "How many days of PTO do I get at Acme Corp?",
            "reference_doc_ids": ["doc_company_policy_01"],
            "variants": [
                "What is the maximum number of paid time off days allowed per year?",
                "Can you tell me the PTO limit for Acme Corp employees?"
            ]
        },
        {
            "id": "q2",
            "query": "Is the SuperWidget 2000 waterproof?",
            "reference_doc_ids": ["doc_company_product_01"],
            "variants": [
                "Can I submerge the SuperWidget 2000 in water?",
                "Does the SuperWidget 2000 survive underwater?"
            ]
        },
        {
            "id": "q3",
            "query": "Who founded Acme Corp and when?",
            "reference_doc_ids": ["doc_company_history_01"],
            "variants": [
                "What year was Acme Corp established and by whom?",
                "Give me the founders and founding year of Acme Corp."
            ]
        }
    ]
    
    write_jsonl(EVAL_QUERIES_FILE, base_queries)
    
    # 2. Counterfactual dataset (modifying context or query)
    print("Generating Counterfactual QA Evaluation Dataset...")
    counterfactual_queries = [
        {
            "id": "cf_q1",
            "base_query_id": "q1",
            "query": "How many days of PTO do I get if I've been at Acme Corp for 10 years?",
            "expected_behavior": "Should gracefully indicate that tenure-based PTO is not in the knowledge base."
        },
        {
            "id": "cf_q2",
            "base_query_id": "q2",
            "query": "Is the SuperWidget 2000 waterproof up to 100 meters?",
            "expected_behavior": "Should state it is only waterproof up to 50 meters, correcting the premise."
        }
    ]
    
    write_jsonl(COUNTERFACTUAL_FILE, counterfactual_queries)
    print("Successfully generated base and counterfactual datasets in data/")

if __name__ == "__main__":
    generate_datasets()
