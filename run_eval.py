import os
import json
import argparse
from app.retrieval import DocumentRetriever
from app.generation import LLMGenerator
from app.rag_pipeline import RAGPipeline, ImprovedRAGPipeline
from app.evaluator import Evaluator

def run_benchmarks(provider="openai", models=None):
    print("======================================================")
    print(f"🚀 LLM EVALUATION & BENCHMARKING (Mode: {provider.upper()})")
    print("======================================================")
    
    # 1. Load Benchmarks
    catalog_path = "data/benchmarks/catalog.json"
    if not os.path.exists(catalog_path):
        print("Catalog not found. Running loader...")
        os.system(".venv/bin/python scripts/load_benchmarks.py")
        
    with open(catalog_path, "r") as f:
        benchmarks = json.load(f)
        
    # Default models if none provided
    if not models:
        models = ["gpt-3.5-turbo"] if provider == "openai" else ["llama3.3"]
        
    all_model_results = {}
    
    for model_name in models:
        print(f"\n" + "#"*60)
        print(f"🤖 EVALUATING MODEL: {model_name}")
        print("#"*60)
        
        model_results = {}
        for ds_name, data in benchmarks.items():
            print(f"\n📁 Dataset: {ds_name}")
            
            corpus = data["corpus"]
            queries = data["queries"]
            
            # Setup Pipelines
            retriever = DocumentRetriever()
            retriever.add_documents(corpus)
            generator = LLMGenerator(provider=provider, model=model_name)
            
            base_pipeline = RAGPipeline(retriever, generator)
            improved_pipeline = ImprovedRAGPipeline(retriever, generator)
            
            base_evaluator = Evaluator(base_pipeline, provider=provider, model=model_name)
            improved_evaluator = Evaluator(improved_pipeline, provider=provider, model=model_name)
            
            print(f"  - Running Baseline...")
            base_results = base_evaluator.evaluate(queries)
            print(f"  - Running Improved...")
            improved_results = improved_evaluator.evaluate(queries)
            
            from app.reporting import Reporter
            reporter = Reporter()
            summary = reporter.generate_summary(base_results, improved_results)
            
            model_results[ds_name] = {
                "baseline": base_results,
                "improved": improved_results,
                "summary": summary
            }
        
        all_model_results[model_name] = model_results
        
        # Save intermediate per-model results
        out_path = f"outputs/reports/models/{model_name}"
        os.makedirs(out_path, exist_ok=True)
        with open(f"{out_path}/eval_results.json", "w") as f:
            json.dump(model_results, f, indent=4)

    # 2. Save global consolidated results
    os.makedirs("outputs/reports", exist_ok=True)
    with open("outputs/reports/consolidated_model_results.json", "w") as f:
        json.dump(all_model_results, f, indent=4)
        
    # Maintain backwards compatibility for the dashboard by saving the first model's results to the old path
    first_model = models[0]
    with open("outputs/reports/eval_results.json", "w") as f:
        json.dump(all_model_results[first_model], f, indent=4)

    print(f"\n✅ All multi-model results saved to outputs/reports/consolidated_model_results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Use local Ollama instead of OpenAI")
    parser.add_argument("--models", nargs="+", help="List of models to benchmark (e.g. llama3.3 phi4 mistral-nemo gemma2)")
    args = parser.parse_args()
    
    mode = "ollama" if args.local else "openai"
    run_benchmarks(provider=mode, models=args.models)
