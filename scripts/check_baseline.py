import os
import sys

# Add parent directory to path to allow importing app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.retrieval import DocumentRetriever
from app.generation import LLMGenerator
from app.rag_pipeline import RAGPipeline

def main():
    print("--- Baseline RAG Pipeline Check ---")
    
    # 1. Setup sample corpus
    sample_docs = [
        {"id": "doc1", "text": "The Eiffel Tower is located in Paris, France. It was completed in 1889."},
        {"id": "doc2", "text": "The Great Wall of China is a series of fortifications that were built across the historical northern borders of ancient Chinese states."},
        {"id": "doc3", "text": "The Louvre Museum is the world's largest art museum and a historic monument in Paris."}
    ]
    
    # 2. Initialize components
    print("Initializing components...")
    from app.db.faiss_store import FAISSStore
    store = FAISSStore()
    retriever = DocumentRetriever(store)
    generator = LLMGenerator()
    pipeline = RAGPipeline(retriever, generator)
    
    # 3. Add documents
    print("Adding documents...")
    pipeline.add_documents(sample_docs)
    
    # 4. Run sample query
    query = "Where is the Eiffel Tower and when was it built?"
    print(f"Running query: {query}")
    result = pipeline.run(query)
    
    print("\n--- Result ---")
    print(f"Query: {result['query']}")
    print(f"Answer: {result['answer']}")
    print("\nRetrieved Context:")
    for i, doc in enumerate(result['retrieved_context']):
        print(f"[{i+1}] {doc['text']}")

if __name__ == "__main__":
    main()
