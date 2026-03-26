from typing import List, Dict, Any
from app.retrieval import DocumentRetriever
from app.generation import LLMGenerator

class RAGPipeline:
    def __init__(self, retriever_model: str = "all-MiniLM-L6-v2", llm_model: str = "gpt-3.5-turbo"):
        self.retriever = DocumentRetriever(retriever_model)
        self.generator = LLMGenerator(llm_model)
        
    def add_documents(self, docs: List[Dict[str, Any]]):
        self.retriever.add_documents(docs)
        
    def run(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        retrieved_docs = self.retriever.search(query, top_k=top_k)
        answer = self.generator.generate_answer(query, retrieved_docs)
        
        return {
            "query": query,
            "answer": answer,
            "retrieved_context": retrieved_docs
        }
