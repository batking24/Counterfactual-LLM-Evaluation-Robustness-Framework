from typing import List, Dict, Any
from app.retrieval import DocumentRetriever
from app.generation import LLMGenerator
from app.reranker import ReRanker
from app.verifier import SelfVerifier

class RAGPipeline:
    def __init__(self, retriever_model: str = "all-MiniLM-L6-v2", llm_model: str = "gpt-3.5-turbo"):
        self.retriever = DocumentRetriever(retriever_model)
        self.generator = LLMGenerator(llm_model)
        
    def add_documents(self, docs: List[Dict[str, Any]]):
        self.retriever.add_documents(docs)
        
    def run(self, query: str, top_k: int = 3, force_empty_context: bool = False, injected_noise: str = None) -> Dict[str, Any]:
        # For evaluation testing
        if force_empty_context:
            retrieved_docs = []
        else:
            retrieved_docs = self.retriever.search(query, top_k=top_k)
            
        if injected_noise and retrieved_docs:
            retrieved_docs[0]['text'] = retrieved_docs[0]['text'] + " " + injected_noise
            
        answer = self.generator.generate_answer(query, retrieved_docs)
        
        return {
            "query": query,
            "answer": answer,
            "retrieved_context": retrieved_docs
        }

class ImprovedRAGPipeline(RAGPipeline):
    def __init__(self, retriever_model: str = "all-MiniLM-L6-v2", llm_model: str = "gpt-3.5-turbo"):
        super().__init__(retriever_model, llm_model)
        self.reranker = ReRanker(cutoff_score=1.5)  # Strict cutoff to prevent hallucination
        self.verifier = SelfVerifier(self.generator)
        
    def run(self, query: str, top_k: int = 5, force_empty_context: bool = False, injected_noise: str = None) -> Dict[str, Any]:
        if force_empty_context:
            retrieved_docs = []
        else:
            retrieved_docs = self.retriever.search(query, top_k=top_k)
            
        if injected_noise and retrieved_docs:
            retrieved_docs[0]['text'] = retrieved_docs[0]['text'] + " " + injected_noise
            
        # 1. Filter and Re-rank
        refined_docs = self.reranker.rerank_and_filter(retrieved_docs, query)
        
        # 2. Base Generation
        initial_answer = self.generator.generate_answer(query, refined_docs)
        
        # 3. Self-Verification
        final_answer = self.verifier.verify_and_correct(query, initial_answer, refined_docs)
        
        return {
            "query": query,
            "answer": final_answer,
            "retrieved_context": refined_docs,
            "initial_answer": initial_answer
        }
