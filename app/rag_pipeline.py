from typing import List, Dict, Any
from app.retrieval import DocumentRetriever
from app.generation import LLMGenerator

class RAGPipeline:
    """Orchestrates the retrieval and generation phases of a RAG system."""
    
    def __init__(self, retriever: DocumentRetriever, generator: LLMGenerator):
        self.retriever = retriever
        self.generator = generator
        
    def add_documents(self, docs: List[Dict[str, Any]]):
        """Add documents to the underlying retriever."""
        self.retriever.add_documents(docs)
        
    def run(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Run the full RAG pipeline: Retrieve -> Generate."""
        # 1. Retrieve
        retrieved_docs = self.retriever.search(query, top_k=top_k)
        
        # 2. Generate
        answer = self.generator.generate_answer(query, retrieved_docs)
        
        return {
            "query": query,
            "answer": answer,
            "retrieved_context": retrieved_docs
        }

from app.reranker import ReRanker
from app.verifier import SelfVerifier
from app.sanitizer import InputSanitizer

class ImprovedRAGPipeline(RAGPipeline):
    """An advanced RAG pipeline with Sanitization, Re-ranking, and Self-Verification."""
    
    def __init__(self, retriever: DocumentRetriever, generator: LLMGenerator):
        super().__init__(retriever, generator)
        self.reranker = ReRanker(cutoff_score=0.1)
        self.verifier = SelfVerifier(generator)
        self.sanitizer = InputSanitizer()
        
    def run(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        # 1. Sanitize Input
        sanitized_query, is_flagged = self.sanitizer.sanitize(query)
        
        if is_flagged:
            # Policy: Refuse or handle specifically
            return {
                "query": query,
                "answer": "I detected a potential safety violation in your input and cannot proceed.",
                "retrieved_context": [],
                "is_adverse": True
            }
            
        # 2. Retrieve
        retrieved_docs = self.retriever.search(sanitized_query, top_k=top_k)
        
        # 3. Re-rank and Filter
        refined_docs = self.reranker.rerank_and_filter(retrieved_docs, sanitized_query)
        
        # 4. Citation-Enforced Generation
        # We override the base prompt by modifying the generator's behavior slightly or providing a specific prompt
        context_text = "\n\n".join([f"[{i+1}] Source (ID: {doc['id']}): {doc['text']}" for i, doc in enumerate(refined_docs)])
        
        enhanced_prompt = f"""
        Answer the query using ONLY the provided context. 
        You MUST cite your sources using [number] notation.
        If the context does not contain the answer, state that.

        Context:
        {context_text}

        Query: {sanitized_query}
        """
        
        # For simplicity, we directly call the client here to avoid modifying generation.py's base prompt
        try:
            response = self.generator.client.chat.completions.create(
                model=self.generator.model,
                messages=[{"role": "user", "content": enhanced_prompt}],
                temperature=0.0
            )
            initial_answer = response.choices[0].message.content
        except Exception as e:
            initial_answer = f"Error: {str(e)}"
        
        # 5. Self-Verification
        final_answer = self.verifier.verify_and_correct(sanitized_query, initial_answer, refined_docs)
        
        return {
            "query": sanitized_query,
            "answer": final_answer,
            "retrieved_context": refined_docs,
            "initial_answer": initial_answer,
            "is_adverse": False
        }
