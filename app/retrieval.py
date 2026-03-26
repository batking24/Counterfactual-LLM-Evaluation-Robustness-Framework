import faiss
import numpy as np
import logging
from typing import List, Dict, Any

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logging.warning("sentence-transformers not installed. Retrieval operations will fail.")
    SentenceTransformer = None

class DocumentRetriever:
    """Simple FAISS-based retriever using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is required for DocumentRetriever.")
        self.encoder = SentenceTransformer(model_name)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.documents = []  # Store original docs for retrieval reference
        
    def add_documents(self, docs: List[Dict[str, Any]]):
        """
        Add documents to the index. 
        docs format: [{"id": "doc1", "text": "Some text", ...}, ...]
        """
        if not docs:
            return
            
        texts = [doc["text"] for doc in docs]
        embeddings = self.encoder.encode(texts, convert_to_numpy=True)
        
        # Add to faiss index
        self.index.add(embeddings)
        self.documents.extend(docs)
        
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve top_k documents for a given query.
        """
        if self.index.ntotal == 0:
            return []
            
        query_emb = self.encoder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_emb, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc["score"] = float(dist)
                results.append(doc)
                
        return results
