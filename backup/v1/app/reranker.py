class ReRanker:
    """Filters and re-ranks FAISS retrieval documents using a secondary pass."""
    
    def __init__(self, cutoff_score: float = 1.6):
        # In FAISS sentence-transformers L2 distance, closer to 0 is better.
        # Anything above 1.6 is highly unlikely to be relevant.
        self.cutoff_score = cutoff_score
        
    def rerank_and_filter(self, retrieved_docs: list, query: str) -> list:
        # 1. Filtering: remove docs with distance > cutoff
        filtered = [doc for doc in retrieved_docs if doc.get("score", 0.0) < self.cutoff_score]
        
        # 2. Re-ranking: sort by distance (lowest first)
        reranked = sorted(filtered, key=lambda x: x.get("score", 0.0))
        return reranked
