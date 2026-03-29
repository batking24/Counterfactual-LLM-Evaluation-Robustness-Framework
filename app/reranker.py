from typing import List, Dict, Any

class ReRanker:
    """Simple heuristic re-ranker to filter results."""
    
    def __init__(self, cutoff_score: float = 0.5):
        self.cutoff_score = cutoff_score
        
    def rerank_and_filter(self, docs: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Filters documents based on a simple keyword match (Phase 1)."""
        query_terms = set(query.lower().split())
        
        refined_docs = []
        for doc in docs:
            doc_text = doc.get("text", "").lower()
            # Simple term frequency/overlap as match score
            match_count = sum(1 for term in query_terms if term in doc_text)
            match_score = match_count / len(query_terms) if query_terms else 0
            
            if match_score >= self.cutoff_score:
                refined_docs.append(doc)
                
        return refined_docs
