from typing import List, Dict, Any

class RetrievalScorer:
    """Evaluates retrieval quality (Recall, Hit Rate)."""
    
    def score(self, retrieved_docs: List[Dict[str, Any]], reference_ids: List[str]) -> dict:
        """Compute hit rate and recall."""
        if not reference_ids:
            return {"hit_rate": 1.0, "recall": 1.0}
            
        retrieved_ids = [doc.get("id") for doc in retrieved_docs]
        hits = [ref_id for ref_id in reference_ids if ref_id in retrieved_ids]
        
        hit_rate = 1.0 if len(hits) > 0 else 0.0
        recall = len(hits) / len(reference_ids)
        
        return {
            "hit_rate": hit_rate,
            "recall": round(recall, 2),
            "num_hits": len(hits)
        }
