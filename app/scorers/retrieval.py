class RetrievalScorer:
    """Evaluates retrieval quality (e.g., hit rate) against ground truth references."""
    
    def score(self, retrieved_docs: list, reference_doc_ids: list) -> dict:
        if not reference_doc_ids:
            return {"hit_rate": 0.0, "precision": 0.0, "recall": 0.0}
            
        retrieved_ids = [doc.get("id") for doc in retrieved_docs]
        hits = set(retrieved_ids).intersection(set(reference_doc_ids))
        
        hit_rate = 1.0 if len(hits) > 0 else 0.0
        precision = len(hits) / len(retrieved_ids) if retrieved_ids else 0.0
        recall = len(hits) / len(reference_doc_ids)
        
        return {
            "hit_rate": hit_rate,
            "precision": round(precision, 2),
            "recall": round(recall, 2)
        }
