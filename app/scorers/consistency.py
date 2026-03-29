from typing import List

class ConsistencyScorer:
    """Evaluates consistency across multiple response variants."""
    
    def score(self, variants: List[str]) -> dict:
        """Score consistency based on exact match or high overlap (Phase 1)."""
        if len(variants) < 2:
            return {"consistency_score": 1.0}
            
        # Simplistic: count unique answers (more sophisticated similarity in Phase 2)
        unique_answers = set([v.strip().lower() for v in variants])
        score = 1.0 / len(unique_answers) if unique_answers else 0.0
        
        return {
            "consistency_score": round(score, 2),
            "num_variants": len(variants),
            "unique_count": len(unique_answers)
        }
