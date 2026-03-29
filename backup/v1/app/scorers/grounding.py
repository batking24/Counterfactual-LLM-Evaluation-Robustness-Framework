class GroundingScorer:
    """Evaluates whether an answer is supported by the retrieved context."""
    
    def score(self, query: str, answer: str, context: list) -> dict:
        # Simplistic heuristic / mock logic for Phase 1.
        # In a full system, this would call an LLM as a judge.
        
        if not context:
            return {"grounding_score": 0.0, "is_supported": False}
            
        context_text = " ".join([c.get("text", "").lower() for c in context])
        answer_lower = answer.lower()
        
        # Super simple token overlap (heuristic grounding)
        answer_tokens = set(answer_lower.split())
        context_tokens = set(context_text.split())
        
        if not answer_tokens:
            return {"grounding_score": 0.0, "is_supported": False}
            
        overlap = len(answer_tokens.intersection(context_tokens)) / len(answer_tokens)
        is_supported = overlap > 0.3  # arbitrary threshold
        
        return {
            "grounding_score": round(overlap, 2),
            "is_supported": is_supported
        }
