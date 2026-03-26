class AttributionScorer:
    """Evaluates fine-grained claim-level attribution."""
    
    def score(self, query: str, answer: str, context: list) -> dict:
        if not context or not answer:
            return {"attribution_score": 0.0, "hallucinated_claims": 1, "total_claims": 1}
            
        context_text = " ".join([c.get("text", "").lower() for c in context])
        
        # Mocking claim extraction process for Phase 3 (Usually done via LLM)
        # We simulate splitting the answer into "claims" via sentence splitting
        claims = [c.strip() for c in answer.lower().split(".") if len(c.strip()) > 5]
        if not claims:
            return {"attribution_score": 0.0, "hallucinated_claims": 0, "total_claims": 0}
            
        supported = 0
        hallucinated = 0
        
        context_tokens = set(context_text.split())
        
        for claim in claims:
            claim_tokens = set(claim.split())
            if not claim_tokens:
                continue
            # Simple heuristic claim overlap
            overlap = len(claim_tokens.intersection(context_tokens)) / len(claim_tokens)
            if overlap > 0.4:
                supported += 1
            else:
                hallucinated += 1
                
        total = supported + hallucinated
        attribution_score = supported / total if total > 0 else 0.0
        
        return {
            "attribution_score": round(attribution_score, 2),
            "hallucinated_claims": hallucinated,
            "total_claims": total
        }
