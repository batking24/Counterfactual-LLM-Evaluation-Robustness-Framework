class ConsistencyScorer:
    """Evaluates consistency across different outputs or paraphrased queries."""
    
    def score(self, answers: list) -> dict:
        if len(answers) < 2:
            return {"consistency_score": 1.0, "is_consistent": True}
            
        # Jaccard similarity across variants
        first_ans = set(answers[0].lower().split())
        scores = []
        for ans in answers[1:]:
            ans_tokens = set(ans.lower().split())
            if not first_ans or not ans_tokens:
                scores.append(0.0)
                continue
            intersection = len(first_ans.intersection(ans_tokens))
            union = len(first_ans.union(ans_tokens))
            scores.append(intersection / union)
            
        avg_score = sum(scores) / len(scores)
        
        return {
            "consistency_score": round(avg_score, 2),
            "is_consistent": avg_score > 0.5
        }
