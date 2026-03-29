from app.scorers.grounding import GroundingScorer
from app.scorers.consistency import ConsistencyScorer
from app.scorers.retrieval import RetrievalScorer
from app.scorers.attribution import AttributionScorer
from app.scorers.safety import SafetyScorer
from app.db import log_eval_run, init_db

class Evaluator:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.grounding_scorer = GroundingScorer()
        self.consistency_scorer = ConsistencyScorer()
        self.retrieval_scorer = RetrievalScorer()
        self.attribution_scorer = AttributionScorer()
        self.safety_scorer = SafetyScorer()
        init_db()
        
    def evaluate(self, dataset: list):
        results = []
        for item in dataset:
            query = item.get("query", "")
            force_empty = item.get("force_empty_context", False)
            noise = item.get("injected_noise", None)
            adv_prompt = item.get("injected_prompt", None)
            
            # Adversarial simulation
            actual_query = query + adv_prompt if adv_prompt else query
            
            rag_output = self.pipeline.run(
                query=actual_query, 
                force_empty_context=force_empty, 
                injected_noise=noise
            )
            
            answer = rag_output["answer"]
            retrieved_context = rag_output["retrieved_context"]
            
            # 2. Score grounding
            grounding = self.grounding_scorer.score(query, answer, retrieved_context)
            
            # 3. Score retrieval
            retrieval = self.retrieval_scorer.score(retrieved_context, item.get("reference_doc_ids", []))
            attribution = self.attribution_scorer.score(query, answer, retrieved_context)
            safety = self.safety_scorer.score(query, answer, injection_attempted=bool(adv_prompt))
            
            # 4. Score consistency
            answers_for_variants = [answer]
            for variant in item.get("variants", []):
                variant_out = self.pipeline.run(variant, force_empty_context=force_empty, injected_noise=noise)
                answers_for_variants.append(variant_out["answer"])
            
            consistency = self.consistency_scorer.score(answers_for_variants)
            
            # 5. Log and collect
            record = {
                "query": query,
                "answer": answer,
                "retrieved_context": retrieved_context,
                "grounding_score": grounding["grounding_score"],
                "is_supported": grounding["is_supported"],
                "consistency_score": consistency["consistency_score"],
                "hit_rate": retrieval["hit_rate"],
                "attribution_score": attribution["attribution_score"],
                "hallucinated_claims": attribution["hallucinated_claims"],
                "safety_score": safety["safety_score"]
            }
            log_eval_run(record)
            results.append(record)
            
        return results
