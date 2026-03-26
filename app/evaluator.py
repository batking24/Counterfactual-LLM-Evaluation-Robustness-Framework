from app.rag_pipeline import RAGPipeline
from app.scorers.grounding import GroundingScorer
from app.scorers.consistency import ConsistencyScorer
from app.scorers.retrieval import RetrievalScorer
from app.db import log_eval_run, init_db

class Evaluator:
    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
        self.grounding_scorer = GroundingScorer()
        self.consistency_scorer = ConsistencyScorer()
        self.retrieval_scorer = RetrievalScorer()
        init_db()
        
    def evaluate(self, dataset: list):
        """
        Evaluates a dataset.
        Format: [{"query": str, "reference_doc_ids": List[str], "variants": List[str]}, ...]
        """
        results = []
        for item in dataset:
            query = item["query"]
            # 1. Run pipeline
            rag_output = self.pipeline.run(query)
            answer = rag_output["answer"]
            retrieved_context = rag_output["retrieved_context"]
            
            # 2. Score grounding
            grounding = self.grounding_scorer.score(query, answer, retrieved_context)
            
            # 3. Score retrieval
            retrieval = self.retrieval_scorer.score(retrieved_context, item.get("reference_doc_ids", []))
            
            # 4. Score consistency
            answers_for_variants = [answer]
            for variant in item.get("variants", []):
                variant_out = self.pipeline.run(variant)
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
                "hit_rate": retrieval["hit_rate"]
            }
            log_eval_run(record)
            results.append(record)
            
        return results
