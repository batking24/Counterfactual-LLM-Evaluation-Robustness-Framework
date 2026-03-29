from typing import List, Dict, Any
from app.rag_pipeline import RAGPipeline
from app.scorers.grounding import GroundingScorer
from app.scorers.consistency import ConsistencyScorer
from app.scorers.retrieval import RetrievalScorer

class Evaluator:
    """Executes evaluation jobs across a dataset of queries."""
    
    def __init__(self, pipeline: RAGPipeline, provider: str = "openai", model: str = None):
        self.pipeline = pipeline
        self.grounding_scorer = GroundingScorer(provider=provider, model=model)
        self.consistency_scorer = ConsistencyScorer()
        self.retrieval_scorer = RetrievalScorer()
        
    def evaluate(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run evaluation on all items in the dataset."""
        results = []
        for stage, item in enumerate(dataset):
            query = item["query"]
            print(f"[{stage+1}/{len(dataset)}] Evaluating: {query[:50]}...")
            
            # 1. Run Pipeline
            rag_output = self.pipeline.run(query)
            
            # 2. Compute Scores
            grounding = self.grounding_scorer.score(
                query, 
                rag_output["answer"], 
                rag_output["retrieved_context"]
            )
            
            retrieval = self.retrieval_scorer.score(
                rag_output["retrieved_context"], 
                item.get("reference_doc_ids", [])
            )
            
            record = {
                "query": query,
                "category": item.get("category"),
                "answer": rag_output["answer"],
                "grounding_score": grounding["grounding_score"],
                "is_supported": grounding["is_supported"],
                "hit_rate": retrieval["hit_rate"],
                "recall": retrieval["recall"],
                "is_adverse": rag_output.get("is_adverse", False)
            }
            results.append(record)
            
        return results
