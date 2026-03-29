import json
from typing import List, Dict, Any

class Reporter:
    """Generates summary reports and failure analysis from evaluation results."""
    
    def generate_summary(self, base_results: List[Dict[str, Any]], improved_results: List[Dict[str, Any]]) -> str:
        """Generates a text-based summary of the comparative evaluation."""
        
        def get_metrics(results):
            if not results: return {"grounding": 0, "hit_rate": 0}
            return {
                "grounding": sum(r["grounding_score"] for r in results) / len(results),
                "hit_rate": sum(r["hit_rate"] for r in results) / len(results)
            }
            
        base_m = get_metrics(base_results)
        improved_m = get_metrics(improved_results)
        
        report = []
        report.append("======================================================")
        report.append("📊 LLM EVALUATION FRAMEWORK - FINAL REPORT")
        report.append("======================================================")
        report.append(f"Total Test Vectors: {len(base_results)}")
        report.append("")
        report.append(f"{'Metric':<20} | {'Baseline':<10} | {'Improved':<10} | {'Delta':<8}")
        report.append("-" * 55)
        
        metrics = [
            ("Avg Grounding", "grounding"),
            ("Avg Hit Rate", "hit_rate")
        ]
        
        for label, key in metrics:
            b, i = base_m[key], improved_m[key]
            d = i - b
            report.append(f"{label:<20} | {b:10.2f} | {i:10.2f} | {d:+8.2f}")
            
        report.append("-" * 55)
        report.append("\nFailure Mode Analysis (Improved Pipeline):")
        
        # Simple failure attribution
        unsupported = [r for r in improved_results if not r["is_supported"]]
        missed = [r for r in improved_results if r["hit_rate"] == 0]
        
        report.append(f"- Hallucination / Unsupported Claims: {len(unsupported)} ({(len(unsupported)/len(improved_results))*100:.1f}%)")
        report.append(f"- Retrieval Misses: {len(missed)} ({(len(missed)/len(improved_results))*100:.1f}%)")
        
        return "\n".join(report)
