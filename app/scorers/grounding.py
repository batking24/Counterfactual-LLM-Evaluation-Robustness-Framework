import os
from openai import OpenAI
from typing import List, Dict, Any

import requests

class GroundingScorer:
    """Evaluates grounding via LLM Judge (supports OpenAI and Ollama)."""
    
    def __init__(self, provider: str = "openai", model: str = None):
        self.provider = provider
        if provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = model or "gpt-3.5-turbo"
        else:
            self.endpoint = "http://localhost:11434/api/generate"
            self.model = model or "llama3"
    
    def score(self, query: str, answer: str, context: List[Dict[str, Any]]) -> dict:
        """Score grounding based on semantic evaluation by an LLM."""
        if not context:
            return {"grounding_score": 0.0, "is_supported": False, "rationale": "No context provided."}
            
        context_text = "\n\n".join([f"DOC {i}: {c.get('text')}" for i, c in enumerate(context)])
        
        prompt = f"""
        You are an evaluator. Given the following context and an answer to a query, determine if the answer is grounded in the context.
        Provide a grounding score between 0.0 and 1.0, where 1.0 means perfectly grounded and 0.0 means completely hallucinated.
        Also provide a brief rationale.

        Context:
        {context_text}

        Query: {query}
        Answer: {answer}

        Output in JSON format with keys: "score" (float) and "rationale" (string).
        """
        
        if self.provider == "openai":
            try:
                from openai import OpenAI
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                data = json.loads(response.choices[0].message.content)
                score = data.get("score", 0.0)
                return {
                    "grounding_score": score,
                    "is_supported": score > 0.7,
                    "rationale": data.get("rationale", "")
                }
            except Exception as e:
                return {"grounding_score": 0.0, "is_supported": False, "rationale": f"OpenAI Judge error: {str(e)}"}
        else:
            try:
                import re
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False
                }
                response = requests.post(self.endpoint, json=payload, timeout=60)
                response.raise_for_status()
                raw_response = response.json().get("response", "{}")
                
                # Robust extraction: find the first { and last }
                try:
                    # Clean markdown if present
                    clean_text = re.sub(r'```json\n?|\n?```', '', raw_response).strip()
                    data = json.loads(clean_text)
                except:
                    # Fallback to regex finding
                    match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                    if match:
                        data = json.loads(match.group())
                    else:
                        raise ValueError(f"No JSON found in response: {raw_response[:100]}")

                score = data.get("score", 0.0)
                return {
                    "grounding_score": float(score),
                    "is_supported": float(score) > 0.7,
                    "rationale": data.get("rationale", "Local evaluation successful")
                }
            except Exception as e:
                # Heuristic fallback if local LLM fails (to avoid 0.0 for everything)
                n_matches = sum(1 for word in set(answer.lower().split()) if any(word in c.get('text', '').lower() for c in context))
                heuristic_score = min(1.0, n_matches / max(1, len(set(answer.lower().split()))))
                return {
                    "grounding_score": heuristic_score, 
                    "is_supported": heuristic_score > 0.5, 
                    "rationale": f"H_Fallback: {str(e)[:50]}"
                }

import json
