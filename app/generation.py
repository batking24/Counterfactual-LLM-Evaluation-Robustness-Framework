import os
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

import json
import requests

class LLMGenerator:
    """Wrapper for LLM generation (supports OpenAI and Ollama)."""
    
    def __init__(self, provider: str = "openai", model: str = None):
        self.provider = provider
        if provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = model or "gpt-3.5-turbo"
        else:
            self.endpoint = "http://localhost:11434/api/generate"
            self.model = model or "llama3"

    def generate_answer(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Generate an answer based on the query and retrieved context."""
        context_text = "\n\n".join([f"Source {i+1}:\n{doc['text']}" for i, doc in enumerate(context)])
        
        prompt = f"""
Use the following context to answer the user query. If the answer is not in the context, say you don't know based on the provided information.

Context:
{context_text}

Query: {query}
"""
        if self.provider == "openai":
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant providing grounded answers."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error during OpenAI generation: {str(e)}"
        else:
            try:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "system": "You are a helpful assistant providing grounded answers."
                }
                response = requests.post(self.endpoint, json=payload, timeout=60)
                response.raise_for_status()
                return response.json().get("response", "No response from Ollama.")
            except Exception as e:
                return f"Error during local (Ollama) generation: {str(e)}"
