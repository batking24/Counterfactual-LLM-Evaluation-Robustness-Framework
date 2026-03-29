import os
import logging
from typing import List, Dict, Optional
import openai

class LLMGenerator:
    """Wrapper for LLM text generation."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logging.warning("OPENAI_API_KEY environment variable is not set. Generation will fail unless using a mock model.")
        self.client = openai.OpenAI(api_key=self.api_key) if self.api_key else None
        
    def generate_answer(self, query: str, context_docs: List[Dict[str, str]] = None) -> str:
        """
        Generate an answer to the query using the provided context documents.
        """
        if not self.client:
            return "Error: OPENAI_API_KEY not configured. Cannot generate answer."
        
        system_prompt = "You are a helpful assistant. Provide an answer based primarily on the retrieved context below if relevant. If you don't know the answer, say so."
        
        context_text = ""
        if context_docs:
            context_text = "\n\nRetrieved Context:\n"
            for i, doc in enumerate(context_docs):
                context_text += f"---\nChunk {i+1}: {doc.get('text', '')}\n"
                
        user_prompt = f"Question: {query}{context_text}\nAnswer:"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error during OpenAI generation: {e}")
            return f"Generation Error: {e}"
