class AdversarialInjector:
    """Injects adversarial noise and prompt injections into the Retrieval/Context pipeline."""
    
    @staticmethod
    def inject_context(context_docs: list, injection_text: str) -> list:
        """Injects malicious or noisy text directly into the retrieved document chunks."""
        if not context_docs:
            return [{"id": "injected_doc", "text": injection_text}]
            
        # Append malicious text to the top-ranked document
        injected_docs = context_docs.copy()
        injected_docs[0]["text"] = f"{injected_docs[0]['text']} {injection_text}"
        return injected_docs
