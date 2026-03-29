import re
from typing import Tuple

class InputSanitizer:
    """Detects and sanitizes potential prompt injection attempts."""
    
    def __init__(self):
        # Heuristic patterns for common injection techniques
        self.injection_patterns = [
            r"ignore previous instructions",
            r"system: ",
            r"user: ",
            r"assistant: ",
            r"override",
            r"dan mode",
            r"jailbreak"
        ]
        
    def sanitize(self, text: str) -> Tuple[str, bool]:
        """
        Returns (sanitized_text, is_flagged).
        Currently flags typical injection keywords.
        """
        is_flagged = False
        lower_text = text.lower()
        
        for pattern in self.injection_patterns:
            if re.search(pattern, lower_text):
                is_flagged = True
                break
                
        # Simple mitigation: strip obvious markers or just flag it
        return text, is_flagged
