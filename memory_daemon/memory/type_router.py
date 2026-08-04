# memory/type_router.py
from typing import Dict, Any, List

class TypeRouter:
    """
    Inspects query text and tokens to determine the most appropriate memory type.
    Routes to: semantic, episodic, procedural, code, science, general.
    """
    def __init__(self):
        self.rules = {
            "semantic": ["what is", "define", "means", "called", "fact", "property"],
            "episodic": ["when", "what happened", "event", "time", "yesterday", "last week", "date"],
            "procedural": ["how to", "steps", "guide", "procedure", "method", "instructions"],
            "code": ["def ", "function ", "class ", "import ", "return ", "snippet", "api"],
            "science": ["formula", "equation", "chemical", "physics", "temperature", "pressure", "mass"]
        }

    def route(self, text: str, tokens: List[str]) -> str:
        """
        Determine memory type based on query text and tokens.
        """
        text_lower = text.lower()
        # Count matches per type
        scores = {t: 0 for t in self.rules}
        for mem_type, keywords in self.rules.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[mem_type] += 1

        # Boost if token matches specific patterns (e.g., code syntax)
        if any(t in tokens for t in ["def", "class", "import", "return"]):
            scores["code"] += 2

        # Return the type with the highest score
        best_type = max(scores, key=scores.get)
        if scores[best_type] == 0:
            return "general"
        return best_type
