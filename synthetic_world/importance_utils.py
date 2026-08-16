import random

# --------------------------------------------------------
# IMPORTANCE CUES (expanded)
# --------------------------------------------------------

IMPORTANCE_CUES = [
    "Remember this: ",
    "Important: ",
    "Critical: ",
    "Key point: ",
    "Vital: ",
    "Crucial: ",
    "Don't forget: ",
    "Noteworthy: ",
    "Significant: ",
    "Priority: ",
    "Urgent: ",
    "Must remember: ",
    "Need to know: ",
    "Essential: ",
    "Major: ",
    "High priority: ",
    "Take note: ",
    "Pay attention: ",
    "This matters: ",
    "Crucial detail: ",
    "Very important: ",
    "Extremely relevant: ",
]

# --------------------------------------------------------
# IMPORTANCE PROBABILITY PER TYPE (configurable)
# --------------------------------------------------------

IMPORTANCE_PROBABILITY = {
    "semantic": 0.15,
    "episodic": 0.25,
    "procedural": 0.20,
    "code": 0.30,
    "science": 0.25,
    "general": 0.10,
}

# --------------------------------------------------------
# IMPORTANCE LEVELS (for more granular control)
# --------------------------------------------------------

IMPORTANCE_LEVELS = {
    "high": 0.50,      # 50% chance
    "medium": 0.25,    # 25% chance
    "low": 0.10,       # 10% chance
}

# --------------------------------------------------------
# CUE PLACEMENT STYLES
# --------------------------------------------------------

def add_importance_cue(text: str, memory_type: str = "general", level: str = "medium") -> str:
    """
    Randomly add an importance cue to text based on memory type and level.

    Args:
        text: The memory text
        memory_type: Type of memory (semantic, episodic, procedural, code, science, general)
        level: Importance level (high, medium, low) — overrides probability if set

    Returns:
        str: Text with optional importance cue
    """
    # Determine probability
    if level in IMPORTANCE_LEVELS:
        prob = IMPORTANCE_LEVELS[level]
    else:
        prob = IMPORTANCE_PROBABILITY.get(memory_type, 0.15)

    # If no cue, return original text
    if random.random() >= prob:
        return text

    # Pick a random cue
    cue = random.choice(IMPORTANCE_CUES)

    # Decide placement
    placement = random.choice(["start", "middle", "end", "after_first_sentence"])

    if placement == "start":
        return f"{cue}{text}"

    elif placement == "after_first_sentence":
        parts = text.split('.', 1)
        if len(parts) > 1 and len(parts[0]) < 100:
            return f"{parts[0]}. {cue}{parts[1].strip()}"
        else:
            return f"{cue}{text}"

    elif placement == "middle":
        # Insert cue somewhere in the middle of the text
        words = text.split()
        if len(words) > 6:
            insert_pos = len(words) // 2
            words.insert(insert_pos, cue.strip())
            return " ".join(words)
        else:
            return f"{cue}{text}"

    else:  # end
        return f"{text} {cue.strip()}"

# --------------------------------------------------------
# BATCH ADD IMPORTANCE CUES
# --------------------------------------------------------

def add_importance_cues_batch(texts: list, memory_type: str = "general", level: str = "medium") -> list:
    """Add importance cues to multiple texts."""
    return [add_importance_cue(text, memory_type, level) for text in texts]

# --------------------------------------------------------
# GET PROBABILITY FOR TYPE
# --------------------------------------------------------

def get_importance_probability(memory_type: str) -> float:
    """Get the importance probability for a memory type."""
    return IMPORTANCE_PROBABILITY.get(memory_type, 0.15)

# --------------------------------------------------------
# SET PROBABILITY FOR TYPE
# --------------------------------------------------------

def set_importance_probability(memory_type: str, probability: float) -> None:
    """Set the importance probability for a memory type."""
    if 0.0 <= probability <= 1.0:
        IMPORTANCE_PROBABILITY[memory_type] = probability
    else:
        raise ValueError("Probability must be between 0.0 and 1.0")

# --------------------------------------------------------
# GET ALL IMPORTANCE CUES
# --------------------------------------------------------

def get_all_importance_cues() -> list:
    """Return all importance cues."""
    return IMPORTANCE_CUES.copy()

# --------------------------------------------------------
# ADD CUSTOM IMPORTANCE CUE
# --------------------------------------------------------

def add_importance_cue_custom(cue: str) -> None:
    """Add a custom importance cue to the list."""
    if cue not in IMPORTANCE_CUES:
        IMPORTANCE_CUES.append(cue)

# --------------------------------------------------------
# EXPORT
# --------------------------------------------------------

__all__ = [
    "IMPORTANCE_CUES",
    "IMPORTANCE_PROBABILITY",
    "IMPORTANCE_LEVELS",
    "add_importance_cue",
    "add_importance_cues_batch",
    "get_importance_probability",
    "set_importance_probability",
    "get_all_importance_cues",
    "add_importance_cue_custom",
]
