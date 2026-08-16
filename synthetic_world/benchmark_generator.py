import argparse
import random
from world import World
from people import PeopleGenerator
from relationships import RelationshipGenerator
from events import EventGenerator
from questions import QuestionGenerator
from exporter import Exporter


# --------------------------------------------------------
# IMPORTANCE CUES
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
]

# Which memory types get importance cues more often
IMPORTANCE_PROBABILITY = {
    "semantic": 0.15,
    "episodic": 0.25,
    "procedural": 0.20,
    "code": 0.30,
    "science": 0.25,
    "general": 0.10,
}


def add_importance_cue(text: str, memory_type: str = "general") -> str:
    """
    Randomly add an importance cue to text based on memory type.
    """
    prob = IMPORTANCE_PROBABILITY.get(memory_type, 0.15)
    
    if random.random() < prob:
        cue = random.choice(IMPORTANCE_CUES)
        # Randomly place cue at start or after first sentence
        if random.random() < 0.3:
            # Place after first sentence
            parts = text.split('.', 1)
            if len(parts) > 1 and len(parts[0]) < 80:
                return f"{parts[0]}. {cue}{parts[1].strip()}"
        # Place at start (default)
        return f"{cue}{text}"
    
    return text


# --------------------------------------------------------
# PIPELINE RUNNER
# --------------------------------------------------------

def build_world(count):
    world = World(seed=42)

    print("\n[1/5] Generating people...")
    PeopleGenerator(world).generate(count=count)

    print("\n[2/5] Generating relationships...")
    RelationshipGenerator(world).generate()

    print("\n[3/5] Generating events...")
    EventGenerator(world).generate()

    print("\n[4/5] Generating questions...")
    QuestionGenerator(world).generate()

    print("\n[5/5] Exporting dataset...")
    Exporter(world).write()

    return world


# --------------------------------------------------------
# CLI ENTRY
# --------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--people",
        type=int,
        default=100,
        help="Number of people to generate"
    )
    parser.add_argument(
        "--export_only",
        action="store_true",
        help="Skip generation and only export (future use)"
    )
    parser.add_argument(
        "--importance-cues",
        action="store_true",
        default=True,
        help="Add importance cues to memories (default: True)"
    )

    args = parser.parse_args()

    print("\n====================================")
    print(" SYNTHETIC WORLD BENCHMARK RUNNER")
    print("====================================\n")
    print(f"People: {args.people}")
    print(f"Importance cues: {args.importance_cues}")
    print("Seed: 42")
    print("\nStarting pipeline...\n")

    world = build_world(args.people)

    print("\nDONE")
    print("\nWorld Summary:")
    world.summary()


if __name__ == "__main__":
    main()
