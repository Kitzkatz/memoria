import json
import csv
import argparse
import random
from world import World
from people import PeopleGenerator
from relationships import RelationshipGenerator
from events import EventGenerator
from questions import QuestionGenerator
from exporter import Exporter
from importance_utils import add_importance_cue, IMPORTANCE_CUES, IMPORTANCE_PROBABILITY


# --------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------

def load_data():
    with open("benchmark_output/benchmark_memories.txt", "r") as f:
        memories = [l.strip() for l in f if l.strip()]

    with open("benchmark_output/benchmark_questions.json", "r") as f:
        questions = json.load(f)

    return memories, questions


# --------------------------------------------------------
# BUILD WORLD
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
# MAIN
# --------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--people", type=int, default=500,
                        help="Number of people to generate (max ~1000 before overlap)")
    parser.add_argument("--no-importance", action="store_true",
                        help="Disable importance cues")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    print("\n====================================")
    print(" SYNTHETIC WORLD BENCHMARK RUNNER")
    print("====================================\n")
    print(f"People: {args.people}")
    print(f"Importance cues: {'OFF' if args.no_importance else 'ON'}")
    print(f"Seed: {args.seed}")
    print("\nGenerating dataset...\n")

    world = build_world(args.people)

    memories, questions = load_data()

    print(f"\nMemories: {len(memories)}")
    print(f"Questions: {len(questions)}")
    print("\nDONE")


if __name__ == "__main__":
    main()
