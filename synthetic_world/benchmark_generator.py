import argparse

from world import World
from people import PeopleGenerator
from relationships import RelationshipGenerator
from events import EventGenerator
from questions import QuestionGenerator
from exporter import Exporter


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

    args = parser.parse_args()

    print("\n====================================")

    print(" SYNTHETIC WORLD BENCHMARK RUNNER")

    print("====================================\n")

    print(f"People: {args.people}")

    print("Seed: 42")

    print("\nStarting pipeline...\n")

    world = build_world(args.people)

    print("\nDONE")

    print("\nWorld Summary:")

    world.summary()


if __name__ == "__main__":

    main()
