from dataclasses import dataclass, field
from collections import defaultdict
import random


# --------------------------------------------------------
# PERSON
# --------------------------------------------------------

@dataclass
class Person:

    person_id: int

    first_name: str

    last_name: str

    age: int

    city: str

    job: str

    phone: str

    address: str

    birthday: str

    food: str

    vehicle: str

    pet: str

    hobbies: list = field(default_factory=list)

    family: list = field(default_factory=list)

    friends: list = field(default_factory=list)

    coworkers: list = field(default_factory=list)

    projects: list = field(default_factory=list)

    appointments: list = field(default_factory=list)

    metadata: dict = field(default_factory=dict)

    @property
    def name(self):

        return f"{self.first_name} {self.last_name}"


# --------------------------------------------------------
# WORLD
# --------------------------------------------------------

class World:

    def __init__(self, seed=42):

        random.seed(seed)

        self.people = []

        self.memories = []

        self.questions = []

        self.truth = []

        #
        # Quick lookup
        #

        self.lookup = {}

        #
        # Statistics
        #

        self.stats = defaultdict(int)

    # ----------------------------------------------------

    def add_person(self, person):

        self.people.append(person)

        self.lookup[person.name] = person

        self.stats["people"] += 1

    # ----------------------------------------------------

    def add_memory(self, text):

        self.memories.append(text)

        self.stats["memories"] += 1

    # ----------------------------------------------------

    def add_memories(self, memories):

        self.memories.extend(memories)

        self.stats["memories"] += len(memories)

    # ----------------------------------------------------

    def add_question(self, question):

        self.questions.append(question)

        self.stats["questions"] += 1

    # ----------------------------------------------------

    def add_truth(self, row):

        self.truth.append(row)

    # ----------------------------------------------------

    def get_random_person(self, exclude=None):

        if not self.people:

            return None

        candidates = self.people

        if exclude:

            candidates = [

                p

                for p in self.people

                if p != exclude

            ]

        if not candidates:

            return None

        return random.choice(candidates)

    # ----------------------------------------------------

    def summary(self):

        print("\n========== WORLD ==========")

        for key, value in sorted(self.stats.items()):

            print(f"{key:15} : {value}")

        print("===========================\n")
