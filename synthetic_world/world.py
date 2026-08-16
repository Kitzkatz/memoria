from dataclasses import dataclass, field
from collections import defaultdict
import random
import hashlib


# --------------------------------------------------------
# PERSON (Expanded)
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
    
    # Expanded fields
    hobbies: list = field(default_factory=list)
    family: list = field(default_factory=list)
    friends: list = field(default_factory=list)
    coworkers: list = field(default_factory=list)
    projects: list = field(default_factory=list)
    appointments: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    # New fields
    email: str = ""
    trait1: str = ""
    trait2: str = ""
    favorite_color: str = ""
    education: str = ""
    years_experience: int = 0

    @property
    def name(self):
        return f"{self.first_name} {self.last_name}"

    @property
    def age_group(self):
        if self.age < 25:
            return "young"
        elif self.age < 40:
            return "adult"
        elif self.age < 60:
            return "middle"
        else:
            return "senior"

    def to_dict(self):
        return {
            "id": self.person_id,
            "name": self.name,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "age": self.age,
            "city": self.city,
            "job": self.job,
            "phone": self.phone,
            "address": self.address,
            "birthday": self.birthday,
            "food": self.food,
            "vehicle": self.vehicle,
            "pet": self.pet,
            "email": self.email,
            "trait1": self.trait1,
            "trait2": self.trait2,
            "hobbies": self.hobbies,
        }

    def __repr__(self):
        return f"Person(name='{self.name}', city='{self.city}', job='{self.job}')"


# --------------------------------------------------------
# WORLD (Enhanced)
# --------------------------------------------------------

class World:
    def __init__(self, seed=42):
        random.seed(seed)
        self.seed = seed
        
        # Core data
        self.people = []
        self.memories = []
        self.questions = []
        self.truth = []
        
        # Lookup indexes
        self.lookup = {}           # name -> Person
        self.lookup_by_id = {}     # id -> Person
        self.lookup_by_city = defaultdict(list)
        self.lookup_by_job = defaultdict(list)
        self.lookup_by_age = defaultdict(list)
        
        # Type-specific storage
        self.memories_by_type = defaultdict(list)
        self.questions_by_type = defaultdict(list)
        
        # Statistics
        self.stats = defaultdict(int)
        
        # Generation metadata
        self.generated_at = None
        self.version = "2.0"

    # ----------------------------------------------------
    # PERSON MANAGEMENT
    # ----------------------------------------------------

    def add_person(self, person):
        self.people.append(person)
        self.lookup[person.name] = person
        self.lookup_by_id[person.person_id] = person
        self.lookup_by_city[person.city].append(person)
        self.lookup_by_job[person.job].append(person)
        self.lookup_by_age[person.age_group].append(person)
        self.stats["people"] += 1

    def get_person_by_name(self, name):
        return self.lookup.get(name)

    def get_person_by_id(self, person_id):
        return self.lookup_by_id.get(person_id)

    def get_people_by_city(self, city):
        return self.lookup_by_city.get(city, [])

    def get_people_by_job(self, job):
        return self.lookup_by_job.get(job, [])

    def get_random_person(self, exclude=None, city=None):
        if not self.people:
            return None
        
        candidates = self.people
        if exclude:
            candidates = [p for p in candidates if p != exclude]
        if city:
            candidates = [p for p in candidates if p.city == city]
        
        if not candidates:
            return None
        return random.choice(candidates)

    def get_random_people(self, count=2, exclude=None):
        """Get multiple random people without duplicates."""
        candidates = [p for p in self.people if p != exclude]
        if len(candidates) < count:
            return candidates
        return random.sample(candidates, count)

    # ----------------------------------------------------
    # MEMORY MANAGEMENT
    # ----------------------------------------------------

    def add_memory(self, text, memory_type="general", metadata=None):
        self.memories.append(text)
        self.memories_by_type[memory_type].append(text)
        self.stats["memories"] += 1
        
        if metadata:
            self.stats[f"memories_{memory_type}"] += 1

    def add_memories(self, memories, memory_type="general"):
        self.memories.extend(memories)
        self.memories_by_type[memory_type].extend(memories)
        self.stats["memories"] += len(memories)
        self.stats[f"memories_{memory_type}"] += len(memories)

    def get_memories_by_type(self, memory_type):
        return self.memories_by_type.get(memory_type, [])

    def get_memory_count_by_type(self):
        return {k: len(v) for k, v in self.memories_by_type.items()}

    # ----------------------------------------------------
    # QUESTION MANAGEMENT
    # ----------------------------------------------------

    def add_question(self, question, question_type="general"):
        self.questions.append(question)
        self.questions_by_type[question_type].append(question)
        self.stats["questions"] += 1

    def add_truth(self, row):
        self.truth.append(row)

    def get_questions_by_type(self, question_type):
        return self.questions_by_type.get(question_type, [])

    # ----------------------------------------------------
    # UTILITY
    # ----------------------------------------------------

    def generate_id(self):
        """Generate a unique ID for a person."""
        return len(self.people)

    def get_unique_name(self, first, last):
        """Generate a unique name combination."""
        base = f"{first} {last}"
        if base not in self.lookup:
            return base
        
        for i in range(1, 10):
            variant = f"{first} {last} {i}"
            if variant not in self.lookup:
                return variant
        
        # If all else fails, use a hash
        unique_id = hashlib.md5(f"{first}{last}{random.random()}".encode()).hexdigest()[:8]
        return f"{first} {last} {unique_id}"

    # ----------------------------------------------------
    # STATISTICS
    # ----------------------------------------------------

    def summary(self):
        print("\n========== WORLD SUMMARY ==========")
        print(f"Version: {self.version}")
        print(f"Seed: {self.seed}")
        print("-" * 35)
        for key, value in sorted(self.stats.items()):
            print(f"{key:20} : {value}")
        print("-" * 35)
        print(f"People by city: {dict(self.lookup_by_city)}")
        print(f"People by job: {dict(self.lookup_by_job)}")
        print("===================================\n")

    def get_stats(self):
        """Return statistics as a dict."""
        return {
            "version": self.version,
            "seed": self.seed,
            **dict(self.stats),
            "people_by_city": {k: len(v) for k, v in self.lookup_by_city.items()},
            "people_by_job": {k: len(v) for k, v in self.lookup_by_job.items()},
            "memories_by_type": self.get_memory_count_by_type(),
        }

    # ----------------------------------------------------
    # EXPORT HELPERS
    # ----------------------------------------------------

    def to_dict(self):
        return {
            "seed": self.seed,
            "version": self.version,
            "stats": self.get_stats(),
            "people": [p.to_dict() for p in self.people],
            "memories": self.memories,
            "memories_by_type": dict(self.memories_by_type),
            "questions": self.questions,
        }

    def __len__(self):
        return len(self.people)

    def __repr__(self):
        return f"World(people={len(self.people)}, memories={len(self.memories)}, questions={len(self.questions)})"
