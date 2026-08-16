import random
from world import Person
from importance_utils import add_importance_cue


FIRST_NAMES = [
    "Alice", "Bob", "Charlie", "David", "Emma",
    "Frank", "Grace", "Henry", "Isabella", "Jack",
    "Kevin", "Liam", "Mason", "Noah", "Olivia",
    "Sophia", "Ava", "Mia", "Ella", "James",
    "Benjamin", "Elijah", "Lucas", "Mason", "Logan",
    "Amelia", "Harper", "Evelyn", "Abigail", "Emily",
    "Elizabeth", "Mila", "Edward", "Avery", "Sofia",
    "Camila", "Aria", "Scarlett", "Victoria", "Madison"
]

LAST_NAMES = [
    "Smith", "Johnson", "Brown", "Wilson", "Taylor",
    "Anderson", "Thomas", "Jackson", "White", "Harris",
    "Martin", "Thompson", "Garcia", "Martinez", "Robinson",
    "Clark", "Rodriguez", "Lewis", "Lee", "Walker",
    "Hall", "Allen", "Young", "Hernandez", "King",
    "Wright", "Lopez", "Hill", "Scott", "Green",
    "Adams", "Baker", "Gonzalez", "Nelson", "Carter",
    "Mitchell", "Perez", "Roberts", "Turner", "Phillips"
]

CITIES = [
    "Detroit", "Chicago", "Boston", "Seattle",
    "Dallas", "Miami", "Denver", "Phoenix",
    "New York", "Los Angeles", "San Francisco", "Portland",
    "Austin", "Nashville", "Atlanta", "Minneapolis",
    "St. Louis", "Kansas City", "Cleveland", "Pittsburgh"
]

JOBS = [
    "roofer", "electrician", "welder", "mechanic",
    "engineer", "teacher", "developer", "technician",
    "architect", "plumber", "carpenter", "painter",
    "gardener", "chef", "pilot", "firefighter",
    "police officer", "nurse", "doctor", "pharmacist",
    "accountant", "lawyer", "consultant", "manager",
    "director", "analyst", "specialist", "therapist"
]

FOODS = [
    "pizza", "tacos", "burgers", "ramen",
    "steak", "sushi", "pasta", "wings",
    "salad", "soup", "sandwich", "burrito",
    "curry", "pho", "dumplings", "fried rice",
    "noodles", "gyro", "kebab", "pancakes",
    "waffles", "bagel", "smoothie", "ice cream",
    "pie", "cake", "cookies", "chips",
    "guacamole", "salsa", "hummus", "falafel"
]

VEHICLES = [
    "blue truck", "red sedan", "black SUV",
    "white van", "silver coupe", "motorcycle",
    "green hatchback", "yellow convertible", "gray minivan",
    "orange sports car", "purple sedan", "brown station wagon",
    "electric car", "hybrid SUV", "pickup truck"
]

PETS = [
    "dog", "cat", "parrot", "rabbit",
    "hamster", "fish", "lizard", "snake",
    "turtle", "ferret", "guinea pig", "chinchilla",
    "horse", "goat", "sheep", "chicken"
]

HOBBIES = [
    "reading", "painting", "hiking", "fishing", "gardening",
    "photography", "biking", "swimming", "cooking", "dancing",
    "yoga", "meditation", "writing", "singing", "playing guitar",
    "woodworking", "knitting", "pottery", "birdwatching", "stargazing"
]

PERSONALITY_TRAITS = [
    "friendly", "outgoing", "reserved", "creative", "logical",
    "adventurous", "cautious", "optimistic", "pessimistic", "hardworking",
    "lazy", "organized", "messy", "quiet", "talkative"
]

EDUCATION_LEVELS = [
    "high school", "associate's degree", "bachelor's degree", 
    "master's degree", "PhD", "trade school"
]

TRAIT_PAIRS = [
    ("friendly", "outgoing"),
    ("creative", "adventurous"),
    ("logical", "reserved"),
    ("hardworking", "organized"),
    ("optimistic", "friendly"),
    ("creative", "messy"),
]


def generate_phone():
    return f"555-{random.randint(100, 999)}-{random.randint(1000, 9999)}"


def generate_birthday():
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    year = random.randint(1940, 2005)
    return f"{month:02d}/{day:02d}/{year}"


def generate_email(name, last_name):
    domains = ["gmail.com", "yahoo.com", "outlook.com", "proton.me"]
    return f"{name.lower()}.{last_name.lower()}@{random.choice(domains)}"


# --------------------------------------------------------
# MAIN GENERATOR
# --------------------------------------------------------

class PeopleGenerator:

    def __init__(self, world):
        self.world = world
        self._used_names = set()

    def _unique_name(self, first, last):
        """Generate a unique name combination."""
        base = f"{first} {last}"
        if base not in self._used_names:
            self._used_names.add(base)
            return base
        
        # If name is taken, try variations
        for i in range(1, 10):
            variant = f"{first} {last} Jr."
            if i > 1:
                variant = f"{first} {last} {i}"
            if variant not in self._used_names:
                self._used_names.add(variant)
                return variant
        
        # If all else fails, use a random unique ID
        unique = f"{first} {last} {random.randint(1000, 9999)}"
        self._used_names.add(unique)
        return unique

    def generate(self, count=100):
        for i in range(count):
            first = random.choice(FIRST_NAMES)
            last = random.choice(LAST_NAMES)
            name = self._unique_name(first, last)
            
            city = random.choice(CITIES)
            job = random.choice(JOBS)
            food = random.choice(FOODS)
            vehicle = random.choice(VEHICLES)
            pet = random.choice(PETS)
            hobby = random.choice(HOBBIES)
            trait1, trait2 = random.choice(TRAIT_PAIRS)
            education = random.choice(EDUCATION_LEVELS)
            years_exp = random.randint(0, 30)

            person = Person(
                person_id=i,
                first_name=first,
                last_name=last,
                age=random.randint(18, 65),
                city=city,
                job=job,
                phone=generate_phone(),
                address=f"{random.randint(100, 9999)} {random.choice(['Main', 'Oak', 'Pine', 'Maple', 'Cedar', 'Elm'])} St",
                birthday=generate_birthday(),
                food=food,
                vehicle=vehicle,
                pet=pet,
                hobbies=[hobby],          # ← FIXED: hobby → hobbies=[hobby]
                trait1=trait1,
                trait2=trait2,
                email=generate_email(first, last),
                education=education,
                years_experience=years_exp,
            )

            self.world.add_person(person)

            # ----------------------------------------------------
            # BASELINE MEMORIES (with importance cues)
            # ----------------------------------------------------
            self.world.add_memories([
                add_importance_cue(
                    f"{person.name} lives in {city}.",
                    memory_type="semantic"
                ),
                add_importance_cue(
                    f"{person.name} works as a {job}.",
                    memory_type="semantic"
                ),
                add_importance_cue(
                    f"{person.name} likes {food}.",
                    memory_type="semantic"
                ),
                add_importance_cue(
                    f"{person.name} drives a {vehicle}.",
                    memory_type="semantic"
                ),
                add_importance_cue(
                    f"{person.name} owns a {pet}.",
                    memory_type="semantic"
                ),
                add_importance_cue(
                    f"{person.name}'s phone number is {person.phone}.",
                    memory_type="semantic"
                ),
                add_importance_cue(
                    f"{person.name}'s birthday is {person.birthday}.",
                    memory_type="semantic"
                ),
                add_importance_cue(
                    f"{person.name} lives at {person.address}.",
                    memory_type="semantic"
                ),
                add_importance_cue(
                    f"{person.name} enjoys {hobby}.",
                    memory_type="semantic"
                ),
                add_importance_cue(
                    f"{person.name} is {trait1} and {trait2}.",
                    memory_type="semantic"
                ),
                add_importance_cue(
                    f"{person.name}'s email is {person.email}.",
                    memory_type="semantic"
                ),
                add_importance_cue(
                    f"{person.name} has a {education}.",
                    memory_type="semantic"
                ),
                add_importance_cue(
                    f"{person.name} has {years_exp} years of experience.",
                    memory_type="semantic"
                ),
            ])

            # ----------------------------------------------------
            # EXTRA RANDOM MEMORIES (creates variety)
            # ----------------------------------------------------
            if random.random() < 0.4:
                extra = random.choice([
                    f"{person.name} was born in {random.choice(CITIES)}.",
                    f"{person.name} graduated from {random.choice(['MIT', 'Stanford', 'Harvard', 'Oxford'])}.",
                    f"{person.name} worked for {random.choice(['Google', 'Amazon', 'Microsoft', 'Tesla'])}.",
                    f"{person.name} speaks {random.choice(['Spanish', 'French', 'German', 'Mandarin'])}.",
                ])
                self.world.add_memory(add_importance_cue(extra, memory_type="semantic"))

        return self.world.people
