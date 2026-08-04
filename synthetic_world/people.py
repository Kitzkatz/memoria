import random
from world import Person


FIRST_NAMES = [
    "Alice", "Bob", "Charlie", "David", "Emma",
    "Frank", "Grace", "Henry", "Isabella", "Jack",
    "Kevin", "Liam", "Mason", "Noah", "Olivia"
]

LAST_NAMES = [
    "Smith", "Johnson", "Brown", "Wilson", "Taylor",
    "Anderson", "Thomas", "Jackson", "White", "Harris"
]

CITIES = [
    "Detroit", "Chicago", "Boston", "Seattle",
    "Dallas", "Miami", "Denver", "Phoenix"
]

JOBS = [
    "roofer", "electrician", "welder",
    "mechanic", "engineer", "teacher",
    "developer", "technician"
]

FOODS = [
    "pizza", "tacos", "burgers", "ramen",
    "steak", "sushi", "pasta", "wings"
]

VEHICLES = [
    "blue truck", "red sedan", "black SUV",
    "white van", "silver coupe", "motorcycle"
]

PETS = [
    "dog", "cat", "parrot", "rabbit",
    "hamster", "fish", "lizard"
]


def generate_phone():

    return f"555-{random.randint(100, 999)}-{random.randint(1000, 9999)}"


def generate_birthday():

    month = random.randint(1, 12)

    day = random.randint(1, 28)

    return f"{month:02d}/{day:02d}"


# --------------------------------------------------------
# MAIN GENERATOR
# --------------------------------------------------------

class PeopleGenerator:

    def __init__(self, world):

        self.world = world

    def generate(self, count=100):

        for i in range(count):

            first = random.choice(FIRST_NAMES)

            last = random.choice(LAST_NAMES)

            city = random.choice(CITIES)

            job = random.choice(JOBS)

            food = random.choice(FOODS)

            vehicle = random.choice(VEHICLES)

            pet = random.choice(PETS)

            person = Person(

                person_id=i,

                first_name=first,

                last_name=last,

                age=random.randint(18, 65),

                city=city,

                job=job,

                phone=generate_phone(),

                address=f"{random.randint(100,999)} Main St",

                birthday=generate_birthday(),

                food=food,

                vehicle=vehicle,

                pet=pet,

            )

            # ----------------------------------------
            # register person
            # ----------------------------------------

            self.world.add_person(person)

            # ----------------------------------------
            # baseline memories (IMPORTANT)
            # ----------------------------------------

            self.world.add_memories([

                f"{person.name} lives in {city}.",

                f"{person.name} works as a {job}.",

                f"{person.name} likes {food}.",

                f"{person.name} drives a {vehicle}.",

                f"{person.name} owns a {pet}.",

                f"{person.name}'s phone number is {person.phone}.",

                f"{person.name}'s birthday is {person.birthday}.",

                f"{person.name} lives at {person.address}.",

            ])

        return self.world.people
