import random
from datetime import datetime, timedelta


class EventGenerator:

    def __init__(self, world):

        self.world = world

    # --------------------------------------------------------
    # MAIN ENTRY
    # --------------------------------------------------------

    def generate(self):

        self._generate_moves()

        self._generate_job_changes()

        self._generate_appointments()

        self._generate_shared_events()

        self._generate_contradictions()

        return self.world

    # --------------------------------------------------------
    # MOVES (location over time)
    # --------------------------------------------------------

    def _generate_moves(self):

        for person in self.world.people:

            if random.random() < 0.35:

                old_city = person.city

                new_person = self.world.get_random_person(exclude=person)

                if new_person:

                    new_city = new_person.city

                else:

                    continue

                if old_city != new_city:

                    person.city = new_city

                    self.world.add_memories([

                        f"{person.name} used to live in {old_city}.",

                        f"{person.name} moved to {new_city}.",

                        f"{person.name} now lives in {new_city}."

                    ])

    # --------------------------------------------------------
    # JOB CHANGES
    # --------------------------------------------------------

    def _generate_job_changes(self):

        for person in self.world.people:

            if random.random() < 0.25:

                old_job = person.job

                new_person = self.world.get_random_person(exclude=person)

                if not new_person:

                    continue

                new_job = new_person.job

                if old_job != new_job:

                    person.job = new_job

                    self.world.add_memories([

                        f"{person.name} used to work as a {old_job}.",

                        f"{person.name} changed jobs to become a {new_job}.",

                        f"{person.name} is now working as a {new_job}."

                    ])

    # --------------------------------------------------------
    # APPOINTMENTS (time-based memories)
    # --------------------------------------------------------

    def _generate_appointments(self):

        for person in self.world.people:

            if random.random() < 0.4:

                friend = self.world.get_random_person(exclude=person)

                if not friend:

                    continue

                days = random.randint(1, 30)

                date = (datetime.utcnow() + timedelta(days=days)).strftime("%Y-%m-%d")

                self.world.add_memories([

                    f"{person.name} has a meeting with {friend.name} on {date}.",

                    f"{person.name} scheduled an appointment with {friend.name}."

                ])

    # --------------------------------------------------------
    # SHARED EVENTS (multi-entity interactions)
    # --------------------------------------------------------

    def _generate_shared_events(self):

        events = [

            "worked on a roofing project",

            "went fishing",

            "attended a workshop",

            "fixed a vehicle",

            "built a shed",

            "played games together",

            "had lunch together"

        ]

        for _ in range(len(self.world.people) // 2):

            p1 = self.world.get_random_person()

            p2 = self.world.get_random_person(exclude=p1)

            if not p1 or not p2:

                continue

            event = random.choice(events)

            self.world.add_memories([

                f"{p1.name} and {p2.name} {event}.",

                f"{p2.name} and {p1.name} {event}."

            ])

    # --------------------------------------------------------
    # CONTRADICTIONS (critical for ranking tests)
    # --------------------------------------------------------

    def _generate_contradictions(self):

        for person in self.world.people:

            if random.random() < 0.2:

                old_food = person.food

                new_food = self._random_food_excluding(old_food)

                self.world.add_memories([

                    f"{person.name} used to like {old_food}.",

                    f"{person.name} now prefers {new_food}."

                ])

                person.food = new_food

    # --------------------------------------------------------
    # UTIL
    # --------------------------------------------------------

    def _random_food_excluding(self, exclude):

        from people import FOODS

        choices = [f for f in FOODS if f != exclude]

        return random.choice(choices) if choices else exclude
