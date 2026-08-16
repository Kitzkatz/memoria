import random
from datetime import datetime, timedelta

# Import from importance_utils (not run_benchmark — avoids circular import)
from importance_utils import add_importance_cue


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
        self._generate_life_events()      # NEW: births, marriages, etc.
        self._generate_achievements()     # NEW: promotions, awards
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
                        add_importance_cue(
                            f"{person.name} used to live in {old_city}.",
                            memory_type="episodic"
                        ),
                        add_importance_cue(
                            f"{person.name} moved to {new_city}.",
                            memory_type="episodic"
                        ),
                        add_importance_cue(
                            f"{person.name} now lives in {new_city}.",
                            memory_type="episodic"
                        )
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
                        add_importance_cue(
                            f"{person.name} used to work as a {old_job}.",
                            memory_type="episodic"
                        ),
                        add_importance_cue(
                            f"{person.name} changed jobs to become a {new_job}.",
                            memory_type="episodic"
                        ),
                        add_importance_cue(
                            f"{person.name} is now working as a {new_job}.",
                            memory_type="episodic"
                        )
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
                    add_importance_cue(
                        f"{person.name} has a meeting with {friend.name} on {date}.",
                        memory_type="episodic"
                    ),
                    add_importance_cue(
                        f"{person.name} scheduled an appointment with {friend.name}.",
                        memory_type="episodic"
                    )
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
            "had lunch together",
            "went to a concert",
            "watched a movie",
            "went hiking",
            "started a business",
            "volunteered together",
            "went on a road trip",
            "bought a house",
            "adopted a pet",
        ]

        for _ in range(len(self.world.people) // 2):
            p1 = self.world.get_random_person()
            p2 = self.world.get_random_person(exclude=p1)
            if not p1 or not p2:
                continue

            event = random.choice(events)
            self.world.add_memories([
                add_importance_cue(
                    f"{p1.name} and {p2.name} {event}.",
                    memory_type="episodic"
                ),
                add_importance_cue(
                    f"{p2.name} and {p1.name} {event}.",
                    memory_type="episodic"
                )
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
                    add_importance_cue(
                        f"{person.name} used to like {old_food}.",
                        memory_type="episodic"
                    ),
                    add_importance_cue(
                        f"{person.name} now prefers {new_food}.",
                        memory_type="episodic"
                    )
                ])

                person.food = new_food

    # --------------------------------------------------------
    # LIFE EVENTS (births, marriages, etc.)
    # --------------------------------------------------------

    def _generate_life_events(self):
        for person in self.world.people:
            if random.random() < 0.15:
                partner = self.world.get_random_person(exclude=person)
                if partner:
                    self.world.add_memories([
                        add_importance_cue(
                            f"{person.name} married {partner.name}.",
                            memory_type="episodic"
                        ),
                        add_importance_cue(
                            f"{partner.name} married {person.name}.",
                            memory_type="episodic"
                        )
                    ])

            if random.random() < 0.1:
                child_name = random.choice(["Liam", "Noah", "Emma", "Olivia", "James", "Charlotte"])
                self.world.add_memories([
                    add_importance_cue(
                        f"{person.name} had a child named {child_name}.",
                        memory_type="episodic"
                    )
                ])

    # --------------------------------------------------------
    # ACHIEVEMENTS (promotions, awards)
    # --------------------------------------------------------

    def _generate_achievements(self):
        achievements = [
            "promoted to senior",
            "won an award",
            "graduated from university",
            "started a new business",
            "published a paper",
            "earned a certification",
            "completed a marathon",
            "won a competition",
            "gave a keynote speech",
        ]

        for person in self.world.people:
            if random.random() < 0.2:
                achievement = random.choice(achievements)
                self.world.add_memories([
                    add_importance_cue(
                        f"{person.name} {achievement}.",
                        memory_type="episodic"
                    )
                ])

    # --------------------------------------------------------
    # UTIL
    # --------------------------------------------------------

    def _random_food_excluding(self, exclude):
        from people import FOODS
        choices = [f for f in FOODS if f != exclude]
        return random.choice(choices) if choices else exclude
