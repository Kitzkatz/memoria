import random


class QuestionGenerator:

    def __init__(self, world):

        self.world = world

    # --------------------------------------------------------
    # MAIN ENTRY
    # --------------------------------------------------------

    def generate(self):

        self._basic_attribute_questions()

        self._relationship_questions()

        self._multi_hop_questions()

        self._temporal_questions()

        self._summary_questions()

        return self.world

    # --------------------------------------------------------
    # BASIC ATTRIBUTE QUESTIONS
    # --------------------------------------------------------

    def _basic_attribute_questions(self):

        for person in self.world.people:

            self.world.add_question({

                "query": f"What does {person.name} like?",

                "expected": person.food

            })

            self.world.add_question({

                "query": f"What does {person.name} drive?",

                "expected": person.vehicle

            })

            self.world.add_question({

                "query": f"Where does {person.name} live?",

                "expected": person.city

            })

            self.world.add_question({

                "query": f"What pet does {person.name} own?",

                "expected": person.pet

            })

    # --------------------------------------------------------
    # RELATIONSHIP QUESTIONS
    # --------------------------------------------------------

    def _relationship_questions(self):

        for person in self.world.people:

            if person.friends:

                friend = random.choice(person.friends)

                self.world.add_question({

                    "query": f"Who is friends with {person.name}?",

                    "expected": friend

                })

            if person.family:

                relative = random.choice(person.family)

                self.world.add_question({

                    "query": f"Who is related to {person.name}?",

                    "expected": relative

                })

            if person.coworkers:

                coworker = random.choice(person.coworkers)

                self.world.add_question({

                    "query": f"Who does {person.name} work with?",

                    "expected": coworker

                })

    # --------------------------------------------------------
    # MULTI-HOP QUESTIONS
    # --------------------------------------------------------

    def _multi_hop_questions(self):

        people = self.world.people

        if len(people) < 3:

            return

        for _ in range(len(people) // 2):

            p1 = random.choice(people)

            if not p1.friends:

                continue

            friend = random.choice(p1.friends)

            self.world.add_question({

                "query": f"Who is friends with {p1.name}'s friend?",

                "expected": friend

            })

    # --------------------------------------------------------
    # TEMPORAL QUESTIONS
    # --------------------------------------------------------

    def _temporal_questions(self):

        for person in self.world.people:

            self.world.add_question({

                "query": f"Where did {person.name} used to live?",

                "expected": person.city

            })

            self.world.add_question({

                "query": f"What job did {person.name} have before?",

                "expected": person.job

            })

    # --------------------------------------------------------
    # SUMMARY QUESTIONS (global reasoning)
    # --------------------------------------------------------

    def _summary_questions(self):

        sample_people = random.sample(

            self.world.people,

            min(10, len(self.world.people))

        )

        for person in sample_people:

            self.world.add_question({

                "query": f"Tell me everything about {person.name}.",

                "expected": person.name

            })

        # system-level reasoning probes

        self.world.add_question({

            "query": "Who owns a truck?",

            "expected": None  # intentionally fuzzy for ranking evaluation

        })

        self.world.add_question({

            "query": "Who lives in Detroit?",

            "expected": None

        })
