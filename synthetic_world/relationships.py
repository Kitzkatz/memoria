import random


class RelationshipGenerator:

    def __init__(self, world):

        self.world = world

    # --------------------------------------------------------
    # MAIN ENTRY
    # --------------------------------------------------------

    def generate(self):

        people = self.world.people

        if len(people) < 2:

            return self.world

        self._generate_friends(people)

        self._generate_families(people)

        self._generate_work_relations(people)

        return self.world

    # --------------------------------------------------------
    # FRIENDSHIPS
    # --------------------------------------------------------

    def _generate_friends(self, people):

        for person in people:

            friends_count = random.randint(1, 4)

            friends = random.sample(

                people,

                min(friends_count, len(people))

            )

            for friend in friends:

                if friend == person:

                    continue

                person.friends.append(friend.name)

                self.world.add_memory(

                    f"{person.name} is friends with {friend.name}."

                )

    # --------------------------------------------------------
    # FAMILY
    # --------------------------------------------------------

    def _generate_families(self, people):

        for person in people:

            # spouse / partner chance

            if random.random() < 0.4:

                partner = self.world.get_random_person(exclude=person)

                if partner:

                    person.family.append(partner.name)

                    partner.family.append(person.name)

                    self.world.add_memory(

                        f"{person.name} is married to {partner.name}."

                    )

            # children / parents relationship approximation

            if random.random() < 0.3:

                parent = self.world.get_random_person(exclude=person)

                if parent:

                    person.family.append(parent.name)

                    self.world.add_memory(

                        f"{person.name} is the child of {parent.name}."

                    )

    # --------------------------------------------------------
    # WORK RELATIONSHIPS
    # --------------------------------------------------------

    def _generate_work_relations(self, people):

        for person in people:

            coworkers_count = random.randint(1, 5)

            coworkers = random.sample(

                people,

                min(coworkers_count, len(people))

            )

            for coworker in coworkers:

                if coworker == person:

                    continue

                person.coworkers.append(coworker.name)

                self.world.add_memory(

                    f"{person.name} works with {coworker.name}."

                )

            # boss relationship (hierarchy simulation)

            if random.random() < 0.7:

                boss = self.world.get_random_person(exclude=person)

                if boss:

                    self.world.add_memory(

                        f"{boss.name} is the boss of {person.name}."

                    )
