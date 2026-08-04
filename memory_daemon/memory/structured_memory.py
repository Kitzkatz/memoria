class StructuredMemory:
    def __init__(self):
        self.slots = {
            "preferences": [],
            "goals": [],
            "projects": [],
            "identity": [],
            "constraints": []
        }

    def store(self, slot, data):
        if slot in self.slots:
            self.slots[slot].append(data)

    def query(self, slot):
        return self.slots.get(slot, [])

    #placeholder function till v4 
