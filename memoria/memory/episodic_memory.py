from datetime import datetime

class EpisodicMemory:
    def __init__(self):
        self.events = []

    def add_event(self, event, outcome=None):
        self.events.append({
            "event": event,
            "outcome": outcome,
            "timestamp": datetime.utcnow().isoformat()
        })

    def recent(self, n=10):
        return self.events[-n:]


#placeholder function till v4 
