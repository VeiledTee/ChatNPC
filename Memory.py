from datetime import datetime


class Fact:
    def __init__(
        self,
        fact_id: int,
        text: str,
        poignancy: float = 0.99,
        importance: int = 5,
        information_type: str = "background",
        expiration=None,
    ):
        self.id = fact_id
        self.text = text
        self.poignancy = poignancy
        self.importance = importance

        self.type = information_type  # background / query / reply / reflection

        self.created = datetime.now()
        self.expiration = expiration
        self.last_accessed = self.created


class Memory:
    def __init__(self):
        pass
