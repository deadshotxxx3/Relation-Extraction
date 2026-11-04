from dataclasses import dataclass


@dataclass
class Dependency:
    dependent: int
    head: int
    relation: str