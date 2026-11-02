from dataclasses import dataclass


@dataclass
class SemEvalExample:
    sentence: str
    index_of_sent: int
    entity1: str
    entity2: str
    entity1_span: tuple[int, int] | None = None
    entity2_span: tuple[int, int] | None = None
    relation: str | None = None
