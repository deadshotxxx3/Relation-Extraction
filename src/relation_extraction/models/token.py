from dataclasses import dataclass


@dataclass
class Token:
    text: str
    token_index: int
    sentence_index: int
    lemma: str | None = None
    pos: str | None = None
    tag: str | None = None
    dep: str | None = None
    head: int | None = None