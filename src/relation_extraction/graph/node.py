from dataclasses import dataclass
from src.relation_extraction.models.token import Token


@dataclass
class Node:
    token: Token