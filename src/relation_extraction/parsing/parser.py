import spacy
from src.relation_extraction.models.token import Token
from src.relation_extraction.models.dependency import Dependency


class Parser:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def parse(self, text: str) -> tuple[list[Token], list[Dependency]]:
        doc = self.nlp(text)
        tokens = [
            Token(
                text=token.text,
                token_index=token.i,
                sentence_index=token.idx,
                lemma=token.lemma_,
                pos=token.pos_,
                tag=token.tag_,
                dep=token.dep_,
                head=token.head.i,
            )
            for token in doc
        ]

        dependencies = [
            Dependency(
                dependent=token.i,
                head=token.head.i,
                relation=token.dep_,
            )
            for token in doc
            if token.head.i != token.i
        ]

        return tokens, dependencies