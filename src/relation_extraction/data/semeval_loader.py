import re

from src.relation_extraction.models.semeval_example import SemEvalExample

ENTITIES_REGEX = r"<e1>(.*?)</e1>|<e2>(.*?)</e2>"

class SemEvalLoader:
    def parse_example(self, text: str, relation: str | None = None) -> SemEvalExample:
        index, text = self._clean_sentence(text)

        entity1 = None
        entity2 = None
        entity1_span = None
        entity2_span = None

        parts = []
        position = 0

        pattern = re.compile(ENTITIES_REGEX)

        for match in pattern.finditer(text):
            parts.append(text[position:match.start()])

            entity_text = match.group(1) or match.group(2)
            entity_start = sum(len(part) for part in parts)

            parts.append(entity_text)

            entity_end = entity_start + len(entity_text)

            if match.group(1) is not None:
                entity1 = entity_text
                entity1_span = (entity_start, entity_end)
            else:
                entity2 = entity_text
                entity2_span = (entity_start, entity_end)

            position = match.end()

        parts.append(text[position:])

        sentence = "".join(parts)

        if entity1 is None or entity2 is None:
            raise ValueError("Sentence must contain both e1 and e2")

        return SemEvalExample(
            sentence=sentence,
            index_of_sent=index,
            entity1=entity1,
            entity2=entity2,
            entity1_span=entity1_span,
            entity2_span=entity2_span,
            relation=relation,
        )

    def _clean_sentence(self, text: str) -> tuple[int, str]:
        text = text.strip()

        match = re.match(r'^(\d+)\s+(.*)$', text)
        index = int(match.group(1))
        text = match.group(2)

        if text.startswith('"') and text.endswith('"'):
            text  = text[1:-1]

        return index, text

    def find_entity_indices(self, tokens, entity_span: tuple[int, int]) -> list[int]:
        start, end = entity_span

        indices = [
            token.token_index for token in tokens
            if token.token_index is not None and self._token_offset_overlaps(
                token.sentence_index, token.sentence_index + len(token.text), start, end)
        ]

        if not indices:
            raise ValueError(f"Entity span {entity_span} does not match any tokens")

        return indices

    def _token_offset_overlaps(self, token_start: int, token_end: int,
                              entity_start: int, entity_end: int) -> bool:
        return token_start < entity_end and token_end > entity_start

    def load_file(self, path: str) -> list[SemEvalExample]:
        examples = []

        with open(path, "r", encoding="utf-8") as file:

            lines = file.readlines()

            for i in range(0, len(lines), 4):
                sentence_line, relation_line = lines[i], lines[i+1]

                sentence_line = sentence_line.strip()
                relation_line = relation_line.strip()

                example = self.parse_example(text=sentence_line, relation=relation_line)
                examples.append(example)

            return examples