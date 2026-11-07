import json
from pathlib import Path

from src.relation_extraction.config import LABEL_TO_ID
from src.relation_extraction.data.semeval_loader import SemEvalLoader
from src.relation_extraction.graph.graph import DependencyGraph
from src.relation_extraction.models.semeval_example import SemEvalExample
from src.relation_extraction.parsing.parser import Parser


def build_record(
    example: SemEvalExample,
    parser: Parser,
    loader: SemEvalLoader,
) -> tuple[dict, bool]:
    label = example.relation
    if label not in LABEL_TO_ID:
        raise ValueError(
            f"Unknown label {label!r} for example id={example.index_of_sent}"
        )

    tokens, dependencies = parser.parse(example.sentence)

    try:
        e1_indices = loader.find_entity_indices(tokens, example.entity1_span)
        e2_indices = loader.find_entity_indices(tokens, example.entity2_span)
        entities_ok = True
    except (ValueError, TypeError):
        e1_indices = []
        e2_indices = []
        entities_ok = False

    if entities_ok:
        graph = DependencyGraph()
        graph.build(tokens, dependencies)
        path = graph.shortest_dependency_path(e1_indices, e2_indices)
    else:
        path = None

    heads = [
        -1 if token.head == token.token_index else token.head
        for token in tokens
    ]

    if path is None:
        sdp = None
        sdp_deps = None
    else:
        sdp = [path[0][0].token.token_index] + [
            target.token.token_index for _, _, target, _ in path
        ]
        sdp_deps = [
            f"{relation}{direction}" for _, relation, _, direction in path
        ]

    record = {
        "id": example.index_of_sent,
        "tokens": [token.text for token in tokens],
        "e1": e1_indices,
        "e2": e2_indices,
        "label": label,
        "label_id": LABEL_TO_ID[label],
        "heads": heads,
        "deprels": [token.dep for token in tokens],
        "pos": [token.pos for token in tokens],
        "sdp": sdp,
        "sdp_deps": sdp_deps,
    }
    return record, entities_ok


def generate_sample(
    input_path: str | Path,
    output_path: str | Path,
    limit: int = 20,
) -> dict:
    loader = SemEvalLoader()
    parser = Parser()

    examples = loader.load_file(str(input_path))[:limit]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    no_path = 0
    failed_entity_ids: list[int] = []

    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            record, entities_ok = build_record(example, parser, loader)
            if not entities_ok:
                failed_entity_ids.append(example.index_of_sent)
            elif record["sdp"] is None:
                no_path += 1
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    return {
        "written": written,
        "no_path": no_path,
        "failed_entity_ids": failed_entity_ids,
    }