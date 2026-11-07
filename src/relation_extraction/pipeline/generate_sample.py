import json
from pathlib import Path

from src.relation_extraction.data.semeval_loader import SemEvalLoader
from src.relation_extraction.parsing.parser import Parser
from src.relation_extraction.graph.graph import DependencyGraph
from src.relation_extraction.pipeline.labels import LABEL_TO_ID


def extract_label(relation_line: str) -> str:
    relation_line = relation_line.strip()
    if not (relation_line.startswith("(") and relation_line.endswith(")")):
        raise ValueError(f"Unexpected relation line: {relation_line!r}")
    parts = relation_line[1:-1].split(",", 2)
    if len(parts) != 3:
        raise ValueError(f"Unexpected relation line: {relation_line!r}")
    return parts[2].strip()


def build_record(example, parser: Parser, loader: SemEvalLoader) -> dict:
    tokens, dependencies = parser.parse(example.sentence)

    e1_indices = loader.find_entity_indices(tokens, example.entity1_span)
    e2_indices = loader.find_entity_indices(tokens, example.entity2_span)

    graph = DependencyGraph()
    graph.build(tokens, dependencies)

    path = graph.shortest_dependency_path(e1_indices, e2_indices)

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

    label = example.relation

    return {
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
    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            record = build_record(example, parser, loader)
            if record["sdp"] is None:
                no_path += 1
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    return {"written": written, "no_path": no_path}