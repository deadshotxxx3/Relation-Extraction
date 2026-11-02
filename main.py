from src.relation_extraction.data.semeval_loader import SemEvalLoader
from src.relation_extraction.parsing.parser import Parser
from src.relation_extraction.graph.graph import DependencyGraph


def main():
    loader = SemEvalLoader()
    parser = Parser()

    examples = loader.load_file(
        "./data/row/TRAIN_FILE.TXT"
    )

    print(f"Loaded {len(examples)} examples")

    for example in examples[:3]:
        tokens, dependencies = parser.parse(example.sentence)

        entity1_indices = loader.find_entity_indices(tokens, example.entity1_span)

        entity2_indices = loader.find_entity_indices(tokens, example.entity2_span)

        graph = DependencyGraph()
        graph.build(tokens, dependencies)

        path = graph.shortest_dependency_path(entity1_indices, entity2_indices)

        print("Sentence:")
        print(example.sentence)

        print()
        print("Entity 1:", example.entity1)
        print("Entity 1 indices:", entity1_indices)

        print("Entity 2:", example.entity2)
        print("Entity 2 indices:", entity2_indices)

        print("Relation:", example.relation)

        print()
        print("Shortest dependency path:")

        for source, relation, target, direction in path:
            print(
                f"{source.token.text} "
                f"{direction} {relation} {direction}"
                f" {target.token.text}"
            )

        print()

if __name__ == "__main__":
    main()