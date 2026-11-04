from src.relation_extraction.data.semeval_loader import SemEvalLoader
from src.relation_extraction.parsing.parser import Parser
from src.relation_extraction.graph.graph import DependencyGraph


def main():
    loader = SemEvalLoader()
    parser = Parser()

    examples = loader.load_file("./data/raw/TRAIN_FILE.TXT")

    print(f"Loaded {len(examples)} examples")
    no_path_count = 0

    for i in range(len(examples)):
        tokens, dependencies = parser.parse(examples[i].sentence)

        entity1_indices = loader.find_entity_indices(tokens, examples[i].entity1_span)

        entity2_indices = loader.find_entity_indices(tokens, examples[i].entity2_span)

        graph = DependencyGraph()
        graph.build(tokens, dependencies)

        path = graph.shortest_dependency_path(entity1_indices, entity2_indices)

        if path is None:
            no_path_count += 1
            continue

        if i < 5:
            print("Sentence:")
            print(examples[i].sentence)

            print()
            print("Entity 1:", examples[i].entity1)
            print("Entity 1 indices:", entity1_indices)

            print("Entity 2:", examples[i].entity2)
            print("Entity 2 indices:", entity2_indices)

            print("Relation:", examples[i].relation)

            print()
            print("Shortest dependency path:")

            for source, relation, target, direction in path:
                if direction == "->":
                    print(
                        f"{source.token.text} "
                        f"--{relation}-> "
                        f"{target.token.text}"
                    )
                else:
                    print(
                        f"{source.token.text} "
                        f"<-{relation}-- "
                        f"{target.token.text}"
                    )

            print()

if __name__ == "__main__":
    main()