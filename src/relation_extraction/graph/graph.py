import networkx as nx

from src.relation_extraction.graph.node import Node
from src.relation_extraction.graph.edge import Edge
from src.relation_extraction.models.token import Token
from src.relation_extraction.models.dependency import Dependency


class DependencyGraph:
    def __init__(self):
        self.nodes: dict[int, Node] = {}
        self.edges: list[Edge] = []
        self.graph = nx.Graph()

    def build(self, tokens: list[Token], dependencies: list[Dependency]) -> None:
        self.nodes = {token.token_index: Node(token=token) for token in tokens}
        self.edges = [
            Edge(
                source=self.nodes[dependency.dependent],
                target=self.nodes[dependency.head],
                relation=dependency.relation,
            )
            for dependency in dependencies
            if dependency.dependent != dependency.head
            and dependency.dependent in self.nodes
            and dependency.head in self.nodes
        ]

        self.graph.add_nodes_from(self.nodes.keys())

        for edge in self.edges:
            source = edge.source.token.token_index
            target = edge.target.token.token_index
            self.graph.add_edge(source, target, relation=edge.relation, source=source, target=target)

    def get_node(self, token_index: int) -> Node | None:
        return self.nodes.get(token_index)

    def get_children(self, token_index: int) -> list[Node]:
        return [
            edge.source for edge in self.edges
            if edge.target.token.token_index == token_index
        ]

    def get_parent(self, token_index: int) -> Node | None:
        for edge in self.edges:
            if edge.source.token.token_index == token_index:
                return edge.target
        return None

    def shortest_dependency_path(self, entity1_indices: list[int], entity2_indices: list[int]) -> list[tuple[Node, str, Node, str]]:
        best_path = None

        for source in entity1_indices:
            for target in entity2_indices:
                try:
                    path = nx.shortest_path(self.graph, source=source, target=target)

                except nx.NetworkXNoPath:
                    continue
                    #чуть позже может быть добавлена отдельная метка для предложений, у которых не пути между сущностями
                if best_path is None or len(path) < len(best_path):
                    best_path = path

        if best_path is None:
            raise nx.NetworkXNoPath("No path between entities")

        result = []

        for i in range(len(best_path) - 1):
            source = best_path[i]
            target = best_path[i + 1]
            edge_data = self.graph[source][target]

            relation = edge_data["relation"]

            original_source = edge_data["source"]
            original_target = edge_data["target"]

            if source == original_source and target == original_target:
                direction = "->"
            else:
                direction = "<-"

            result.append((self.nodes[source], relation, self.nodes[target], direction))

        return result