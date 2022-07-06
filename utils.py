from typing import List
import numpy as np
import networkx as nx
import copy


def string2number(labels: List[str]) -> np.ndarray:
    mapping = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
    n = len(labels)
    N = len(labels[0])
    labels_temp = copy.copy(labels)
    labels = np.zeros((n, N), dtype=np.int8)
    for i in range(n):
        for j in range(N):
            labels[i, j] = mapping[labels_temp[i][j]]

    return labels


def number2string(labels: np.ndarray) -> List[str]:
    """
    Return the Pauli labels from the number convention , e.g. [0, 2, 1, 3], to the string convention 'IYXZ'.
    """
    mapping = ['I', 'X', 'Y', 'Z']

    shape = np.shape(labels)
    if len(shape) == 2:
        n, N = shape
    else:
        labels = labels.reshape(1, -1)
        n, N = np.shape(labels)

    labels_string = []
    for i in range(n):
        string = ''
        for j in range(N):
            string += mapping[labels[i, j]]
        labels_string.append(string)
    return labels_string


def add_edge(G: nx.Graph, node1: int, node2: int) -> None:
    if node2 < node1:
        node1, node2 = node2, node2

    if node1 != node2:
        edges = list(G.edges())
        if (node1, node2) in edges:
            last_weight = nx.get_edge_attributes(G, "weight")[(node1, node2)]
        else:
            last_weight = 0

        G.add_edge(node1, node2, weight=last_weight + 1)
    else:
        G.add_node(node1)
