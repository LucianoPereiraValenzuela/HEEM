from typing import List
import numpy as np
import networkx as nx
import copy
import joblib
import contextlib


def random_labels(n: int, N: int) -> np.ndarray:
    """
    Return a total of n Pauli strings of N qubits.
    """
    labels = []
    for i in range(n):
        labels.append(np.random.randint(4, size=N))
    return np.array(labels)


def string2number(labels: List[str]) -> np.ndarray:
    mapping = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}

    n = len(labels)  # Number of labels
    N = len(labels[0])  # Number of qubits

    labels_temp = copy.copy(labels)
    labels = np.zeros((n, N), dtype=np.int8)  # Since the indices < 256, use int8 to reduce memory usage
    for i in range(n):
        for j in range(N):
            labels[i, j] = mapping[labels_temp[i][j]]

    return labels


def number2string(labels: np.ndarray) -> List[str]:
    """
    Return the Pauli labels from the number convention , e.g. [0, 2, 1, 3], to the string convention 'IYXZ'.
    """
    mapping = ['I', 'X', 'Y', 'Z']

    # If only one label is provided, then reshape to (1, N)
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
    """
    Add an edge in the graph between node1 and node2 with width equals to one. If the edge already exists, then the
    weight increases by one.
    """
    if node2 < node1:
        node1, node2 = node2, node2  # Ensure that node1 < node2

    if node1 != node2:  # Do not include self edges
        edges = list(G.edges())
        if (node1, node2) in edges:
            last_weight = nx.get_edge_attributes(G, "weight")[(node1, node2)]
        else:
            last_weight = 0

        G.add_edge(node1, node2, weight=last_weight + 1)
    else:
        G.add_node(node1)


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar given as argument. Code extracted from the
    stack overflow response: https://stackoverflow.com/a/58936697/8237804
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
