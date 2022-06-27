import numpy as np
import networkx as nx
from itertools import permutations
import copy
from tqdm.auto import tqdm

# TODO: All required functions for grouping with and without entanglement. Also, a class with some of the most used
#  methods and attributes

"""
In order to simplify the programming, we use a numerical encoding to identify each Pauli string with an integer
I-> 0, X-> 1, Y-> 2, Z-> 3
Then, for example, XIZY would be mapped to the array [1, 0, 3, 2].

Similarly, we map measurements into numbers:
TPBX -> 1, TPBY -> 2, TPBZ -> 3, Bell -> 4, OmegaX -> 5, OmegaY -> 6, OmegaZ -> 7, chi -> 8, chi_tilde -> 9
(Note: if some measurement output has the number '0' it means that any measure is valid.)

Finally, we build lists of compatibility, one for each measurement. The list of compatibility of the measurement k 
should contain the arrays assigned to the Pauli strings that are compatible with that measurement. 
For instance, if we consider the measure 4 (Bell) its list of compatibility should contain [0,0], [1,1], [2,2], [3,3],
because the Bell measurement is compatible with II, XX, YY and ZZ, respectively.

Thus, when checking the compatibility of the strings v_i and v_j with the measurement k on the qubits (l, m), what we 
should do is checking if [v_i(l),v_i(m)] and [v_j(l),v_j(m)] are both in the compatibility list of the measurement k.
For example, if we had v_i=YIZZ=[2, 0, 3, 3] and v_j=XIZY=[1, 0, 3, 2], and we wanted to check if theses strings are
compatible with the measurement 4 (Bell) on the qubits (2, 3). What we have to do is checking if [v_i(2), v_i(3)]=[3, 3]
and [v_j(2),v_j(2)]=[3, 2] are in the compatibility list of the measurement 4. As this compatibility list is 
Comp_4=[[0, 0], [1, 1], [2, 2], [3, 3]], we have that [v_i(3), v_i(4)] belongs to Comp_4 but [v_j(3),v_j(4)] does not.
In consequence, the measurement 4 on the qubits (3, 4) is not compatible with v_i and v_j. 
"""

COMP_LIST = [[[]], [[0], [1]], [[0], [2]], [[0], [3]],  # One qubit measurements (0, 1, 2, 3)
             [[0, 0], [1, 1], [2, 2], [3, 3]],  # Bell (4)
             [[0, 0], [1, 1], [2, 3], [3, 2]],  # OmegaX (5)
             [[0, 0], [2, 2], [1, 3], [3, 1]],  # OmegaY (6)
             [[0, 0], [3, 3], [1, 2], [2, 1]],  # OmegaZ (7)
             [[0, 0], [1, 2], [2, 3], [3, 1]],  # Chi (8)
             [[0, 0], [2, 1], [3, 2], [1, 3]]]  # Chi tilde (9)

# Number of simultaneous measurements (0, 1 or 2)
LENGTH_MEAS = [len(x[0]) for x in COMP_LIST]


def PauliGraph(PS, print_progress=False):
    """
    Construction of the Pauli Graph.

    Parameters
    ----------
    PS: ndarray (n, N) # TODO: Which representation is used, labels or numbers?
        Each row represents a Pauli string, and each column represents a qubit. Thus, n is the number of Pauli strings
        and N is the number of qubits.
    print_progress: bool (optional, default=False)
        If true, print a tqdm progress bar for the calculation of the graph.

    Return
    ------
    PG: nx.Graph
        The Pauli graph represents the non commutativity of the n Pauli strings. Its nodes are Pauli strings, and its
        edges exist if and only if two nodes are NOT qubit-wise commutative. Two strings are qubit-wise commutative if
        for each qubit at least one of these conditions is True:
            a) both strings have the same factor,
            b) one of the strings has a factor I ([0] in our encoding).
    """

    # Number of strings
    n = np.size(PS[:, 0])

    pbar = tqdm(total=n, desc='Computing Pauli graph', disable=not print_progress)

    PG = nx.Graph()
    PG.add_nodes_from(np.arange(n))

    for i in range(n):  # Loop over each Pauli string v_i
        v_i = PS[i, :]

        for j in range(i + 1, n):  # Nested loop over the following Pauli strings v_j
            v_j = PS[j, :]
            compatiblequbits = np.logical_or.reduce((v_i == v_j, v_i == 0, v_j == 0))
            if not np.all(compatiblequbits):  # If at least one of the qubits shared by the PS is not commutative
                PG.add_edges_from([(i, j)])  # add an edge in the Pauli Graph
            pbar.update()
        pbar.close()

    return PG
