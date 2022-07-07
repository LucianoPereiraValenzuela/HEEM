from typing import Union, List, Optional, Dict, Tuple, Iterable
import numpy as np
import networkx as nx
from itertools import permutations
import copy
from tqdm.auto import tqdm
from time import time
import pickle
import matplotlib.pyplot as plt

from qiskit.compiler import transpile as transpile_qiskit
from qiskit.opflow.primitive_ops import TaperedPauliSumOp, PauliSumOp

from molecules import extract_Paulis
from utils import string2number, number2string, add_edge

"""
In order to simplify the programming, we use a numerical encoding to identify each Pauli string with an integer
I-> 0, X-> 1, Y-> 2, Z-> 3
Then, for example, XIZY would be mapped to the array [1, 0, 3, 2].

Similarly, we map measurements into numbers:
TPBX -> 1, TPBY -> 2, TPBZ -> 3, Bell -> 4, OmegaX -> 5, OmegaY -> 6, OmegaZ -> 7, chi -> 8, chi_tilde -> 9

Finally, we build lists of compatibility, one for each measurement. The list of compatibility of the measurement k 
should contain the arrays assigned to the Pauli strings that are compatible with that measurement. For instance, 
if we consider the measure 4 (Bell) its list of compatibility should contain [0,0], [1,1], [2,2], [3,3], because the
Bell measurement is compatible with II, XX, YY and ZZ, respectively.
"""

COMP_LIST = [[[]], [(0,), (1,)], [(0,), (2,)], [(0,), (3,)],  # One qubit measurements [I(O), X(1), Y(2), Z(3)]
             [(0, 0), (1, 1), (2, 2), (3, 3)],  # Bell (4)
             [(0, 0), (1, 1), (2, 3), (3, 2)],  # OmegaX (5)
             [(0, 0), (2, 2), (1, 3), (3, 1)],  # OmegaY (6)
             [(0, 0), (3, 3), (1, 2), (2, 1)],  # OmegaZ (7)
             [(0, 0), (1, 2), (2, 3), (3, 1)],  # Chi (8)
             [(0, 0), (2, 1), (3, 2), (1, 3)]]  # Chi tilde (9)

# Number of simultaneous measurements (0, 1 or 2)
LENGTH_MEAS = [len(x[0]) for x in COMP_LIST]

PartialMeasurement = Tuple[int, List[int]]
MoleculeType = Union[PauliSumOp, TaperedPauliSumOp]


def build_pauli_graph(labels: Union[np.ndarray, List[str], MoleculeType], print_progress: bool = False) -> nx.Graph:
    """
    Construction of the Pauli Graph.

    Parameters
    ----------
    labels: ndarray (n, N) or list[str] or MoleculeType
        Each row represents a Pauli string, and each column represents a qubit. Thus, n is the number of Pauli strings
        and N is the number of qubits. The labels are represented with indices.
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

    labels, _ = pauli_labels_numbers(labels)  # Ensure the number convention

    # Number of strings
    n = np.size(labels[:, 0])

    PG = nx.Graph()
    PG.add_nodes_from(np.arange(n))

    pbar = tqdm(range(n), desc='Pauli graph', disable=not print_progress)
    for i, v_i in zip(pbar, labels):  # Loop over each Pauli string
        for j in range(i + 1, n):  # Loop over the next Pauli strings
            v_j = labels[j, :]
            compatible_qubits = np.logical_or.reduce((v_i == v_j, v_i == 0, v_j == 0))
            if not np.all(compatible_qubits):  # If one of the qubits shared by the Pauli label, then is not commutative
                PG.add_edge(i, j)

    return PG


def empty_factors() -> Dict[str, int]:
    """
    Create an empty dictionary with one entry for each possible 2 qubit combination (II, IX, ...) --> F['00']  0,
    F['01'] = 0, .... Each entry will be filled with the number of times that two qubits in N pauli strings have that
    factor.

    Returns
   -------
    F: dict
        Empy directory with the keys '00', '01', '02', ..., '10', '11', ...
    """
    F = {str(i) + str(j): 0 for i in range(4) for j in range(4)}
    return F


def compatible_measurements_1q(measurement: int, f: np.ndarray) -> int:
    """
    Given a measurement and an array of one-qubit factors, calculates the number of compatible measurements that can
    be made with that measurement in those factors.

    Parameters
    ----------
    measurement: int
        Index of the desired one-qubit measurement. It must be in [1, 3]
    f: ndarray
        Pauli string to measure

    Returns
    -------
    n_compatibilities:  int
        Number of compatible measurements
    """
    compatible_factors = np.size(np.argwhere(f == 0)) + np.size(np.argwhere(f == measurement))
    n_compatibilities = int(compatible_factors * (compatible_factors - 1) / 2)

    return n_compatibilities


def compatible_measurements_2q(measurement: int, f: Dict[str, int]) -> int:
    """
    Given a measurement and a dictionary of two-qubit factors, calculates the number of compatible measurements
    that can be made with that measurement in those factors.

    Parameters
    ----------
    measurement: int
        Index of the desired two-qubit measurement. It must be between [4, 9].
    f: dict
        Dictionary with compatible measurement. For more info see the function empty_dict_factors.

    Returns
    -------
    n_compatibilities:  int
        Number of compatible measurements.
    """

    pairs = []
    for pair_labels in COMP_LIST[measurement]:  # Iterate over the pair of compatible Pauli labels
        pairs.append(str(pair_labels[0]) + str(pair_labels[1]))

    counts = 0
    for pair in pairs:
        counts += f[pair]

    n_compatibilities = int(counts * (counts - 1) / 2)

    return n_compatibilities


def compatible_measurements(labels: np.ndarray, T: Optional[Union[List[int], Iterable]] = None,
                            connectivity_graph: Optional[nx.Graph] = None, one_qubit: bool = True,
                            two_qubit: bool = True) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Given a set of 'n' Pauli Strings with 'N' qubits, returns three arrays regarding the compatibilities of the
    measurements. C is the number of two_qubit measurements for each pair of qubits. CM is the number of times a given
    measurement can be applied. Finally, CQ is the number of times a given qubit must be measured.

    Parameters
    ----------
    labels: ndarray (n, N)
        Pauli strings, each row represents a Pauli string and each column represents a qubit.
    T: list (optional, default=None)
        Map from theoretical qubits to physical qubits. If T[i] = j it means that the i-th theoretical qubit is
        mapped to the j-th physical qubit. If not provided, the mapping T[i] = i is assumed.
    connectivity_graph: nx.Graph (optional, default=None)
        Connectivity of the chip. Each edge represents a pair of well-connected qubits. If not provided, an all-to-
        all device is assumed.
    one_qubit: bool (optional, default=True)
        Measure compatible 1 qubit measurements for each qubit. The data is saved in CQ
    two_qubit: bool (optional, default=True)
        Measure compatible 2 qubit measurements for each measure. The data is saved in CM.

    Returns
    -------
    C:  ndarray (n, N)
        Compatibility matrix
    CM: list
        Number of times that qubits of two pauli strings are compatible with each measurement, given that T have been
        chosen as the map from theoretical qubits to physical qubits.
    CQ: list(N)
        The element CQ[i] contains the number of times that the qubit i can participate in a joint measurement with any
        other qubit though any measurement, given that T have been chosen as the map from theoretical qubits to physical
        qubits. It is the sum of the i_th row/column of the matrix C, excluding the -1 of the diagonal, the number of
        compatibilities due to one-qubit measurements.

    """

    n, N = np.shape(labels)

    C = np.diag(np.ones(N) * np.nan)  # Compatibility matrix
    CM = [0] * 10  # Compatible measurements
    CQ = [0] * N  # Compatible qubits

    if T is None:
        T = list(range(N))

    if connectivity_graph is None:
        connectivity = list(permutations(range(N), 2))
    else:
        connectivity = connectivity_graph.edges()

    for i in range(N):
        if two_qubit:
            for j in range(i + 1, N):
                if (T[i], T[j]) in connectivity:  # Connected qubits
                    PSij = labels[:, [i, j]]

                    F = empty_factors()
                    for label in PSij:  # Generate factors list
                        F[str(label[0]) + str(label[1])] += 1

                    for measurement in range(4, 10):  # Iterate over 2-qubits measurements
                        n_counts = compatible_measurements_2q(measurement, F)
                        C[i, j] += n_counts
                        C[j, i] += n_counts
                        CM[measurement] += n_counts

        if one_qubit:
            CQ[i] += np.sum(C[i, :])
            for measurement in range(1, 4):  # Iterate over 1-qubit measurements
                n_counts = compatible_measurements_1q(measurement, labels[:, i])
                CQ[i] += n_counts
                CM[measurement] += n_counts

    # TODO: Is it possible that CM gives preference to one-qubit measurements over two-qubit ones?
    return C, CM, CQ


def check_degree(connectivity_graph: nx.Graph, T: List[int], i: int, C: np.ndarray) -> None:
    """
    Check if physical qubit T[i] is connected to other ones. If not, modify the compatibility matrix and set the row and
    the column associate to the qubit to np.nan.
    """
    if nx.degree(connectivity_graph, T[i]) == 0:
        C[i, :] = np.nan
        C[:, i] = np.nan


def only_one_assigned(i: int, j: int, T: List[int], AQ: List[int], C: np.ndarray, connectivity_graph: nx.Graph) -> None:
    if i in AQ:
        assigned = i
        not_assigned = j
    else:
        assigned = j
        not_assigned = i

    # Associate the not assigned theoretical qubit to the first neighbor available
    neighbor = list(connectivity_graph[T[assigned]])[0]
    if neighbor not in T:
        T[not_assigned] = neighbor
        AQ.append(not_assigned)
        C[[i, j], [j, i]] = np.nan

        # If the neighbors of the assigned qubit are already assigned, remove the edge in the graph
        neighbors_2 = list(connectivity_graph[neighbor])
        for neighbor_2 in neighbors_2:
            if neighbor_2 in T:
                connectivity_graph.remove_edge(neighbor_2, neighbor)
                check_degree(connectivity_graph, T, T.index(neighbor_2), C)
        check_degree(connectivity_graph, T, not_assigned, C)


def transpile_connected(connectivity_graph: nx.Graph, C: np.ndarray) -> List[int]:
    """
    Construct a theoretical-physical map for the qubits, ensuring that the graphs in the chip connectivity are
    connected. For details on the arguments and the return, check the documentation of the transpile function.
    """
    C = copy.copy(C)
    connectivity_graph = copy.deepcopy(connectivity_graph)

    N = len(C)  # Number of qubits
    AQ = []  # Assigned qubits
    T = [-1] * N

    # Start with the two qubits with the highest compatibility
    i, j = np.unravel_index(np.nanargmax(C), [N, N])
    ii, jj = list(connectivity_graph.edges())[0]

    # Update the map
    T[i] = ii
    T[j] = jj

    # Add qubits to the assigned list
    AQ.append(i)
    AQ.append(j)

    C[[i, j], [j, i]] = np.nan

    # Remove used edges
    connectivity_graph.remove_edge(ii, jj)
    check_degree(connectivity_graph, T, i, C)
    check_degree(connectivity_graph, T, j, C)

    while len(AQ) < N:  # While not all qubits assigned
        i, j = np.unravel_index(np.nanargmax(C[AQ, :]), [N, N])
        i = AQ[i]

        if j in AQ:
            C[[i, j], [j, i]] = np.nan
        else:
            only_one_assigned(i, j, T, AQ, C, connectivity_graph)

    return T


def transpile_disconnected(connectivity_graph: nx.Graph, C: np.ndarray) -> List[int]:
    C = copy.copy(C)
    connectivity_graph = copy.deepcopy(connectivity_graph)

    N = len(C)
    AQ = []
    T = [-1] * N

    while len(AQ) < N:
        i, j = np.unravel_index(np.nanargmax(C), [N, N])

        if (i in AQ) and (j in AQ):
            C[[i, j], [j, i]] = np.nan
        elif (i not in AQ) and (j not in AQ):
            C[[i, j], [j, i]] = np.nan
            for ii, jj in connectivity_graph.edges():
                if (ii not in T) and (jj not in T):
                    T[i] = ii
                    T[j] = jj
                    AQ.append(i)
                    AQ.append(j)

                    for node in ii, jj:
                        neighbors = list(connectivity_graph[node])
                        for neighbor in neighbors:
                            if neighbor in T:
                                connectivity_graph.remove_edge(neighbor, node)
                                check_degree(connectivity_graph, T, T.index(neighbor), C)
                        check_degree(connectivity_graph, T, T.index(node), C)

                    break

        else:
            only_one_assigned(i, j, T, AQ, C, connectivity_graph)

    return T


def transpile(connectivity_graph: nx.Graph, C: np.ndarray, connected: bool) -> List[int]:
    """
    Construct a theoretical-physical mapping for the qubits taking into account the chip connectivity
    Parameters
    ----------
    connectivity_graph: nx.Graph
        Pauli graph
    C:  ndarray (N, N)
        Compatibility matrix. More info in function compatibilities().
    connected: bool
        If True the transpile algorithm ensures that the subgraph of the theoretical qubits in the chip is connected.
        More info in function groupingWithOrder()

    Returns
    -------
    T: list
        T is the theo-phys map chosen. T[i] = j means that the i-theoretical qubit is mapped to the j-physical qubit.
    """

    if connected:
        return transpile_connected(connectivity_graph, C)
    else:
        return transpile_disconnected(connectivity_graph, C)


def measurement_assignment(Vi: np.ndarray, Vj: np.ndarray, Mi: Union[List, List[PartialMeasurement]], AM: List[int],
                           connectivity_graph: nx.Graph, OQ: List[int], T: List[int]) -> Tuple[
    List[PartialMeasurement], bool]:
    """
    Try to assign a series of measurements so the Pauli labels Vi and Vj can be measured simultaneously. This function
    follows one of two different paths according to the input Mi:
        A) If Mi is an empty list the function assigns measurements Mi that are both compatible with Vi and Vj.

        B) If Mi is not complete, i.e. there are qubits with no assigned measurements, first the function checks if
        the currently assigned measurements Mi are compatible with Vj. If this is true then in tries to assign to the
        remaining qubits measurements that are both compatible with Vi and Vj.

    In both cases the function returns S = True iff the updated Mi is full and compatible with Vi and Vj. Else, it
    returns S = False and an unchanged Mi.

    This is the Algorithm 2 of https://arxiv.org/abs/1909.09119.

    Parameters
    ----------
    Vi: ndarray
       ith Pauli string in our numerical encoding
    Vj: ndarray
        jth Pauli string in our numerical encoding
    Mi: list[tuple]
        Current assignment of the measurement of the group i. It is a list of partial measurements. Each partial
        measurements is a tuple of two elements. The first of these elements encodes the partial measurement assigned
        and the second the qubits where it should be performed.
    AM: list
        Admissible measurements considered. Regarding our numerical encoding, it is a list of integers from 1 to 9.
        The order of the list encodes the preference of measurement assignment. For instance, the first measurement
        appearing in this list will be the one that would be preferentially assigned.
    connectivity_graph: nx.Graph
        Connectivity of the chip. Each edge represents a pair of well-connected qubits.
    OQ: List
        Order of qubits that the algorithm should follow in each iteration.
    T: List
        Theo-phys map chosen. T[i] = j means that the i-theoretical qubit is mapped to the j-physical qubit.

    Returns
    -------
    UMi: list[list]
        Updated Mi. If the algorithm fails, UMi will be equal to Mi.
    S: bool
        If the algorithm has succeeded, i.e., if Mi has been updated in a way such the group of Vi and Vj are compatible
    """

    # Check if the current assignment of Mi is compatible with Vj. If so, U contains the qubits where Mi does not act.
    U = OQ.copy()
    for PM in Mi:
        if tuple(Vj[PM[1]]) not in COMP_LIST[PM[0]]:
            return Mi, False
        else:
            for k in PM[1]:
                U.remove(k)

    common_factors = np.argwhere(Vi == Vj)
    for k in common_factors:
        if k in U:
            U.remove(k)

    """
    U contains the qubits where Mi does no act and the factors of Vi and Vj are not equal. It is in the qubits of U
    where partial measurements have to be assigned to make the strings of Vi and Vj compatible.
    """

    """
    Next loop tries to update the measurement Mi on the qubits in U. To do so it runs through the admissible
    partial measurements AM. For each of those measurements, the loop runs through all the possible set of qubits
    where the measurement can act (perm). For each element 'per' in 'perm' the code checks if 'per' is a set of
    well-connected qubits. Finally, it is checked if the measurement on those qubits is compatible with the string Vi
    and Vj (if so, by construction, the measurement will be compatible with all strings of group Vi and with Vj). If
    there is success, UMi is updated with that partial measurement, the qubits where this partial measurement are 
    deleted of U, and we begin again if U is not empty. If we managed to empty U, the update would have succeeded
    """

    WC = [list(x) for x in connectivity_graph.edges()]

    UMi = Mi[:]
    while len(U) != 0:
        for Eps in AM:  # Admissible measurements
            if len(U) >= LENGTH_MEAS[Eps]:  # Only enters if there is enough qubits to measure
                for per in permutations(U, LENGTH_MEAS[Eps]):  # Check each permutation between the remained qubits
                    per = list(per)
                    connected = True
                    if LENGTH_MEAS[Eps] >= 2:
                        Tper = [int(T[per[0]]), int(T[per[1]])]
                        connected = Tper in WC  # Connectivity check

                    if connected:
                        # Compatibility check
                        if (tuple(Vi[per]) in COMP_LIST[Eps]) and (tuple(Vj[per]) in COMP_LIST[Eps]):
                            UMi.append((Eps, per))
                            for k in per:
                                U.remove(k)
                            break
                else:
                    continue
                break
        else:
            return Mi, False
    return UMi, True


def grouping_entangled(labels: np.ndarray, pauli_graph: nx.Graph, connectivity_graph: Optional[nx.Graph] = None,
                       connected_graph: bool = True, print_progress: bool = False, transpiled_order: bool = True) -> \
        Tuple[List[List[int]], List[List[PartialMeasurement]], List[int]]:
    """
    Given a set of Pauli strings, groups them using entangled measurements. The chip connectivity can be taken into
    account to avoid two-qubit measurements between non-connected qubits.

    Parameters
    ----------
    labels: ndarray (n, N)
        A total of n Pauli strings for N qubits. The Pauli labels are in the number convention.
    pauli_graph: nx.Graph
       Pauli graph
    connectivity_graph: nx.Graph (optional, default=None)
        Graph that represents the connectivity of the chip. If not provided, an all-to-all device is assumed.
    connected_graph: bool (optional, default=False)
        If True the transpile algorithm ensures that the subgraph of the theoretical qubits in the chip is connected.
        Else, the algorithm does not ensure that the subgraph of the theoretical qubits in the chip is connected,
        instead tries to optimize omega(T) in a greedy way.
    print_progress: bool (optional, default=False)
        If true, print the progress of the pauli graph and the grouping.
    transpiled_order: bool (optional, default=None)
        If True, order the qubits and the measurements depending on the compatibility matrices. Otherwise, the order
        of the measurements is arbitrary.

    Returns
    -------
    groups: list[list[int]]
        Indices of the grouped Pauli labels
    measurements: list[list[tuple[int, list]]
        Measurements for each group of Pauli labels. Each grouped measurements is constituted with multiple one qubit
        and two qubit partial measurements. The first index denotes the partial measurement to perform, and the
        second one the qubits to measure.
    T: list[int]
        T is the theo-phys map chosen. T[i]=j means that the i-theoretical qubit is mapped to the j-physical qubit.

    ¡Important!: In 'Measurement' output the indexes that we use to refer to the qubits are theoretical indexes, not the
    correspondent physical indexes, i.e., if we have the i-theoretical qubit is mapped to the j-physical qubit through
    T, in other words T[i]=j, we use the index i and not the j to refer to that qubit.
    """
    n, N = np.shape(labels)

    if connectivity_graph is None:  # If no connectivity, all-to-all assumed
        connectivity_graph = nx.Graph()
        connectivity_graph.add_edges_from(list(permutations(list(range(N)), 2)))
    elif type(connectivity_graph) is list:  # If a list is provided, construct the graph
        temp = connectivity_graph
        connectivity_graph = nx.Graph()
        connectivity_graph.add_edges_from(temp)

    if len(connectivity_graph.nodes) < N:
        raise Exception('The number of qubits in the device is not high enough. Use a bigger device.')

    # Paulis in descending order of compatible measurements
    SV = sorted(pauli_graph.degree, key=lambda x: x[1], reverse=True)
    SV = [x[0] for x in SV]

    if transpiled_order:
        C, _, _ = compatible_measurements(labels, one_qubit=False)
        T = transpile(connectivity_graph, C, connected_graph)
        _, CM, CQ = compatible_measurements(labels, T=T, connectivity_graph=connectivity_graph)

        AM = list(np.argsort(CM[1:])[::-1] + 1)
        OQ = list(np.argsort(CQ)[::-1])
    else:
        AM = [4, 6, 7, 8, 9, 5, 1, 2, 3]  # Preference the 2-qubit measurements over the 1-qubit ones
        OQ = list(range(N))  # default order for the qubits
        T = list(range(N))  # Naive theoretical physical qubit with T[i] = i

    groups = []
    measurements = []

    pbar = tqdm(total=n, desc='Grouping entangled', disable=not print_progress)
    while len(SV) > 0:
        pbar.update()
        i = SV.pop(0)  # Run the Pauli strings in decreasing order of CQ
        Mi = []
        GroupMi = [i]
        for j in SV:  # Try to make the group of the string i as large as possible
            Mi, S = measurement_assignment(labels[i], labels[j], Mi, AM, connectivity_graph, OQ, T)
            if S:  # Success grouping label i and j
                pbar.update()
                GroupMi.append(j)

        for index in GroupMi[1:]:  # Remove grouped Pauli labels
            SV.remove(index)

        QWM = list(range(N))  # If a qubit has no measurement assigned, then use 1-qubit measurement
        for partial_measurement in Mi:
            for qubit in partial_measurement[1]:
                QWM.remove(qubit)

        for qubit in QWM:
            Mi.append((max(labels[GroupMi, qubit]), [qubit]))

        groups.append(GroupMi)
        measurements.append(Mi)
    pbar.close()

    return groups, measurements, T


def grouping_tpb(labels: np.ndarray, print_progress: bool = False, pauli_graph: Optional[nx.Graph] = None) -> Tuple[
    List[List[int]], List[List[PartialMeasurement]]]:
    """
    Construction of the TPB groups, i.e., the groups when considering the TPB basis.

    Parameters
    ----------
    labels: ndarray (n, N)
        A total of n Pauli strings for N qubits. The Pauli labels are in the number convention.
    print_progress: bool (optional, default=False)
        If True, print the progress bar for the Pauli graph building.
    pauli_graph: nx.Graph (optional, default=None)
        If the Pauli graph is already computed, then it can be used here and not computed twice

    Returns
    -------
    groups: list[list]
        Indices of the grouped Pauli labels
    measurements: list[list[list]]
        Measurements for each group of Pauli labels. Each grouped measurements is constituted with multiple one qubit
        and two qubit partial measurements. The first index denotes the partial measurement to perform, and the
        second one the qubits to measure.
    """

    if pauli_graph is None:
        pauli_graph = build_pauli_graph(labels, print_progress=print_progress)

    # Graph coloring code of networkx. By default, it uses LDFC strategy
    coloring = nx.coloring.greedy_color(pauli_graph)

    # Obtain grouped Pauli labels
    groups = [[k for k, v in coloring.items() if v == color] for color in set(coloring.values())]

    # TPB measurements assignment
    measurements = []
    for group in groups:
        Mi = []
        for k in range(labels.shape[1]):  # Iterate over the qubits
            Mi.append((max(labels[group, k]), [k]))
        measurements.append(Mi)

    return groups, measurements


def test_grouping_paulis(labels: np.ndarray, groups: List[List[int]]) -> bool:
    """
    Check that all Pauli labels are measured one, and only one time.

    Parameters
    ----------
    labels: ndarray (n, N)
        A total of n Pauli strings for N qubits. The Pauli labels are in the number convention.
    groups: list[list]
        Indices of the grouped Pauli labels

    Returns
    -------
    If the test passes, True is returned. Otherwise, and exception is raised.
    """

    n = len(labels)

    pauli_indices = list(range(n))
    index = None
    try:
        for group in groups:
            for index in group:
                pauli_indices.remove(index)  # Remove index of measured Pauli label
    except ValueError:  # If try to remove twice the same index
        raise Exception('The Pauli label ({}) is measured at least two times.'.format(labels[index]))

    # If some index has not been removed
    if len(pauli_indices) > 0:
        raise Exception('The following Pauli indices are not measured {}'.format(pauli_indices))

    return True  # Test passed


def test_grouping_measurements(labels: np.ndarray, groups: List[List[int]],
                               measurements: List[List[PartialMeasurement]]) -> bool:
    """
    Check that all the measurements correctly corresponds to the grouping labels, and no error has occurred during the
    grouping algorithm.

    Parameters
    ----------
    labels: ndarray (n, N)
        A total of n Pauli strings for N qubits. The Pauli labels are in the number convention.
    groups: list[list]
        Indices of the grouped Pauli labels
    measurements: list[list[tuple[int, list[int]]]]
        Measurements for each group of Pauli labels. Each grouped measurements is constituted with multiple one qubit
        and two qubit partial measurements. The first index denotes the partial measurement to perform, and the
        second one the qubits to measure.

    Returns
    -------
    If the test passes, True is returned. Otherwise, and exception is raised.
    """
    for group, measurement in zip(groups, measurements):  # Iterate over the groups
        grouped_labels = labels[group]
        for partial_measurement in measurement:  # Iterate over the partial measurements
            index_measure, qubits = partial_measurement
            for grouped_label in grouped_labels:  # Check each grouped Pauli label
                if list(grouped_label[qubits]) == [0] * len(qubits):  # If the identity is applied, continue
                    continue
                # If the Pauli is not compatible with the assigned measurement
                elif tuple(grouped_label[qubits]) not in COMP_LIST[index_measure]:
                    raise Exception(
                        'The Pauli label {} is incompatible with the measurement {} over the qubit(s) {}'.format(
                            grouped_label, index_measure, *qubits))
    return True  # Test passed


def test_connectivity(measurements: List[List[PartialMeasurement]], T: List[int],
                      connectivity: List[List[int]]) -> bool:
    """
    Test that entangled measurements are only provided between well-connected qubits.

    Parameters
    ----------
    measurements: list[list[list]]
        Measurements for each group of Pauli labels. Each grouped measurements is constituted with multiple one qubit
        and two qubit partial measurements. The first index denotes the partial measurement to perform, and the
        second one the qubits to measure.
    T: List
        T is the theo-phys map chosen. T[i]=j means that the i-theoretical qubit is mapped to the j-physical qubit.
    connectivity: list
        Connectivity of the chip.

    Returns
    -------
    If the test passes, True is returned. Otherwise, and exception is raised.
    """
    for measurement in measurements:
        for partial_measurement in measurement:
            if partial_measurement[0] > 3:  # Only check two qubit measurements
                phy_qubits = [T[q] for q in partial_measurement[1]]
                if phy_qubits not in connectivity:
                    raise Exception('Entangled measurement between non-connected '
                                    'qubits {} and {}'.format(phy_qubits[0], phy_qubits[1]))
    return True  # Test passed


def pauli_labels_numbers(labels: Union[np.ndarray, List[str], MoleculeType],
                         coeffs: Optional[Union[List[complex], np.ndarray]] = None) -> Tuple[
    np.ndarray, Union[List[complex], np.ndarray]]:
    """
    Return the Pauli labels in the number convention.
    """
    if type(labels[0]) == str:
        labels = string2number(labels)
        if coeffs is None:
            coeffs = [0] * len(labels)
    elif (type(labels[0]) == np.ndarray) or (type(labels[0]) == list):
        labels = np.array(labels, dtype=np.int8)
        if coeffs is None:
            coeffs = [0] * len(labels)
    elif (type(labels) is TaperedPauliSumOp) or (type(labels) is PauliSumOp):
        labels, coeffs = extract_Paulis(labels)
        labels = string2number(labels)
    else:
        raise Exception('Pauli Labels\' type not implemented. Please, use a list of strings, a numpy array within'
                        ' the number convention, a PauliSumOp, or a TaperedPauliSumOp')

    return labels, coeffs


class Grouping:
    def __init__(self, labels: Optional[Union[np.ndarray, List[str]]] = None,
                 connectivity: Optional[Union[List[int], Tuple[int, int]]] = None, tests: bool = True,
                 connected_graph: bool = True, print_progress: bool = True, method: str = 'HEEM',
                 transpiled_order: bool = True, pauli_graph: Optional[nx.Graph] = None, load: Optional[str] = None,
                 coeffs: Optional[Union[List[complex], np.ndarray]] = None) -> None:
        if load is None:
            self._labels, self._coeffs = pauli_labels_numbers(labels, coeffs)  # Reformat labels and obtain coeffs

            self._connectivity = connectivity
            self._tests = tests
            self._connected_graph = connected_graph
            self._print_progress = print_progress
            self._transpiled_order = transpiled_order
            self._pauli_graph = pauli_graph

            self.T = None
            self.groups = None
            self.measurements = None
            self.grouping_time = 0
            self.n_groups = 0

            self._method = method

            if self._method == 'TPB':
                self._entangled = False
            elif self._method == 'EM':
                self._entangled = True
                self._connectivity = None
            elif self._method == 'HEEM':
                self._entangled = True
                if self._connectivity is None:
                    print('HEEM grouping method have been chosen, but no chip connectivity provided.'
                          ' Switching to EM assuming an all-to-all connectivity')
            else:
                raise Exception(
                    '{} not implemented. Please, use one of these methods: TPB, EM or HEEM.'.format(self._method))

            self._n_qubits = len(self._labels[0])

            if self._connectivity is None:
                self._connectivity = permutations(range(self._n_qubits), 2)
            self._connectivity = [list(x) for x in self._connectivity]

            self._connectivity_graph = nx.Graph()
            self._connectivity_graph.add_edges_from(self._connectivity)

        else:
            self._load_data(load)

    def group(self):
        self._check_pauli_graph()

        t0 = time()
        if self._entangled:  # Use entangled measurements
            self.groups, self.measurements, self.T = grouping_entangled(self._labels, self._pauli_graph,
                                                                        connectivity_graph=self._connectivity_graph,
                                                                        connected_graph=self._connected_graph,
                                                                        print_progress=self._print_progress,
                                                                        transpiled_order=self._transpiled_order)
        else:  # Only 1-qubit measurements
            self.groups, self.measurements = grouping_tpb(self._labels, print_progress=self._print_progress,
                                                          pauli_graph=self._pauli_graph)
            self.T = list(range(self._n_qubits))

        tf = time()

        self.grouping_time = tf - t0
        self.n_groups = len(self.groups)

        if self._tests:
            test_grouping_measurements(self._labels, self.groups, self.measurements)
            test_grouping_paulis(self._labels, self.groups)
            test_connectivity(self.measurements, self.T, self._connectivity)

    def labels_string(self):
        return number2string(self._labels)

    def save_data(self, file_name: str) -> None:
        file_name += '.pickle'
        with open(file_name, 'wb') as file:
            pickle.dump(self.__dict__, file)

    def _load_data(self, file_name: str) -> None:
        file_name += '.pickle'
        with open(file_name, 'rb') as file:
            self.__dict__.update(pickle.load(file))

    def _check_grouping(self) -> None:
        if self.groups is None:
            self.group()

    def _check_pauli_graph(self) -> None:
        if self._pauli_graph is None:
            self._pauli_graph = build_pauli_graph(self._labels, print_progress=self._print_progress)

    def draw_entangled_measurements(self, seed=0) -> nx.Graph:
        self._check_grouping()

        plt.figure()
        G = nx.Graph()

        for measurement in self.measurements:
            for partial_measurement in measurement:
                if partial_measurement[0] > 3:
                    add_edge(G, self.T[partial_measurement[1][0]], self.T[partial_measurement[1][1]])

        e = list(G.edges())
        pos = nx.spring_layout(G, seed=seed)

        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")

        nx.draw_networkx_edges(G, pos, edgelist=e, width=3)
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels)

        return G

    def draw_transpiled_chip(self, seed=0) -> nx.Graph:
        self._check_grouping()

        plt.figure()

        G = nx.Graph()
        for edge in self._connectivity:
            try:
                if edge[0] < edge[1]:
                    add_edge(G, edge[0], edge[1])
            except ValueError:
                pass

        e_all = list(G.edges())

        e_transpiled = []
        for node1, node2 in e_all:
            try:
                if node1 < node2:
                    if (node1 in self.T) and (node2 in self.T):
                        e_transpiled.append((node1, node2))
            except ValueError:
                pass

        pos = nx.spring_layout(G, seed=seed)
        pos2 = {}
        mapping = {}
        for key in pos.keys():
            new_key = key
            if key in self.T:
                new_key = str(key) + '/' + str(self.T.index(key))
            pos2[new_key] = pos[key]
            mapping[key] = new_key
        nx.relabel_nodes(G, mapping, copy=False)

        nx.draw_networkx_nodes(G, pos2)
        nx.draw_networkx_labels(G, pos2, font_size=12, font_family="sans-serif")

        nx.draw_networkx_edges(G, pos, edgelist=e_all, width=3)
        nx.draw_networkx_edges(G, pos, edgelist=e_transpiled, width=3, edge_color='red')

        return G

    def draw_pauli_graph(self, color: Optional[bool] = True, names: Optional[bool] = True) -> None:
        self._check_pauli_graph()

        names_map = {}
        color_map = None

        if names:
            names_temp = number2string(self._labels)
            for i in range(len(names_temp)):
                names_map[i] = names_temp[i]

        if color:
            self._check_grouping()

            color_map = [0] * len(self._labels)
            for i in range(self.n_groups):
                grouped = self.groups[i]
                for node in grouped:
                    color_map[node] = i

            color_map = np.array(color_map)

        nx.draw(self._pauli_graph, labels=names_map, with_labels=True, node_color=color_map)

    def draw_measurements(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        measurements_count = [0] * len(COMP_LIST)
        for measure in self.measurements:
            for partial_measure in measure:
                measure_index = partial_measure[0]
                measurements_count[measure_index] += 1

        measurements_count = measurements_count[1:]  # Do not plot the identity

        labels = ['X', 'Y', 'Z', 'Bell', r'$\Omega_X$', r'$\Omega_Y$', r'$\Omega_Z$', r'$\chi$', r'$\tilde{\chi}$']
        ax.bar(np.arange(3), measurements_count[:3])
        ax.bar(np.arange(3, 9), measurements_count[3:])
        ax.set_xticks(np.arange(9))
        ax.set_xticklabels(labels)

    def n_cnots(self, coupling_map: List[List[int]]):
        from heem import measure_circuit_factor

        circuits = [measure_circuit_factor(measure, self._n_qubits, make_measurements=False) for measure in
                    self.measurements]
        circuits_transpiled = transpile_qiskit(circuits, coupling_map=coupling_map, initial_layout=self.T[::-1])

        n_cnots = 0
        for circuit in circuits_transpiled:
            try:
                n_cnots += circuit.count_ops()['cx']
            except KeyError:
                pass

        return n_cnots

    def _shuffle_labels(self) -> None:
        order_paulis = np.arange(len(self._labels))
        np.random.shuffle(order_paulis)
        self._labels = self._labels[order_paulis]

    def _shuffle_qubits(self) -> None:
        order_qubits = np.arange(self._n_qubits)
        np.random.shuffle(order_qubits)
        self._labels = self._labels[:, order_qubits]
