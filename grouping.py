import numpy as np
import networkx as nx
from itertools import permutations
import copy
from tqdm.auto import tqdm

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

COMP_LIST = [[[]], [[0], [1]], [[0], [2]], [[0], [3]],  # One qubit measurements (0, 1, 2, 3)
             [[0, 0], [1, 1], [2, 2], [3, 3]],  # Bell (4)
             [[0, 0], [1, 1], [2, 3], [3, 2]],  # OmegaX (5)
             [[0, 0], [2, 2], [1, 3], [3, 1]],  # OmegaY (6)
             [[0, 0], [3, 3], [1, 2], [2, 1]],  # OmegaZ (7)
             [[0, 0], [1, 2], [2, 3], [3, 1]],  # Chi (8)
             [[0, 0], [2, 1], [3, 2], [1, 3]]]  # Chi tilde (9)

# Number of simultaneous measurements (0, 1 or 2)
LENGTH_MEAS = [len(x[0]) for x in COMP_LIST]


def build_pauli_graph(PS, print_progress=False):
    """
    Construction of the Pauli Graph.

    Parameters
    ----------
    PS: ndarray (n, N)
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

    # Number of strings
    n = np.size(PS[:, 0])

    PG = nx.Graph()
    PG.add_nodes_from(np.arange(n))

    pbar = tqdm(range(n), desc='Pauli graph', disable=not print_progress)
    for i in pbar:  # Loop over each Pauli string v_i
        v_i = PS[i, :]
        for j in range(i + 1, n):  # Nested loop over the following Pauli strings v_j
            v_j = PS[j, :]
            compatible_qubits = np.logical_or.reduce((v_i == v_j, v_i == 0, v_j == 0))
            if not np.all(compatible_qubits):  # If at least one of the qubits shared by the PS is not commutative
                PG.add_edges_from([(i, j)])  # add an edge in the Pauli Graph

    return PG


def empty_factors():
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


def compatible_measurements_1q(measurement, F):
    """
    Given a measurement and an array of one-qubit factors, calculates the number of compatible measurements that can
    be made with that measurement in those factors.

    Parameters
    ----------
    measurement: int
        Index of the desired one-qubit measurement. It must be in [1, 3]
    F: ndarray
        Pauli string to measure

    Returns
    -------
    n_compatibilities:  int
        Number of compatible measurements
    """
    compatible_factors = np.size(np.argwhere(F == 0)) + np.size(np.argwhere(F == measurement))
    n_compatibilities = compatible_factors * (compatible_factors - 1) / 2

    return n_compatibilities


def compatible_measurements_2q(measurement, F):
    """
    Given a measurement and a dictionary of two-qubit factors 'F', calculates the number of compatible measurements
    that can be made with that measurement in those factors.

    Parameters
    ----------
    measurement: int
        Index of the desired two-qubit measurement. It must be between [4, 9].
    F: dict
        Dictionary with compatible measurement. For more info see the function empty_dict_factors.

    Returns
    -------
    n_compatibilities:  int
        Number of compatible measurements.
    """

    pair = []
    for pair_labels in COMP_LIST[measurement]:  # Iterate over the pair of compatible Pauli labels
        pair.append(str(pair_labels[0]) + str(pair_labels[1]))

    counts = 0
    for pair_temp in pair:
        counts += F[pair_temp]

    n_compatibilities = counts * (counts - 1) / 2

    return n_compatibilities


def compatible_measurements(labels, T=None, connectivity_graph=None, one_qubit=True, two_qubit=True):
    """
    Given a set of 'n' Pauli Strings with 'N' qubits, returns three arrays regarding the compatibilities of the
    measurements. C is the number of two_qubit measurements for each pair of qubits. CM is the number of times a given
    measurement can be applied. Finally, CQ is the number of times a given qubit must be measured.

    Parameters
    ----------
    labels: ndarray (n, N)
        Pauli strings, each row represents a Pauli string and each column represents a qubit.
    T: list
        Map from theoretical qubits to physical qubits. If T[i] = j it means that the i-th theoretical qubit is
        mapped to the j-th physical qubit.
    connectivity_graph: nx.Graph
        Connectivity graph of the chip. Its vertices represent physical qubits and its edges physical connections
        between them.
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

    C = np.diag(-1 * np.ones(N, dtype=int))
    CM = [0] * 10
    CQ = [0] * N

    if T is None:
        T = list(range(N))
        connectivity_graph = nx.Graph()
        connectivity_graph.add_edges_from(list(permutations(list(range(N)), 2)))

    for i in range(N):  # First qubit
        if two_qubit:
            for j in range(i + 1, N):  # Second qubit
                if [T[i], T[j]] in connectivity_graph.edges():  # Connected qubits
                    PSij = labels[:, [i, j]]

                    F = empty_factors()
                    for label in PSij:  # Generate factors list
                        F[str(label[0]) + str(label[1])] += 1

                    for measurement in range(4, 10):  # Fill compatibility measurements between qubits i and j
                        n_counts = compatible_measurements_2q(measurement, F)

                        C[i, j] += n_counts
                        C[j, i] += n_counts
                        CM[measurement] += n_counts

        if one_qubit:
            CQ[i] = 1 + np.sum(C[i, :])
            for measurement in range(1, 4):
                n_counts = compatible_measurements_1q(measurement, labels[:, i])
                CQ[i] += n_counts
                CM[measurement] += n_counts

    return C, CM, CQ


def check_degree(connectivity_graph, T, i, C):
    """
    Check if physical qubit T[i] is connected to other ones. If not, modify the compatibility matrix
    """
    if nx.degree(connectivity_graph, T[i]) == 0:
        C[i, :] = -1
        C[:, i] = -1


def only_one_assigned(i, j, T, AQ, C, connectivity_graph):
    if i in AQ:
        assigned = i
        not_assigned = j
    else:
        assigned = j
        not_assigned = i

    # Assign the not assigned theoretical qubit to the first neighbor available
    neighbor = list(connectivity_graph.neighbors(T[assigned]))[0]
    if neighbor not in T:
        T[not_assigned] = neighbor
        AQ.append(not_assigned)
        C[i, j] = -1
        C[j, i] = -1

        # If the neighbors_2 of the just assigned qubit are also assigned,
        # remove the edge in the graph because it is not available
        neighbors_2 = copy.copy(connectivity_graph.neighbors(neighbor))
        for neighbor_2 in neighbors_2:
            if neighbor_2 in T:
                connectivity_graph.remove_edge(neighbor_2, neighbor)
                check_degree(connectivity_graph, T, T.index(neighbor_2), C)
        check_degree(connectivity_graph, T, not_assigned, C)


def transpile_connected(connectivity_graph, C):
    """
    Construct a theoretical-physical map for the qubits, ensuring that the graphs in the chip connectivity are
    connected. For details on the arguments and the return, check the documentation of the transpile function.
    """
    C = copy.copy(C)
    connectivity_graph = copy.deepcopy(connectivity_graph)

    N = len(C)  # Number of qubits
    AQ = []  # Assigned qubits
    T = [None] * N

    # Start with the two qubits with the highest compatibility
    i, j = np.unravel_index(np.argmax(C), [N, N])

    ii, jj = list(connectivity_graph.edges())[0]

    # Update the map
    T[i] = ii
    T[j] = jj

    AQ.append(i)
    AQ.append(j)

    C[i, j] = -1
    C[j, i] = -1

    # Remove used edges
    connectivity_graph.remove_edge(ii, jj)
    check_degree(connectivity_graph, T, i, C)
    check_degree(connectivity_graph, T, j, C)

    while len(AQ) < N:
        Cp = C[AQ, :]
        i, j = np.unravel_index(np.argmax(Cp), [N, N])
        i = AQ[i]

        if i in AQ and j in AQ:
            C[i, j] = -1
            C[j, i] = -1
        elif (i in AQ) or (j in AQ):
            only_one_assigned(i, j, T, AQ, C, connectivity_graph)

    return T


def transpile_disconnected_2(connectivity_graph, C):
    """
    Construct a theoretical-physical map for the qubits. For details on the arguments and the return, check the documentation of the transpile function.
    """
    C = copy.copy(C)
    connectivity_graph = copy.deepcopy(connectivity_graph)

    N = len(C)
    AQ = []
    T = [None] * N

    while len(AQ) < N:
        i, j = np.unravel_index(np.argmax(C), [N, N])

        if (i in AQ) and (j in AQ):
            C[i, j] = -1
            C[j, i] = -1
        elif (i not in AQ) and (j not in AQ):
            ii, jj = list(connectivity_graph.edges())[0]

            C[i, j] = -1
            C[j, i] = -1

            if (ii not in T) and (jj not in T):
                T[i] = ii
                T[j] = jj
                AQ.append(i)
                AQ.append(j)

                for node in ii, jj:
                    neighbors = copy.copy(connectivity_graph.neighbors(node))
                    for neighbor in neighbors:
                        if neighbor in T:
                            connectivity_graph.remove_edge(neighbor, node)
                            check_degree(connectivity_graph, T, T.index(neighbor), C)
                    check_degree(connectivity_graph, T, T.index(node), C)

        else:
            only_one_assigned(i, j, T, AQ, C, connectivity_graph)
    return T


def transpile_disconnected(connectivity_graph, C):
    C = copy.copy(C)
    connectivity_graph = copy.deepcopy(connectivity_graph)

    N = len(C)
    AQ = []
    T = [None] * N

    while len(AQ) < N:
        i, j = np.unravel_index(np.argmax(C), [N, N])

        if (i in AQ) and (j in AQ):
            C[i, j] = -1
            C[j, i] = -1
        elif (i not in AQ) and (j not in AQ):
            C[i, j] = -1
            C[j, i] = -1
            for ii, jj in connectivity_graph.edges():
                if (ii not in T) and (jj not in T):
                    T[i] = ii
                    T[j] = jj
                    AQ.append(i)
                    AQ.append(j)

                    for node in ii, jj:
                        neighbors = copy.copy(connectivity_graph.neighbors(node))
                        for neighbor in neighbors:
                            if neighbor in T:
                                connectivity_graph.remove_edge(neighbor, node)
                                check_degree(connectivity_graph, T, T.index(neighbor), C)
                        check_degree(connectivity_graph, T, T.index(node), C)

                    break

        else:
            only_one_assigned(i, j, T, AQ, C, connectivity_graph)

    return T


def transpile(connectivity_graph, C, connected):
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


def measurement_assignment(Vi, Vj, Mi, AM, WC, OQ, T):
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
    Mi: list[list]
        Current assignment of the measurement of the group i. It is a list of partial measurements. Each partial
        measurements is a list of two elements. The first of these elements encodes the partial measurement assigned and
        the second the qubits where it should be performed.
    AM: list
        Admissible measurements considered. Regarding our numerical encoding, it is a list of integers from 1 to 9.
        The order of the list encodes the preference of measurement assignment. For instance, the first measurement
        appearing in this list will be the one that would be preferentially assigned.
    WC: list[list]
        Well-connected qubits. Each element denotes a pair of connected qubits
    OQ: list
        Order of qubits that the algorithm should follow in each iteration.
    T: list
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
        if list(Vj[PM[1]]) not in COMP_LIST[PM[0]]:
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

    UMi = Mi[:]
    while len(U) != 0:
        for Eps in AM:  # Admissible measurement loop
            if len(U) >= LENGTH_MEAS[Eps]:
                for per in permutations(U, LENGTH_MEAS[Eps]):  # Possible qubits loop
                    per = list(per)
                    if LENGTH_MEAS[Eps] >= 2:
                        Tper = (int(T[per[0]]), int(T[per[1]]))
                    else:
                        Tper = 0
                    if (Tper in WC) or (LENGTH_MEAS[Eps] == 1):  # Connectivity check
                        # Compatibility check
                        if (list(Vi[per]) in COMP_LIST[Eps]) and (list(Vj[per]) in COMP_LIST[Eps]):
                            UMi.append([Eps, per])
                            for k in per:
                                U.remove(k)
                            break
                else:
                    continue
                break
        else:
            return Mi, False
    return UMi, True


def grouping_entangled(labels, connectivity=None, connected_graph=False, print_progress=False, pauli_graph=None):
    """
    Given a set of Pauli strings, groups them using entangled measurements. The chip connectivity can be taken into
    account to avoid two-qubit measurements between non-connected qubits.

    Parameters
    ----------
    labels: ndarray (n, N)
        A total of n Pauli strings for N qubits. The Pauli labels are in the number convention.
    connectivity: nx.Graph (optional, default=None)
        Graph that represents the connectivity of the chip. If not provided, an all-to-all device is assumed.
    connected_graph: bool (optional, default=False)
        If True the transpile algorithm ensures that the subgraph of the theoretical qubits in the chip is connected.
        Else, the algorithm does not ensure that the subgraph of the theoretical qubits in the chip is connected,
        instead tries to optimize omega(T) in a greedy way.
    print_progress: bool (optional, default=False)
        If true, print the progress of the pauli graph and the grouping.
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
    T: list
        T is the theo-phys map chosen. T[i]=j means that the i-theoretical qubit is mapped to the j-physical qubit.

    ¡Important!: In 'Measurement' output the indexes that we use to refer to the qubits are theoretical indexes, not the
    correspondent physical indexes, i.e., if we have the i-theoretical qubit is mapped to the j-physical qubit through
    T, in other words T[i]=j, we use the index i and not the j to refer to that qubit.
    """
    n, N = np.shape(labels)

    if connectivity is None:
        connectivity = nx.Graph()
        connectivity.add_edges_from(list(permutations(list(range(N)), 2)))

    if type(connectivity) == nx.classes.graph.Graph:
        pass
    elif type(connectivity) == list:
        temp = copy.copy(connectivity)
        connectivity = nx.Graph()
        connectivity.add_edges_from(temp)

    if len(connectivity.nodes) < len(labels[0]):
        raise Exception('The number of qubits in the device is not high enough. Use a bigger device.')

    # TODO: When the class is implemented, extract the Pauli graph out of the grouping functions
    if pauli_graph is None:
        pauli_graph = build_pauli_graph(labels, print_progress=print_progress)

    # Qubits in descending order of compatible measurements
    SV = sorted(pauli_graph.degree, key=lambda x: x[1], reverse=True)
    SV = [x[0] for x in SV]

    WC = list(connectivity.edges)  # list of pairs of well-connected qubits

    C, _, _ = compatible_measurements(labels, one_qubit=False)
    T = transpile(connectivity, C, connected_graph)
    _, CM, CQ = compatible_measurements(labels, T=T, connectivity_graph=connectivity)

    AM = list(np.argsort(CM[1:])[::-1] + 1)
    OQ = list(np.argsort(CQ)[::-1])

    groups = []
    measurements = []

    pbar = tqdm(total=n, desc='Grouping entangled', disable=not print_progress)
    while len(SV) > 0:
        pbar.update()
        i = SV.pop(0)  # We run the Pauli strings in a decreasing order of CQ.
        Mi = []
        GroupMi = [i]
        for j in SV:  # We try to make the group of the string i as big as possible
            Mi, S = measurement_assignment(labels[i], labels[j], Mi, AM, WC, OQ, T)
            if S:
                pbar.update()
                GroupMi.append(j)

        for index in GroupMi[1:]:
            SV.remove(index)

        QWM = list(range(N))  # Qubits without a Measurement assigned by Mi.
        for PM in Mi:
            for s in PM[1]:
                QWM.remove(s)

        for q in QWM:
            TPBq = max(labels[GroupMi, q])
            Mi.append([TPBq, [q]])

        groups.append(GroupMi)
        measurements.append(Mi)
    pbar.close()

    return groups, measurements, T


def grouping_TPB(labels, print_progress=False):
    """
    Construction of the TPB groups, i.e., the groups when considering the TPB basis.

    Parameters
    ----------
    labels: ndarray (n, N)
        A total of n Pauli strings for N qubits. The Pauli labels are in the number convention.
    print_progress: bool (optional, default=False)
        If True, print the progress bar for the Pauli graph building.

    Returns
    -------
    groups: list[list]
        Indices of the grouped Pauli labels
    measurements: list[list[list]]
        Measurements for each group of Pauli labels. Each grouped measurements is constituted with multiple one qubit
        and two qubit partial measurements. The first index denotes the partial measurement to perform, and the
        second one the qubits to measure.
    """
    PG = build_pauli_graph(labels, print_progress=print_progress)
    coloring = nx.coloring.greedy_color(PG)  # Graph coloring code of networkx. By default, it uses LDFC strategy.

    # Obtain grouped Pauli labels
    groups = [[k for k, v in coloring.items() if v == color] for color in set(coloring.values())]

    # TPB measurements assignment
    measurements = []
    for group in groups:
        Mi = []
        for k in range(labels.shape[1]):  # Iterate over the qubits
            Mi.append([max(labels[group, k]), [k]])
        measurements.append(Mi)

    return groups, measurements


def test_grouping_paulis(labels, groups):
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

    n_labels = len(labels)

    pauli_indices = list(range(n_labels))
    try:
        for group in groups:
            for index in group:
                pauli_indices.remove(index)  # Remove index of measured Pauli label
    except ValueError:  # If try to remove twice the same index
        raise Exception('The Pauli label ({}) is measured at least two times.'.format(labels[index]))

    # If some index has not been removed
    if len(pauli_indices) > 0:
        raise Exception('The following Pauli indices are not measured {}'.format(pauli_indices))

    return True


def test_grouping_measurements(labels, groups, measurements):
    """
    Check that all the measurements correctly corresponds to the grouping labels, and no error has occurred during the
    grouping algorithm.

    Parameters
    ----------
    labels: ndarray (n, N)
        A total of n Pauli strings for N qubits. The Pauli labels are in the number convention.
    groups: list[list]
        Indices of the grouped Pauli labels
    measurements: list[list[list]]
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
                # If the Pauli not compatible with measurement
                elif list(grouped_label[qubits]) not in COMP_LIST[index_measure]:
                    raise Exception(
                        'The Pauli label {} is incompatible with the measurement {} over the qubit(s) {}'.format(
                            grouped_label, index_measure, *qubits))
    return True


def test_connectivity(measurements, T, connectivity):
    """
    Test that entangled measurements are only provided between well-connected qubits.

    Parameters
    ----------
    measurements: list[list[list]]
        Measurements for each group of Pauli labels. Each grouped measurements is constituted with multiple one qubit
        and two qubit partial measurements. The first index denotes the partial measurement to perform, and the
        second one the qubits to measure.
    T: list
        T is the theo-phys map chosen. T[i]=j means that the i-theoretical qubit is mapped to the j-physical qubit.
    connectivity: list
        List that contains the connectivity of the chip.

    Returns
    -------
    If the test passes, True is returned. Otherwise, and exception is raised.
    """
    for measurement in measurements:
        for partial_measurement in measurement:
            if partial_measurement[0] > 4:  # Only check two qubit measurements
                phy_qubits = [T[q] for q in partial_measurement[1]]
                if tuple(phy_qubits) not in connectivity:
                    raise Exception('Entangled measurement between non-connected '
                                    'qubits {} and {}'.format(phy_qubits[0], phy_qubits[1]))
    return True
