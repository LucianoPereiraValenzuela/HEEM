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
(Note: if some measurement output has the number '0' it means that any measure is valid.)

Finally, we build lists of compatibility, one for each measurement. The list of compatibility of the measurement k 
should contain the arrays assigned to the Pauli strings that are compatible with that measurement. 
For instance, if we consider the measure 4 (Bell) its list of compatibility should contain [0,0], [1,1], [2,2], [3,3],
because the Bell measurement is compatible with II, XX, YY and ZZ, respectively.

Thus, when checking the compatibility of the strings v_i and v_j with the measurement k on the qubits (l, m), what we 
should do is checking if [v_i(l),v_i(m)] and [v_j(l),v_j(m)] are both in the compatibility list of the measurement k.
For example, if we had v_i = YIZZ = [2, 0, 3, 3] and v_j = XIZY = [1, 0, 3, 2], and we wanted to check if theses strings
are compatible with the measurement 4 (Bell) on the qubits (2, 3). What we have to do is checking if 
[v_i(2), v_i(3)] = [3, 3] and [v_j(2), v_j(2)] = [3, 2] are in the compatibility list of the measurement 4. As this
compatibility list is Comp_4 = [[0, 0], [1, 1], [2, 2], [3, 3]], we have that [v_i(3), v_i(4)] belongs to Comp_4 but
[v_j(3), v_j(4)] does not. In consequence, the measurement 4 on the qubits (3, 4) is not compatible with v_i and v_j. 
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


def pauli_graph(PS, print_progress=False):
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

    pbar = tqdm(range(n), desc='Computing Pauli graph', disable=not print_progress)
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


def compatibilities(PS):
    """
    Given a set of 'n' Pauli Strings (PS) with 'N' qubits, returns three arrays regarding the compatibilities of the
    measurements. The element (i, j) of the compatibility matrix contains the number of times that the qubits i and j
    of one string are jointly measurable (compatible) with the qubits i and j of other string. For example, the pauli
    strings [1, 1] and [2, 2] will produce C[0, 1] = 1 because the qubits 0 and 1 of those pauli strings are jointly
    measurable only with the bell measurement. This matrix is symmetric, and the diagonal factors set to -1.

    Parameters
    ----------
    PS: ndarray (n, N)
        Pauli strings, each row represents a Pauli string and each column represents a qubit.

    Returns
    -------
    C:  ndarray (n, N)
        Compatibility matrix¡.
    """

    n, N = np.shape(PS)  # Number of Pauli labels and number of qubits

    C = np.diag(-1 * np.ones(N))

    for i in range(N):  # First qubit
        for j in range(i + 1, N):  # Second qubit
            PSij = PS[:, [i, j]]

            F = empty_factors()
            for label in PSij:  # Generate factors list
                F[str(label[0]) + str(label[1])] += 1

            for measurement in range(4, 10):  # Fill compatibility measurements between qubits i and j
                n_counts = compatible_measurements_2q(measurement, F)

                C[i, j] += n_counts
                C[j, i] += n_counts

    return C


# TODO: Documentation of transpile
# TODO: Revisit these two functions
def transpile_connected(G, C):
    """

    """
    C = copy.copy(C)
    G = copy.deepcopy(G)

    N = np.shape(C)[0]
    AQ = []
    T = [None] * N

    # First we assign two qubits, then we build the map from them ensuring that the resulting graph is connected
    i, j = np.unravel_index(np.argmax(C), [N, N])
    for ii, jj in G.edges():
        # Update the map
        T[i] = ii
        T[j] = jj
        AQ.append(i)
        AQ.append(j)
        C[i, j] = -1
        C[j, i] = -1

        # Remove used edges
        G.remove_edge(ii, jj)
        if nx.degree(G, ii) == 0:
            C[i, :] = -1
            C[:, i] = -1
        if nx.degree(G, jj) == 0:
            C[j, :] = -1
            C[:, j] = -1

        break

    while len(AQ) < N:
        Cp = C[AQ, :]
        i, j = np.unravel_index(np.argmax(Cp), [N, N])
        i = AQ[i]

        if i in AQ and j in AQ:
            C[i, j] = -1
            C[j, i] = -1

        elif i in AQ or j in AQ:
            if i in AQ:
                assigned = i
                not_assigned = j
            else:
                assigned = j
                not_assigned = i

            # Assign the not assigned theoretical qubit to the first neighbor available
            for neighbor in G.neighbors(T[assigned]):
                if neighbor not in T:
                    T[not_assigned] = neighbor
                    AQ.append(not_assigned)
                    C[i, j] = -1
                    C[j, i] = -1

                    # If the neighbors_2 of the just assigned qubit are also assigned,
                    # remove the edge in the graph because it is not available
                    neighbors_2 = copy.copy(G.neighbors(T[not_assigned]))
                    for neighbor_2 in neighbors_2:  # Loop 1
                        if neighbor_2 in T:
                            G.remove_edge(neighbor_2, neighbor)
                            if nx.degree(G, neighbor_2) == 0:
                                s = T.index(neighbor_2)
                                C[s, :] = -1
                                C[:, s] = -1

                    if nx.degree(G, neighbor) == 0:
                        C[not_assigned, :] = -1
                        C[:, not_assigned] = -1

                    break
    return T


def transpile_disconnected(G, C):
    """

    """
    C = copy.copy(C)
    G = copy.deepcopy(G)

    N = np.shape(C)[0]
    AQ = []
    T = [None] * N

    while len(AQ) < N:
        i, j = np.unravel_index(np.argmax(C), [N, N])

        if i in AQ and j in AQ:
            C[i, j] = -1
            C[j, i] = -1
        elif (i not in AQ) and (j not in AQ):
            success = False
            for ii, jj in G.edges():
                if ii not in T and jj not in T:
                    T[i] = ii
                    T[j] = jj
                    AQ.append(i)
                    AQ.append(j)
                    C[i, j] = -1
                    C[j, i] = -1
                    success = True

                    for node in ii, jj:
                        neighbors = copy.copy(G.neighbors(node))
                        for neighbor in neighbors:  # Loop 1
                            if neighbor in T:
                                G.remove_edge(neighbor, node)
                                if nx.degree(G, neighbor) == 0:
                                    s = T.index(neighbor)
                                    C[s, :] = -1
                                    C[:, s] = -1
                        if nx.degree(G, node) == 0:
                            C[T.index(node), :] = -1
                            C[:, T.index(node)] = -1

                    break

            if not success:
                C[i, j] = -1
                C[j, i] = -1

        elif i in AQ or j in AQ:  # if we reach this point, then only one of i and j will be in AQ.
            if i in AQ:
                assigned = i
                not_assigned = j
            else:
                assigned = j
                not_assigned = i

            # Assign the not assigned theoretical qubit to the first neighbor available
            for neighbor in G.neighbors(T[assigned]):
                if neighbor not in T:
                    T[not_assigned] = neighbor
                    AQ.append(not_assigned)
                    C[i, j] = -1
                    C[j, i] = -1

                    # If the neighbors_2 of the just assigned qubit are also assigned,
                    # remove the edge in the graph because it is not available
                    neighbors_2 = copy.copy(G.neighbors(neighbor))
                    for neighbor_2 in neighbors_2:  # Loop 1
                        if neighbor_2 in T:
                            G.remove_edge(neighbor_2, neighbor)
                            if nx.degree(G, neighbor_2) == 0:
                                s = T.index(neighbor_2)
                                C[s, :] = -1
                                C[:, s] = -1

                    if nx.degree(G, neighbor) == 0:
                        C[not_assigned, :] = -1
                        C[:, not_assigned] = -1

                    break
    return T


def transpile(G, C, connected=False):
    """

    Parameters
    ----------
    G: nx.Graph
        Pauli graph
    C:  ndarray (N, N)
        Compatibility matrix. More info in function compatibilities().
    connected: bool (optional, default=False)
        If True the transpile algorithm ensures that the subgraph of the theoretical qubits in the chip is connected.
        More info in function groupingWithOrder()

    Returns
    -------
    T: list
        T is the theo-phys map chosen. T[i] = j means that the i-theoretical qubit is mapped to the j-physical qubit.
    """

    if connected:
        return transpile_connected(G, C)
    else:
        return transpile_disconnected(G, C)


# TODO: Revisit compatibilities functions
def transpiled_compatibilities(PS, T, G):
    """
    Given a set of 'n' Pauli Strings (PS) with 'N' qubits, returns three arrays regarding the compatibilities of the
    measurements.

    Parameters
    ----------
    PS: ndarray (n, N)
        Pauli strings, each row represents a Pauli string and each column represents a qubit.
    T: list
        Map from theoretical qubits to physical qubits. If T[i] = j it means that the i-th theoretical qubit is
        mapped to the j-th physical qubit.
    G: nx.Graph
        Connectivity graph of the chip. Its vertices represent physical qubits and its edges physical connections
        between them.

    Returns
    -------
    C: ndarray (n, N)
        The element C[i,j] contains the number of times that the qubits i and j of one string are jointly measurable
        (compatible) with the qubits i and j of other string, given that T have been chosen as the map from theoretical
        qubits to physical qubits. Symmetric matrix whose diagonal elements are -1.
    CM: dict
        Number of times that qubits of two pauli strings are compatible with each measurement, given that T have been
        chosen as the map from theoretical qubits to physical qubits.
    CQ: ndarray (N)
        The element CQ[i] contains the number of times that the qubit i can participate in a joint measurement with any
        other qubit though any measurement, given that T have been chosen as the map from theoretical qubits to physical
        qubits. It is the sum of the i_th row/column of the matrix C, excluding the -1 of the diagonal, the number of
        compatibilities due to one-qubit measurements.
    """

    n, N = np.shape(PS)

    C = np.diag(-1 * np.ones(N))
    CM = [0] * 10
    CQ = np.zeros(N)

    for i in range(N):  # First qubit
        for j in range(i + 1, N):  # Second qubit
            if [T[i], T[j]] in G.edges():
                PSij = PS[:, [i, j]]

                F = empty_factors()
                for s in range(n):  # Generate factors list
                    label = str(PSij[s, 0]) + str(PSij[s, 1])
                    F[label] += 1

                for measurement in range(4, 10):  # Fill compatibility measurements between qubits i and j
                    n_counts = compatible_measurements_2q(measurement, F)

                    C[i, j] += n_counts
                    C[j, i] += n_counts
                    CM[measurement] += n_counts

        CQ[i] = 1 + np.sum(C[i, :])
        for measurement in range(1, 4):
            n_counts = compatible_measurements_1q(measurement, PS[:, i])
            CQ[i] += n_counts
            CM[measurement] += n_counts

    return C, CM, CQ


# TODO: Redo the documentation
def measurement_assignment_with_order(Vi, Vj, Mi, AM, WC, OQ, T):
    """
    This function follows one of two different paths according to the input Mi:
        A) If Mi is an empty list the function assigns measurements Mi that are both compatible with Vi and Vj.

        B) If Mi is not complete, i.e. there are qubits with no assigned measurements, first the function checks if
        the currently assigned measurements Mi are compatible with Vj. If this is true then in tries to assign to the
        remaining qubits measurements that are both compatible with Vi and Vj.

    In both cases A and B the function returns S = True iff the updated Mi is full and compatible with Vi and Vj. Else,
    it returns S = False and an unchanged Mi.

    This program is the Algorithm 2 of https://arxiv.org/abs/1909.09119.

    Parameters
    ----------
    Vi: ndarray
       i Pauli string in our numerical encoding
    Vj: ndarray
        j Pauli string in our numerical encoding.
    Mi: list
        Current assignment of the measurement of the group i. It is a list of partial measurements. Each partial
        measurements is a list of two elements. The first of these elements encodes the partial measurement assigned and
        the second the qubits where it should be performed.
    AM: list
        Admissible measurements considered. Regarding our numerical encoding, it is a list of
        integers from 1 to 9. The order of the list encodes the preference of measurement assignment. For instance, the
        first measurement appearing in this list will be the one that would be preferentially assigned.
    WC: list[list]
        Well-connected qubits. Each element denotes a pair of connected qubits
    OQ: list
        Order of qubits that the algorithm should follow in each iteration.
    T: list
        Theo-phys map chosen. T[i] = j means that the i-theoretical qubit is mapped to the j-physical qubit.

    Returns
    -------
    UMi: list
        Updated Mi. If the algorithm fails, UMi will be equal to Mi.
    S: bool
        If the algorithm has succeeded, i.e., if Mi has been updated in a way such the group of Vi and Vj are compatible
    """

    # The first loop checks if the current assignment of Mi is compatible with Vj.
    # If Mi is compatible with Vj, the array U will contain the qubits where Mi does not act.

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
    After the second loop U contains the qubits where Mi does no act and the factors of Vi and Vj are not equal.
    Thus, it is in the qubits of U where partial measurements have to be assigned to make the strings of Vi and Vj
    compatible.
    """

    """
    The third loop tries to update the measurement Mi on the qubits in U. To do so it runs through the admissible
    partial measurements AM. For each of those measurements, the loop runs through all the possible set of qubits
    where the measurement can act (perm). For each element 'per' in 'perm' the code checks if 'per' is a set of
    well-connected qubits. Finally, it is checked if the measurement on those qubits is compatible with the string Vi
    and Vj (if so, by construction, the measurement will be compatible with all strings of group Vi and with Vj). If
    there is success in this last check, UMi is updated with that partial measurement, the qubits where this partial
    measurement are deleted of U, and we begin again if U is not empty. If we managed to empty U, the update would
    have succeeded
    """

    UMi = Mi[:]

    while len(U) != 0:
        for Eps in AM:  # Admissible measurement loop
            if len(U) >= LENGTH_MEAS[Eps]:
                perm = list(permutations(U, LENGTH_MEAS[Eps]))
                for per in perm:  # Possible qubits loop
                    if LENGTH_MEAS[Eps] >= 2:
                        Tper = (int(T[per[0]]), int(T[per[1]]))
                    else:
                        Tper = 0
                    if (Tper in WC) or (LENGTH_MEAS[Eps] == 1):  # Connectivity check
                        # Compatibility check
                        if (list(Vi[tuple([per])]) in COMP_LIST[Eps]) and (list(Vj[tuple([per])]) in COMP_LIST[Eps]):
                            UMi.append([Eps, list(per)])
                            for k in per:
                                U.remove(k)
                            break
                else:
                    continue
                break
        else:
            return Mi, False

    return UMi, True


# TODO: Is this the best name for the grouping?
def groupingWithOrder(PS, G=None, connected=False, print_progress=False):
    """
    Given a set of Pauli strings (PS), this function makes groups of PS assigning taking into account the chip's
    connectivity.

    Parameters
    ----------
    PS: array (n, M)
        Pauli strings, each row represents a Pauli string while each the column represents a qubit. Thus, n is the
        number of Pauli strings and N is the number of qubits.
    G: nx.Graph or list (optional, default=None)
        Graph or list that represents the connectivity of the chip. If not provided, an all-to-all device is assumed.
    connected: bool (optional, default=False)
        If True the transpile algorithm ensures that the subgraph of the theoretical qubits in the chip is connected.
        Else, algorithm does not ensure that the subgraph of the theoretical qubits in the chip is connected, instead
        tries to optimize omega(T) in a greedy way.
    print_progress: bool (optional, default=False)
        If true, print the progress of the pauli graph and the grouping.

    Returns
    -------
    Groups: list
        The element in the position i is a list with the indexes of strings assigned to group i, i.e, the strings of
        group i.
    Measurement: list
        The element in position i is a list which represents measurement assigned to the group i. Each of these list is
        a list of partial measurements. Each partial measurements is a list of two elements. The first of these elements
        encodes the partial measurement assigned and the second the qubits where it should be performed.
    T: list
        T is the theo-phys map chosen. T[i]=j means that the i-theoretical qubit is mapped to the j-physical qubit.

    ¡Important!: In 'Measurement' output the indexes that we use to refer to the qubits are theoretical indexes, not the
    correspondent physical indexes, i.e., if we have the i-theoretical qubit is mapped to the j-physical qubit through
    T, in other words T[i]=j, we use the index i and not the j to refer to that qubit.
    """
    n, N = np.shape(PS)

    if G is None:
        G = nx.Graph()
        G.add_edges_from(list(permutations(list(range(N)), 2)))

    if type(G) == nx.classes.graph.Graph:
        pass
    elif type(G) == list:
        temp = copy.copy(G)
        G = nx.Graph()
        G.add_edges_from(temp)

    if len(G.nodes) < len(PS[0]):
        raise Exception('The number of qubits in the device is not high enough. Use a bigger device.')

    PG = pauli_graph(PS, print_progress=print_progress)
    SV = sorted(PG.degree, key=lambda x: x[1], reverse=True)

    WC = list(G.edges)  # list of pairs of well-connected qubits
    AS = []  # List of strings with assigned measurement
    C = compatibilities(PS)
    T = transpile(G, C, connected)
    CT, CM, CQ = transpiled_compatibilities(PS, T, G)
    CMlist = [CM[str(i)] for i in range(1, 10)]  # TODO: Check if this variable is useful

    AM = [i[0] + 1 for i in sorted(enumerate(CMlist), key=lambda x: x[1], reverse=True)]
    OQ = [i[0] for i in sorted(enumerate(list(CQ)), key=lambda x: x[1], reverse=True)]

    Groups = []
    Measurements = []

    pbar = tqdm(range(n), desc='Computing grouping', disable=not print_progress)
    for k in pbar:
        i = SV[k][0]  # We run the Pauli strings in a decreasing order of CQ.
        if i not in AS:  # If we enter to this loop, the i string will have its own group.
            Mi = []
            GroupMi = [i]
            AS.append(i)
            for m in range(n):  # We try to make the group of the string i as big as possible
                j = SV[m][0]
                if j not in AS:
                    Mi, S = measurement_assignment_with_order(PS[i, :], PS[j, :], Mi, AM, WC, OQ, T)
                    if S:
                        AS.append(j)
                        GroupMi.append(j)

            # Mi completion
            QWM = list(np.arange(N))  # Qubits without a Measurement assigned by Mi.
            for PM in Mi:
                for s in PM[1]:
                    QWM.remove(s)
            for q in QWM:
                TPBq = max(PS[GroupMi, q])
                Mi.append([TPBq, [q]])

            """
            In this loop we complete the measurement Mi, as it might not assign a partial measurement to each qubit.
            The qubits where Mi does not assign a partial measurement will satisfy that all factors of the 
            strings of the group are equal. Thus, a TPB should be assigned in those qubits. We proceed in a similar way
            as we did in the tpb_grouping code.
            """

            Groups.append(GroupMi)
            Measurements.append(Mi)

    return Groups, Measurements, T


def tpb_grouping(labels, print_progress=False):
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
    PG = pauli_graph(labels, print_progress=print_progress)
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
