#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import networkx as nx
from itertools import permutations
from multiprocessing import Pool
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook
import sys
from utils import Label2Chain, sort_solution, unpack_functions, isnotebook
from qiskit.opflow.list_ops import SummedOp
from qiskit.quantum_info import Pauli
from qiskit.opflow.primitive_ops import PauliOp
import copy

"""
See the report for context of this code. 

In order to simplify the programming, we use a numerical encoding. We identify each Pauli string with an integer

I-> 0, X-> 1, Y-> 2, Z-> 3. 

Then, for example, $XIZY$ would be mapped to the array [1,0,3,2]. 

Similarly, we  map  measurements into numbers:
TPBX-> 1, TPBY-> 2, TPBZ-> 3, Bell->4, OmegaX-> 5, OmegaY-> 6, OmegaZ-> 7, chi-> 8, \chi'->9.

Note: if some measurement output has the number '0' it means that any measure is valid.

Finally, we build lists of compatibility, one for each measurement. The list of compatibility of the measurement k
should contain the arrays assigned to the Pauli strings that are compatible with the measurement k. 
For instance, if we consider the measure 4 (the Bell measure) its list of compatibility should contain 
[0,0], [1,1], [2,2], [3,3], because the Bell measurement is compatible with II,XX,YY and ZZ. Thus the compatibility 
lists are:

    Comp_1={I, X} = {[0],[1]},    Comp_2={I, Y} = {[0],[2]},    Comp_3={I, Z} = {[0],[3]},
    Comp_4={II,XX,YY,ZZ} = {[0,0],[1,1],[2,2],[3,3]},    Comp_5={II,XX,YZ,ZY} = {[0,0],[1,1],[2,3],[3,2]},
    Comp_6={II,YY,XZ,ZX} = {[0,0],[2,2],[1,3],[3,1]},    Comp_7={II,ZZ,XY,YX} = {[0,0],[3,3],[1,2],[2,1]},
    Comp_8={II,XY,YX,ZX} = {[0,0],[1,2],[2,1],[3,1]},    Comp_9={II,YX,ZY,XZ} = {[0,0],[2,1],[3,2],[1,3]}.


Thus, when checking the compatibility of the strings v_i and v_j with the measurement k on the qubits (l,m),
what we should do is checking if [v_i(l),v_i(m)] and [v_j(l),v_j(m)] are both in the compatibility list 
of the measurement k. For example, if we had v_i=YIZZ=[2,0,3,3] and v_j=XIZY=[1,0,3,2] and we wanted to check
if theses strings are compatible with the measurement 4 (the Bell measurement) on the qubits (3,4), what we have
to do is checking if [v_i(3),v_i(4)]=[3,3] and [v_j(3),v_j(4)]=[3,2] are in the compatibility list of the 
measurement 4. As this compatibility list is Comp_4={[0,0],[1,1],[2,2],[3,3]}, we have that [v_i(3),v_i(4)] belongs
to Comp_4 but [v_j(3),v_j(4)] does not. In consequence, the measurement 4 on the qubits (3,4) is not compatible with 
v_i and v_j. 

v2 Changes
----------
1.- Changed the variable name "length" to "len_meas".
"""

# The compatibility lists are implemented manually
# We construct two lists with 9 elements each. The first one with all the available measurements, sorted as explained
# above, and the second specifying the length of the measure (number of qubits to measure)

Comp = []
Comp.append([[]])  # This empty entry is to fix notation. This way, we preserve the exposed encoding.
Comp.append([[0], [1]])
Comp.append([[0], [2]])
Comp.append([[0], [3]])
Comp.append([[0, 0], [1, 1], [2, 2], [3, 3]])
Comp.append([[0, 0], [1, 1], [2, 3], [3, 2]])
Comp.append([[0, 0], [2, 2], [1, 3], [3, 1]])
Comp.append([[0, 0], [3, 3], [1, 2], [2, 1]])
Comp.append([[0, 0], [1, 2], [2, 3], [3, 1]])
Comp.append([[0, 0], [2, 1], [3, 2], [1, 3]])

len_meas = [len(x[0]) for x in Comp]


def PauliGraph(PS):
    """
    Construction of the Pauli Graph

    Parameters
    ----------
    PS: ndarray (n, N)
        PS are the Pauli strings. Each row represents a Pauli string, and each column represents a qubit. Thus, n is the
        number of Pauli strings and N is the number of qubits.

    Return
    ------
    PG: graph
        The Pauli graph represents the noncommutativity of the n Pauli strings. 
        Its nodes are Pauli strings, and its edges exist if and only if two nodes are NOT qubit-wise commutative.
        Two strings are qubit-wise commutative if for each qubit at least one of these conditions is True: 
        a) both strings have the same factor,
        b) one of the strings has a factor I ( [0] in our encoding).
        
        
    v2 Changes
    ----------
    1.- Extended description of 'PG: graph'.
    2.- Changed compatibility check, now it is twice as fast.
    """

    n = np.size(PS[:, 0])
    
    PG = nx.Graph()
    PG.add_nodes_from(np.arange(n))  # Assigns a node to each Pauli string
    
    for i in range(n):  # Loop over each Pauli string v_i
        v_i = PS[i, :]

        for j in range(i + 1, n): # Nested loop over the folowing Pauli strings v_j
            v_j = PS[j, :]
            compatiblequbits = np.logical_or.reduce((v_i == v_j, v_i == 0, v_j == 0)) 
            if np.any(compatiblequbits == False) :     # If at least one of the qubits shared by the PS is not commutative
                PG.add_edges_from([(i, j)])            # add an edge in the Pauli Graph
    return PG


def colorgroups(colordict):
    """
    Construction of the TPB groups from the color dictionary.

    Parameters
    ----------
    colordict: dictionary
        The keys are the indexes of the Pauli strings. The values are the colors assigned to the Pauli string.
        The rule is that two Pauli strings have a different color if their nodes in the Pauli graph are connected.

    Return
    ------
    Groups: list
        The element in the position i is a list with the indexes of strings assigned to color i, i.e, the group of
        strings with color i.
        
        
    v2 Changes
    ----------
    1.- Extended description of 'colordict: dictionary'.
    """
    colorarray = np.array(list(colordict.items()))
    keys = np.array(colorarray[:, 0])
    values = np.array(colorarray[:, 1])
    Groups = []
    for i in range(max(values) + 1):
        groupi = list(keys[np.nonzero(values == i)])
        Groups.append(groupi)
    return Groups


def TPBgrouping(PS):
    """
    Construction of the TPB groups, i.e., the groups when considering the TPB basis.

    Parameters
    ----------
    PS: array (n, N)
        Pauli strings, each row represents a Pauli string, each column represents a qubit. 
        Thus, n is the number of Pauli strings and N is the number of qubits.

    Returns
    -------
    Color: dictionary
        The value assigned to the key i is the color assigned to the string i
    Groups: list
        The element in the position i is a list with the indexes of strings assigned to color i, i.e, the group of
        strings with color i.
    Measurement: list
        The element in position i is a list which represents measurement assigned to the group i. Each of these list is
        a list of partial measurements. Each partial measurements is a list of two elements. The first of these elements
        encodes the partial measurement assigned and the second the qubits where it should performed.
    """
    N = np.size(PS[0, :])
    
    PG = PauliGraph(PS)
    Color = nx.coloring.greedy_color(PG)  # Graph coloring code of networkx. By default it uses LDFC strategy.
    Groups = colorgroups(Color)  # Groups of strings with the same color assigned
    # TPB measurements assignment
    Measurements = []
    for i in range(len(Groups)):
        Mi = []
        for k in range(N):
            Mi.append([max(PS[Groups[i], k]), [k]])
        Measurements.append(Mi)

    """
    This loop is to assign the measurements to each group. In order to do so we run through all groups. 
    Given a group, we run through all qubits. For each qubit, we assign a TPB measurement to the group.
    With that purpose, we extract the k factors of all strings of the group. They will be the same Pauli operator
    and/or the identity. Thus, regarding our numerical encoding, we assign, to the group, the measurement
    max(PS[Groups[i],k]) in the position k.
    """

    return Color, Groups, Measurements


def MeasurementAssignment(Vi, Vj, Mi, AM, WC):
    """
    
    This function regards the assignment of admissible and efficient measurements Mi to the pauli strings Vi and Vj. 
    
    Admisible and efficient (compatible) means that the code tries to add to Mi admissible measurements
    (AM is the list of admissible measurements, given by the entangled measurements considered) involving well connected 
    qubits (WC is the list of pairs of directly connected qubits in the quantum processor).
    
    This function follows one of two different paths according to the input Mi:
    
    A) If Mi is an empty list the function assigns measurements Mi that are both compatible with Vi and Vj.
    
    B) If Mi is not complete, i.e. there are qubits with no assigned measurements, first the function checks if 
    the currently assigned measurements Mi are compatible with Vj. If this is true then in tries to assign to the remaining
    qubits measurements that are both compatible with Vi and Vj.
    
    In both cases A and B the function returns S=True iff the updated Mi is full and compatible with Vi and Vj, else, 
    it returns S=False and an unchanged Mi.
    
    This program is the Algorithm 2 of https://arxiv.org/abs/1909.09119.

    Parameters
    ----------
    Vi: array
        It is the array associated to the i Pauli string, according to our numerical encoding.
    Vj: array
        It is the array associated to the j Pauli string, according to our numerical encoding.
    Mi: list
        Current assignment of the measurement of the group i. It is a list of partial measurements. Each partial
        measurements is a list of two elements. The first of these elements encodes the partial measurement assigned and
        the second the qubits where it should performed.
    AM: list
        It is the list of the admissible measurements considered. Regarding our numerical encoding, it is a list of
        integers from 1 to 9. The order of the list encodes the preference of measurement assignment. For instance, the
        first measurement appearing in this list will be the one that would be preferentially assigned.
    WC: list
        It is a list of tuples. Each tuple represents a set of well connected qubits.

    Returns
    -------
    UMi: list
        Updated Mi. If the algorithm fails, UMi will be equal to Mi.
    S: bool
        If the algorithm has succeeded. In other words, if Mi has been updated in a way such the group of Vi and Vj are
        compatible, S=True. Otherwise, S=False.
        
    v2 Changes    
    ----------
    1.- Modified and extended the description of function.
    2.- Modified the comments of the function.
    """
    
    # The first loop checks if the current assignment of Mi is compatible with Vj. If not, the programm returns Mi and
    # S=False. If Mi is compatible with Vj, the array U will contain the qubits where Mi does not act.

    N = np.size(Vi)
    U = list(np.arange(N))
    # Check of Vj compatibility with current Mi
    for PM in Mi:
        if list(Vj[PM[1]]) not in Comp[PM[0]]:
            return Mi, False
        else:
            for k in PM[1]:
                U.remove(k)

    commonfactors = np.argwhere(Vi == Vj)
    for k in commonfactors:
        if k in U:
            U.remove(k)

    # After the second loop U contains the qubits where Mi does no act and the factors of Vi and Vj are not equal.
    # Thus, it is in the qubits of U where partial measurements have to be assigned to make the strings of Vi and Vj 
    # compatible.
    
    # The third loop tries to update the measurement Mi on the qubits in U. To do so it runs through the admissible 
    # partial measurements AM (admissible measurements loop). For each of those measurements, the loop runs through all 
    # the possible set of qubits where the measurement can act (perm) (possible qubits loop). For each element 'per' in 
    # 'perm' the code checks if 'per' is a set of well connected qubits (connectivity check). Finally, it is checked if
    # the measurement on those qubits is compatible with the string Vi and Vj (if so, by construction, the measurement 
    # will be compatible with all strings of group Vi and with Vj)(compatibility check). If there is success in this last 
    # check, UMi is updated with that partial measurement, the qubits where this partial measurement are deleted of U 
    # and we begin again if U is not empty. If we managed to empty U, the update would have succeeded and we would return
    # UMi, S=True. If there is no success, Mi, S=False are returned.

    # We will update Mi in the following loop. We create UMi because the loop might fail in the middle, thus,
    # an unchaged Mi should be returned.
    UMi = Mi[:]

    # UMi updating loop
    while len(U) != 0:
        for Eps in AM:  # Admissible measurement loop
            if len(U) >= len_meas[Eps]: 
                perm = list(permutations(U, len_meas[Eps]))
                for per in perm:  # Possible qubits loop
                    if (per in WC) or (len_meas[Eps] == 1):  # Connectivity check
                        if (list(Vi[tuple([per])]) in Comp[Eps]) and (
                                list(Vj[tuple([per])]) in Comp[Eps]):  # Compatibility check
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


def grouping(PS, AM, WC=None):
    """
    Given a set of Pauli strings (PS), this function makes groups of PS assigning the admissible measurements (AM)
    on the well connected qubits (WC).

    Parameters
    ----------
    PS: array (n, M)
        Pauli strings, each row represents a Pauli string while each the column represents a qubit. Thus, n is the
        number of Pauli strings and N is the number of qubits.
    AM: list
        List of the admissible measurements considered. Regarding our numerical encoding, it is a list of
        integers from 1 to 9. The order of the list encodes the preference of measurement assignment. For instance, the
        first measurement appearing in this list will be the one that would be preferentially assigned.
    WC: list (optional)
        List of tuples. Each tuple represents a set of well connected qubits. If WC is not provided, and all-to-
        all connectivity is assumed.

    Returns
    -------
    Groups: list
        The element in the position i is a list with the indexes of strings assigned to group i, i.e, the strings of
        group i.
    Measurement: list
        The element in position i is a list which represents measurement assigned to the group i. Each of these list is
        a list of partial measurements. Each partial measurements is a list of two elements. The first of these elements
        encodes the partial measurement assigned and the second the qubits where it should performed.
    """
    PG = PauliGraph(PS)
    SV = sorted(PG.degree, key=lambda x: x[1], reverse=True)  # Sorted Vertices by decreasing degree.
    n = np.size(PS[:, 0])
    N = np.size(PS[0, :])
    if WC is None:
        WC = list(permutations(list(range(N)), 2))
    AS = []  # List of strings with assigned measurement
    Groups = []
    Measurements = []
    for k in range(n):
        i = SV[k][0]  # We run the nodes in a decreasing order of degree according to Pauli graph, as LDFC does.
        if i not in AS:  # If we enter to this loop, the i string will have its own group.
            Mi = []
            GroupMi = [i]
            AS.append(i)
            for l in range(n):  # We try to make the group of the string i as big as possible
                j = SV[l][0]
                if j not in AS:
                    Mi, S = MeasurementAssignment(PS[i, :], PS[j, :], Mi, AM, WC)
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
            as we did in the TPBgrouping code.
            """
            Groups.append(GroupMi)
            Measurements.append(Mi)

    return Groups, Measurements

def n_groups(PS, AM, WC):
    """
    Compute the number of groups for a given order or Paulis strings, admissible measurements and connectivity.

    Parameters
    ----------
    PS: array (n, M)
        Pauli strings, each row represents a Pauli string while each the column represents a qubit. Thus, n is the
        number of Pauli strings and N is the number of qubits.
    AM: list
        It is the list of the admissible measurements considered. Regarding our numerical encoding, it is a list of
        integers from 1 to 9. The order of the list encodes the preference of measurement assignment. For instance, the
        first measurement appearing in this list will be the one that would be preferentially assigned.
    WC: list
        It is a list of tuples. Each tuple represents a set of well connected qubits.

    Return
    ------
    int: Number of groups
    """
    Groups, _ = grouping(PS, AM, WC)
    return len(Groups)


def grouping_shuffle(operator, AM, WC, n_mc=500, progress_bar=True):
    """
    Shuffle the order for the pauli strings randomly a given number of times and choose the ordering with less number of
    groups.

    Parameters
    ----------
    operator: SumOp
        Operator with the Pauli strings to group.
    AM: list
        It is the list of the admissible measurements considered. Regarding our numerical encoding, it is a list of
        integers from 1 to 9. The order of the list encodes the preference of measurement assignment. For instance, the
        first measurement appearing in this list will be the one that would be preferentially assigned.
    WC: list
        It is a list of tuples. Each tuple represents a set of well connected qubits.

    n_mc: int (optional)
        Number of Monte Carlo random orderings.
    progress_bar: Bool (optional)
        If True then print the progress bar for the computation of random orderings. If False, then nothing is print.

    Returns
    -------
        Groups: list
        The element in the position i is a list with the indexes of strings assigned to group i, i.e, the strings of
        group i.
    Measurement: list
        The element in position i is a list which represents measurement assigned to the group i. Each of these list is
        a list of partial measurements. Each partial measurements is a list of two elements. The first of these elements
        encodes the partial measurement assigned and the second the qubits where it should performed.
    # operator: SumOp
        # Rearrange Pauli strings that obtain the best grouping for the number of Monte Carlo shots provided.
    """

    PS, weigths, labels = Label2Chain(operator)

    orders = []
    order = np.arange(len(PS))  # Default order
    args = []
    results = []

    for i in range(n_mc):
        if i != 0:
            np.random.shuffle(order)  # Shuffle randomly the Pauli strings
        orders.append(np.copy(order))
        args.append([i, n_groups, [PS[order], AM, WC]])

    if progress_bar:  # initialize the progress bar, depending if the instance is in a Jupyter notebook or not
        if isnotebook():
            pbar = tqdm_notebook(total=n_mc, desc='Computing optimal order')
        else:
            pbar = tqdm(total=n_mc, desc='Computing optimal order', file=sys.stdout, ncols=90,
                        bar_format='{l_bar}{bar}{r_bar}')

    pool = Pool()  # Initialize the multiprocessing
    for i, result in enumerate(pool.imap_unordered(unpack_functions, args, chunksize=1), 1):
        results.append(result)  # Save the result
        if progress_bar:
            pbar.update()

    if progress_bar:
        pbar.close()

    pool.terminate()
    results = sort_solution(results)  # Sort the async results

    number_groups = []
    for result in results:
        number_groups.append(result)

    index = np.argmin(number_groups)

    print('The original order gives {} groups'.format(number_groups[0]))
    print('The best order found gives {} groups'.format(number_groups[index]))

    order = orders[index]

    operator = SummedOp([PauliOp(Pauli(labels[order[j]]), weigths[order[j]]) for j in range(len(order))])

    Groups, Measurements = grouping(PS[order], AM, WC)  # Obtain the groups and measurements for the best case
    
    # Remap the Pauli strings so the initial order is conserved
    for i in range(len(Groups)):
        for j in range(len(Groups[i])):
            Groups[i][j] = order[Groups[i][j]]

    return Groups, Measurements #operator


#%% Renovations for the grouping algorithm after Qiskit hackaton

def empty_dict_factors():
    ''' Create an empty dictionary with one entry for each possible 2 qubit combination (II, IX, ...) --> F['00']=0, F['01']=0, ...
    Each entry will be filled with the number of times that two qubits in N pauli strings have that factor. 
    '''
    F={}
    for i in range(4):
        for j in range(4):
            F[str(i)+str(j)] = 0
            
    return F


def empty_dict_compatible_measurements():
    ''' Create an empty dictionary with one entry for each measurement (Bell, Ωx, ...) --> F['0']=0, F['1']=0, ...
    Each entry will be filled with the number of times that each measurement creates a compatible measurement between pairs of qubits.
    '''
    CM = {}
    for i in range(0,10):
            CM[str(i)] = 0
            
    return CM


def number_of_compatible_measurements(m,F):
    """
    Given a measurement 'm' and a dictionary of two-qubit factors 'F', calcultes the number of compatible measurements that can be made with 
    that measurement in those factors.

    Parameters
    ----------
    m: integer in [4,9]
    F: dictionary, see 'empty_dict_factors'

    Returns
    -------
    n:  integer
        Number of compatible measurements with 'm' on the list 'F'.  
        
    """
    pair = []
    for i in range(4):
        pair.append( str(Comp[m][i][0])+str(Comp[m][i][1]) )
        
    n_compatibilites = (1/2)*( (F[pair[0]]+F[pair[1]]+F[pair[2]]+F[pair[3]])**2 - (F[pair[0]]+F[pair[1]]+F[pair[2]]+F[pair[3]]) )
        
    return n_compatibilites

def number_of_compatible_measurements_onequbit(m,F):
    """
    Given a measurement 'm' and a array of one-qubit factors 'F', calcultes the number of compatible measurements that can be made with 
    that measurement in those factors

    Parameters
    ----------
    m: integer in [1,3]
    F: array

    Returns
    -------
    n:  integer
        Number of compatible measurements with 'm' on the list 'F'.  
        
    """
    compatiblefactors=np.size(np.argwhere(F==0))+np.size(np.argwhere(F==m))
    n_compatibilities=compatiblefactors*(compatiblefactors-1)/2
        
    return n_compatibilities

  
def compatiblities(PS): # Algorithm 1 of Fran's notes.
    """
    Given a set of 'n' Pauli Strings (PS) with 'N' qubits, returns three arrays regarding the compatibilites of the measurements.

    Parameters
    ----------
    PS: ndarray (n, N)
        PS are the Pauli strings. Each row represents a Pauli string, and each column represents a qubit.

    Returns
    -------
    C:  ndarray (n, N) Symmetric matrix whose diagonal elements are -1.
        The element C[i,j] contains the number of times that 
        the qubits i and j of one string are jointly measurable (compatible) with the qubits i and j of other string.
        For example, the pauli strings [1,1] and [2,2] will produce C[0,1] = 1 because the qubits 0 and 1 of those
        pauli strings are jointly measurable only with the bell measurement.
        
    """
    
    n = np.shape(PS)[0]
    N = np.shape(PS)[1]
    
    C  = np.diag( -1*np.ones(N) )
    
    for i in range(N):         # First qubit
        for j in range(i+1,N): # Second qubit
            PSij = PS[:,[i,j]]
            CMij = empty_dict_compatible_measurements()
            
            F = empty_dict_factors()
            for s in range(n): # Generate factors list
                label = str(PSij[s,0]) + str(PSij[s,1])
                F[label] += 1
            
            for m in range(4,10): # Fill compatibility measurements between qubits i and j
                CMij[str(m)] = number_of_compatible_measurements(m,F)
            
                C[i,j]     += CMij[str(m)]
                C[j,i]     += CMij[str(m)]
                    
    return C


def transpile_HEEM(G,C, connected=False): 
    
    C  = copy.copy(C)
    G  = copy.deepcopy(G)
    
    N  = np.shape(C)[0]
    AQ = []
    T  = [None]*N
    
    if connected == False:
        while len(AQ) < N:
            i,j = np.unravel_index(np.argmax(C),[N,N])
            
            if i in AQ and j in AQ:
                C[i,j] = -1
                C[j,i] = -1
                
            elif i in AQ or j in AQ:
                
                if i in AQ:
                    assigned     = i
                    not_assigned = j
                else:
                    assigned     = j
                    not_assigned = i
                
                # Assign the not assigned theoretical qubit to the first neighbor availeable
                for neighbor in G.neighbors(T[assigned]):
                    if neighbor not in T:
                        T[not_assigned] = neighbor
                        AQ.append(not_assigned)
                        C[i,j] = -1
                        C[j,i] = -1
                
                        #If the neighbors_2 of the just assigned qubit are also assigned, 
                        # remove the edge in the graph because it is not availeable
                        neighbors_2 = copy.copy( G.neighbors( neighbor ) )
                        for neighbor_2 in neighbors_2: #Loop 1
                            if neighbor_2 in T:
                                G.remove_edge(neighbor_2, neighbor)
                                if nx.degree(G,neighbor_2) == 0:
                                    s=T.index(neighbor_2)
                                    C[s,:] = -1
                                    C[:,s] = -1
                                            
                        if nx.degree(G,neighbor) == 0:
                            C[not_assigned,:] = -1
                            C[:,not_assigned] = -1

                        break
                    
            else:
                success = False
                for I,J in G.edges():
                    if I not in T and J not in T:
                        T[i] = I
                        T[j] = J
                        AQ.append(i)
                        AQ.append(j)
                        C[i,j] = -1
                        C[j,i] = -1
                        success = True
                        
                        for node in I,J:
                            neighbors = copy.copy( G.neighbors(node) )
                            for neighbor in neighbors: #Loop 1
                                if neighbor in T:
                                    G.remove_edge(neighbor, node)
                                    if nx.degree(G,neighbor) == 0:
                                        s = T.index(neighbor)
                                        C[s,:] = -1
                                        C[:,s] = -1
                        if nx.degree(G,neighbor) == 0:
                            C[T.index(node),:] = -1
                            C[:,T.index(node)] = -1
                                        
                        break
                    
                if success == False:
                    C[i,j] = -1
                    C[j,i] = -1
                    
    elif connected == True :
        # First we assign two qubits, then we build the map from them ensuring that the resulting graph is connected
        i,j = np.unravel_index(np.argmax(C),[N,N])
        for I,J in G.edges():
            # Update the map
            T[i] = I
            T[j] = J
            AQ.append(i)
            AQ.append(j)
            C[i,j] = -1
            C[j,i] = -1
            
            #Remove used edges
            G.remove_edge(I, J)
            if nx.degree(G,I) == 0:
                C[i,:] = -1
                C[:,i] = -1
            if nx.degree(G,J) == 0:
                C[j,:] = -1
                C[:,j] = -1
            
            break
        
        while len(AQ) < N:
            Cp = C[AQ,:]
            i,j = np.unravel_index(np.argmax(Cp),[N,N])
            i = AQ[i]
            
            if i in AQ and j in AQ:
                C[i,j] = -1
                C[j,i] = -1
                
            elif i in AQ or j in AQ:
                if i in AQ:
                    assigned     = i
                    not_assigned = j
                else:
                    assigned     = j
                    not_assigned = i
                
                # Assign the not assigned theoretical qubit to the first neighbor availeable
                for neighbor in G.neighbors(T[assigned]):
                    if neighbor not in T:
                        T[not_assigned] = neighbor
                        AQ.append(not_assigned)
                        C[i,j] = -1
                        C[j,i] = -1
                        success = True
                
                        #If the neighbors_2 of the just assigned qubit are also assigned, 
                        # remove the edge in the graph because it is not availeable
                        neighbors_2 = copy.copy( G.neighbors(T[not_assigned]) )
                        for neighbor_2 in neighbors_2: #Loop 1
                            if neighbor_2 in T:
                                G.remove_edge(neighbor_2, neighbor)
                                if nx.degree(G,neighbor_2) == 0: 
                                    s = T.index(neighbor_2)
                                    C[s,:] = -1
                                    C[:,s] = -1                
                            
                        if nx.degree(G,neighbor) == 0:
                            C[not_assigned,:] = -1
                            C[:,not_assigned] = -1
                    
                        break
                    
        
                
    return T
    
    
#%% Compatibility matrix test
# for k in range(1000):
#     print(k)
#     N=6
#     PS = np.random.randint(0,4,[4,N])
#     # PS = np.array([[0,1,1,3,3,1],[2,2,1,2,0,1],[1,3,3,3,2,0],[0,1,2,1,1,2]])
#     C, _, _ = compatiblities(PS)
    
#     # Transpile test
#     G = nx.Graph()
#     G.add_nodes_from(np.arange(7))
#     G.add_edges_from([(0, 1)])
#     G.add_edges_from([(0, 2)])
#     G.add_edges_from([(0, 3)])
#     G.add_edges_from([(2, 4)])
#     G.add_edges_from([(3, 4)])
#     G.add_edges_from([(4, 5)])
#     G.add_edges_from([(4, 6)])
#     G.add_edges_from([(6, 7)])
#     # nx.draw_networkx(G)
    
#     T = transpile_HEEM(G,C,connected=bool(True*np.mod(k,2)))


def Tcompatiblities(PS,T,G): # Algorithm 4 of Fran's notes.
    """
    Given a set of 'n' Pauli Strings (PS) with 'N' qubits, returns three arrays regarding the compatibilites of the measurements.
    
    Parameters
    ----------
    PS: ndarray (n, N)
        PS are the Pauli strings. Each row represents a Pauli string, and each column represents a qubit.
        
    T: list 
        T is the map from theoretical qubits to physical qubits. If T[i]=j it means that the i-th theoretical qubit is 
        mapped to a the j-th physical qubit. 
        
    G: graph
        G is the connectivity graph of the chip. It vertices represents physical qubits and its edges physical connections
        between them.

    Returns
    -------
    CT:  ndarray (n, N) Symmetric matrix whose diagonal elements are -1.
        The element C[i,j] contains the number of times that 
        the qubits i and j of one string are jointly measurable (compatible) with the qubits i and j of other string,
        given that T have been choosen as the map from theoretical qubits to physical qubits.
    
    CTM: dictionary of 9 entries, one for each measurement
        Number of times that qubits of two pauli strings are compatible with each measurement, given that T have been 
        choosen as the map from theoretical qubits to physical qubits. If one measurement has a large value 
        in this dictionary it means that many pairs of qubits can be jointly measured in two pauli strings
        with that measurement.
        
    CTQ: ndarray (N)
        The element CQ[i] cointains the number of times that the qubit i can participate in a joint measurement
        with any other qubit thorugh any measurement, given that T have been choosen as the map from theoretical qubits to physical qubits . It is the sum of the i_th row/column of the matrix C, excluding 
        the -1 of the diagonal plus the number of compatibilities due to one-qubit measurements.
        
    """
    
    n = np.shape(PS)[0]
    N = np.shape(PS)[1]
    
    C  = np.diag( -1*np.ones(N) )
    CM = empty_dict_compatible_measurements()
    CQ = np.zeros(N)
    
    for i in range(N):         # First qubit
        for j in range(i+1,N): # Second qubit
            if [T[i],T[j]] in G.edges():
                PSij = PS[:,[i,j]]
                CMij = empty_dict_compatible_measurements()

                F = empty_dict_factors()
                for s in range(n): # Generate factors list
                    label = str(PSij[s,0]) + str(PSij[s,1])
                    F[label] += 1

                for m in range(4,10): # Fill compatibility measurements between qubits i and j
                    CMij[str(m)] = number_of_compatible_measurements(m,F)

                    C[i,j]     += CMij[str(m)]
                    C[j,i]     += CMij[str(m)]
                    CM[str(m)] += CMij[str(m)]
        
        CQ[i] = 1 + np.sum(C[i,:])
        for m in range(1,4):
            CMi = empty_dict_compatible_measurements()
            CMi[str(m)]=number_of_compatible_measurements_onequbit(m,PS[:,i])
            CQ[i]=CQ[i]+CMi[str(m)]
            CM[str(m)] += CMi[str(m)]
    CT=C
    CTM=CM
    CTQ=CQ
    return CT, CTM, CTQ


# In[68]:


def MeasurementAssignmentWithOrder(Vi, Vj, Mi, AM, WC, OQ):
    """
    
    This function regards the assignment of admissible and efficient measurements Mi to the pauli strings Vi and Vj. 
    
    Admisible and efficient (compatible) means that the code tries to add to Mi admissible measurements
    (AM is the list of admissible measurements, ordered by preference) involving well connected 
    qubits (WC is the list of pairs of directly connected qubits in the quantum processor).
    
    This function follows one of two different paths according to the input Mi:
    
    A) If Mi is an empty list the function assigns measurements Mi that are both compatible with Vi and Vj.
    
    B) If Mi is not complete, i.e. there are qubits with no assigned measurements, first the function checks if 
    the currently assigned measurements Mi are compatible with Vj. If this is true then in tries to assign to the remaining
    qubits measurements that are both compatible with Vi and Vj.
    
    In both cases A and B the function returns S=True iff the updated Mi is full and compatible with Vi and Vj, else, 
    it returns S=False and an unchanged Mi.
    
    This program is the Algorithm 2 of https://arxiv.org/abs/1909.09119.

    Parameters
    ----------
    Vi: array
        It is the array associated to the i Pauli string, according to our numerical encoding.
    Vj: array
        It is the array associated to the j Pauli string, according to our numerical encoding.
    Mi: list
        Current assignment of the measurement of the group i. It is a list of partial measurements. Each partial
        measurements is a list of two elements. The first of these elements encodes the partial measurement assigned and
        the second the qubits where it should performed.
    AM: list
        It is the list of the admissible measurements considered. Regarding our numerical encoding, it is a list of
        integers from 1 to 9. The order of the list encodes the preference of measurement assignment. For instance, the
        first measurement appearing in this list will be the one that would be preferentially assigned.
    WC: list
        It is a list of tuples. Each tuple represents a set of well connected qubits.
    OQ: list 
        It is a list of integers. It represents the order of qubits that the algorithm should follow in each iteration.

    Returns
    -------
    UMi: list
        Updated Mi. If the algorithm fails, UMi will be equal to Mi.
    S: bool
        If the algorithm has succeeded. In other words, if Mi has been updated in a way such the group of Vi and Vj are
        compatible, S=True. Otherwise, S=False.
        
    v2 Changes    
    ----------
    1.- Modified and extended the description of function.
    2.- Modified the comments of the function.
    """
    
    # The first loop checks if the current assignment of Mi is compatible with Vj. If not, the programm returns Mi and
    # S=False. If Mi is compatible with Vj, the array U will contain the qubits where Mi does not act.

    N = np.size(Vi)
    U = OQ.copy()
    # Check of Vj compatibility with current Mi
    for PM in Mi:
        if list(Vj[PM[1]]) not in Comp[PM[0]]:
            return Mi, False
        else:
            for k in PM[1]:
                U.remove(k)

    commonfactors = np.argwhere(Vi == Vj)
    for k in commonfactors:
        if k in U:
            U.remove(k)

    # After the second loop U contains the qubits where Mi does no act and the factors of Vi and Vj are not equal.
    # Thus, it is in the qubits of U where partial measurements have to be assigned to make the strings of Vi and Vj 
    # compatible.
    
    # The third loop tries to update the measurement Mi on the qubits in U. To do so it runs through the admissible 
    # partial measurements AM (admissible measurements loop). For each of those measurements, the loop runs through all 
    # the possible set of qubits where the measurement can act (perm) (possible qubits loop). For each element 'per' in 
    # 'perm' the code checks if 'per' is a set of well connected qubits (connectivity check). Finally, it is checked if
    # the measurement on those qubits is compatible with the string Vi and Vj (if so, by construction, the measurement 
    # will be compatible with all strings of group Vi and with Vj)(compatibility check). If there is success in this last 
    # check, UMi is updated with that partial measurement, the qubits where this partial measurement are deleted of U 
    # and we begin again if U is not empty. If we managed to empty U, the update would have succeeded and we would return
    # UMi, S=True. If there is no success, Mi, S=False are returned.

    # We will update Mi in the following loop. We create UMi because the loop might fail in the middle, thus,
    # an unchaged Mi should be returned.
    UMi = Mi[:]

    # UMi updating loop
    while len(U) != 0:
        for Eps in AM:  # Admissible measurement loop
            if len(U) >= len_meas[Eps]: 
                perm = list(permutations(U, len_meas[Eps]))
                for per in perm:  # Possible qubits loop
                    if len_meas[Eps]>=2:
                        Tper=(int(T[per[0]]),int(T[per[1]]))
                    else:
                        Tper=0
                    if (Tper in WC) or (len_meas[Eps] == 1):  # Connectivity check
                        if (list(Vi[tuple([per])]) in Comp[Eps]) and (
                                list(Vj[tuple([per])]) in Comp[Eps]):  # Compatibility check
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


def groupingWithOrder(PS, G,connected=False):
    """
    Given a set of Pauli strings (PS), this function makes groups of PS assigning taking into account the chip's
    connectivity.

    Parameters
    ----------
    PS: array (n, M)
        Pauli strings, each row represents a Pauli string while each the column represents a qubit. Thus, n is the
        number of Pauli strings and N is the number of qubits.
    G: graph
        Graph that represents the connectivity of the chip
    connected: boolean
        If connected=True the transpile_HEEM algorithm ensures that the subgraph of the theoretical qubits in the 
        chip is connected. If connected=False the transpile_HEEM algorithm does not ensure that the the subgraph of
        the theoretical qubits in the chip is connected, instead tries to optimize \omega(T) in a greedy way.

    Returns
    -------
    Groups: list
        The element in the position i is a list with the indexes of strings assigned to group i, i.e, the strings of
        group i.
    Measurement: list
        The element in position i is a list which represents measurement assigned to the group i. Each of these list is
        a list of partial measurements. Each partial measurements is a list of two elements. The first of these elements
        encodes the partial measurement assigned and the second the qubits where it should performed.
    T: list
        T is the theo-phys map chosen. T[i]=j means that the i-theoretical qubit is mapped to the j-physical qubit.
    
    ¡Important!: In 'Measurement' output the indexes that we use to refer to the qubits are theoretical indexes, 
    not the correspondant physical indexes (i.e., if we have the i-theoretical qubit is mapped to the j-physical qubit
    through T, in other words T[i]=j, we use the index i and not the j to refer to that qubit)
        
        
        """
    PG = PauliGraph(PS)
    SV = sorted(PG.degree, key=lambda x: x[1], reverse=True)
    n = np.size(PS[:, 0])
    N = np.size(PS[0, :])
    WC=list(G.edges) #list of pairs of well connected qubits
    AS = []  # List of strings with assigned measurement
    C=compatiblities(PS)
    T=transpile_HEEM(G,C, connected)
    CT,CM,CQ=Tcompatiblities(PS,T,G)
    CMlist=[]
    for i in range(1,10):
        CMlist.append(CM[str(i)])
    AM=[i[0] for i in sorted(enumerate(CMlist), key=lambda k: k[1], reverse=True)]
    AM=[x+1 for x in AM]
    OQ=[i[0] for i in sorted(enumerate(list(CQ)), key=lambda k: k[1], reverse=True)]
    Groups = []
    Measurements = []
    for k in range(n):
        i = SV[k][0]  # We run the Pauli strings in a decreasing order of CQ.
        if i not in AS:  # If we enter to this loop, the i string will have its own group.
            Mi = []
            GroupMi = [i]
            AS.append(i)
            for l in range(n):  # We try to make the group of the string i as big as possible
                j = SV[l][0]
                if j not in AS:
                    Mi, S = MeasurementAssignmentWithOrder(PS[i, :], PS[j, :], Mi, AM, WC, OQ)
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
            as we did in the TPBgrouping code.
            """
            Groups.append(GroupMi)
            Measurements.append(Mi)

    return Groups, Measurements, T

