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

    Comp_1={I, X} = {[0],[1]}, Comp_2={I, Y} = {[0],[2]}, Comp_3={I, Z} = {[0],[3]},
    Comp_4={II,XX,YY,ZZ} = {[0,0],[1,1],[2,2],[3,3]}, Comp_5={II,XX,YZ,ZY} = {[0,0],[1,1],[2,3],[3,2]},
    Comp_6={II,YY,XZ,ZX} = {[0,0],[2,2],[1,3],[3,1]}, Comp_7={II,ZZ,XY,YX} = {[0,0],[3,3],[1,2],[2,1]},
    Comp_8={II,XY,YX,ZX} = {[0,0],[1,2],[2,1],[3,1]}, Comp_9={II,YX,ZY,XZ} = {[0,0],[2,1],[3,2],[1,3]}.


Thus, when checking the compatibility of the strings v_i and v_j with the measurement k on the qubits (l,m),
what we should do is checking if [v_i(l),v_i(m)] and [v_j(l),v_j(m)] are both in the compatibility list 
of the measurement k. For example, if we had v_i=YIZZ=[2,0,3,3] and v_j=XIZY=[1,0,3,2] and we wanted to check
if theses strings are compatible with the measurement 4 (the Bell measurement) on the qubits (3,4), what we have
to do is checking if [v_i(3),v_i(4)]=[3,3] and [v_j(3),v_j(4)]=[3,2] are in the compatibility list of the 
measurement 4. As this compatibility list is Comp_4={[0,0],[1,1],[2,2],[3,3]}, we have that [v_i(3),v_i(4)] belongs
to Comp_4 but [v_j(3),v_j(4)] does not. In consequence, the measurement 4 on the qubits (3,4) is not compatible with 
v_i and v_j. 
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

length = [len(x[0]) for x in Comp]


def PauliGraph(PS):
	"""
    Construction of the Pauli Graph

    Parameters
    ----------
    PS: ndarray (n, N)
        PS are the Pauli strings.Each row represents a Pauli string, and each column represents a qubit. Thus, n is the
        number of Pauli strings and N is the number of qubits.

    Return
    ------
    PG: graph
        The Pauli graph corresponding with the n Pauli strings given as input.
    """

	n = np.size(PS[:, 0])
	N = np.size(PS[0, :])
	PG = nx.Graph()
	PG.add_nodes_from(np.arange(n))  # Assigns a node to each Pauli string
	for i in range(n):  # Run a loop over each Pauli string v_i
		v_i = PS[i, :]

		for j in range(i + 1, n):
			v_j = PS[j, :]
			qubits = np.arange(N)
			# TPB compatibility checking
			noncommonqubits = np.delete(qubits,
			                            np.argwhere(v_i == v_j))  # Qubits corresponding to the noncommon factors
			vi = v_i[noncommonqubits]
			vj = v_j[noncommonqubits]
			if (vi * vj != 0).any():
				PG.add_edges_from([(i, j)])
			"""
            To check compatibility it has to be verified if for each noncommon qubit at least one of the factos is I.
            Regarding our numerical encoding, one of the factors is I iff one of the factors is 0, i. e., the factors
            k of strings v_i and v_j (if they are different) are compatible iff v_i[k] and v_j[k] iff v_i[k]v_j[k]==0.
            """
	return PG


def colorgroups(colordict):
	"""
    Construction of the TPB groups from the color dictionary.

    Parameters
    ----------
    colordict: dictionary
		The keys are the indexes of the Pauli strings. The values are the colors assigned to the Pauli string.

    Return
    ------
    Groups: list
		The element in the position i is a list with the indexes of strings assigned to color i, i.e, the group of
		 strings with color i.
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
        Pauli strings, each row represents a Pauli string while each the column represents a qubit. Thus, n is the
        number of Pauli strings and N is the number of qubits.

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
	PG = PauliGraph(PS)
	Color = nx.coloring.greedy_color(PG)  # Graph coloring code of networkx. By default it uses LDFC strategy.
	Groups = colorgroups(Color)  # Groups of strings with the same color assigned
	N = np.size(PS[0, :])
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
    Given a current assignment of measurements Mi to the group of string Vi, this code checks if Mi can be updated
    so Vj becomes compatible with the group of string Vi. To do so, the code tries to add to Mi admissible measurements
    (AM is the list of admissible measurements) involving well connected qubits (WC is the list of pairs of well
    connected qubits).

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
    """

	N = np.size(Vi)
	U = list(np.arange(N))
	# Check of Vj compatibility with current Mi
	for PM in Mi:
		if list(Vj[PM[1]]) not in Comp[PM[0]]:
			return Mi, False
		else:
			for k in PM[1]:
				U.remove(k)

	"""
    This loop checks if the current assignment of Mi is compatible with Vj. If not, the programm returns Mi and
    S=0(fail) If Mi is compatible with Mi, the array U, after this loop, will contain the qubits where Mi does not act.
    """

	commonfactors = np.argwhere(Vi == Vj)
	for k in commonfactors:
		if k in U:
			U.remove(k)

	"""
    After this loop U contains the qubits where Mi does no act and the strings of Vi and Vj does not coincide.
    Thus, is in the qubits of U where partial measurements have to be assigned to make the strings of Vi and Vj 
    compatible.
    """

	# We will update Mi in the following loop. We create UMi because the loop may fail in the middle, so Mi should be
	# return and UMi would be not equal to Mi.
	UMi = Mi[:]

	# Mi updating loop
	while len(U) != 0:
		for Eps in AM:  # Admissible measurement loop
			if len(U) >= length[Eps]:
				perm = list(permutations(U, length[Eps]))
				for per in perm:  # Possible qubits loop
					if (per in WC) or (length[Eps] == 1):  # Connectivity check
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

	"""
    This loops tries to update the measurement Mi on the qubits enlisted in U. To do so it runs through the admissible 
    partial measurements AM (admissible measurements loop). For each of those measurements, the loop runs through all 
    the possible set of qubits where the measurement can act (perm) (possible qubits loop). For each element 'per' in 
    'perm' the code checks if 'per' is a set of well connected qubits (connectivity check). Finally, it is checked if
    the measurement on those qubits is compatible with the string Vi and Vj (if so, by construction, the measurement 
    will be compatible with all strings of group Vi and with Vj)(compatibility check). If there is success in this last 
    check, UMi is updated with that partial measurement, the qubits where this partial measurement are deleted of U 
    and we begin again if U is not empty. If we managed to empty U, the update would have succeeded and we would return
    UMi, S=1. If there is no success, Mi, S=0 are returned.
    """

	return UMi, True


def grouping(PS, AM, WC=None):
	"""
    Given a set of Pauli strings (PS), this function make groups of Paulis assigning the admissible measurements
    given as an input (AM) on the well connected qubits (WC).

    Parameters
    ----------
    PS: array (n, M)
        Pauli strings, each row represents a Pauli string while each the column represents a qubit. Thus, n is the
        number of Pauli strings and N is the number of qubits.
    AM: list
        It is the list of the admissible measurements considered. Regarding our numerical encoding, it is a list of
        integers from 1 to 9. The order of the list encodes the preference of measurement assignment. For instance, the
        first measurement appearing in this list will be the one that would be preferentially assigned.
    WC: list (optional)
        It is a list of tuples. Each tuple represents a set of well connected qubits. If WC is not provided, and all-to-
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
	#	Rearrange Pauli strings that obtain the best grouping for the number of Monte Carlo shots provided.
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

#	operator = SummedOp([PauliOp(Pauli(labels[order[j]]), weigths[order[j]]) for j in range(len(order))])

	Groups, Measurements = grouping(PS[order], AM, WC)  # Obtain the groups and measurements for the best case

    # Remap the Pauli strings so the initial order is conserved
	for i in range(len(Groups)):
		for j in range(len(Groups[i])):
			Groups[i][j] = order[Groups[i][j]]

	return Groups, Measurements
