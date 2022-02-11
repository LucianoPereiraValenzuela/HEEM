"""
Variational Quantum Eigensolver algorithm with Hardware Efficient Entangled measurements.
"""

import numpy as np 
import matplotlib.pyplot as plt 
from qiskit.providers.aer import AerSimulator 
from qiskit import IBMQ, Aer 
from qiskit.circuit.library import EfficientSU2 
from qiskit.algorithms import NumPyMinimumEigensolver 
from qiskit.algorithms.optimizers import SPSA 
from qiskit.compiler import transpile 

def main(backend, 
         user_messenger,
         pars  = None,
         grouping = 'TPB',
         iters = 10,
         shots = 8192,
         conectivity = None ):
    
    energies = []
    def callback( ExpVal, params ):
        energies.append(ExpVal)
    
    H, circuit_init = H2O(initial_state=True) 
    min_energy = NumPyMinimumEigensolver( ).compute_minimum_eigenvalue(H).eigenvalue
    num_qubits = H.num_qubits 
    optimizer = SPSA( maxiter=iters, last_avg=1  )

    # quantum_instance = QuantumInstance( backend, shots = shots )
    
    solver = VQE( circuit_init, 
                optimizer, 
                params = pars, 
                backend = backend, 
                conectivity=conectivity, 
                back = callback, 
                grouping = grouping )
    results = solver.compute_minimum_eigenvalue( H )

    return results_to_dict(results, energies)

def results_to_dict(results, energies):
    
    results_dict = {
        'values' : energies
        }

    return results_dict

from qiskit.circuit.library import EfficientSU2 
from qiskit.compiler import transpile 

class VQE :

    def __init__( self, 
                    circuit_init,
                    optimizer,
                    params,
                    backend,
                    shots = 2**10,
                    grouping = 'HEEM',
                    conectivity = None,
                    back = None ):

        self.circuit_init = circuit_init
        #self._num_params = variational_circuit.num_parameters 
        self._optimizer = optimizer
        self._params = params 
        self._grouping = grouping
        self._backend = backend
        self._shots = shots

        if conectivity is None :
            conectivity = get_backend_connectivity( backend )
        self._conectivity = conectivity
        self._back = back


    def compute_minimum_eigenvalue( self, H ):

        self.circuits( H )

        results = self._optimizer.optimize( self._num_params, self.energy_evaluation, initial_point=self._params)

        return results

    def energy_evaluation( self, params ):
        circuits_tp = [ circuit.assign_parameters(params) for circuit in self._circuits ]
        circuits_tp = transpile( circuits_tp, backend=self._backend, initial_layout=range(self._num_qubits) )
        counts = self._backend.run( circuits_tp, shots=self._shots ).result().get_counts()
        probs = [ post_process_results( count, self._num_qubits, self._shots ) for count in counts ]
        ExpVal = 0
        for j in range(len(probs)):
            ExpVal += np.sum(self._prob2Exp[j]@probs[j])

        if self._back is not None:
            self._back( ExpVal, params ) 

        return ExpVal

    def grouping( self, H ):

        self._num_qubits = H.num_qubits 
        paulis, coeff, labels = Label2Chain( H )

        if self._grouping == 'TPB' :
            Color, self._Groups, self._Measurements = TPBgrouping( paulis )
            self._layaout = range(self._num_qubits)
        elif self._grouping == 'EM' :
            self._Groups, self._Measurements, self._layaout = groupingWithOrder( paulis )
            self._layaout = self._layaout[::-1]
        elif self._grouping == 'HEEM' :
            self._Groups, self._Measurements, self._layaout = groupingWithOrder( paulis, G=self._conectivity )
            self._layaout = self._layaout[::-1]

        self._prob2Exp = probability2expected( coeff, labels, self._Groups, self._Measurements)


    def circuits( self, H ):

        self.grouping( H )

        self._variational_circuit = self.hardware_efficient_circuit()

        self._circuits = [ transpile( measure_circuit_factor( measure , self._num_qubits 
                            ), initial_layout=self._layaout ).compose( self._variational_circuit, front=True  )
                            for measure in self._Measurements ]  


    def hardware_efficient_circuit( self ):
        
        WC = [ indx for indx in self._conectivity if indx[0]<indx[1] ]
        variational_circuit = self.circuit_init.compose( EfficientSU2( self._num_qubits, reps=1, entanglement=WC ) ) 
        self._num_params = variational_circuit.num_parameters 
        return variational_circuit



import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

# Maps for the order of measurements in each basis
maps = [np.array(['XX', 'YY', 'ZZ', 'II']),  # Bell
        np.array(['XX', 'YZ', 'ZY', 'II']),  # Omega xx
        np.array(['YY', 'XZ', 'ZX', 'II']),  # Omega yy
        np.array(['ZZ', 'XY', 'YX', 'II']),  # Omega zz
        np.array(['XY', 'YZ', 'ZX', 'II']),  # Chi
        np.array(['YX', 'ZY', 'XZ', 'II'])]  # Chi_prime

# Factors for expected value of one qubit, and two qubits (in the correct order for each basis)
factors_list = [[np.array([1, -1]), np.array([1, 1])],  # One qubit
                [np.array([1, -1, 1, -1]), np.array([-1, 1, 1, -1]), np.array([1, 1, -1, -1]), np.array([1, 1, 1, 1])],
                # Bell
                [np.array([1, -1, -1, 1]), np.array([-1, 1, -1, 1]), np.array([-1, -1, 1, 1]), np.array([1, 1, 1, 1])],
                # Omega xx
                [np.array([1, -1, -1, 1]), np.array([1, -1, 1, -1]), np.array([1, 1, -1, -1]), np.array([1, 1, 1, 1])],
                # Omega yy
                [np.array([1, 1, -1, -1]), np.array([-1, 1, -1, 1]), np.array([-1, 1, 1, -1]), np.array([1, 1, 1, 1])],
                # Omega zz
                [np.array([1, -1, 1, -1]), np.array([-1, 1, 1, -1]), np.array([1, 1, -1, -1]), np.array([1, 1, 1, 1])],
                # Chi
                [np.array([-1, 1, 1, -1]), np.array([1, 1, -1, -1]), np.array([1, -1, 1, -1]),
                 np.array([1, 1, 1, 1])]]  # Chi_prime


def post_process_results(result, n_q, NUM_SHOTS):
	"""
	Transform the counts obtained when running the backend with multiple classical register into an array of
	probabilities.

	Parameters
	----------
	result: dict('str': int)
		Dictionary in which the keys are the results of a given experiment, with each classical register separated by a
		space. In the values of the dictionaries are saved the number of times a given result is obtained.
	n_q: int
		Number of measured qubits. This values does not have to coincide with the total number of qubits in the circuit.
	NUM_SHOTS: int
		Number of shots for the measurements.

	Return
	------
	probabilities: (2 ** n_q) array(float)
		Probabilities for each result, where each index correspond to the given result in decimal.
	"""

	# Initialize list for the results and the counts
	labels = []
	counts = []
	for key in result.keys():
		labels.append(key.replace(' ', ''))  # Join all the classical register in one single string with no spaces
		counts.append(result[key])

	# Initialize the array of probabilities with all the results in 0
	probabilities = np.zeros(2 ** n_q)
	for j in range(len(labels)):
		# Transform the result from binary to decimal, and save the probability
		probabilities[int(labels[j], 2)] += counts[j] / NUM_SHOTS

	return probabilities


def generate_diagonal_factors(*factors):
	"""
	Generate the diagonal part of the tensor product of matrices that represent the basis in which each qubit (or pair
	of qubits) has been measured. The tensor product is obtained by computing the Kronecker product. This product for a
	pair of square matrices A (m x m) and B(n x n) is given by:

	A x B = | a_11 B   . . .   a_1m B |
			|   .      .         .    |
			|   .        .       .    |
			|   .         .      .    |
			| a_m1 B   . . .   a_mm B |

	Since in our case all the matrices are diagonals, we don't have to compute the entire matrix, and just obtained its
	diagonal factors as:
	diag(A x B) = (a_11 * diag(B), ..., a_mm * diag(B))
	Parameter
	---------
	factors: list(int)
		List in which each element is another list with the diagonal part of each matrix

	Return
	------
	diagonal_factor: array(int)
		Diagonal elements of the tensor product
	"""
	factors = factors[::-1]  # Invert the order to ensure the correct Kronecker product
	diagonal_factor = factors[0]  # Initialize the diagonal factors as the diagonal of the first matrix
	for i in range(1, len(factors)):  # Run over all the indices, except the first one
		temp = np.array([])  # Temporary variable to create the new diagonal factors
		for j in range(len(diagonal_factor)):  # Run over all the elements of the current diagonal factors
			temp = np.hstack([temp, diagonal_factor[j] * factors[i]])  # Append a_jj * B
		diagonal_factor = temp[:]  # Save the computed diagonal factors

	return diagonal_factor


# DEPRECATED
# def measure_circuit_factor(measurements, n_qubits):
# 	"""
# 	Function to create the circuit needed to obtain a given group of measurements. Each measurement will be save in an
# 	individual classical register. Each measurement coded with an int number. The available measurements are:
# 	0 -> Identity
# 	1 -> X
# 	2 -> Y
# 	3 -> Z
# 	4 -> Bell
# 	5 -> Omega_xx
# 	6 -> Omega_yy
# 	7 -> Omega_zz
# 	8 -> Chi
# 	9 -> Chi_prime
#
# 	To ensure the correct functionality of this function each qubit can only be in one of the measurements, so it's only
# 	measured once. If a qubit is not provided, then it is not measured.
#
# 	Parameters
# 	----------
# 	measurements: list(list(int, list(int)))
# 		List with all the measurements. Each measured is a list in which the first index in the int encoding the
# 		measurement, and the second element is another list with the indices of the measured qubits. The convention for
# 		the indices of the qubits is opposite to the qiskit convention. Here the qubit with higher weight is named as
# 		q_0.
# 	n_qubits: int
# 		Total number of qubits in the circuit. This values does not have to coincide with the number of measured qubits.
#
# 	Returns
# 	-------
# 	circuit: quantum circuit
# 		Circuit (including quantum and classical registers) with the gates needed to perform the measurements.
# 	n_measures: int
# 		Number of measured qubits
# 	"""
# 	# Initialize the number of measured qubits to 0 and a list with the classical registers for each measurement
# 	n_measures = 0
# 	classical_registers = []
# 	for measure in measurements:
# 		if measure[0] != 0:  # If the operation is not the identity
# 			classical_registers.append(ClassicalRegister(len(measure[1])))
# 			n_measures += len(measure[1])
#
# 	# Create the quantum circuit
# 	qr = QuantumRegister(n_qubits)
# 	circuit = QuantumCircuit(qr, *classical_registers)
#
# 	counter = 0  # Index for the classical register
# 	for measure in measurements:  # Iterate over all the measurements
# 		measure_label, qubits = measure  # Extract the index of the measurement and the measured qubits
# 		qubits = np.abs(np.array(qubits) - n_qubits + 1)  # Goes to the qiskit convention
# 		qubits = sorted(qubits)  # Ensure the order of the qubits of entangled measurements
# 		if measure_label == 0:
# 			# No measurement
# 			continue
# 		elif measure_label == 1:
# 			# X Circuit
# 			circuit.h(qubits)
# 		elif measure_label == 2:
# 			# Y Circuit
# 			circuit.sdg(qubits)
# 			circuit.h(qubits)
# 			pass
# 		elif measure_label == 3:
# 			# Z Circuit
# 			pass
# 		elif measure_label == 4:
# 			# Bell Circuit
# 			circuit.cnot(qubits[0], qubits[1])
# 			circuit.h(qubits[0])
# 		elif measure_label == 5:
# 			# Omega xx Circuit
# 			circuit.s(qubits)
# 			circuit.h(qubits[0])
# 			circuit.cnot(qubits[0], qubits[1])
# 			circuit.h(qubits[0])
# 		elif measure_label == 6:
# 			# Omega yy Circuit
# 			circuit.h(qubits[0])
# 			circuit.cnot(qubits[0], qubits[1])
# 			circuit.h(qubits[0])
# 		elif measure_label == 7:
# 			# Omega zz Circuit
# 			circuit.s(qubits[0])
# 			circuit.cnot(qubits[0], qubits[1])
# 			circuit.h(qubits[0])
# 		elif measure_label == 8:
# 			# Chi Circuit
# 			circuit.u2(np.pi / 2, np.pi, qubits[0])
# 			circuit.cnot(qubits[0], qubits[1])
# 			circuit.h(qubits[0])
# 		elif measure_label == 9:
# 			# Chi_prime Circuit
# 			circuit.u2(0, np.pi / 2, qubits[0])
# 			circuit.cnot(qubits[0], qubits[1])
# 			circuit.h(qubits[0])
# 		# circuit.barrier(range(n_qubits))
# 		circuit.measure(qubits, classical_registers[counter])
# 		counter += 1
#
# 	return circuit, n_measures


def measure_circuit_factor(measurements, n_qubits, make_measurements=True):
	"""
	This functions differs from the original one in the way to map the measurements to the classical register. In order
	to include Measurement Error Mitigation, so all the qubits are measured in the correct order.

	Function to create the circuit needed to obtain a given group of measurements. Each measurement will be save in an
	individual classical register. Each measurement coded with an int number. The available measurements are:
	0 -> Identity
	1 -> X
	2 -> Y
	3 -> Z
	4 -> Bell
	5 -> Omega_xx
	6 -> Omega_yy
	7 -> Omega_zz
	8 -> Chi
	9 -> Chi_prime

	To ensure the correct functionality of this function each qubit can only be in one of the measurements, so it's only
	measured once. If a qubit is not provided, then it is not measured.

	Parameters
	----------
	measurements: list(list(int, list(int)))
		List with all the measurements. Each measured is a list in which the first index in the int encoding the
		measurement, and the second element is another list with the indices of the measured qubits. The convention for
		the indices of the qubits is opposite to the qiskit convention. Here the qubit with higher weight is named as
		q_0.
	n_qubits: int
		Total number of qubits in the circuit. This values does not have to coincide with the number of measured qubits.
	make_measurements: bool (optional)
		If True, include measurement gates at the end of the circuit.

	Returns
	-------
	circuit: quantum circuit
		Circuit (including quantum and classical registers) with the gates needed to perform the measurements.
	"""
	# Create the quantum circuit
	qr = QuantumRegister(n_qubits)
	cr = ClassicalRegister(n_qubits)
	circuit = QuantumCircuit(qr, cr)

	for measure in measurements:  # Iterate over all the measurements
		measure_label, qubits = measure  # Extract the index of the measurement and the measured qubits

		qubits = list(np.abs(np.array(qubits) - n_qubits + 1))[::-1]  # Goes to the qiskit convention

		if measure_label == 0:
			# No measurement
			pass
		elif measure_label == 1:
			# X Circuit
			circuit.h(qubits)
		elif measure_label == 2:
			# Y Circuit
			circuit.sdg(qubits)
			circuit.h(qubits)
		elif measure_label == 3:
			# Z Circuit
			pass
		elif measure_label == 4:
			# Bell Circuit
			circuit.cnot(qubits[0], qubits[1])
			circuit.h(qubits[0])
		elif measure_label == 5:
			# Omega xx Circuit
			circuit.s(qubits)
			circuit.h(qubits[0])
			circuit.cnot(qubits[0], qubits[1])
			circuit.h(qubits[0])
		elif measure_label == 6:
			# Omega yy Circuit
			circuit.h(qubits[0])
			circuit.cnot(qubits[0], qubits[1])
			circuit.h(qubits[0])
		elif measure_label == 7:
			# Omega zz Circuit
			circuit.s(qubits[0])
			circuit.cnot(qubits[0], qubits[1])
			circuit.h(qubits[0])
		elif measure_label == 8:
			# Chi Circuit
			circuit.u2(np.pi / 2, np.pi, qubits[0])
			circuit.cnot(qubits[0], qubits[1])
			circuit.h(qubits[0])
		elif measure_label == 9:
			# Chi_prime Circuit
			circuit.u2(0, np.pi / 2, qubits[0])
			circuit.cnot(qubits[0], qubits[1])
			circuit.h(qubits[0])

		if make_measurements:
			circuit.measure(qubits, qubits)

	return circuit


def probability2expected(Pauli_weights, Pauli_labels, Groups, Measurements, shift=True):
	"""
	Compute the prefactors for computing the expected value of a given Hamiltonian with the probabilities measured based
	on some grouping of measurements.

	Parameters
	----------
	Pauli_weights: list (complex)
		Weights of each pauli string in the Hamiltonian.
	Pauli_labels: list (str)
		Pauli string in the str convention.
	Groups: list(list(int))
		List in which each element is represented the indices of pauli string that are measured simultaneously.
	Measurements: list(list(int, list(int)))
		List with all the measurements. Each measurement is a list in which the first index is an int encoding the
		measurement, and the second element is another list with the indices of the measured qubits. The convention for
		the indices of the qubits is opposite to the qiskit convention. Here the qubit with higher weight is named as
		q_0.
	shift: (optional) bool
		Change between qubits numbering conventions.

	Return
	------
	diagonal_factor_all: list(array(int))
		Diagonal factors for all the tensor products. Each element in the list represents the diagonal factors for a
		given group of measurements.
	"""
	n_qubits = len(Pauli_labels[0])
	diagonal_factors_all = []  # Initialize the list with all the diagonal factors

	for measurements, group in zip(Measurements, Groups):  # Iterate over all the measurements
		# Pauli weights and string in each measurement
		pauli_weights = [Pauli_weights[i] for i in group]
		pauli_labels = [Pauli_labels[i] for i in group]
		diagonal_factors_temp = []  # Initialize the diagonal factors for the given group

		for i in range(len(pauli_labels)):  # Iterate over all the measured pauli strings
			diagonal_factors = []  # Initialize the diagonal factors for one pauli string
			for j in range(len(measurements)):  # Iterate over all the measurements in a given group
				index_measure, qubits = measurements[j]
				if 0 < index_measure <= 3:  # Single qubit measurement
					if pauli_labels[i][qubits[0]] == 'I':  # If the identity is grouped with another measurement
						diagonal_factors.append(factors_list[0][1])
					else:  # If a Pauli matrix is measured
						diagonal_factors.append(factors_list[0][0])
				elif index_measure > 3:  # Entangled qubits measurement
					# Obtain the tensor product of pauli matrices measured
					measure_string = pauli_labels[i][qubits[0]] + pauli_labels[i][qubits[1]]
					map_basis = maps[index_measure - 4]  # Map of tensor products of the entangled basis
					index = np.where(map_basis == measure_string)[0][0]  # Index in the map
					diagonal_factors.append(factors_list[index_measure - 3][index])
				else:
					diagonal_factors.append(factors_list[0][1])

			# Generate the product tensor of all the diagonal factors
			diagonal_factors = generate_diagonal_factors(*diagonal_factors)
			if shift:
				chain_qubits = []
				for j in range(len(measurements)):  # Iterate over all the measurements in a given group
					index_measure, qubits = measurements[len(measurements) - 1 - j]
					qubits = np.abs(np.array(qubits) - n_qubits + 1)

					for qubit in qubits:
						chain_qubits.append(qubit)

				permutations = swaps(chain_qubits)
				for permutation in permutations:
					diagonal_factors = permute_indices(diagonal_factors,
					                                   permutation[0], permutation[1],
					                                   n_qubits)
			diagonal_factors_temp.append(diagonal_factors * pauli_weights[i])

		diagonal_factors_all.append(np.array(diagonal_factors_temp))

	return diagonal_factors_all

import numpy as np
import networkx as nx
from itertools import permutations
import copy

# from multiprocessing import Pool
# from tqdm import tqdm
# from tqdm.notebook import tqdm_notebook
# import sys
# from utils import Label2Chain, sort_solution, unpack_functions, isnotebook
# from qiskit.opflow.list_ops import SummedOp
# from qiskit.quantum_info import Pauli
# from qiskit.opflow.primitive_ops import PauliOp


"""
See the report for context of this code. 

In order to simplify the programming, we use a numerical encoding. We identify each Pauli string with an integer

I-> 0, X-> 1, Y-> 2, Z-> 3. 

Then, for example, $XIZY$ would be mapped to the array [1,0,3,2]. 

Similarly, we  map  measurements into numbers:
TPBX-> 1, TPBY-> 2, TPBZ-> 3, Bell->4, OmegaX-> 5, OmegaY-> 6, OmegaZ-> 7, chi-> 8, chi'->9.

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

Comp = [[[]],
        [[0], [1]],
        [[0], [2]],
        [[0], [3]],
        [[0, 0], [1, 1], [2, 2], [3, 3]],
        [[0, 0], [1, 1], [2, 3], [3, 2]],
        [[0, 0], [2, 2], [1, 3], [3, 1]],
        [[0, 0], [3, 3], [1, 2], [2, 1]],
        [[0, 0], [1, 2], [2, 3], [3, 1]],
        [[0, 0], [2, 1], [3, 2], [1, 3]]]

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

		for j in range(i + 1, n):  # Nested loop over the following Pauli strings v_j
			v_j = PS[j, :]
			compatiblequbits = np.logical_or.reduce((v_i == v_j, v_i == 0, v_j == 0))
			if not np.all(compatiblequbits):  # If at least one of the qubits shared by the PS is not commutative
				PG.add_edges_from([(i, j)])  # add an edge in the Pauli Graph
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

	Admissible and efficient (compatible) means that the code tries to add to Mi admissible measurements
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

	# The first loop checks if the current assignment of Mi is compatible with Vj. If not, the program returns Mi and
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
	# an unchanged Mi should be returned.
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


def grouping(PS, AM=None, WC=None):
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

	if AM is None:
		AM = [4, 6, 7, 8, 9, 5, 3, 2, 1]

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
			for m in range(n):  # We try to make the group of the string i as big as possible
				j = SV[m][0]
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


### DEPRECATED ###
# def grouping_shuffle(operator, AM, WC, n_mc=500, progress_bar=True):
# 	"""
# 	Shuffle the order for the pauli strings randomly a given number of times and choose the ordering with less number of
# 	groups.
#
# 	Parameters
# 	----------
# 	operator: SumOp
# 		Operator with the Pauli strings to group.
# 	AM: list
# 		It is the list of the admissible measurements considered. Regarding our numerical encoding, it is a list of
# 		integers from 1 to 9. The order of the list encodes the preference of measurement assignment. For instance, the
# 		first measurement appearing in this list will be the one that would be preferentially assigned.
# 	WC: list
# 		It is a list of tuples. Each tuple represents a set of well connected qubits.
#
# 	n_mc: int (optional)
# 		Number of Monte Carlo random orderings.
# 	progress_bar: Bool (optional)
# 		If True then print the progress bar for the computation of random orderings. If False, then nothing is print.
#
# 	Returns
# 	-------
# 		Groups: list
# 		The element in the position i is a list with the indexes of strings assigned to group i, i.e, the strings of
# 		group i.
# 	Measurement: list
# 		The element in position i is a list which represents measurement assigned to the group i. Each of these list is
# 		a list of partial measurements. Each partial measurements is a list of two elements. The first of these elements
# 		encodes the partial measurement assigned and the second the qubits where it should performed.
# 	# operator: SumOp
# 		# Rearrange Pauli strings that obtain the best grouping for the number of Monte Carlo shots provided.
# 	"""
#
# 	PS, weights, labels = Label2Chain(operator)
#
# 	orders = []
# 	order = np.arange(len(PS))  # Default order
# 	args = []
# 	results = []
#
# 	for i in range(n_mc):
# 		if i != 0:
# 			np.random.shuffle(order)  # Shuffle randomly the Pauli strings
# 		orders.append(np.copy(order))
# 		args.append([i, n_groups, [PS[order], AM, WC]])
#
# 	if progress_bar:  # initialize the progress bar, depending if the instance is in a Jupyter notebook or not
# 		if isnotebook():
# 			pbar = tqdm_notebook(total=n_mc, desc='Computing optimal order')
# 		else:
# 			pbar = tqdm(total=n_mc, desc='Computing optimal order', file=sys.stdout, ncols=90,
# 			            bar_format='{l_bar}{bar}{r_bar}')
# 	else:
# 		pbar = None
#
# 	pool = Pool()  # Initialize the multiprocessing
# 	for i, result in enumerate(pool.imap_unordered(unpack_functions, args, chunksize=1), 1):
# 		results.append(result)  # Save the result
# 		if progress_bar:
# 			pbar.update()
#
# 	if progress_bar:
# 		pbar.close()
#
# 	pool.terminate()
# 	results = sort_solution(results)  # Sort the async results
#
# 	number_groups = []
# 	for result in results:
# 		number_groups.append(result)
#
# 	index = np.argmin(number_groups)
#
# 	print('The original order gives {} groups'.format(number_groups[0]))
# 	print('The best order found gives {} groups'.format(number_groups[index]))
#
# 	order = orders[index]
#
# 	# operator = SummedOp([PauliOp(Pauli(labels[order[j]]), weights[order[j]]) for j in range(len(order))])
#
# 	Groups, Measurements = grouping(PS[order], AM, WC)  # Obtain the groups and measurements for the best case
#
# 	# Remap the Pauli strings so the initial order is conserved
# 	for i in range(len(Groups)):
# 		for j in range(len(Groups[i])):
# 			Groups[i][j] = order[Groups[i][j]]
#
# 	return Groups, Measurements  # operator


# %% Renovations for the grouping algorithm after Qiskit hackathon

def empty_dict_factors():
	"""
	Create an empty dictionary with one entry for each possible 2 qubit combination (II, IX, ...) --> F['00']=0,
	F['01']=0, ...
	Each entry will be filled with the number of times that two qubits in N pauli strings have that factor.

	Returns
	-------

	"""
	F = {}
	for i in range(4):
		for j in range(4):
			F[str(i) + str(j)] = 0

	return F


def empty_dict_compatible_measurements():
	"""
	Create an empty dictionary with one entry for each measurement (Bell, Ωx, ...) --> F['0']=0, F['1']=0, ...
	Each entry will be filled with the number of times that each measurement creates a compatible measurement between
	pairs of qubits.

	Returns
	-------

	"""

	CM = {}
	for i in range(0, 10):
		CM[str(i)] = 0

	return CM


def number_of_compatible_measurements(m, F):
	"""
	Given a measurement 'm' and a dictionary of two-qubit factors 'F', calculates the number of compatible measurements
	that can be made with that measurement in those factors.

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
		pair.append(str(Comp[m][i][0]) + str(Comp[m][i][1]))

	temp = F[pair[0]] + F[pair[1]] + F[pair[2]] + F[pair[3]]
	n_compatibilities = (1 / 2) * (temp ** 2 - temp)

	return n_compatibilities


def number_of_compatible_measurements_onequbit(m, F):
	"""
	Given a measurement 'm' and a array of one-qubit factors 'F', calculates the number of compatible measurements
	that can be made with that measurement in those factors

	Parameters
	----------
	m: integer in [1,3]
	F: array

	Returns
	-------
	n:  integer
		Number of compatible measurements with 'm' on the list 'F'.

	"""
	compatiblefactors = np.size(np.argwhere(F == 0)) + np.size(np.argwhere(F == m))
	n_compatibilities = compatiblefactors * (compatiblefactors - 1) / 2

	return n_compatibilities


def compatibilities(PS):  # Algorithm 1 of Fran's notes.
	"""
	Given a set of 'n' Pauli Strings (PS) with 'N' qubits, returns three arrays regarding the compatibilities of the
	measurements.

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

	C = np.diag(-1 * np.ones(N))

	for i in range(N):  # First qubit
		for j in range(i + 1, N):  # Second qubit
			PSij = PS[:, [i, j]]
			CMij = empty_dict_compatible_measurements()

			F = empty_dict_factors()
			for s in range(n):  # Generate factors list
				label = str(PSij[s, 0]) + str(PSij[s, 1])
				F[label] += 1

			for m in range(4, 10):  # Fill compatibility measurements between qubits i and j
				CMij[str(m)] = number_of_compatible_measurements(m, F)

				C[i, j] += CMij[str(m)]
				C[j, i] += CMij[str(m)]

	return C


def transpile_HEEM(G, C, connected=False):
	C = copy.copy(C)
	G = copy.deepcopy(G)

	N = np.shape(C)[0]
	AQ = []
	T = [None] * N

	if not connected:
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
	else:
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


# %% Compatibility matrix test
# for k in range(1000):
#     print(k)
#     N=6
#     PS = np.random.randint(0,4,[4,N])
#     # PS = np.array([[0,1,1,3,3,1],[2,2,1,2,0,1],[1,3,3,3,2,0],[0,1,2,1,1,2]])
#     C, _, _ = compatibilities(PS)

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


def Tcompatiblities(PS, T, G):  # Algorithm 4 of Fran's notes.
	"""
	Given a set of 'n' Pauli Strings (PS) with 'N' qubits, returns three arrays regarding the compatibilities of the
	measurements.

	Parameters
	----------
	PS: ndarray (n, N)
		PS are the Pauli strings. Each row represents a Pauli string, and each column represents a qubit.

	T: list
		T is the map from theoretical qubits to physical qubits. If T[i]=j it means that the i-th theoretical qubit is
		mapped to a the j-th physical qubit.

	G: graph
		G is the connectivity graph of the chip. It vertices represents physical qubits and its edges physical
		connections between them.

	Returns
	-------
	CT:  ndarray (n, N) Symmetric matrix whose diagonal elements are -1.
		The element C[i,j] contains the number of times that
		the qubits i and j of one string are jointly measurable (compatible) with the qubits i and j of other string,
		given that T have been chosen as the map from theoretical qubits to physical qubits.

	CTM: dictionary of 9 entries, one for each measurement
		Number of times that qubits of two pauli strings are compatible with each measurement, given that T have been
		chosen as the map from theoretical qubits to physical qubits. If one measurement has a large value
		in this dictionary it means that many pairs of qubits can be jointly measured in two pauli strings
		with that measurement.

	CTQ: ndarray (N)
		The element CQ[i] contains the number of times that the qubit i can participate in a joint measurement
		with any other qubit though any measurement, given that T have been chosen as the map from theoretical qubits
		to physical qubits . It is the sum of the i_th row/column of the matrix C, excluding
		the -1 of the diagonal plus the number of compatibilities due to one-qubit measurements.
	"""

	n = np.shape(PS)[0]
	N = np.shape(PS)[1]

	C = np.diag(-1 * np.ones(N))
	CM = empty_dict_compatible_measurements()
	CQ = np.zeros(N)

	for i in range(N):  # First qubit
		for j in range(i + 1, N):  # Second qubit
			if [T[i], T[j]] in G.edges():
				PSij = PS[:, [i, j]]
				CMij = empty_dict_compatible_measurements()

				F = empty_dict_factors()
				for s in range(n):  # Generate factors list
					label = str(PSij[s, 0]) + str(PSij[s, 1])
					F[label] += 1

				for m in range(4, 10):  # Fill compatibility measurements between qubits i and j
					CMij[str(m)] = number_of_compatible_measurements(m, F)

					C[i, j] += CMij[str(m)]
					C[j, i] += CMij[str(m)]
					CM[str(m)] += CMij[str(m)]

		CQ[i] = 1 + np.sum(C[i, :])
		for m in range(1, 4):
			CMi = empty_dict_compatible_measurements()
			CMi[str(m)] = number_of_compatible_measurements_onequbit(m, PS[:, i])
			CQ[i] = CQ[i] + CMi[str(m)]
			CM[str(m)] += CMi[str(m)]
	CT = C
	CTM = CM
	CTQ = CQ
	return CT, CTM, CTQ


def MeasurementAssignmentWithOrder(Vi, Vj, Mi, AM, WC, OQ, T):
	"""

	This function regards the assignment of admissible and efficient measurements Mi to the pauli strings Vi and Vj.

	Admissible and efficient (compatible) means that the code tries to add to Mi admissible measurements
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
	T: list
		T is the theo-phys map chosen. T[i]=j means that the i-theoretical qubit is mapped to the j-physical qubit.

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

	# The first loop checks if the current assignment of Mi is compatible with Vj. If not, the program returns Mi and
	# S=False. If Mi is compatible with Vj, the array U will contain the qubits where Mi does not act.

	# N = np.size(Vi)
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
	# an unchanged Mi should be returned.
	UMi = Mi[:]

	# UMi updating loop
	while len(U) != 0:
		for Eps in AM:  # Admissible measurement loop
			if len(U) >= len_meas[Eps]:
				perm = list(permutations(U, len_meas[Eps]))
				for per in perm:  # Possible qubits loop
					if len_meas[Eps] >= 2:
						Tper = (int(T[per[0]]), int(T[per[1]]))
					else:
						Tper = 0
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


def groupingWithOrder(PS, G=None, connected=False):
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
		the theoretical qubits in the chip is connected, instead tries to optimize omega(T) in a greedy way.

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
	not the correspondent physical indexes (i.e., if we have the i-theoretical qubit is mapped to the j-physical qubit
	through T, in other words T[i]=j, we use the index i and not the j to refer to that qubit)
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
		# G.add_nodes_from(range(np.max(temp) + 1))
		G.add_edges_from(temp)

	if len(G.nodes) < len(PS[0]):
		raise Exception('The number of qubits in the device is not high enough. Use a bigger device.')

	PG = PauliGraph(PS)
	SV = sorted(PG.degree, key=lambda x: x[1], reverse=True)
	# n = np.size(PS[:, 0])
	# N = np.size(PS[0, :])

	WC = list(G.edges)  # list of pairs of well connected qubits
	AS = []  # List of strings with assigned measurement
	C = compatibilities(PS)
	T = transpile_HEEM(G, C, connected)
	CT, CM, CQ = Tcompatiblities(PS, T, G)
	CMlist = []
	for i in range(1, 10):
		CMlist.append(CM[str(i)])

	AM = [i[0] for i in sorted(enumerate(CMlist), key=lambda x: x[1], reverse=True)]
	AM = [x + 1 for x in AM]
	OQ = [i[0] for i in sorted(enumerate(list(CQ)), key=lambda x: x[1], reverse=True)]
	Groups = []
	Measurements = []
	for k in range(n):
		i = SV[k][0]  # We run the Pauli strings in a decreasing order of CQ.
		if i not in AS:  # If we enter to this loop, the i string will have its own group.
			Mi = []
			GroupMi = [i]
			AS.append(i)
			for m in range(n):  # We try to make the group of the string i as big as possible
				j = SV[m][0]
				if j not in AS:
					Mi, S = MeasurementAssignmentWithOrder(PS[i, :], PS[j, :], Mi, AM, WC, OQ, T)
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

	# # Ensure the order for entangled measurements
	# for i in range(len(Measurements)):
	# 	for j in range(len(Measurements[i])):
	# 		Measurements[i][j][1] = sorted(Measurements[i][j][1])

	return Groups, Measurements, T


import numpy as np
import matplotlib.pyplot as plt
from qiskit.opflow.primitive_ops import PauliOp
from qiskit.opflow.list_ops import SummedOp
from qiskit.quantum_info import Pauli
from qiskit.opflow.primitive_ops.pauli_sum_op import PauliSumOp
from qiskit.opflow.primitive_ops.tapered_pauli_sum_op import TaperedPauliSumOp
from qiskit_nature.circuit.library import HartreeFock
from qiskit_nature.transformers.second_quantization.electronic import FreezeCoreTransformer
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.mappers.second_quantization import ParityMapper, JordanWignerMapper, BravyiKitaevMapper
from qiskit_nature.converters.second_quantization.qubit_converter import QubitConverter
from qiskit_nature.drivers.second_quantization import PySCFDriver
import os
import copy
import sys
from networkx import is_connected
from datetime import datetime


# from IPython import get_ipython


def HeisenbergHamiltonian(J=1, H=1, num_qubits=2, neighbours=None):
	"""
	Qiskit operator of the 3-D Heisenberg Hamiltonian of a lattice of spins.

	H = - J Σ_j ( X_j X_{j+1} + Y_j Y_{j+1} + Z_j Z_{j+1} ) - H Σ_j Z_j

	Parameters
	----------
	J: float
		Coupling constant.
	H: float
		External magnetic field.
	num_qubits: int.
		Number of qubits.
	neighbours: list(tuples).
		Coupling between the spins.

	Return
	------
	Hamiltonian: SummedOp
		Heisenberg Hamiltonian of the system.
	"""

	if neighbours is None:
		neighbours = [(0, 1)]

	num_op = num_qubits + 3 * len(neighbours)
	Hamiltonian_op_x = []
	Hamiltonian_op_z = []
	Hamiltonian_coeff = num_qubits * [-H] + num_op * [-J]

	for idx in range(num_qubits):
		op_x = np.zeros(num_qubits)
		op_z = np.zeros(num_qubits)
		op_z[idx] = 1
		Hamiltonian_op_x.append(op_x.copy())
		Hamiltonian_op_z.append(op_z.copy())

	for idx in neighbours:
		op_x = np.zeros(num_qubits)
		op_z = np.zeros(num_qubits)
		op_x[idx[0]] = 1
		op_x[idx[1]] = 1
		Hamiltonian_op_x.append(op_x.copy())
		Hamiltonian_op_z.append(op_z.copy())
		op_z[idx[0]] = 1
		op_z[idx[1]] = 1
		Hamiltonian_op_x.append(op_x.copy())
		Hamiltonian_op_z.append(op_z.copy())
		op_x[idx[0]] = 0
		op_x[idx[1]] = 0
		Hamiltonian_op_x.append(op_x.copy())
		Hamiltonian_op_z.append(op_z.copy())

	Hamiltonian = SummedOp(
		[PauliOp(Pauli((Hamiltonian_op_z[j], Hamiltonian_op_x[j])), Hamiltonian_coeff[j]) for j in range(num_op)])

	return Hamiltonian


def RandomHamiltonian(num_qubits=2, num_paulis=4):
	idxs = np.random.randint(2, size=(2, num_qubits, num_paulis))

	Hamiltonian = SummedOp([PauliOp(Pauli((idxs[0, :, j], idxs[1, :, j])), 1) for j in range(num_paulis)])

	return Hamiltonian


def Label2Chain(QubitOp):
	"""
	Transform a string of Pauli matrices into a numpy array.
	'I' --> 0
	'X' --> 1
	'Y' --> 2
	'Z' --> 3

	Parameters
	----------
	QubitOp: SummedOp.

	Returns
	-------
	ops: ndarray(Pauli operators) (number_of_operators, number_of_qubits)
	coeff: list(float)
		Coefficients of each Pauli operator.
	label: list(str)
		Pauli strings
	"""
	Dict = {'I': 0,
	        'X': 1,
	        'Y': 2,
	        'Z': 3}

	if type(QubitOp) == PauliSumOp or type(QubitOp) == TaperedPauliSumOp:
		QubitOp = QubitOp.to_pauli_op()

	label = []
	ops = []
	coeff = []

	for idx in QubitOp.oplist:
		label_temp = idx.primitive.to_label()
		label.append(label_temp)
		ops.append([Dict.get(s) for s in label_temp])
		coeff.append(idx.coeff)

	return np.array(ops), coeff, label


def from_string_to_numbers(pauli_labels):
	"""
	Function that transform a set of pauli string from the str convention ('IXYZ'), to the number convention (0123).

	Parameter
	---------
	pauli_labels: list
		List with the pauli string written as a string.

	Return
	------
	PS: array
		Pauli strings in the number convention.
	"""
	map_str_int = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}  # Map between str and int conventions
	PS = []  # Initialize the Pauli string for int convention

	for label in pauli_labels:  # Iterate over all the given pauli strings
		temp = []
		for letter in label:  # Map each element of a Pauli string
			temp.append(map_str_int[letter])

		PS.append(np.array(temp))
	return np.array(PS)


def get_backend_connectivity(backend):
	"""
	Get the connected qubit of q backend. Has to be a quantum computer.

	Parameters
	----------
	backend: qiskit.backend

	Return
	------
	connexions: (list)
		List with the connected qubits
	"""
	defaults = backend.defaults()
	connexions = [indx for indx in defaults.instruction_schedule_map.qubits_with_instruction('cx')]
	return connexions


def H2(distance=None, freeze_core=True, remove_orbitals=False, initial_state=False, operator=True,
       mapper_type='ParityMapper'):
	"""
	Qiskit operator of the LiH

	Parameters
	----------
	distance: float (optional)
		Distance between atoms of Li and H
	freeze_core: Bool (optional)
		If freeze some cores that do highly impact in the energy
	remove_orbitals: Bool (optional)
		Remove some orbitals that do no impact in the energy
	initial_state: Bool (optional)
		Return the initial Hartree Fock state
	operator: Bool (optional)

	mapper_type: str (optional)
		Type of mapping between orbitals and qubits. Available options:
			'ParityMapper'
			'JordanWignerMapper'
			'BravyiKitaevMapper'

	Returns
	-------
	qubit_op: SummedOp
		Pauli strings and coefficients for the Hamiltonian
	init_state: QuantumCircuit (if initial_state=True)
		Quantum Circuit with the initial state given by Hartree Fock
	"""

	if distance is None:
		distance = .761

	molecule = 'H .0 .0 .0; H .0 .0 ' + str(distance)

	try:
		driver = PySCFDriver(molecule)
	except Exception:
		from qiskit_nature.drivers.second_quantization.pyquanted import PyQuanteDriver
		driver = PyQuanteDriver(molecule)

	# qmolecule = driver.run()

	if remove_orbitals is False:
		Transformer = FreezeCoreTransformer(freeze_core=freeze_core)
	else:
		Transformer = FreezeCoreTransformer(freeze_core=freeze_core, remove_orbitals=remove_orbitals)

	problem = ElectronicStructureProblem(driver, transformers=[Transformer])

	# Generate the second-quantized operators
	second_q_ops = problem.second_q_ops()

	# Hamiltonian
	main_op = second_q_ops[0]

	# Setup the mapper and qubit converter
	if mapper_type == 'ParityMapper':
		mapper = ParityMapper()
	elif mapper_type == 'JordanWignerMapper':
		mapper = JordanWignerMapper()
	elif mapper_type == 'BravyiKitaevMapper':
		mapper = BravyiKitaevMapper()
	else:
		# TODO: Raise an error
		return None

	# The fermionic operators are mapped
	converter = QubitConverter(mapper=mapper, two_qubit_reduction=True)

	if operator is False:
		return converter, problem
	else:
		particle_number = problem.grouped_property_transformed.get_property("ParticleNumber")
		num_particles = (particle_number.num_alpha, particle_number.num_beta)
		num_spin_orbitals = particle_number.num_spin_orbitals
		qubit_op = converter.convert(main_op, num_particles=num_particles)
		if initial_state is False:
			return qubit_op
		else:
			init_state = HartreeFock(num_spin_orbitals, num_particles, converter)
			return qubit_op, init_state


def LiH(distance=None, freeze_core=True, remove_orbitals=None, initial_state=False, operator=True,
        mapper_type='ParityMapper'):
	"""
	Qiskit operator of the LiH

	Parameters
	----------
	distance: float (optional)
		Distance between atoms of Li and H
	freeze_core: Bool (optional)
		If freeze some cores that do highly impact in the energy
	remove_orbitals: Bool (optional)
		Remove some orbitals that do no impact in the energy
	initial_state: Bool (optional)
		Return the initial Hartree Fock state
	operator: Bool (optional)

	mapper_type: str (optional)
		Type of mapping between orbitals and qubits. Available options:
			'ParityMapper'
			'JordanWignerMapper'
			'BravyiKitaevMapper'

	Returns
	-------
	qubit_op: SummedOp
		Pauli strings and coefficients for the Hamiltonian
	init_state: QuantumCircuit (if initial_state=True)
		Quantum Circuit with the initial state given by Hartree Fock
	"""

	if distance is None:
		distance = 1.5474

	if remove_orbitals is None:
		remove_orbitals = [3, 4]

	molecule = 'Li 0.0 0.0 0.0; H 0.0 0.0 ' + str(distance)

	try:
		driver = PySCFDriver(molecule)
	except Exception:
		from qiskit_nature.drivers.second_quantization.pyquanted import PyQuanteDriver
		driver = PyQuanteDriver(molecule)

	# qmolecule = driver.run()

	if remove_orbitals is False:
		Transformer = FreezeCoreTransformer(freeze_core=freeze_core)
	else:
		Transformer = FreezeCoreTransformer(freeze_core=freeze_core, remove_orbitals=remove_orbitals)

	problem = ElectronicStructureProblem(driver, transformers=[Transformer])

	# Generate the second-quantized operators
	second_q_ops = problem.second_q_ops()

	# Hamiltonian
	main_op = second_q_ops[0]

	# Setup the mapper and qubit converter
	if mapper_type == 'ParityMapper':
		mapper = ParityMapper()
	elif mapper_type == 'JordanWignerMapper':
		mapper = JordanWignerMapper()
	elif mapper_type == 'BravyiKitaevMapper':
		mapper = BravyiKitaevMapper()
	else:
		return None

	# The fermionic operators are mapped
	converter = QubitConverter(mapper=mapper, two_qubit_reduction=True)

	if operator is False:
		return converter, problem
	else:
		# # The fermionic operators are mapped to qubit operators
		# num_particles = (problem.grouped_property_transformed.get_property("ParticleNumber").num_alpha,
		#                  problem.grouped_property_transformed.get_property("ParticleNumber").num_beta)
		# num_spin_orbitals = 2 * problem.molecule_data_transformed.num_molecular_orbitals

		particle_number = problem.grouped_property_transformed.get_property("ParticleNumber")
		num_particles = (particle_number.num_alpha, particle_number.num_beta)
		num_spin_orbitals = particle_number.num_spin_orbitals
		qubit_op = converter.convert(main_op, num_particles=num_particles)
		if initial_state is False:
			return qubit_op
		else:
			init_state = HartreeFock(num_spin_orbitals, num_particles, converter)
			return qubit_op, init_state


def BeH2(distance=None, freeze_core=True, remove_orbitals=None, initial_state=False, operator=True,
         mapper_type='ParityMapper'):
	"""
	Qiskit operator of the BeH2

	Parameters
	----------
	distance: float (optional)
		Distance between atoms of Be and H
	freeze_core: Bool (optional)
		If freeze some cores that do highly impact in the energy
	remove_orbitals: Bool (optional)
		Remove some orbitals that do no impact in the energy
	initial_state: Bool (optional)
		Return the initial Hartree Fock state
	operator: Bool (optional)

	mapper_type: str (optional)
		Type of mapping between orbitals and qubits. Available options:
			'ParityMapper'
			'JordanWignerMapper'
			'BravyiKitaevMapper'

	Returns
	-------
	qubit_op: SummedOp
		Pauli strings and coefficients for the Hamiltonian
	init_state: QuantumCircuit (if initial_state=True)
		Quantum Circuit with the initial state given by Hartree Fock
	"""

	if distance is None:
		distance = 1.339

	if remove_orbitals is None:
		remove_orbitals = [3, 6]

	molecule = 'H 0.0 0.0 -' + str(distance) + '; Be 0.0 0.0 0.0; H 0.0 0.0 ' + str(distance)

	try:
		driver = PySCFDriver(molecule)
	except Exception:
		from qiskit_nature.drivers.second_quantization.pyquanted import PyQuanteDriver
		driver = PyQuanteDriver(molecule)

	# qmolecule = driver.run()
	if remove_orbitals is False:
		Transformer = FreezeCoreTransformer(freeze_core=freeze_core)
	else:
		Transformer = FreezeCoreTransformer(freeze_core=freeze_core, remove_orbitals=remove_orbitals)

	problem = ElectronicStructureProblem(driver, transformers=[Transformer])

	# Generate the second-quantized operators
	second_q_ops = problem.second_q_ops()

	# Hamiltonian
	main_op = second_q_ops[0]

	# Setup the mapper and qubit converter
	if mapper_type == 'ParityMapper':
		mapper = ParityMapper()
	elif mapper_type == 'JordanWignerMapper':
		mapper = JordanWignerMapper()
	elif mapper_type == 'BravyiKitaevMapper':
		mapper = BravyiKitaevMapper()
	else:
		return None

	# The fermionic operators are mapped
	converter = QubitConverter(mapper=mapper, two_qubit_reduction=True)

	if operator is False:
		return converter, problem
	else:
		# num_particles = (problem.grouped_property_transformed.get_property("ParticleNumber").num_alpha,
		#                  problem.grouped_property_transformed.get_property("ParticleNumber").num_beta)

		particle_number = problem.grouped_property_transformed.get_property("ParticleNumber")
		num_particles = (particle_number.num_alpha, particle_number.num_beta)
		num_spin_orbitals = particle_number.num_spin_orbitals
		qubit_op = converter.convert(main_op, num_particles=num_particles)
		if initial_state is False:
			return qubit_op
		else:
			init_state = HartreeFock(num_spin_orbitals, num_particles, converter)
			return qubit_op, init_state


def H2O(distance=None, freeze_core=True, remove_orbitals=None, initial_state=False, operator=True,
        mapper_type='ParityMapper'):
	"""
	Qiskit operator of the BeH2

	Parameters
	----------
	distance: float (optional)
		Distance between atoms of Be and H
	freeze_core: Bool (optional)
		If freeze some cores that do highly impact in the energy
	remove_orbitals: Bool (optional)
		Remove some orbitals that do no impact in the energy
	initial_state: Bool (optional)
		Return the initial Hartree Fock state
	operator: Bool (optional)

	mapper_type: str (optional)
		Type of mapping between orbitals and qubits. Available options:
			'ParityMapper'
			'JordanWignerMapper'
			'BravyiKitaevMapper'

	Returns
	-------
	qubit_op: SummedOp
		Pauli strings and coefficients for the Hamiltonian
	init_state: QuantumCircuit (if initial_state=True)
		Quantum Circuit with the initial state given by Hartree Fock
	"""

	if distance is None:
		distance = 0.9573

	if remove_orbitals is None:
		remove_orbitals = [4]

	dist1 = distance * 0.757 / 0.9573
	dist2 = distance * 0.586 / 0.9573
	molecule = 'O 0.0 0.0 0.0; H ' + str(dist1) + ' ' + str(dist2) + ' 0.0; H -' + str(dist1) + ' ' + str(
		dist2) + ' 0.0'

	try:
		driver = PySCFDriver(molecule)
	except Exception:
		from qiskit_nature.drivers.second_quantization.pyquanted import PyQuanteDriver
		driver = PyQuanteDriver(molecule)

	# qmolecule = driver.run()
	if remove_orbitals is False:
		Transformer = FreezeCoreTransformer(freeze_core=freeze_core)
	else:
		Transformer = FreezeCoreTransformer(freeze_core=freeze_core, remove_orbitals=remove_orbitals)

	problem = ElectronicStructureProblem(driver, transformers=[Transformer])

	# Generate the second-quantized operators
	second_q_ops = problem.second_q_ops()

	# Hamiltonian
	main_op = second_q_ops[0]

	# Setup the mapper and qubit converter
	if mapper_type == 'ParityMapper':
		mapper = ParityMapper()
	elif mapper_type == 'JordanWignerMapper':
		mapper = JordanWignerMapper()
	elif mapper_type == 'BravyiKitaevMapper':
		mapper = BravyiKitaevMapper()
	else:
		return None

	# The fermionic operators are mapped
	converter = QubitConverter(mapper=mapper, two_qubit_reduction=True)

	if not operator:
		return converter, problem
	else:
		# num_particles = (problem.grouped_property_transformed.get_property("ParticleNumber").num_alpha,
		#                 problem.grouped_property_transformed.get_property("ParticleNumber").num_beta)

		particle_number = problem.grouped_property_transformed.get_property("ParticleNumber")
		num_particles = (particle_number.num_alpha, particle_number.num_beta)
		num_spin_orbitals = particle_number.num_spin_orbitals
		qubit_op = converter.convert(main_op, num_particles=num_particles)
		if initial_state is False:
			return qubit_op
		else:
			init_state = HartreeFock(num_spin_orbitals, num_particles, converter)
			return qubit_op, init_state


def CH4(distance=None, freeze_core=True, remove_orbitals=None, initial_state=False, operator=True,
        mapper_type='ParityMapper'):
	"""
	Qiskit operator of the CH4

	Parameters
	----------
	distance: float (optional)
		Distance between atoms of Be and H
	freeze_core: Bool (optional)
		If freeze some cores that do highly impact in the energy
	remove_orbitals: Bool (optional)
		Remove some orbitals that do no impact in the energy
	initial_state: Bool (optional)
		Return the initial Hartree Fock state
	operator: Bool (optional)

	mapper_type: str (optional)
		Type of mapping between orbitals and qubits. Available options:
			'ParityMapper'
			'JordanWignerMapper'
			'BravyiKitaevMapper'

	Returns
	-------
	qubit_op: SummedOp
		Pauli strings and coefficients for the Hamiltonian
	init_state: QuantumCircuit (if initial_state=True)
		Quantum Circuit with the initial state given by Hartree Fock
	"""

	if distance is None:
		distance = 0.9573

	if remove_orbitals is None:
		remove_orbitals = [7, 8]

	#          H(1)
	#          O
	#   H(2)      H(3)   H(4)

	theta = 109.5
	r_inf = distance * np.cos(np.deg2rad(theta - 90))
	height_low = distance * np.sin(np.deg2rad(theta - 90))

	H1 = np.array([0, 0, distance])
	H2 = np.array([r_inf, 0, -height_low])
	H3 = np.array([-r_inf * np.cos(np.pi / 3), r_inf * np.sin(np.pi / 3), -height_low])
	H4 = np.array([-r_inf * np.cos(np.pi / 3), -r_inf * np.sin(np.pi / 3), -height_low])

	molecule = 'O 0 0 0; H {}; H {}; H {}; H {}'.format(str(H1)[1:-1], str(H2)[1:-1], str(H3)[1:-1], str(H4)[1:-1])

	try:
		driver = PySCFDriver(molecule)
	except Exception:
		from qiskit_nature.drivers.second_quantization.pyquanted import PyQuanteDriver
		driver = PyQuanteDriver(molecule)

	if remove_orbitals is False:
		Transformer = FreezeCoreTransformer(freeze_core=freeze_core)
	else:
		Transformer = FreezeCoreTransformer(freeze_core=freeze_core, remove_orbitals=remove_orbitals)

	problem = ElectronicStructureProblem(driver, transformers=[Transformer])

	# Generate the second-quantized operators
	second_q_ops = problem.second_q_ops()

	# Hamiltonian
	main_op = second_q_ops[0]

	# Setup the mapper and qubit converter
	if mapper_type == 'ParityMapper':
		mapper = ParityMapper()
	elif mapper_type == 'JordanWignerMapper':
		mapper = JordanWignerMapper()
	elif mapper_type == 'BravyiKitaevMapper':
		mapper = BravyiKitaevMapper()
	else:
		return None

	# The fermionic operators are mapped
	converter = QubitConverter(mapper=mapper, two_qubit_reduction=True)

	if not operator:
		return converter, problem
	else:
		particle_number = problem.grouped_property_transformed.get_property("ParticleNumber")
		num_particles = (particle_number.num_alpha, particle_number.num_beta)
		num_spin_orbitals = particle_number.num_spin_orbitals
		qubit_op = converter.convert(main_op, num_particles=num_particles)
		if initial_state is False:
			return qubit_op
		else:
			init_state = HartreeFock(num_spin_orbitals, num_particles, converter)
			return qubit_op, init_state


def C2H2(distance=None, freeze_core=True, remove_orbitals=None, initial_state=False, operator=True,
         mapper_type='ParityMapper'):
	"""
	Qiskit operator of the C2H2

	Parameters
	----------
	distance: float (optional)
		Distance between atoms of Be and H
	freeze_core: Bool (optional)
		If freeze some cores that do highly impact in the energy
	remove_orbitals: Bool (optional)
		Remove some orbitals that do no impact in the energy
	initial_state: Bool (optional)
		Return the initial Hartree Fock state
	operator: Bool (optional)

	mapper_type: str (optional)
		Type of mapping between orbitals and qubits. Available options:
			'ParityMapper'
			'JordanWignerMapper'
			'BravyiKitaevMapper'

	Returns
	-------
	qubit_op: SummedOp
		Pauli strings and coefficients for the Hamiltonian
	init_state: QuantumCircuit (if initial_state=True)
		Quantum Circuit with the initial state given by Hartree Fock
	"""

	if distance is None:
		distance = [1.2, 1.06]

	if remove_orbitals is None:
		remove_orbitals = [11]

	#   H(1)  C(1)  C(2)  H(2)

	H1 = str(np.array([0, 0, 0]))[1:-1]
	C1 = str(np.array([0, 0, distance[1]]))[1:-1]
	C2 = str(np.array([0, 0, distance[1] + distance[0]]))[1:-1]
	H2 = str(np.array([0, 0, 2 * distance[1] + distance[0]]))[1:-1]

	molecule = 'H {}; C {}; C {}; H {}'.format(H1, C1, C2, H2)

	try:
		driver = PySCFDriver(molecule)
	except Exception:
		from qiskit_nature.drivers.second_quantization.pyquanted import PyQuanteDriver
		driver = PyQuanteDriver(molecule)

	if remove_orbitals is False:
		Transformer = FreezeCoreTransformer(freeze_core=freeze_core)
	else:
		Transformer = FreezeCoreTransformer(freeze_core=freeze_core, remove_orbitals=remove_orbitals)

	problem = ElectronicStructureProblem(driver, transformers=[Transformer])

	# Generate the second-quantized operators
	second_q_ops = problem.second_q_ops()

	# Hamiltonian
	main_op = second_q_ops[0]

	# Setup the mapper and qubit converter
	if mapper_type == 'ParityMapper':
		mapper = ParityMapper()
	elif mapper_type == 'JordanWignerMapper':
		mapper = JordanWignerMapper()
	elif mapper_type == 'BravyiKitaevMapper':
		mapper = BravyiKitaevMapper()
	else:
		return None

	# The fermionic operators are mapped
	converter = QubitConverter(mapper=mapper, two_qubit_reduction=True)

	if not operator:
		return converter, problem
	else:
		particle_number = problem.grouped_property_transformed.get_property("ParticleNumber")
		num_particles = (particle_number.num_alpha, particle_number.num_beta)
		num_spin_orbitals = particle_number.num_spin_orbitals
		qubit_op = converter.convert(main_op, num_particles=num_particles)
		if initial_state is False:
			return qubit_op
		else:
			init_state = HartreeFock(num_spin_orbitals, num_particles, converter)
			return qubit_op, init_state


def molecules(molecule_name, distance=None, freeze_core=True, remove_orbitals=None, operator=True, initial_state=False,
              mapper_type='ParityMapper'):
	molecule_name = molecule_name.lower()

	if molecule_name == 'h2':
		return H2(distance=distance, freeze_core=freeze_core, remove_orbitals=remove_orbitals,
		          initial_state=initial_state, operator=operator, mapper_type=mapper_type)
	elif molecule_name == 'lih':
		return LiH(distance=distance, freeze_core=freeze_core, remove_orbitals=remove_orbitals,
		           initial_state=initial_state, operator=operator, mapper_type=mapper_type)
	elif molecule_name == 'beh2':
		return BeH2(distance=distance, freeze_core=freeze_core, remove_orbitals=remove_orbitals,
		            initial_state=initial_state, operator=operator, mapper_type=mapper_type)
	elif molecule_name == 'h2o':
		return H2O(distance=distance, freeze_core=freeze_core, remove_orbitals=remove_orbitals,
		           initial_state=initial_state, operator=operator, mapper_type=mapper_type)
	elif molecule_name == 'ch4':
		return CH4(distance=distance, freeze_core=freeze_core, remove_orbitals=remove_orbitals,
		           initial_state=initial_state, operator=operator, mapper_type=mapper_type)
	elif molecule_name == 'c2h2':
		return C2H2(distance=distance, freeze_core=freeze_core, remove_orbitals=remove_orbitals,
		            initial_state=initial_state, operator=operator, mapper_type=mapper_type)
	else:
		raise Exception('The molecule {} is not implemented.'.format(molecule_name))


# DEPRECATED
# def unpack_functions(pack):
# 	"""
# 	Unpack the list where the first element is the index of the async execution, the second index in the function to
# 	run, the third index are the function variables, and the last index (if provided) are the optional arguments.
#
# 	Parameter
# 	---------
# 	pack: list
# 		List with all the data
#
# 	 Return
# 	 ------
# 	 Result of the function
# 	"""
# 	if len(pack) < 4:  # If no optional arguments are provided
# 		pack.append({})
# 	return [pack[0], pack[1](*pack[2], **pack[3])]
#
#
# def sort_solution(data):
# 	"""
# 	Function to sort the data obtained for a parallel computation
#
# 	Parameter
# 	---------
# 	data: list
# 		List in which each entry represents one solution of the parallel computation. The elements are
# 		also list which contains in the first element the index and in the second one the result of the computation.
#
# 	Return
# 	------
# 	List with the data sorted
# 	"""
# 	n = len(data)  # Extract the number of computations done
# 	sorted_sol = [None] * n  # Empty list with the correct number of elements
# 	for i in range(n):  # Iterate over all the elements
# 		index = data[i][0]  # Obtain the index of the result
# 		temp = data[i][1]  # Obtain the result
# 		sorted_sol[index] = temp  # Save the result in the correct element
#
# 	return sorted_sol
#
#
# def isnotebook():
# 	"""
# 	Check if the script is been running in a jupyter notebook instance
#
# 	Return
# 	------
# 	True is the instance is a Jupyter notebook, false in other cases
# 	"""
# 	try:
# 		shell = get_ipython().__class__.__name__
# 		if shell == 'ZMQInteractiveShell':
# 			return True  # Jupyter notebook or qtconsole
# 		elif shell == 'TerminalInteractiveShell':
# 			return False  # Terminal running IPython
# 		else:
# 			return False  # Other type (?)
# 	except NameError:
# 		return False  # Probably standard Python interpreter
#

def permute_indices(diagonal_factors, qubit0, qubit1, n_qubits):
	"""
	Permute the diagonal factors indices by the interchange of qubit_0 <---> qubit_1, maintaining all other indices the
	same.

	Parameters
	----------
	diagonal_factors: ndarray (2 ** n_qubits)
		Diagonal factors for the computation of the expected energy
	qubit0: int
		Index of the first qubit to swap
	qubit1: int
		Index of the second qubit to swap
	n_qubits: int
		Number of qubits in the circuit

	Return
	------
	temp: ndarray (2 ** n_qubits)
		Refactor diagonal factors
	"""
	temp = np.zeros(2 ** n_qubits)

	# Iterate over all the possible outputs of the circuit
	for i in range(len(temp)):
		new = bin(i)[2:]  # New index in binary
		if len(new) < n_qubits:  # Complete with 0's if the index is not of the correct size
			new = ''.join(['0']) * (n_qubits - len(new)) + new
		old = swapPositions(new, qubit0, qubit1)  # Swap the indices of qubit_0 and qubit_1
		temp[int(new, 2)] = diagonal_factors[int(old, 2)]  # Copy the old diagonal factor in the new position

	return temp


def swapPositions(str_variable, pos1, pos2):
	"""
	Swap the position of two indices of a given string.

	Parameters
	----------
	str_variable: str
		String to interchange the indices. The length must be >= max(pos1, pos2)
	pos1: int
		Index of the first element to swap
	pos2: int
		Index of the second element to swap

	Return
	------
	Reformat string with the given swaps
	"""
	list_variable = list(str_variable)
	list_variable[pos1], list_variable[pos2] = list_variable[pos2], list_variable[pos1]
	return ''.join(list_variable)


def swaps(arr, reverse=True):
	"""
	Compute the needed swaps of two elements to sort a given array in descending (or ascending) order.

	Parameters
	----------
	arr: list
		Original array with unsorted numbers [0, 1, ...., len(arr) - 1]. A given element can not appear twice.
	reverse: bool (optional, default=True)
		If reverse=True, sort in descending order, if reverse=False, sort in ascending order

	Returns
	-------
	swaps: ndarray (n, 2)
		Array containing the indices needed to perform a total of n swaps. Each swap corresponds to a given row. The
		swaps must be performed in the correct order, starting from swaps[0], and finish in swaps[-1].
	"""
	# If descending order, reverse the order of the original array
	if reverse:
		arr = arr[::-1]
	n = len(arr)  # Number of elements
	swaps_list = []  # List with the swaps

	# Start the algorithm
	i = 0
	while i < n:
		if arr[i] != i:  # If the element is not in the correct locations
			swaps_list.append(np.array([i, arr[i]]))
			# Interchange the element with the correct element in a given location
			arr[arr[i]], arr[i] = arr[i], arr[arr[i]]
		else:
			i += 1

	swaps_list = np.array(swaps_list)

	# If descending order, transform the indices in each swap. E.g. if N = 3: 0 --> |0 - 3 + 1| = 2, 1 -> 1 and 2 -> 0
	if reverse:
		swaps_list = np.abs(swaps_list - n + 1)

	return swaps_list


def change_order_qubitop(qubit_op, order_paulis, order_qubits):
	"""

	Parameters
	----------
	qubit_op
	order_paulis
	order_qubits

	Returns
	-------

	"""
	paulis, coeffs, labels = Label2Chain(qubit_op)

	operators = []
	for i in range(len(labels)):
		coeff = coeffs[order_paulis[i]]
		label = labels[order_paulis[i]]

		temp = ''
		for j in range(len(order_qubits)):
			temp += label[order_qubits[j]]

		operators.append(PauliOp(Pauli(temp), coeff))

	new_qubit_op = SummedOp(operators)
	return new_qubit_op


def number2SummedOp(labels, coeffs):
	operators = []
	for label, coeff in zip(labels, coeffs):
		operators.append(PauliOp(Pauli(label), coeff))

	new_qubit_op = SummedOp(operators)
	return new_qubit_op


def question_overwrite(name):
	"""
	Make a question if you want to overwrite a certain file that already exists in the directory. There is only two
	possible answers y -> yes (true) or	n -> no (False). If the answer is non of this two the question is repeated until
	a good answer is given.

	Parameter
	----------
	name: (Str)
		Name of the file to overwrite

	Return
	------
	(Bool)
		Answer given by the user
	"""

	temp = input('Do you want to overwrite the file ({})?  [y]/n: '.format(name))  # Ask for an answer by keyword input

	if temp == 'y' or temp == '':
		return True
	elif temp == 'n':
		return False
	else:  # If the answer is not correct
		print('I didn\'t understand your answer.')
		return question_overwrite(name)  # The function will repeat until a correct answer if provided


def save_figure(fig, file_dic):
	fig.savefig(file_dic, bbox_inches="tight",
	            dpi=600)  # Save the figure with the corresponding file direction and the correct extension


def save_data(data, file_dic):
	np.save(file_dic, data)


def save_object(object_save, name, overwrite=None, extension=None, dic=None, prefix='', back=0,
                temp=False, index=0, ask=True, extent=False, silent=False):
	"""
	Save a given figure or date. We must introduce the name with which we want to save the file. If the file already
	exist, then we will be asked if we want to overwrite it. We can also change the	extension used to save the image.
	This function has a protection for not overwriting and save a temp file if the overwriting question is not asked.

	Parameters
	----------
	object_save: (fig or dic)
		Matplotlib figure or dictionary with the data to save
	name: (str)
		String with the name of the file in which save the data
	overwrite: Optional (bool)
		Condition to overwrite or not the data. If a value is not given then the function will ask by default
	extension: Optional (str)
		Extension of the save folder. By default extension='npy' for data, and extension='pdf' for figures
	dic: Optional (str)
		Directory to save the data. By default dic='data/
	prefix: Optional (str)
		Prefix for the folder 'data/'. This must be used if the function is called from a sub folder or is the target
		folder have a complex name. By default prefix=''
	back: Optional (int)
		Number of times to go back from the script that call this function until reach the parent folder of the target
		directory. By default back=0
	temp: (bool)
		If a temp file is saved. Only works with overwrite=None
	index: (int)
		Index to include in the file name is the previous onw is already occupied
	ask: (bool)
		Condition than controls if the question to overwrite is done.
	"""

	if str(type(object_save)) == "<class 'matplotlib.figure.Figure'>":
		object_type = 'figure'
		save_function = save_figure
	else:
		object_type = 'data'
		save_function = save_data

	if dic is None:
		dic = '../data/'

	dic = '../' * back + dic

	if extension is None:
		if object_type == 'figure':
			extension = 'pdf'
		else:
			extension = 'npy'

	file_dic = prefix + dic + name  # Complete the directory of the file including its extension

	if object_type == 'data':
		if index != 0:  # If an index is specified
			file_dic += ' (' + str(index) + ')'  # Include the index in the same

	if overwrite is None:  # If the user does not give a preference for the overwriting
		if os.path.isfile(file_dic + '.' + extension):  # If the file exists in the folder
			if temp:
				save_function(object_save, file_dic + '_temp' + '.' + extension)
			if ask:  # The function will ask if the user want to overwrite the file
				overwrite = question_overwrite(file_dic + '.' + extension)
			else:
				overwrite = False
		else:
			overwrite = True  # If the file does not exist, them the figure will be saved

	if overwrite:  # Depending on the answer of the user
		save_function(object_save, file_dic + '.' + extension)
		if not silent:
			print(object_type, 'saved as', file_dic + '.' + extension)
	elif extent:
		# If the user does not want to over write, a copy is saved
		# The copy will include the typical (1) at the end of the name, if this already exist then (2) without asking.
		# If the file also exist then (3)
		# and so on until an empty number is reached.
		save_object(object_save, name, index=index + 1, ask=False, temp=False)

	if os.path.isfile(file_dic + '_temp.npy'):
		os.remove(file_dic + '_temp.npy')  # If the file is correctly saved the temp file is removed


def n_groups_shuffle(paulis, G, seed, shuffle_paulis=True, shuffle_qubits=True, x=1, n_max=10000, n_delete=0,
                     connected=False, full_output=False, grouping_method='Entangled', order=True, AM=None,
                     minimal_output=False):
	num_qubits = len(paulis[0])
	G_new = copy.deepcopy(G)

	if x < 1 or n_delete > 0:
		edges = list(G_new.edges())

		if x < 1:
			n_delete = int((1 - x) * len(edges))

		indices_delete = np.random.default_rng().choice(len(edges), size=n_delete, replace=False)
		for index in indices_delete:
			G_new.remove_edge(*edges[index])

		if not is_connected(G_new):
			if n_max == 0:
				return np.nan, None, None, G_new
			else:
				return n_groups_shuffle(paulis, G, seed, shuffle_paulis=shuffle_paulis,
				                        shuffle_qubits=shuffle_qubits, x=x, n_max=n_max - 1, n_delete=n_delete)

	np.random.seed(seed)
	order_paulis = np.arange(len(paulis))
	order_qubits = np.arange(num_qubits)
	if shuffle_paulis:
		np.random.shuffle(order_paulis)

	if shuffle_qubits:
		np.random.shuffle(order_qubits)

		temp = copy.deepcopy(paulis)
		for i in range(len(order_qubits)):
			paulis[:, i] = temp[:, order_qubits[i]]

	if grouping_method.lower() == 'entangled':
		if order:
			Groups, Measurements, T = groupingWithOrder(paulis[order_paulis], G_new, connected=connected)

		else:
			Groups, Measurements = grouping(paulis[order_paulis], AM=AM, WC=list(G_new.edges))

	elif grouping_method.lower() == 'tpb':
		_, Groups, Measurements = TPBgrouping(paulis)
	else:
		raise Exception('Invalid grouping method.')

	if minimal_output:
		return len(Groups)

	output = [len(Groups), order_paulis, order_qubits, G_new]
	if full_output:
		output.append(Groups)
		output.append(Measurements)

		try:
			output.append(T)
		except NameError:
			output.append(None)

	return output


def unconnected_measurements(WC, Measurements, T=None):
	if T is None:
		T = np.arange(np.max(WC) + 1)

	counter = 0
	for Groups in Measurements:
		for measurement in Groups:
			if measurement[0] > 3:
				qubit_1_teo, qubit_2_teo = measurement[1]
				qubit_1_phys = T[qubit_1_teo]
				qubit_2_phys = T[qubit_2_teo]
				if (qubit_1_phys, qubit_2_phys) not in WC:
					counter += 1

	return counter


def number_cnots(circuits, qi):
	circuits = qi.transpile(circuits)

	n_cnots = 0
	for circuit in circuits:
		try:
			n_cnots += circuit.count_ops()['cx']
		except KeyError:
			pass

	return n_cnots


def current_time():
	now = datetime.now()
	return now.strftime("%d/%m/%Y, %H:%M:%S")
