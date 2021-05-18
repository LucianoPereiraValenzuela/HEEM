import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit import Aer, transpile, assemble
from qiskit.circuit.library import EfficientSU2
from qiskit.algorithms.optimizers import COBYLA
from qiskit.aqua.operators.legacy import WeightedPauliOperator
from qiskit.quantum_info import Pauli
from qiskit.aqua.algorithms import NumPyEigensolver

backend = Aer.get_backend("qasm_simulator")  # Backend for simulation
NUM_SHOTS = 10000  # Number of shots for each circuit
n_rep = 1  # Number of repetition for the variational circuit (n_rep + 1: layers of R_y and R_z, n_rep: layers of CNOT)
N = 5  # Number of qubits
num_vars = N * 2 * (n_rep + 1)  # Number of parameters of the variational circuit, one parameters for each R_z and R_y
bound_limits = [(0, 2 * np.pi)] * num_vars  # Lower and upper bound limits for each parameter \theta_i: [0, 2 * pi]
energy_list = []  # List with the energies of each iteration of the VQE algorithm


def create_pauli_qiskit(pauli):
	"""
	Transform a Pauli Weight in form of dictionary into a qiskit operator. To preforms this transformation we construct
	a list, in which each elements is another list of the form [weight, Pauli].

	Parameters
	----------
	pauli (dic{'str': complex}): Each keys is a string representing the Pauli chains, e.g., 'IXXZ', and the values are
							   given by the weight of each Pauli string.

	Returns
	-------
	(WeightedPauliOperator) qiskit operator

	"""
	temp = []  # List in which save the weights and the Pauli strings
	for key in pauli.keys():  # Iterate over all the Pauli strings
		temp.append([pauli[key], Pauli(key)])
	return WeightedPauliOperator(temp)  # Transform the list into a qiskit operator


def get_var_form(params, reps=n_rep):
	"""
	Generate the variational circuit using a efficientSU2 circuit, with linear entanglements between qubits (the cnot
	gates are only applied to neighbours qubits). Each repetition of the circuit is formed by a R_y and R_z to all the
	qubits, and then the cnot gates.

	Parameters
	----------
	param (numpy.array): Array with the values of the angles of each one qubit rotation gate
	reps (Optional, int): Number of repetitions for the variational circuit

	Return
	------
	(qiskit.circuit) of N qubits and N classical registers, with the variational form
	"""
	qc = QuantumCircuit(N, N)  # Create a quantum circuit with classical registers, so later we can concatenate circuits

	# Create variational circuit (without classical register), and substitute the parameters of each gate
	qc_temp = EfficientSU2(N, entanglement='linear', reps=reps).bind_parameters(params)

	return qc.compose(qc_temp)  # Add a  classical register to the circuit


def generate_diagonal_factors(n):
	"""
	Generate the diagonal terms for the tensor product of Z_1 x Z_2 ... Z_n

	Parameters
	----------
	n (int): Number of tensor products

	Return
	------
	factor (numpy.array): Array of int representing diag(Z_1 x Z_2 x ... x Z_n)
	"""
	factors = np.array([1, -1])  # Initialize the diag terms with the diagonal of the Z Pauli matrix
	for _ in range(n - 1):  # Iterate n - 1 times
		factors = np.hstack([factors, factors * -1])  # Append the same array multiplied by -1
	return factors


def get_distribution(counts, n_qubits):
	"""
	Create a ordered list with the probability distribution for the output of a circuit simulation

	Parameters
	----------
	counts (dic): Counts with the keys in binary
	n_qubits (int): Number of measured qubits, this value does not have to match with the number of digits in the keys

	Return
	------
	probabilities (numpy.array): Array with the probabilities ordered such that the prob. for
								 '000...0' -> probabilities[0], and for '111...1' -> probabilities[-1].
	"""
	probabilities = np.zeros(2 ** n_qubits)  # Array of zeros and with the correct size for the measured qubits
	for key in counts.keys():  # Iterate over the measured outputs
		# Transform the key from binary to decimal, and them save the probability
		probabilities[int(key, 2)] = counts[key] / NUM_SHOTS
	return probabilities


def measure_circuit_factory(pauli_string):
	"""
	Create the measurement circuit needed for a given pauli string. All the measurements are done in the z axis, so if
	we need to measure in other axis, we need to rotate the axis. The qubits with a identity matrix in the Pauli string,
	are not measured. The classical registers are ordered in such a way that always measure in the indexes with lower
	weight. For example, if the given Pauli string is 'YXIZI', the correspondence between qubit (q_i) and the classical
	register (c_i) is given by:
	Qubit  -> Classical register
	q_0    -> None
	q_1    -> c_0
	q_2    -> None
	q_3    -> c_1
	q_4    -> c_2

	Parameters
	----------
	pauli_string (int): Pauli string in the form, e.g., 'IXZZI'. The left most index correspond to the last qubit, with
						higher weight in the binary representation.

	Return
	------
	measure_circuit (qiskit.circuit): Circuit with the needed transformation so all the required qubits are measured in
									  the z axis.
	counter (int): Number of measured qubits.
	"""
	pauli_string = pauli_string[::-1]  # Invert the Pauli string, so the first index corresponds to the first qubit
	n = len(pauli_string)  # Total number of qubits in the circuit
	counter = 0  # Variable to save the number of measured qubits, that will be >= n.
	measure_circuit = QuantumCircuit(n, n)  # Create a blank circuit with n quantum and classical registers
	for q in range(n):  # Iterate over the indices of the Pauli string
		# Change basis of measurement
		if pauli_string[q] == 'Z':  # If the measurement is in the z axis, no action is needed
			pass
		elif pauli_string[q] == 'X':  # If the measurement is in the x axis, apply a Hadamard gate
			measure_circuit.h(q)
		elif pauli_string[q] == 'Y':  # If the measurement is in the y axis, go to the x basis by S^+, and later to z
			measure_circuit.sdg(q)
			measure_circuit.h(q)
		elif pauli_string[q] == 'I':  # If there is no measurement, continue with the next qubit
			continue
		else:  # If the Pauli string has a type, return None
			print('Wrong Pauli measure name')
			return None

		measure_circuit.measure(q, counter)  # Measure the qubit in the smallest free index
		counter += 1  # Increase by one the number of measured qubits
	return measure_circuit, counter


def objective_function(params):
	"""
	Compute the energy of the variational circuit with a given set of parameters for the rotation angles

	Parameters
	----------
	params (numpy.array): Array with the angles of the rotation gates for the variation circuit
	"""

	energy = 0  # Initialize the energy in 0

	for key in pauli_weights.keys():  # Iterate over the pauli string in the Pauli weight

		qc = get_var_form(params)  # Obtain a quantum circuit instance from the parameters
		mc, n_measures = measure_circuit_factory(key)  # Obtain the measurement circuit from the Pauli string
		qc_final = qc.compose(mc)  # Combine both circuits

		# Execute the quantum circuit to obtain the probability distribution associated with the current parameters
		t_qc = transpile(qc_final, backend)
		q_obj = assemble(t_qc, shots=NUM_SHOTS)
		counts = backend.run(q_obj).result().get_counts(qc_final)

		distribution = get_distribution(counts, n_measures)  # Convert the measured counts into a probability vector

		# Weight each probability by the diagonal factor, them sum all of them, and later multiply by the Pauli Weight
		energy += np.sum(distribution * generate_diagonal_factors(n_measures)) * pauli_weights[key]

	energy_list.append(energy)  # Append the new computed energy

	# Print the iteration of the VQE and the energy
	print('Iteration {}, Energy: {:.3f}'.format(len(energy_list), energy))

	return energy


# ------------  Heisenberg model  ------------
h = -1  # External magnetic field
J_x = 1
J_y = 2
J_z = 3
J_s = [J_x, J_y, J_z]
paulis = ['X', 'Y', 'Z']  # Name of Pauli matrices
pauli_weights = {}  # Dic for the Pauli Weights of the model

# External magnetic field
for i in range(N):
	index = ['I'] * N  # List of the form ['I', 'I', ...., 'I']
	index[i] = 'Z'  # Change the i index for Z, e.g., ['I', ..., 'Z', ..., 'I']
	pauli_weights[''.join(index)] = -1 / 2 * h  # Join the letters in the list and create new key in the dic

# Nearest neighbors interaction
for alpha in range(3):  # Iterate over three directions alpha: [X, Y, Z]
	for i in range(N):  # Iterate over all spins
		index = ['I'] * N
		index[i] = paulis[alpha]
		index[(i + 1) % N] = paulis[alpha]  # Interaction of the same type (alpha_i, alpha_{i+1}), with PBC
		pauli_weights[''.join(index)] = -1 / 2 * J_s[alpha]

# Solve the system exactly
pauli_weights_qiskit = create_pauli_qiskit(pauli_weights)
result = NumPyEigensolver(pauli_weights_qiskit).run()
exact_energy = np.real(result.eigenvalues)[0]
print('The exact energy is {:.3f}'.format(exact_energy))

optimizer = COBYLA(maxiter=500, tol=1e-4, disp=True)  # Set up classical optimizer for VQE

initial_params = np.random.rand(num_vars)  # Initialize the parameters for the variational circuit with random values

# Optimize the variational circuit to obtain the minimum energy of a given Hamiltonian
ret = optimizer.optimize(num_vars=num_vars, objective_function=objective_function, initial_point=initial_params,
                         variable_bounds=bound_limits)

# Plot the results
plt.figure()
plt.plot(energy_list, 'ro', label='VQE')
plt.hlines(exact_energy, 0, len(energy_list), linestyles='--', colors='g', label='Exact')
plt.xlim(1, len(energy_list) - 1)
plt.ylabel('Energy')
plt.xlabel('Iteration')
