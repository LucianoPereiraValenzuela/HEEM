import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit import Aer, transpile, assemble
from qiskit.circuit.library import EfficientSU2
from qiskit.algorithms.optimizers import COBYLA
from qiskit.aqua.operators.legacy import WeightedPauliOperator
from qiskit.quantum_info import Pauli
from qiskit.aqua.algorithms import NumPyEigensolver

backend = Aer.get_backend("qasm_simulator")
NUM_SHOTS = 10000
n_rep = 1
N = 4
num_vars = N * 2 * (n_rep + 1)
bound_limits = [(0, 2 * np.pi)] * num_vars
energy_list = []


def creatre_Pauli_qiskit(pauli):
	temp = []
	for key in pauli.keys():
		temp.append([pauli[key], Pauli(key)])
	return WeightedPauliOperator(temp)


def get_var_form(params, reps=n_rep):
	qc = QuantumCircuit(N, N)
	qc_temp = EfficientSU2(N, entanglement='linear', reps=reps).bind_parameters(params)
	return qc.compose(qc_temp)


# Heisenberg model
h = -1
J_x = 1
J_y = 2
J_z = 3
J_s = [J_x, J_y, J_z]
paulis = ['X', 'Y', 'Z']
pauli_weights = {}

# External magnetic field
for i in range(N):
	index = ['I'] * N
	index[i] = 'Z'
	pauli_weights[''.join(index)] = -1 / 2 * h

# Nearest neighbors interaction
for alpha in range(3):
	for i in range(N):
		index = ['I'] * N
		index[i] = paulis[alpha]
		index[(i + 1) % N] = paulis[alpha]
		pauli_weights[''.join(index)] = -1 / 2 * J_s[alpha]

pauli_weights_qiskit = creatre_Pauli_qiskit(pauli_weights)
result = NumPyEigensolver(pauli_weights_qiskit).run()
exact_energy = np.real(result.eigenvalues)[0]
print('The exact energy is {:.3f}'.format(exact_energy))


# Generate diagonal terms for Z x Z x ....  x Z^(n)
def generate_factors(n):
	factors = np.array([1, -1])
	for i in range(n - 1):
		factors = np.hstack([factors, factors * -1])
	return factors


def get_distribution(counts, n_qubits):
	probabilities = np.zeros(2 ** n_qubits)
	for key in counts.keys():
		probabilities[int(key, 2)] = counts[key] / NUM_SHOTS

	return probabilities


def measure_circuit_factory(pauli_pair):
	pauli_pair = pauli_pair[::-1]
	n = len(pauli_pair)  # Number of qubits
	counter = 0
	measure_circuit = QuantumCircuit(n, n)
	for i in range(n):
		# Change basis of measurement
		if pauli_pair[i] == 'Z':
			pass
		elif pauli_pair[i] == 'X':
			measure_circuit.h(i)
		elif pauli_pair[i] == 'Y':
			measure_circuit.sdg(i)
			measure_circuit.h(i)
		elif pauli_pair[i] == 'I':
			continue
		else:
			print('Wrong Pauli measure name')
			return None

		measure_circuit.measure(i, counter)
		counter += 1
	return measure_circuit, counter


def objective_function(params):
	energy = 0
	for key in pauli_weights.keys():
		# Obtain a quantum circuit instance from the parameters
		qc = get_var_form(params)
		mc, n_measures = measure_circuit_factory(key)
		qc_final = qc.compose(mc)

		# Execute the quantum circuit to obtain the probability distribution associated with the current parameters
		t_qc = transpile(qc_final, backend)
		qobj = assemble(t_qc, shots=NUM_SHOTS)
		counts = backend.run(qobj).result().get_counts(qc_final)

		# Obtain the counts for each measured state, and convert those counts into a probability vector
		distribution = get_distribution(counts, n_measures)
		energy += np.sum(distribution * generate_factors(n_measures)) * pauli_weights[key]

	energy_list.append(energy)
	print('Iteration {}, Energy: {:.3f}'.format(len(energy_list), energy))

	return energy


optimizer = COBYLA(maxiter=500, tol=1e-4, disp=True)

params = np.random.rand(num_vars)
ret = optimizer.optimize(num_vars=num_vars, objective_function=objective_function, initial_point=params,
						 variable_bounds=bound_limits)

print("Parameters Found:", ret[0])
print("Number of iterations:", ret[-1])

plt.figure()
plt.plot(energy_list, 'ro', label='VQE')
plt.hlines(exact_energy, 0, len(energy_list), linestyles='--', colors='g', label='Exact')
plt.xlim(1, len(energy_list) - 1)
plt.ylabel('Energy')
plt.xlabel('Iteration')
