import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import Aer, transpile, assemble
from qiskit.aqua.components.optimizers import COBYLA

np.random.seed(999999)
target_distr = np.random.rand(2)
# We now convert the random vector into a valid probability vector
target_distr /= sum(target_distr)

backend = Aer.get_backend("qasm_simulator")
NUM_SHOTS = 10000


# TODO: 4 qubits, change var form
def get_var_form(params):
	qr = QuantumRegister(2, name="q")
	cr = ClassicalRegister(2, name='c')
	qc = QuantumCircuit(qr, cr)
	qc.h(qr[0])
	qc.u3(params[0], params[1], params[2], qr[0])
	# qc.measure(qr, cr[0])  # Delete this line
	return qc, cr, qr

# Heisenberg model
N = 4
h = 1
J_x = 2
J_y = 3
J_z = 4
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


# TODO: Measure the energy (Pauli graph) (David)
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

	return energy


optimizer = COBYLA(maxiter=500, tol=1e-4)

params = np.random.rand(3)
ret = optimizer.optimize(num_vars=3, objective_function=objective_function, initial_point=params)

qc = get_var_form(ret[0])
t_qc = transpile(qc, backend)
qobj = assemble(t_qc, shots=NUM_SHOTS)
counts = backend.run(qobj).result().get_counts(qc)
output_distr = get_probability_distribution(counts)

print("Target Distribution:", target_distr)
print("Obtained Distribution:", output_distr)
print("Output Error (Manhattan Distance):", ret[1])
print("Parameters Found:", ret[0])
print("Number of iterations:", ret[-1])
