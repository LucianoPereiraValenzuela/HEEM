
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import Aer, transpile, assemble
from qiskit.aqua.components.optimizers import COBYLA


# %%
# TODO: 4 qubits, change var form
def get_var_form(params, n_ry, n_q, Full_ent=False):
	qr = QuantumRegister(n_q, name="q")
	cr = ClassicalRegister(n_q, name='c')
	qc = QuantumCircuit(qr, cr)
	# qc = QuantumCircuit(n_q, n_q)
	for i in range(n_ry):
		# RYs
		for j in range(n_q):
			qc.ry(params[j, i], qr[j])
		# CNOTs
		for j in range(n_q):
			if j < n_q - 1:
				qc.cnot(qr[j], qr[j + 1])
		if Full_ent == True:
			qc.cnot(qr[0], qr[2])
			qc.cnot(qr[1], qr[3])
			qc.cnot(qr[0], qr[3])
	return qc, cr, qr


# TODO: Write this function
def meausure_gates(qr, cr, qc, gates):
	for gate, i in enumerate(gates):
		if gate == 'X':
			qc.x(qr[i])
		elif gate == 'Y':
			qc.y(qr[i])
		else:
			pass
	# qc.x(qr[1])
	qc.measure(qr, cr)
	# qc.measure_all()
	return qc


# TODO: Review
def get_probability_distribution(counts):
	output_distr = [v / NUM_SHOTS for v in counts.values()]
	if len(output_distr) == 1:
		output_distr.append(1 - output_distr[0])
	return output_distr


# TODO: Measure the energy (Pauli graph) (David)
def objective_function(params):
	# Obtain a quantum circuit instance from the parameters
	qc = get_var_form(params)
	# Execute the quantum circuit to obtain the probability distribution associated with the current parameters
	t_qc = transpile(qc, backend)
	qobj = assemble(t_qc, shots=NUM_SHOTS)
	result = backend.run(qobj).result()
	# Obtain the counts for each measured state, and convert those counts into a probability vector
	output_distr = get_probability_distribution(result.get_counts(qc))
	# Calculate the cost as the distance between the output distribution and the target distribution
	cost = sum([np.abs(output_distr[i] - target_distr[i]) for i in range(2)])
	return cost


# %% Test

np.random.seed(999999)
target_distr = np.random.rand(2)
# We now convert the random vector into a valid probability vector
target_distr /= sum(target_distr)

backend = Aer.get_backend("qasm_simulator")
NUM_SHOTS = 10000
n_ry = 2
n_q = 4

params = np.random.rand(n_q, n_ry)

optimizer = COBYLA(maxiter=500, tol=1e-4)

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
