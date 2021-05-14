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
