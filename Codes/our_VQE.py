import numpy as np

np.random.seed(999999)
target_distr = np.random.rand(2)
# We now convert the random vector into a valid probability vector
target_distr /= sum(target_distr)

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister


def get_var_form(params):
	qr = QuantumRegister(1, name="q")
	cr = ClassicalRegister(1, name='c')
	qc = QuantumCircuit(qr, cr)
	qc.u3(params[0], params[1], params[2], qr[0])
	qc.measure(qr, cr[0])
	return qc


from qiskit import Aer, transpile, assemble

backend = Aer.get_backend("qasm_simulator")
NUM_SHOTS = 10000


def get_probability_distribution(counts):
	output_distr = [v / NUM_SHOTS for v in counts.values()]
	if len(output_distr) == 1:
		output_distr.append(1 - output_distr[0])
	return output_distr


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

random_parameters = np.random.rand(3)
cost_random = objective_function(random_parameters)