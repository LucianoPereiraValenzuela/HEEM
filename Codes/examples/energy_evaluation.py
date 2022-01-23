import sys
import warnings

sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import getopt
from time import time

from qiskit import IBMQ
from qiskit import Aer, QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import random_statevector
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.utils.quantum_instance import QuantumInstance
from qiskit.compiler import transpile
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter

from utils import molecules, Label2Chain, get_backend_connectivity, save_object, number_cnots
from HEEM_VQE_Functions import measure_circuit_factor, probability2expected, post_process_results
from GroupingAlgorithm import groupingWithOrder, TPBgrouping

os.environ['QISKIT_IN_PARALLEL'] = 'TRUE'


def compute_energy_exact(seed):
	# state_0 = random_circuit(n_qubits, 1, seed=seed)

	np.random.seed(seed)
	state_0 = QuantumCircuit(n_qubits)
	state_0.initialize(random_statevector(2 ** n_qubits).data)

	# Exact
	_, Groups_TPB, Measurements_TPB = TPBgrouping(paulis)
	prob2Exp_TPB = probability2expected(coeff, labels, Groups_TPB, Measurements_TPB)

	circuits_TPB = []
	for j, measure in enumerate(Measurements_TPB):
		circuit, n_measure_temp = measure_circuit_factor(measure, n_qubits, make_measurements=False)
		circuit = circuit.compose(state_0, front=True)
		circuit.save_statevector()
		circuits_TPB.append(circuit)
	circuits_TPB = transpile(circuits_TPB, simulator)

	result_TPB = simulator.run(circuits_TPB).result()
	exact_energy = 0
	for j in range(len(Measurements_TPB)):
		prob = np.abs(result_TPB.get_statevector(circuits_TPB[j])) ** 2
		exact_energy += np.sum(prob2Exp_TPB[j] @ prob)

	return exact_energy, state_0


def compute_energy(qi, groups, measurements, state_0):
	prob2Exp = probability2expected(coeff, labels, groups, measurements)

	circuits = []
	for measure in measurements:
		circuit, n_measure_temp = measure_circuit_factor(measure, n_qubits)
		circuit = circuit.compose(state_0, front=True)
		circuits.append(circuit)

	counts = qi.execute(circuits).get_counts()
	if len(circuits) == 1:
		counts = [counts]
	probabilities = [post_process_results(counts[j], circuits[j].num_clbits,
	                                      qi.run_config.shots) for j in range(len(counts))]
	energy = 0
	for j in range(len(probabilities)):
		energy += np.sum(prob2Exp[j] @ probabilities[j])

	return energy


def compare_grouping(initial_state, energy_exact, seed):
	np.random.seed(seed)

	energy_EM = compute_energy(qi_EM, Groups_EM, Measurements_EM, initial_state)
	error_EM = np.abs((energy_EM - energy_exact) / energy_exact)

	energy_HEEM = compute_energy(qi_HEEM, Groups_HEEM, Measurements_HEEM, initial_state)
	error_HEEM = np.abs((energy_HEEM - energy_exact) / energy_exact)

	return error_EM, error_HEEM


def cnots(qi, measurements):
	circuits = []
	for measure in measurements:
		circuit, n_measure_temp = measure_circuit_factor(measure, n_qubits)
		circuits.append(circuit)

	return number_cnots(circuits, qi)


# ------------  Default parameters calculation  --------------------
molecule_name = 'LiH'
n_jobs = -1
NUM_SHOTS = 2 ** 15
n_states = 24
threshold = 1e-2

name_backend = 'ibmq_montreal'
backend_parallel = 'loky'
# ---------------------------------------------------------
message_help = 'energy_evaluation.py -m <molecule ({})> -j <#JOBS ({})> -s <# shots ({})> ' \
               '-N <# runs ({})>,'.format(molecule_name,
                                          n_jobs,
                                          NUM_SHOTS,
                                          n_states)

try:
	argv = sys.argv[1:]
	opts, args = getopt.getopt(argv, "hm:j:s:N:")
except getopt.GetoptError:
	print(message_help)
	sys.exit(2)

for opt, arg in opts:
	if opt == '-h':
		print(message_help)
		sys.exit(2)
	elif opt == '-m':
		molecule_name = str(arg)
	elif opt == '-j':
		n_jobs = int(arg)
	elif opt == '-s':
		NUM_SHOTS = int(arg)
	elif opt == '-N':
		n_states = int(arg)

if __name__ == '__main__':
	start = time()
	file_name = 'energy_evaluation_' + molecule_name

	if n_jobs == -1:
		n_jobs = os.cpu_count()

	n_states = int(np.ceil(n_states / n_jobs) * n_jobs)

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		IBMQ.load_account()

	provider = IBMQ.get_provider(hub='ibm-q-csic', group='internal', project='iff-csic')
	backend_device = provider.get_backend(name_backend)
	WC_device = get_backend_connectivity(backend_device)
	simulator = Aer.get_backend('aer_simulator')  # Backend for simulation

	device = QasmSimulator.from_backend(backend_device)
	coupling_map = device.configuration().coupling_map
	noise_model = NoiseModel.from_backend(device)
	basis_gates = noise_model.basis_gates

	print('Comparing energy for', molecule_name)
	try:
		qubit_op = np.load('../data/big_molecules.npy', allow_pickle=True).item()[molecule_name]
		print('Data loaded')
	except KeyError:
		qubit_op = molecules(molecule_name)

	paulis, coeff, labels = Label2Chain(qubit_op)

	n_qubits = qubit_op.num_qubits

	G_device = nx.Graph()
	G_device.add_edges_from(WC_device)

	Groups_EM, Measurements_EM, layout_EM = groupingWithOrder(paulis, connected=True)
	Groups_HEEM, Measurements_HEEM, layout_HEEM = groupingWithOrder(paulis, G_device, connected=True)

	# mitigation = CompleteMeasFitter
	mitigation = None

	# EM
	qi_EM = QuantumInstance(backend=simulator,
	                        coupling_map=coupling_map,
	                        noise_model=noise_model,
	                        basis_gates=basis_gates,
	                        shots=NUM_SHOTS,
	                        measurement_error_mitigation_cls=mitigation)
	qi_EM.set_config(initial_layout=layout_EM)
	n_cnots_EM = cnots(qi_EM, Measurements_EM)

	# HEEM
	qi_HEEM = QuantumInstance(backend=simulator,
	                          coupling_map=coupling_map,
	                          noise_model=noise_model,
	                          basis_gates=basis_gates,
	                          shots=NUM_SHOTS,
	                          measurement_error_mitigation_cls=mitigation)
	qi_HEEM.set_config(initial_layout=layout_HEEM)
	n_cnots_HEEM = cnots(qi_HEEM, Measurements_HEEM)

	pbar = tqdm(range(n_states), desc='Loading exact energies', ncols=90)
	results = Parallel(n_jobs=n_jobs, backend=backend_parallel)(
		delayed(compute_energy_exact)(None) for _ in pbar)

	init_states = []
	exact_energies = []
	for i in range(n_states):
		if np.abs(results[i][0]) > threshold:
			exact_energies.append(results[i][0])
			init_states.append(results[i][1])

	N = len(exact_energies)
	pbar = tqdm(range(N), desc='Comparing groupings', ncols=90)
	results = Parallel(n_jobs=n_jobs, backend=backend_parallel)(
		delayed(compare_grouping)(init_states[i], exact_energies[i], None) for i in pbar)

	errors_EM = np.zeros(N)
	errors_HEEM = np.zeros(N)
	for i in range(N):
		errors_EM[i], errors_HEEM[i] = results[i]

	print('Average error for EM: {:.3f} %'.format(np.average(errors_EM) * 100))
	print('Average error for HEEM: {:.3f} %'.format(np.average(errors_HEEM) * 100))

	parameters = {'molecule': molecule_name, 'device': name_backend, 'NUM_SHOTS': NUM_SHOTS, 'n_states': n_states}
	data = {'EM': errors_EM, 'HEEM': errors_HEEM, 'parameters': parameters, 'n_cnots_EM': n_cnots_EM,
	        'n_cnots_HEEM': n_cnots_HEEM}
	save_object(data, file_name, overwrite=True)

	print('With EM there are a total of {} cnots gates'.format(n_cnots_EM))
	print('With HEEM there are a total of {} cnots gates'.format(n_cnots_HEEM))

	print('The code takes {:.3f} min'.format((time() - start) / 60))
