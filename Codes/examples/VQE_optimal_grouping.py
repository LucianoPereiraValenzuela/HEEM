import sys

sys.path.append('../')

import numpy as np

from VQE import VQE
from utils import number2SummedOp, save_object, question_overwrite

import os
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings

from qiskit import IBMQ
from qiskit import Aer
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.utils.quantum_instance import QuantumInstance
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SPSA, COBYLA
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from qiskit.circuit.library import EfficientSU2

save = True
clean_start = False


def callback(evals, params, mean, deviation):
	stream = getattr(sys, "stdout")
	print("{}, {}".format(evals, mean), file=stream)
	stream.flush()


def run_VQE(solver, qubitOp, seed, nmax=100):
	np.random.seed(seed)

	solution = False
	while not solution and nmax > 0:
		try:
			solver.compute_minimum_eigenvalue(qubitOp)
			solution = True
		except Exception:
			print('Trying again...')
			nmax -= 1

	if not solution:
		return None

	return solver.energies


os.environ['QISKIT_IN_PARALLEL'] = 'TRUE'

if len(sys.argv) == 1:
	# print('Introduce a molecule to compute the best grouping')
	# molecule = None
	molecule = 'H2O'
else:
	molecule = str(sys.argv[1])

print('Computing optimal VQE for', molecule)

N_backup = os.cpu_count()
N_runs = N_backup * 20

NUM_SHOTS = 2 ** 13  # Number of shots for each circuit
maxiter = 100

with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	IBMQ.load_account()

provider = IBMQ.get_provider(hub='ibm-q-csic', group='internal', project='iff-csic')
name_backend = 'ibmq_montreal'
backend_device = provider.get_backend(name_backend)
backend_sim = Aer.get_backend('aer_simulator')

device = QasmSimulator.from_backend(backend_device)
coupling_map = device.configuration().coupling_map
noise_model = NoiseModel.from_backend(device)
basis_gates = noise_model.basis_gates

qi = QuantumInstance(backend=backend_sim,
					 coupling_map=coupling_map,
					 noise_model=noise_model,
					 basis_gates=basis_gates,
					 shots=NUM_SHOTS,
					 measurement_error_mitigation_cls=CompleteMeasFitter)
# cals_matrix_refresh_period=15)

data = np.load('../data/optimal_grouping_' + molecule + '_' + name_backend + '.npy', allow_pickle=True).item()
qubit_op = number2SummedOp(data['optimal_labels'], data['optimal_coeffs'])

result_exact = NumPyMinimumEigensolver().compute_minimum_eigenvalue(qubit_op)
init_state = data['init_state']

num_qubits = qubit_op.num_qubits
ansatz = init_state.compose(EfficientSU2(num_qubits, ['ry', 'rz'], entanglement='linear', reps=1))
num_var = ansatz.num_parameters
initial_params = [0.1] * num_var

optimizer = SPSA(maxiter=maxiter, last_avg=1)
# optimizer = COBYLA(maxiter=maxiter)

file_name = 'VQE_' + molecule + '_' + name_backend

N_chunks = int(np.ceil(N_runs / N_backup))

if clean_start and save:
	try:
		os.remove('../data/' + file_name + '.npy')
	except Exception:
		pass

pbar = tqdm(total=N_chunks, desc='Computing VQE', file=sys.stdout, ncols=90,
			bar_format='{l_bar}{bar}{r_bar}')
for i in range(0, N_chunks):
	initial = i * N_backup
	final = min(initial + N_backup, N_runs)

	solvers_TPB = [VQE(ansatz, optimizer, initial_params, grouping='TPB',
					   quantum_instance=qi, callback=callback) for _ in range(final - initial)]

	solvers_EM = [VQE(ansatz, optimizer, initial_params, grouping='Entangled',
					  quantum_instance=qi, callback=callback) for _ in range(final - initial)]

	solvers_HEEM = [VQE(ansatz, optimizer, initial_params, grouping='Entangled',
						quantum_instance=qi, connectivity=coupling_map, callback=callback) for _ in
					range(final - initial)]

	energies_TPB = Parallel(n_jobs=-1)(delayed(run_VQE)(solver, qubit_op, None) for solver in solvers_TPB)
	print('TPB Completed')
	energies_EM = Parallel(n_jobs=-1)(delayed(run_VQE)(solver, qubit_op, None) for solver in solvers_EM)
	print('EM Completed')
	energies_HEEM = Parallel(n_jobs=-1)(delayed(run_VQE)(solver, qubit_op, None) for solver in solvers_HEEM)
	print('HEEM Completed')

	if save:

		if i == 0:
			try:
				data = np.load('../data/' + file_name + '.npy', allow_pickle=True).item()

				if len(data['TPB'][0]) < len(energies_TPB[0]):
					os.remove('../data/' + file_name + '.npy')
				else:
					save = question_overwrite(file_name)

			except Exception:
				pass

		try:
			data = np.load('../data/' + file_name + '.npy', allow_pickle=True).item()
			print('Loaded previous data')
			start = 0
		except:
			data = {'TPB': energies_TPB[0], 'EM': energies_EM[0], 'HEEM': energies_HEEM[0]}
			print('New data created')
			start = 1

		for j in range(start, len(energies_HEEM)):
			data['TPB'] = np.vstack([data['TPB'], energies_TPB[j]])
			data['EM'] = np.vstack([data['EM'], energies_EM[j]])
			data['HEEM'] = np.vstack([data['HEEM'], energies_HEEM[j]])

		save_object(data, file_name, overwrite=True)

	print('-' * 90)
	pbar.update()
pbar.close()
