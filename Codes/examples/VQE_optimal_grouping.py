import sys
import warnings
from sympy.utilities.exceptions import SymPyDeprecationWarning
from h5py.h5py_warnings import H5pyDeprecationWarning

warnings.filterwarnings("ignore", category=SymPyDeprecationWarning)
warnings.filterwarnings("ignore", category=H5pyDeprecationWarning)
sys.path.append('../')

with warnings.catch_warnings():
	warnings.simplefilter("ignore")

	import numpy as np
	from VQE import VQE
	from utils import save_object, change_order_qubitop, current_time

	import os
	from tqdm import tqdm
	from joblib import Parallel, delayed
	import getopt

	from qiskit import IBMQ
	from qiskit import Aer
	from qiskit.providers.aer import QasmSimulator
	from qiskit.providers.aer.noise import NoiseModel
	from qiskit.utils.quantum_instance import QuantumInstance
	from qiskit.algorithms import NumPyMinimumEigensolver
	from qiskit.algorithms.optimizers import SPSA
	from qiskit.circuit.library import EfficientSU2


def callback(evals, params, mean, deviation):
	stream = getattr(sys, "stdout")
	print("{}, {}".format(evals, mean), file=stream)
	stream.flush()


def run_VQE(solver, qubitOp, seed):
	np.random.seed(seed)
	solver.compute_minimum_eigenvalue(qubitOp)

	return solver.energies


os.environ['QISKIT_IN_PARALLEL'] = 'TRUE'

# ------------  Default parameters calculation  --------------------
save = True
clean_start = True

molecule_name = 'LiH'
n_jobs = -1
NUM_SHOTS = 2 ** 14
N_runs = 12
maxiter = 300

name_backend = 'ibmq_montreal'
backend_parallel = 'loky'
optimizer = SPSA
noise = True
# ---------------------------------------------------------
message_help = 'Some error in input: VQE_optimal_grouping.py -m <molecule ({})> -j <#JOBS ({})> -s <# shots ({})> ' \
               '-N <# runs ({})>, -i <# iterations ({})>, -c <clear start>'.format(molecule_name, n_jobs, NUM_SHOTS,
                                                                                   N_runs, maxiter, clean_start)

try:
	argv = sys.argv[1:]
	opts, args = getopt.getopt(argv, "hm:j:s:N:i:c:")
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
		N_runs = int(arg)
	elif opt == '-i':
		maxiter = int(arg)
	elif opt == '-c':
		clean_start = bool(arg)  # This is wrong

if noise:
	file_name = 'VQE_' + molecule_name + '_' + name_backend
else:
	file_name = 'VQE_noise_free_' + molecule_name + '_' + name_backend

if n_jobs == -1:
	n_jobs = os.cpu_count()

N_runs = int(np.ceil(N_runs / n_jobs) * n_jobs)

if __name__ == '__main__':
	print('Code starts at:', current_time())
	print('Computing optimal VQE for', molecule_name)

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		IBMQ.load_account()

	provider = IBMQ.get_provider(hub='ibm-q-csic', group='internal', project='iff-csic')
	backend_device = provider.get_backend(name_backend)
	backend_sim = Aer.get_backend('aer_simulator')

	device = QasmSimulator.from_backend(backend_device)
	coupling_map = device.configuration().coupling_map
	noise_model = NoiseModel.from_backend(device)
	basis_gates = noise_model.basis_gates

	if noise:
		qi = QuantumInstance(backend=backend_sim, coupling_map=coupling_map, noise_model=noise_model,
		                     basis_gates=basis_gates, shots=NUM_SHOTS)
	else:
		qi = QuantumInstance(backend=backend_sim)

	data_groups = np.load('../data/optimal_grouping_' + molecule_name + '_' + name_backend + '.npy',
	                      allow_pickle=True).item()

	# I fucked up with the water molecule, so I compute the operator again
	if molecule_name.lower() == 'h20':
		from utils import molecules

		qubit_op = molecules(molecule_name)
	else:
		qubit_op = data_groups['qubit_op']

	result_exact = NumPyMinimumEigensolver().compute_minimum_eigenvalue(qubit_op)

	qubit_op_TPB = change_order_qubitop(qubit_op, data_groups['TPB']['paulis_order'], data_groups['TPB']['qubit_order'])
	qubit_op_EM = change_order_qubitop(qubit_op, data_groups['EM']['paulis_order'], data_groups['EM']['qubit_order'])
	qubit_op_HEEM = change_order_qubitop(qubit_op, data_groups['HEEM']['paulis_order'],
	                                     data_groups['HEEM']['qubit_order'])

	print('The exact energy is: {:.3f}'.format(result_exact.eigenvalue.real))

	init_state = data_groups['TPB']['init_state']
	num_qubits = qubit_op.num_qubits
	ansatz = init_state.compose(EfficientSU2(num_qubits, ['ry', 'rz'], entanglement='linear', reps=1))
	num_var = ansatz.num_parameters
	initial_params = [0.1] * num_var

	optimizer = optimizer(maxiter=maxiter)

	if clean_start and save:
		try:
			os.remove('../data/' + file_name + '.npy')
			print('Data removed')
		except FileNotFoundError:
			pass

	N_batches = int(N_runs / n_jobs)

	pbar = tqdm(total=N_batches, desc='Computing VQE', file=sys.stdout, ncols=90, bar_format='{l_bar}{bar}{r_bar}')
	for i in range(N_batches):
		initial = i * n_jobs
		final = min(initial + n_jobs, N_runs)

		solvers_TPB = [VQE(ansatz, optimizer, initial_params, grouping='TPB', quantum_instance=qi, callback=callback,
		                   Groups=data_groups['TPB']['Groups'], Measurements=data_groups['TPB']['Measurements']) for _
		               in range(final - initial)]

		solvers_EM = [
			VQE(ansatz, optimizer, initial_params, grouping='Entangled', quantum_instance=qi, callback=callback,
			    Groups=data_groups['EM']['Groups'], Measurements=data_groups['EM']['Measurements'],
			    layout=data_groups['EM']['T']) for _ in range(final - initial)]

		solvers_HEEM = [
			VQE(ansatz, optimizer, initial_params, grouping='Entangled', quantum_instance=qi, connectivity=coupling_map,
			    callback=callback, Groups=data_groups['HEEM']['Groups'],
			    Measurements=data_groups['HEEM']['Measurements'], layout=data_groups['HEEM']['T']) for _ in
			range(final - initial)]

		energies_TPB = Parallel(n_jobs=n_jobs, backend=backend_parallel)(
			delayed(run_VQE)(solver, qubit_op_TPB, None) for solver in solvers_TPB)
		print('TPB Completed')
		energies_EM = Parallel(n_jobs=n_jobs, backend=backend_parallel)(
			delayed(run_VQE)(solver, qubit_op_EM, None) for solver in solvers_EM)
		print('EM Completed')
		energies_HEEM = Parallel(n_jobs=n_jobs, backend=backend_parallel)(
			delayed(run_VQE)(solver, qubit_op_HEEM, None) for solver in solvers_HEEM)
		print('HEEM Completed')

		if save:
			try:
				data = np.load('../data/' + file_name + '.npy', allow_pickle=True).item()
				print('Loaded previous data')
				start = 0
			except FileNotFoundError:
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

	print('Code ends at:', current_time())
