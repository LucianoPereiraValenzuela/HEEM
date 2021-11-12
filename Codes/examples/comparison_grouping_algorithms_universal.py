import sys

sys.path.append('../')

from utils import save_object, get_backend_connectivity, n_groups_shuffle
import numpy as np
import itertools
import os
from qiskit import IBMQ
import warnings
from joblib import Parallel, delayed
from tqdm import tqdm
import networkx as nx
from time import time

# Load data from arguments in terminal

if len(sys.argv) < 2:
	print('Introduce number of qubits and number of jobs')
	# n_qubits = 4
	# n_jobs = 24
	sys.exit()
else:
	n_qubits = int(sys.argv[1])
	n_jobs = int(sys.argv[2])
	n_shots_shuffle = int(sys.argv[3])
	n_shots_paulis = int(sys.argv[4])

# ------------  Parameters calculation  --------------------
clear_start = False

# n_qubits = 6
# n_jobs = 4

n_paulis = 10 ** 4
n_shots_shuffle = n_jobs * n_shots_shuffle
# n_shots_paulis = n_shots_and_paulis

name_backend = 'ibmq_montreal'
backend_parallel = 'multiprocessing'
file_name = 'comparison_grouping_algorithms'
# ---------------------------------------------------------


# Start calculation
if __name__ == '__main__':
	print('Computing grouping for {} qubits and {} Pauli strings'.format(n_qubits, min(n_paulis, 4 ** n_qubits)))
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		IBMQ.load_account()

	provider = IBMQ.get_provider(hub='ibm-q-csic', group='internal', project='iff-csic')
	backend = provider.get_backend(name_backend)
	WC_device = get_backend_connectivity(backend)

	G_device = nx.Graph()
	G_device.add_nodes_from(range(n_qubits))
	G_device.add_edges_from(WC_device)

	pauli_numbers = np.arange(4)

	labels = ['naive', 'order_disconnected', 'order_connected']

	if clear_start:
		try:
			os.remove('../data/' + file_name + '.npy')
		except OSError:
			pass

	pbar1 = tqdm(range(n_shots_paulis), desc='Grouping Pauli chains', ncols=90, file=sys.stdout)
	for _ in pbar1:
		n_groups = {}
		times = {}
		PS = np.array(list(itertools.product(pauli_numbers, repeat=n_qubits)))
		np.random.shuffle(PS)
		PS = PS[: min(n_paulis, len(PS))]

		start = time()
		pbar = tqdm(range(n_shots_shuffle), desc='   Grouping', ncols=90, file=sys.stdout, leave=False)
		results = Parallel(n_jobs=n_jobs, backend=backend_parallel)(
			delayed(n_groups_shuffle)(PS, G_device, None, order=False) for _ in pbar)
		n_groups['naive'] = [result[0] for result in results]
		times['naive'] = time() - start

		start = time()
		pbar = tqdm(range(n_shots_shuffle), desc='   Grouping + order', ncols=90, file=sys.stdout, leave=False)
		results = Parallel(n_jobs=n_jobs, backend=backend_parallel)(
			delayed(n_groups_shuffle)(PS, G_device, None, order=True, connected=False) for _ in pbar)
		n_groups['order_disconnected'] = [result[0] for result in results]
		times['order_disconnected'] = time() - start

		start = time()
		pbar = tqdm(range(n_shots_shuffle), desc='   Grouping + order + connected', ncols=90, file=sys.stdout,
					leave=False)
		results = Parallel(n_jobs=n_jobs, backend=backend_parallel)(
			delayed(n_groups_shuffle)(PS, G_device, None, order=True, connected=True) for _ in pbar)
		n_groups['order_connected'] = [result[0] for result in results]
		times['order_connected'] = time() - start

		try:
			data = np.load('../data/' + file_name + '.npy', allow_pickle=True).item()
		except OSError:
			data = {}

		if str(n_qubits) not in data:
			parameters = {'n_paulis': n_paulis, 'n_shots_paulis': n_shots_paulis, 'name_backend': name_backend}
			data[str(n_qubits)] = {'times': {'naive': 0, 'order_disconnected': 0, 'order_connected': 0},
								   'parameters': parameters}

		for label in labels:
			data[str(n_qubits)][label] = np.append(data[str(n_qubits)].setdefault(label, []), n_groups[label])
			data[str(n_qubits)]['times'][label] = data[str(n_qubits)]['times'].setdefault(label, 0) + times[label]

		save_object(data, file_name, overwrite=True, silent=True)
