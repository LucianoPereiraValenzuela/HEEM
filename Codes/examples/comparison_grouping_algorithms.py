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

# ------------  Parameters calculation  --------------------
clear_start = True

n_qubits = 4
n_paulis = 10 ** 3
n_shots_shuffle = 10 ** 3
n_shots_paulis = 2

N_paulis_max = 10 ** 4

name_backend = 'ibmq_montreal'
backend_parallel = 'multiprocessing'
file_name = 'comparison_grouping_algorithms'
# ---------------------------------------------------------

if __name__ == '__main__':
	# Start calculation
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

	try:
		os.remove('../data/' + file_name + '.npy')
	except:
		pass

	pbar1 = tqdm(range(n_shots_paulis), desc='Grouping different Pauli chains', file=sys.stdout, ncols=90)
	for _ in range(n_shots_paulis):
		n_groups = {}
		times = {}
		PS = np.array(list(itertools.product(pauli_numbers, repeat=n_qubits)))
		np.random.shuffle(PS)
		PS = PS[: min(N_paulis_max, len(PS))]

		start = time()
		pbar2 = tqdm(range(n_shots_shuffle), desc='   Grouping', file=sys.stdout, ncols=90)
		results = Parallel(n_jobs=-1, backend=backend_parallel)(
			delayed(n_groups_shuffle)(PS, G_device, None, order=False) for _ in pbar2)
		n_groups['naive'] = [result[0] for result in results]
		times['naive'] = time() - start

		start = time()
		pbar3 = tqdm(range(n_shots_shuffle), desc='   Grouping + order', file=sys.stdout, ncols=90)
		results = Parallel(n_jobs=-1, backend=backend_parallel)(
			delayed(n_groups_shuffle)(PS, G_device, None, order=True, connected=False) for _ in pbar3)
		n_groups['order_disconnected'] = [result[0] for result in results]
		times['order_disconnected'] = time() - start

		start = time()
		pbar4 = tqdm(range(n_shots_shuffle), desc='   Grouping + order + connected', file=sys.stdout,
					 ncols=90)
		results = Parallel(n_jobs=-1, backend=backend_parallel)(
			delayed(n_groups_shuffle)(PS, G_device, None, order=True, connected=True) for _ in pbar4)
		n_groups['order_connected'] = [result[0] for result in results]
		times['order_connected'] = time() - start

		try:
			data = np.load('../data/' + file_name + '.npy', allow_pickle=True).item()
		except:
			data = {'naive': np.array([]), 'order_disconnected': np.array([]), 'order_connected': np.array([]),
					'time_naive': 0, 'time_order_disconnected': 0, 'time_order_connected': 0}

		for label in labels:
			data[label] = np.append(data[label], n_groups[label])
			data['time_' + label] += times[label]

		save_object(data, file_name, overwrite=True)

		pbar1.update()
	pbar1.close()
