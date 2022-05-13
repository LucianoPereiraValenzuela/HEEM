import sys

sys.path.append('../')

from utils import save_object, get_backend_connectivity, n_groups_shuffle, molecules, Label2Chain
import numpy as np
import getopt
import os
from qiskit import IBMQ
import warnings
from joblib import Parallel, delayed
from tqdm import tqdm
import networkx as nx
from time import time


def groupings(G_device, n):
	n_groups = {}
	times_grouping = {}

	start = time()

	pbar = tqdm(range(n), desc='  Grouping', ncols=90, file=sys.stdout)
	n_groups['naive'] = Parallel(n_jobs=n_jobs, backend=backend_parallel)(
		delayed(n_groups_shuffle)(paulis, G_device, None, shuffle_qubits=False, order=False, minimal_output=True) for _
		in pbar)
	times_grouping['naive'] = time() - start

	start = time()
	pbar = tqdm(range(n), desc='  Grouping + order', ncols=90)
	n_groups['order_disconnected'] = Parallel(n_jobs=n_jobs, backend=backend_parallel)(
		delayed(n_groups_shuffle)(paulis, G_device, None, shuffle_qubits=False, order=True, connected=False,
		                          minimal_output=True) for _ in pbar)
	times_grouping['order_disconnected'] = time() - start

	start = time()
	pbar = tqdm(range(n), desc='  Grouping + order + connected ', ncols=90)
	n_groups['order_connected'] = Parallel(n_jobs=n_jobs, backend=backend_parallel)(
		delayed(n_groups_shuffle)(paulis, G_device, None, shuffle_qubits=False, order=True, connected=True,
		                          minimal_output=True) for _ in pbar)
	times_grouping['order_connected'] = time() - start

	return n_groups, times_grouping


# ------------  Default parameters calculation  --------------------
molecule_name = 'C2H2'
n_jobs = -1
N = 100

backend_parallel = 'multiprocessing'
name_backend_huge = 'ibmq_brooklyn'
name_backend_big = 'ibmq_montreal'
name_backend_small = 'ibmq_guadalupe'

# ---------------------------------------------------------
message_help = 'grouping_devices' + '.py -m <molecule ({})> -j <#JOBS ({})>  -N <# shots ({})>'.format(molecule_name,
                                                                                                       n_jobs, N)

try:
	argv = sys.argv[1:]
	opts, args = getopt.getopt(argv, "hm:j:N:")
except getopt.GetoptError:
	print('Wrong argument.')
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
	elif opt == '-N':
		N = int(arg)

if n_jobs == -1:
	n_jobs = os.cpu_count()

N = int(np.ceil(N / n_jobs) * n_jobs)

file_name = 'grouping_devices_only_paulis_' + molecule_name

# Start calculation
if __name__ == '__main__':
	qubit_op = molecules(molecule_name)
	n_qubits = qubit_op.num_qubits
	paulis, _, _ = Label2Chain(qubit_op)

	print(
		'Grouping the {} molecule, with {} Pauli strings of {} qubits.'.format(molecule_name, len(qubit_op), n_qubits))

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		IBMQ.load_account()

	provider = IBMQ.get_provider(hub='ibm-q-csic', group='internal', project='iff-csic')

	backend = provider.get_backend(name_backend_huge)
	WC_device = get_backend_connectivity(backend)

	G_device_huge = nx.Graph()
	G_device_huge.add_edges_from(WC_device)

	backend = provider.get_backend(name_backend_big)
	WC_device = get_backend_connectivity(backend)

	G_device_big = nx.Graph()
	G_device_big.add_nodes_from(range(n_qubits))
	G_device_big.add_edges_from(WC_device)

	backend = provider.get_backend(name_backend_small)
	WC_device = get_backend_connectivity(backend)

	G_device_small = nx.Graph()
	G_device_small.add_nodes_from(range(n_qubits))
	G_device_small.add_edges_from(WC_device)

	labels = ['naive', 'order_disconnected', 'order_connected']

	start_upper = time()

	print('Grouping with huge device')
	n_groups_huge, times_huge = groupings(G_device_huge, N)

	print('-' * 90)

	print('Grouping with big device')
	n_groups_big, times_big = groupings(G_device_big, N)

	print('-' * 90)

	print('Grouping with small device')
	n_groups_small, times_small = groupings(G_device_small, N)

	print('\tmolecule: {} ({} qubits)'.format(molecule_name, n_qubits))
	print('huge device: {} ({} qubits)'.format(name_backend_huge, len(G_device_huge.nodes)))
	print('big device: {} ({} qubits)'.format(name_backend_big, len(G_device_big.nodes)))
	print('small device: {} ({} qubits)'.format(name_backend_small, len(G_device_small.nodes)))
	print('-' * 45)

	for label in labels:
		print(label)
		print(
			'huge: in average {:.3f} groups, in {:.3f} s'.format(np.mean(n_groups_huge[label]), times_huge[label] / N))
		print('big: in average {:.3f} groups, in {:.3f} s'.format(np.mean(n_groups_big[label]), times_big[label] / N))
		print('small: in average {:.3f} groups, in {:.3f} s'.format(np.mean(n_groups_small[label]),
		                                                            times_small[label] / N))
		print('-' * 45)

	parameters = {'name_huge': name_backend_huge, 'name_big': name_backend_big, 'name_small': name_backend_small,
	              'molecule_name': molecule_name, 'n_qubits_molecule': n_qubits}
	times = {'huge': times_huge, 'big': times_big, 'small': times_small}
	data = {'huge': n_groups_huge, 'big': n_groups_big, 'small': n_groups_small, 'parameters': parameters,
	        'time': times}

	print('The code takes {:.3f} min.'.format((time() - start_upper) / 60))

	save_object(data, file_name, overwrite=True)
