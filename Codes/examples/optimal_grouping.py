import sys

sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
from utils import get_backend_connectivity, Label2Chain, n_groups_shuffle, change_order_qubitop, save_object
from utils import H2, LiH, BeH2, H2O
from qiskit import IBMQ
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed
from time import time as timer

N_test = 1000
time_save = 15  # (min)
total_time = 10 * 60  # (min)
N_total = None
# N_total = 2e4


backend_parallel = 'multiprocessing'

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-csic', group='internal', project='iff-csic')
name_backend = 'ibmq_montreal'
backend = provider.get_backend(name_backend)
WC_device = get_backend_connectivity(backend)

if len(sys.argv) == 1:
	print('Introduce a molecule to compute the best grouping')
	molecule = None
	file_name = None

	# molecule = H2O
	# file_name = 'optimal_grouping_' + 'H2O' + '_' + name_backend
	# print('Optimizing grouping for {} in {}'.format('H2O', name_backend))

else:
	molecule_name = str(sys.argv[1])

	file_name = 'optimal_grouping_' + molecule_name + '_' + name_backend
	print('Optimizing grouping for {} in {}'.format(molecule_name, name_backend))

	if molecule_name.lower() == 'h2':
		molecule = H2
	elif molecule_name.lower() == 'lih':
		molecule = LiH
	elif molecule_name.lower() == 'beh2':
		molecule = BeH2
	elif molecule_name.lower() == 'h2o':
		molecule = H2O
	else:
		molecule = None

qubit_op = molecule()
num_qubits = qubit_op.num_qubits
paulis, _, _ = Label2Chain(qubit_op)
print('There are {} Pauli strings of {} qubits.'.format(len(qubit_op), num_qubits))

G_device = nx.Graph()
G_device.add_nodes_from(range(num_qubits))
G_device.add_edges_from(WC_device)

start = timer()
pbar = tqdm(range(N_test), desc='Test grouping', file=sys.stdout, ncols=90)
results = Parallel(n_jobs=-1, backend=backend_parallel)(
	delayed(n_groups_shuffle)(paulis, G_device, None) for i in pbar)
total_time_test = timer() - start

n_groups = [result[0] for result in results]
index_min = np.argmin(n_groups)
n_min, optimal_paulis_order, optimal_qubit_order = results[index_min][:3]

new_qubit_op = change_order_qubitop(qubit_op, optimal_paulis_order, optimal_qubit_order)
paulis, coeffs, labels = Label2Chain(new_qubit_op)

average_time = total_time_test / N_test / 60  # (min)
batch_size = int(time_save / average_time)
if N_total is None:
	N_total = int(total_time / average_time)
print('There will be a total of {} shots in {} batches'.format(N_total, int(np.ceil(N_total / batch_size))))

n_until_save = 0
for i in range(0, int(np.ceil(N_total / batch_size))):
	initial = i * batch_size
	final = initial + batch_size
	if final >= N_total:
		final = N_total

	n_batch = int(final - initial)
	n_until_save += n_batch

	pbar = tqdm(range(n_batch), desc='Computing batch {} / {}'.format(i + 1, int(np.ceil(N_total / batch_size))),
	            file=sys.stdout, ncols=90)
	results = Parallel(n_jobs=-1, backend=backend_parallel)(
		delayed(n_groups_shuffle)(paulis, G_device, None) for i in pbar)

	n_groups = [result[0] for result in results]
	if np.min(n_groups) < n_min:
		index_min = np.argmin(n_groups)
		n_min, optimal_paulis_order, optimal_qubit_order = results[index_min][:3]

		new_qubit_op = change_order_qubitop(qubit_op, optimal_paulis_order, optimal_qubit_order)
		paulis, coeffs, labels = Label2Chain(new_qubit_op)

	try:
		previous_data = np.load('../data/' + file_name + '.npy', allow_pickle=True).item()
		n_min_previous = previous_data['n_min']
		total_previous = previous_data['N_total']
	except:
		n_min_previous = np.inf
		total_previous = 0

	if n_min < n_min_previous:
		data = {'optimal_coeffs': coeffs, 'optimal_labels': labels, 'N_total': n_until_save + total_previous,
		        'n_min': n_min}
		save_object(data, file_name, overwrite=True)
		n_until_save = 0

print('The minimum number of groups is {}'.format(n_min))
