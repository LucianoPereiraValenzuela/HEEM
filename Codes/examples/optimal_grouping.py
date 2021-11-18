import sys

sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
from utils import get_backend_connectivity, Label2Chain, n_groups_shuffle, save_object, molecules
from qiskit import IBMQ
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed
from time import time as timer
import warnings
import getopt
import os
from itertools import permutations

# ------------  Default parameters calculation  --------------------
molecule_name = 'BeH2'
N_test = 100
n_jobs = -1
time_save = 0.5  # (min)
total_time = 3  # (min)

file_name = 'optimal_grouping'
name_backend = 'ibmq_montreal'
backend_parallel = 'multiprocessing'

# ---------------------------------------------------------
message_help = file_name + '.py -m <molecule ({})> -j <#JOBS ({})> -t <time save (min) ({})> ' \
                           '-T <time total (min) ({})> -N <# tests ({})>, '.format(molecule_name, n_jobs, time_save,
                                                                                   total_time, N_test)

try:
	argv = sys.argv[1:]
	opts, args = getopt.getopt(argv, "hm:j:t:T:N:")
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
	elif opt == '-t':
		time_save = int(arg)
	elif opt == '-T':
		total_time = int(arg)
	elif opt == '-N':
		N_test = int(arg)
	elif opt == '-c':
		clear_start = bool(arg)

if n_jobs == -1:
	n_jobs = os.cpu_count()

N_test = int(np.ceil(N_test / n_jobs) * n_jobs)

file_name = 'optimal_grouping_' + molecule_name + '_' + name_backend

labels = ['TPB', 'EM', 'HEEM']

# Start calculation
if __name__ == '__main__':
	print('Optimizing grouping for {} in {}'.format(molecule_name, name_backend))
	results = {}
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		IBMQ.load_account()

	provider = IBMQ.get_provider(hub='ibm-q-csic', group='internal', project='iff-csic')
	name_backend = 'ibmq_montreal'
	backend = provider.get_backend(name_backend)
	WC_device = get_backend_connectivity(backend)

	qubit_op, init_state = molecules(molecule_name, initial_state=True)
	num_qubits = qubit_op.num_qubits
	paulis, _, _ = Label2Chain(qubit_op)
	print('There are {} Pauli strings of {} qubits.'.format(len(qubit_op), num_qubits))

	G_device = nx.Graph()
	G_device.add_edges_from(WC_device)

	G_ideal = nx.Graph()
	G_ideal.add_edges_from(list(permutations(list(range(num_qubits)), 2)))

	start = timer()
	pbar = tqdm(range(N_test), desc='TPB: Test grouping', file=sys.stdout, ncols=90)
	Parallel(n_jobs=n_jobs, backend=backend_parallel)(
		delayed(n_groups_shuffle)(paulis, None, None, grouping_method='tpb') for i in pbar)
	total_time_test = timer() - start

	n_min = {}
	for label in labels:
		n_min[label] = np.infty

	average_time = total_time_test / N_test / 60 * 3  # (min)
	batch_size = int(np.ceil(int(time_save / average_time) / n_jobs) * n_jobs)
	N_total = int(np.ceil(int(total_time / average_time) / n_jobs) * n_jobs)
	N_batches = int(np.ceil(N_total / batch_size))
	print('There will be a total of {} shots in {} batches'.format(N_total, N_batches))

	n_until_save = 0
	n_groups = {}
	for i in range(N_batches):
		initial = i * batch_size
		final = initial + batch_size
		if final >= N_total:
			final = N_total

		n_batch = int(final - initial)
		n_until_save += n_batch

		# TPB Grouping
		pbar = tqdm(range(n_batch),
		            desc='TPB: Computing batch {} / {}'.format(i + 1, int(np.ceil(N_total / batch_size))),
		            file=sys.stdout, ncols=90)
		results['TPB'] = Parallel(n_jobs=n_jobs, backend=backend_parallel)(
			delayed(n_groups_shuffle)(paulis, None, None, grouping_method='tpb', full_output=True) for i in pbar)
		n_groups['TPB'] = [result[0] for result in results['TPB']]

		# EM Grouping
		pbar = tqdm(range(n_batch),
		            desc='EM: Computing batch {} / {}'.format(i + 1, int(np.ceil(N_total / batch_size))),
		            file=sys.stdout, ncols=90)
		results['EM'] = Parallel(n_jobs=n_jobs, backend=backend_parallel)(
			delayed(n_groups_shuffle)(paulis, G_ideal, None, grouping_method='entangled', order=False,
			                          full_output=True) for i in pbar)
		n_groups['EM'] = [result[0] for result in results['EM']]

		# HEEM Grouping
		pbar = tqdm(range(n_batch),
		            desc='HEEM: Computing batch {} / {}'.format(i + 1, int(np.ceil(N_total / batch_size))),
		            file=sys.stdout, ncols=90)
		results['HEEM'] = Parallel(n_jobs=n_jobs, backend=backend_parallel)(
			delayed(n_groups_shuffle)(paulis, G_device, None, grouping_method='entangled', order=True, connected=True,
			                          full_output=True)
			for i in pbar)
		n_groups['HEEM'] = [result[0] for result in results['HEEM']]

		for label in labels:
			if np.min(n_groups[label]) < n_min[label]:
				index_min = np.argmin(n_groups[label])
				n_min[label] = n_groups[label][index_min]

				paulis_order, qubit_order = results[label][index_min][1:3]
				Groups, Measurements, T = results[label][index_min][4:]

			# new_qubit_op = change_order_qubitop(qubit_op, paulis_order, qubit_order)
			try:
				previous_data = np.load('../data/' + file_name + '.npy', allow_pickle=True).item()
				n_min_previous = previous_data[label]['n_min']
				total_previous = previous_data[label]['N_total']
			except:
				n_min_previous = np.inf
				total_previous = 0

			if n_min[label] < n_min_previous:
				try:
					total_data = np.load('../data/' + file_name + '.npy', allow_pickle=True).item()
				except:
					total_data = {'qubit_op': qubit_op}

				data = {'Groups': Groups, 'Measurements': Measurements, 'T': T,
				        'N_total': n_until_save + total_previous, 'init_state': init_state,
				        'paulis_order': paulis_order, 'qubit_order': qubit_order, 'n_min': n_min[label]}

				total_data[label] = data

				save_object(total_data, file_name, overwrite=True)
				n_until_save = 0

	for label in labels:
		print('The minimum number of groups for {} is {}'.format(label, n_min[label]))
