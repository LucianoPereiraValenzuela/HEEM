import sys

sys.path.append('../')

from GroupingAlgorithm import grouping, groupingWithOrder
from utils import save_object, get_backend_connectivity
import numpy as np
import itertools
import os
from time import time as timer
from qiskit import IBMQ
import warnings

N_paulis_max = 10 ** 4

with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	IBMQ.load_account()

provider = IBMQ.get_provider(hub='ibm-q-csic', group='internal', project='iff-csic')
name_backend = 'ibmq_montreal'
backend = provider.get_backend(name_backend)
WC_device = get_backend_connectivity(backend)

file_name = 'UniversalGroups'

pauli_numbers = np.arange(4)

try:
	os.remove('../data/' + file_name + '.npy')
except:
	pass

for n_qubits in range(2, 10):
	print('Computing universal grouping for {} qubits'.format(n_qubits))

	PS = np.array(list(itertools.product(pauli_numbers, repeat=n_qubits)))
	np.random.shuffle(PS)
	PS = PS[: min(N_paulis_max, len(PS))]

	# G = nx.complete_graph(n_qubits)
	# WC = list(G.edges())

	start = timer()
	Groups1, Meas1 = grouping(PS, [4, 6, 7, 8, 9, 5, 3, 2, 1], WC_device)
	print('    Completed grouping w/o order with {} groups in {:.3f} min'.format(len(Groups1), (timer() - start) / 60))

	start = timer()
	Groups2, Meas2, T2 = groupingWithOrder(PS, WC_device, False)
	print('    Completed grouping w/ order w/o connected with {} groups in {:.3f} min'.format(len(Groups2),
																							  (timer() - start) / 60))

	start = timer()
	Groups3, Meas3, T3 = groupingWithOrder(PS, WC_device, True)
	print('    Completed grouping w/ order w/ connected with {} groups in {:.3f} min'.format(len(Groups3),
																							 (timer() - start) / 60))

	try:
		data = np.load('../data/' + file_name + '.npy', allow_pickle=True).item()
	except:
		data = {'naive': [], 'order_disconnected': [], 'order_connected': []}

	data['naive'].append({'groups': Groups1, 'measurements': Meas1})
	data['order_disconnected'].append({'groups': Groups2, 'measurements': Meas2, 'mapping': T2})
	data['order_connected'].append({'groups': Groups3, 'measurements': Meas3, 'mapping': T3})

	save_object(data, file_name, overwrite=True)
