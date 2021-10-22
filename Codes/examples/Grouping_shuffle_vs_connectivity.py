import sys

sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
from GroupingAlgorithm import groupingWithOrder
from utils import Label2Chain, H2O, save_object
from joblib import delayed, Parallel
import networkx as nx
from itertools import permutations
from tqdm.auto import tqdm
import copy

sys.setrecursionlimit(10 ** 9)


def n_groups_shuffle(paulis, G, seed, shuffle_paulis=True, shuffle_qubits=True, x=1, n_max=10000, n_delete=0):
	G_new = copy.deepcopy(G)

	if x < 1 or n_delete > 0:
		edges = list(G_new.edges())

		if x < 1:
			n_delete = int((1 - x) * len(edges))

		indices_delete = np.random.default_rng().choice(len(edges), size=n_delete, replace=False)
		for index in indices_delete:
			G_new.remove_edge(*edges[index])

		if not nx.is_connected(G_new):
			if n_max == 0:
				return np.nan, None, None, G_new
			else:
				return n_groups_shuffle(paulis, G, seed, shuffle_paulis=shuffle_paulis,
				                        shuffle_qubits=shuffle_qubits, x=x, n_max=n_max - 1, n_delete=n_delete)

	np.random.seed(seed)
	order_paulis = np.arange(len(paulis))
	order_qubits = np.arange(num_qubits)
	if shuffle_paulis:
		np.random.shuffle(order_paulis)

	if shuffle_qubits:
		np.random.shuffle(order_qubits)

		temp = copy.deepcopy(paulis)
		for j in range(len(order_qubits)):
			paulis[:, j] = temp[:, order_qubits[j]]

	Groups_HEEM, _, _ = groupingWithOrder(paulis[order_paulis], G_new)
	return len(Groups_HEEM), order_paulis, order_qubits, G_new


qubit_op = H2O()
num_qubits = qubit_op.num_qubits
paulis, _, _ = Label2Chain(qubit_op)
print('There are {} Pauli strings of {} qubits.'.format(len(qubit_op), num_qubits))

WC_ideal = list(permutations(list(range(num_qubits)), 2))

G_ideal = nx.Graph()
G_ideal.add_nodes_from(range(num_qubits))
G_ideal.add_edges_from(WC_ideal)

backend_parallel = 'multiprocessing'

n = num_qubits
k = 2
total_edges = int(np.math.factorial(n) / (np.math.factorial(n - k) * 2))

n_x = 20
N = 3000
x_vec = np.linspace((num_qubits - 1) / total_edges, 1, n_x)

n_groups_list = []
optimal_order_paulis = []
optimal_order_qubits = []
optimal_graph = []

pbar_outer = tqdm(range(n_x), desc='Connectivity', file=sys.stdout, ncols=90,
                  bar_format='{l_bar}{bar}{r_bar}', position=0)

for i in pbar_outer:
	pbar_inner = tqdm(range(N), desc='Shuffling', file=sys.stdout, ncols=90,
	                  bar_format='{l_bar}{bar}{r_bar}', position=1)

	results = Parallel(n_jobs=-1, backend=backend_parallel)(
		delayed(n_groups_shuffle)(paulis, G_ideal, None, x=x_vec[i]) for j in
		pbar_inner)

	print('-' * 90)

	n_groups = [results[i][0] for i in range(N)]

	delete_indices = np.where(np.isnan(n_groups))[0]
	for j, index in enumerate(delete_indices):
		n_groups.pop(index - j)
		results.pop(index - j)

	n_groups_list.append(n_groups)

	index_min = np.argmin(n_groups)
	optimal_order_paulis.append(results[index_min][1])
	optimal_order_qubits.append(results[index_min][2])
	optimal_graph.append(results[index_min][3])

n_std = np.zeros(n_x)
n_avg = np.zeros(n_x)
n_min = np.zeros(n_x)
n_max = np.zeros(n_x)

for i in range(n_x):
	n_std[i] = np.std(n_groups_list[i])
	n_avg[i] = np.mean(n_groups_list[i])
	n_min[i] = np.min(n_groups_list[i])
	n_max[i] = np.max(n_groups_list[i])

fig, ax = plt.subplots()
ax.plot(x_vec, n_avg)
ax.fill_between(x_vec, n_avg - n_std, n_avg + n_std, alpha=0.25)
ax.plot(x_vec, n_min, '--')
ax.plot(x_vec, n_max, '--')
ax.set_xlabel('x')
ax.set_ylabel('# of groups')
ax.set_xlim([x_vec[0], x_vec[-1]])

fig.show()

file = 'H20_grouping_shuffle_ideal_vs_connectivity'

data_save = {'x_vec': x_vec, 'n_groups': n_groups_list, 'optimal_order_paulis': optimal_order_paulis,
             'optimal_order_qubits': optimal_order_qubits, 'optimal_graph': optimal_graph}

save_object(data_save, file, overwrite=True)
