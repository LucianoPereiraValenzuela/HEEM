import sys

sys.path.append('../')

import numpy as np
import networkx as nx
import warnings
from qiskit import IBMQ
import os
import getopt
# from tqdm.auto import tqdm
# from npy_append_array import NpyAppendArray

from utils import molecules, Label2Chain, get_backend_connectivity, flatten_measurements, flatten_groups, \
    flatten_prob2Exp
from GroupingAlgorithm import groupingWithOrder, TPBgrouping
# from HEEM_VQE_Functions import probability2expected_parallel

try:
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, "hm:j:N:")
except getopt.GetoptError:
    print('Wrong argument.')
    sys.exit(2)

molecule_name = 'H2'
for opt, arg in opts:
    if opt == '-m':
        molecule_name = str(arg)

prefix = '../data/groupings/'
try:
    os.mkdir(prefix + molecule_name)
except FileExistsError:
    pass

max_size = 10

try:
    qubit_op = np.load('../data/big_molecules.npy', allow_pickle=True).item()[molecule_name]
    print('Data loaded')
except KeyError:
    print('Computing molecule')
    qubit_op = molecules(molecule_name)

paulis, coeffs, labels = Label2Chain(qubit_op)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    IBMQ.load_account()

provider_CSIC = IBMQ.get_provider(hub='ibm-q-csic', group='internal', project='iff-csic')
name_backend = 'ibmq_montreal'
backend = provider_CSIC.get_backend(name_backend)
WC_device = get_backend_connectivity(backend)

G_device = nx.Graph()
G_device.add_edges_from(WC_device)


def grouping(method):
    if method == 'TPB':
        _, Groups, Measurements = TPBgrouping(paulis, print_progress=True)
    elif method == 'EM':
        Groups, Measurements, layout = groupingWithOrder(paulis, connected=True, print_progress=True)
    elif method == 'HEEM':
        Groups, Measurements, layout = groupingWithOrder(paulis, G_device, connected=True, print_progress=True)

    groups = flatten_groups(Groups)
    np.save(prefix + molecule_name + '/Groups_' + method + '.npy', np.array(groups, dtype='int32'))
    del groups

    measure_index, qubits = flatten_measurements(Measurements)
    np.save(prefix + molecule_name + '/Measurements_' + method + '.npy', np.array(measure_index, dtype='int8'))
    np.save(prefix + molecule_name + '/Measurements_qubits_' + method + '.npy', np.array(qubits, dtype='uint8'))
    del (measure_index, qubits)

    if method != 'TPB':
        np.save(prefix + molecule_name + '/layout_' + method + '.npy', np.array(layout, dtype=object))

    # n_runs = int(np.ceil(len(Groups) / max_size))
    #
    # progress_bar_bool = len(Groups) < max_size
    #
    # file_name_diagonals = molecule_name + '/prob2Exp_diagonals_' + method + '.npy'
    # file_name_factors = molecule_name + '/prob2Exp_factors_' + method + '.npy'
    #
    # pbar = tqdm(total=len(Groups), desc='Computing diagonal factors')
    # for i in range(n_runs):
    #     initial = i * max_size
    #     final = min((i + 1) * max_size, len(Groups))
    #
    #     prob2Exp = probability2expected_parallel(-1, coeffs, labels, Groups[initial:final],
    #                                              Measurements[initial:final],
    #                                              print_progress=progress_bar_bool, shift=False)
    #
    #     diagonals_0 = [temp[0] for temp in prob2Exp]
    #     factors_0 = [temp[1] for temp in prob2Exp]
    #
    #     diagonals_0, factors_0 = flatten_prob2Exp(diagonals_0, factors_0)
    #     diagonals_0 = np.array(diagonals_0, dtype='int8')
    #     factors_0 = np.array(factors_0)
    #
    #     with NpyAppendArray(file_name_diagonals) as npaa:
    #         npaa.append(diagonals_0)
    #
    #     with NpyAppendArray(file_name_factors) as npaa:
    #         npaa.append(factors_0)
    #
    #     del diagonals_0, factors_0, prob2Exp
    #
    #     pbar.update(final - initial)
    # pbar.close()

    # return Groups, Measurements, layout, prob2Exp


methods = ['TPB', 'EM', 'HEEM']
for method in methods:
    print('\n\nComputing ' + method)
    grouping(method)
