import sys

sys.path.append('../')

import numpy as np
import networkx as nx
import warnings
from qiskit import IBMQ
import os
import getopt

from utils import molecules, Label2Chain, get_backend_connectivity, flatten_measurements, flatten_groups
from GroupingAlgorithm import groupingWithOrder, TPBgrouping

try:
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, "m:")
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

qubit_op = molecules(molecule_name, load=True)
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


methods = ['TPB', 'EM', 'HEEM']
for method in methods:
    print('\n\nComputing ' + method)
    grouping(method)
