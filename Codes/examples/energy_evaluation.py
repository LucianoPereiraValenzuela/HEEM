import sys
import warnings

sys.path.append('../')
warnings.filterwarnings('ignore')

import numpy as np
from tqdm.auto import tqdm
from os import cpu_count
import getopt

from qiskit import IBMQ
from qiskit.providers.aer import AerSimulator

from utils import molecules, Label2Chain, load_grouping_data
from HEEM_VQE_Functions import measure_circuit_factor, probability2expected_binary, post_process_results_3


def compute_energy(groups, measurements, batch_size, layout=None):
    circuits = [measure_circuit_factor(measurement, n_qubits, measure_all=False) for measurement in measurements]
    shots = NUM_SHOTS // len(groups)

    if batch_size == -1:
        batch_size = len(circuits)
    n_batches = int(np.ceil(len(circuits) / batch_size))

    counts = []
    pbar = tqdm(range(len(circuits)), desc='Simulating circuits')
    for i in range(n_batches):
        initial = i * batch_size
        final = min(initial + batch_size, len(circuits))

        if layout is not None:
            temp = simulator.run(circuits[initial:final], shots=shots,
                                 initial_layout=layout[::-1]).result().get_counts()
        else:
            temp = simulator.run(circuits[initial:final], shots=shots).result().get_counts()

        if type(temp) != list:
            temp = [temp]
        counts += temp

        pbar.update(final - initial)
    pbar.close()

    energy_temp = []
    pbar = tqdm(range(len(circuits)), desc='Computing energy')
    for i in pbar:
        counts_indices, counts_values = post_process_results_3(counts[i])

        diagonals, factors = probability2expected_binary(coeffs, labels, [groups[i]], [measurements[i]], shift=False)
        diagonals = [(~diagonal * 2 - 1).astype('int8') for diagonal in diagonals[0][:, counts_indices]]
        energy_temp.append(np.sum((diagonals * np.array(factors[0])[:, None]) * counts_values[None, :]) / shots)

    return np.sum(energy_temp)


def compute_error(method):
    print('\n')
    print(f'Computing {method}')

    if method == 'TPB':
        Groups, Measurements = load_grouping_data(molecule_name, method)
        Layout = None
    else:
        Groups, Measurements, Layout = load_grouping_data(molecule_name, method)

    print('Number of groups:', len(Groups))

    energy = compute_energy(Groups, Measurements, Batch_size, Layout)
    error = np.abs((energy - energy_exact) / energy_exact)

    print('Relative error: {:.3f} %'.format(error * 100))


try:
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, "m:")
except getopt.GetoptError:
    print('Wrong argument.')
    sys.exit(2)

molecule_name = 'C2H2'
for opt, arg in opts:
    if opt == '-m':
        molecule_name = str(arg)

print('The chosen molecule is ' + molecule_name)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    IBMQ.load_account()

provider_CSIC = IBMQ.get_provider(hub='ibm-q-csic', group='internal', project='iff-csic')

name_backend = 'ibmq_montreal'
backend = provider_CSIC.get_backend(name_backend)

simulator = AerSimulator.from_backend(backend)  # Backend for simulation
simulator.set_options(max_parallel_experiments=0)

NUM_SHOTS = 2 ** 22
print(f'Total number of shots (equally distributed across circuits): {NUM_SHOTS}')
# batch_size = cpu_count() * 4
Batch_size = -1

qubit_op = molecules(molecule_name)
paulis, coeffs, labels = Label2Chain(qubit_op)

n_qubits = qubit_op.num_qubits

print(f'{len(paulis)} total Pauli strings')
print(f'{n_qubits} qubits')

energy_exact = 0
for i in range(len(labels)):
    label = labels[i]
    if 'X' not in label and 'Y' not in label:
        energy_exact += coeffs[i]
print('\nExact energy: {}'.format(energy_exact))

methods = ['TPB', 'EM', 'HEEM']
for method in methods:
    compute_error(method)
