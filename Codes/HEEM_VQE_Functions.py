import os

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from utils import permute_indices, swaps, tqdm_joblib
from tqdm.auto import tqdm
from joblib import Parallel, delayed

# Maps for the order of measurements in each basis
# maps = [np.array(['XX', 'YY', 'ZZ', 'II']),  # Bell
#         np.array(['XX', 'YZ', 'ZY', 'II']),  # Omega xx
#         np.array(['YY', 'XZ', 'ZX', 'II']),  # Omega yy
#         np.array(['ZZ', 'XY', 'YX', 'II']),  # Omega zz
#         np.array(['XY', 'YZ', 'ZX', 'II']),  # Chi
#         np.array(['YX', 'ZY', 'XZ', 'II'])]  # Chi_prime

maps = [{'XX': 0, 'YY': 1, 'ZZ': 2, 'II': 3},  # Bell
        {'XX': 0, 'YZ': 1, 'ZY': 2, 'II': 3},  # Omega xx
        {'YY': 0, 'XZ': 1, 'ZX': 2, 'II': 3},  # Omega yy
        {'ZZ': 0, 'XY': 1, 'YX': 2, 'II': 3},  # Omega zz
        {'XY': 0, 'YZ': 1, 'ZX': 2, 'II': 3},  # Chi
        {'YX': 0, 'ZY': 1, 'XZ': 2, 'II': 3}]  # Chi_prime

# Factors for expected value of one qubit, and two qubits (in the correct order for each basis)
factors_list = [[np.array([1, -1], dtype='int8'), np.array([1, 1], dtype='int8')],  # One qubit
                [np.array([1, -1, 1, -1], dtype='int8'), np.array([-1, 1, 1, -1], dtype='int8'),
                 np.array([1, 1, -1, -1], dtype='int8'), np.array([1, 1, 1, 1], dtype='int8')],  # Bell
                [np.array([1, -1, -1, 1], dtype='int8'), np.array([-1, 1, -1, 1], dtype='int8'),
                 np.array([-1, -1, 1, 1], dtype='int8'), np.array([1, 1, 1, 1], dtype='int8')],  # Omega xx
                [np.array([1, -1, -1, 1], dtype='int8'), np.array([1, -1, 1, -1], dtype='int8'),
                 np.array([1, 1, -1, -1], dtype='int8'), np.array([1, 1, 1, 1], dtype='int8')],  # Omega yy
                [np.array([1, 1, -1, -1], dtype='int8'), np.array([-1, 1, -1, 1], dtype='int8'),
                 np.array([-1, 1, 1, -1], dtype='int8'), np.array([1, 1, 1, 1], dtype='int8')],  # Omega zz
                [np.array([1, -1, 1, -1], dtype='int8'), np.array([-1, 1, 1, -1], dtype='int8'),
                 np.array([1, 1, -1, -1], dtype='int8'), np.array([1, 1, 1, 1], dtype='int8')],  # Chi
                [np.array([-1, 1, 1, -1], dtype='int8'), np.array([1, 1, -1, -1], dtype='int8'),
                 np.array([1, -1, 1, -1], dtype='int8'), np.array([1, 1, 1, 1], dtype='int8')]]  # Chi_prime

factors_list_bool = []
for factors in factors_list:
    factors_list_bool.append([])
    for factor in factors:
        temp = ~np.array((factor + 1), dtype=bool)

        factors_list_bool[-1].append(temp)


def post_process_results(result, n_q, NUM_SHOTS):
    """
    Transform the counts obtained when running the backend with multiple classical register into an array of
    probabilities.

    Parameters
    ----------
    result: dict('str': int)
        Dictionary in which the keys are the results of a given experiment, with each classical register separated by a
        space. In the values of the dictionaries are saved the number of times a given result is obtained.
    n_q: int
        Number of measured qubits. This values does not have to coincide with the total number of qubits in the circuit.
    NUM_SHOTS: int
        Number of shots for the measurements.

    Return
    ------
    probabilities: (2 ** n_q) array(float)
        Probabilities for each result, where each index correspond to the given result in decimal.
    """

    # Initialize list for the results and the counts
    labels = []
    counts = []
    for key in result.keys():
        labels.append(key.replace(' ', ''))  # Join all the classical register in one single string with no spaces
        counts.append(result[key])

    # Initialize the array of probabilities with all the results in 0
    probabilities = np.zeros(2 ** n_q)
    for j in range(len(labels)):
        # Transform the result from binary to decimal, and save the probability
        probabilities[int(labels[j], 2)] += counts[j] / NUM_SHOTS

    return probabilities


def generate_diagonal_factors(*factors, print_progress=False):
    """
    Generate the diagonal part of the tensor product of matrices that represent the basis in which each qubit (or a pair
    of qubits) has been measured. The tensor product is obtained by computing the Kronecker product. This product for a
    pair of square matrices A (m x m) and B(n x n) is given by:

    A x B = | a_11 B   . . .   a_1m B |
            |   .      .         .    |
            |   .        .       .    |
            |   .         .      .    |
            | a_m1 B   . . .   a_mm B |

    Since in our case all the matrices are diagonals, we don't have to compute the entire matrix, and just obtained its
    diagonal factors as:
    diag(A x B) = (a_11 * diag(B), ..., a_mm * diag(B))
    Parameter
    ---------
    factors: list(int)
        List in which each element is another list with the diagonal part of each matrix

    Return
    ------
    diagonal_factor: array(int)
        Diagonal elements of the tensor product
    """
    factors = factors[::-1]  # Invert the order to ensure the correct Kronecker product
    diagonal_factor = factors[0]  # Initialize the diagonal factors as the diagonal of the first matrix

    pbar = None
    if print_progress:
        pbar = tqdm(total=len(factors) - 1, desc='Computing Kronecker product')

    for i in range(1, len(factors)):  # Run over all the indices, except the first one
        diagonal_factor = np.kron(diagonal_factor, factors[i])

        if print_progress:
            pbar.update()

    if print_progress:
        pbar.close()

    return diagonal_factor.astype('int8')


def generate_diagonal_factors_binary(*factors, print_progress=False):
    """
    Refactor of the function generate_diagonal_factors, but working with binary. Originally we have to compute the
    Kronecker product of 1's and -1's. Obtaining the following truth table (left). However, the same can be obtained
    with boolean variable, under a sum in Z2 (right)

    1 x 1   = 1            0 + 0 = 0
    1 x -1  = -1           0 + 1 = 1
    -1 x 1  = -1           1 + 0 = 1
    -1 x -1 = 1            1 + 1 = 0

    The equivalence is (1 = 0) and (-1 = 1). This equivalence should be restored once the diagonals factors are used to
    compute the expected value of an operator.

    Parameter
    ---------
    factors: list(bool)
        List in which each element is another list with the diagonal part of each matrix

    Return
    ------
    diagonal_factor: array(int)
        Diagonal elements of the tensor product
    """
    factors = list(factors)
    diagonal_factor = factors.pop(0)  # Initialize the diagonal factors as the diagonal of the first matrix

    pbar = None
    if print_progress:
        pbar = tqdm(total=len(factors) - 1, desc='Computing Kronecker product')

    while len(factors) != 0:  # Run over all the indices, except the first one
        v2 = factors.pop(0)
        diagonal_factor = v2[0] ^ diagonal_factor

        n = len(diagonal_factor)
        for j in range(1, len(v2)):
            if v2[j] == v2[0]:
                diagonal_factor = np.hstack([diagonal_factor, diagonal_factor[:n]])
            else:
                diagonal_factor = np.hstack([diagonal_factor, ~diagonal_factor[:n]])

        if print_progress:
            pbar.update()

    if print_progress:
        pbar.close()

    return diagonal_factor


def measure_circuit_factor(measurements, n_qubits, make_measurements=True, measure_all=True):
    """
    This functions differs from the original one in the way to map the measurements to the classical register. In order
    to include Measurement Error Mitigation, so all the qubits are measured in the correct order.

    Function to create the circuit needed to obtain a given group of measurements. Each measurement will be saved in an
    individual classical register. Each measurement coded with an int number. The available measurements are:
    0 -> Identity
    1 -> X
    2 -> Y
    3 -> Z
    4 -> Bell
    5 -> Omega_xx
    6 -> Omega_yy
    7 -> Omega_zz
    8 -> Chi
    9 -> Chi_prime

    To ensure the correct functionality of this function each qubit can only be in one of the measurements, so it's only
    measured once. If a qubit is not provided, then it is not measured.

    Parameters
    ----------
    measurements: list(list(int, list(int)))
        List with all the measurements. Each measured is a list in which the first index in the int encoding the
        measurement, and the second element is another list with the indices of the measured qubits. The convention for
        the indices of the qubits is opposite to the qiskit convention. Here the qubit with higher weight is named as
        q_0.
    n_qubits: int
        Total number of qubits in the circuit. This values does not have to coincide with the number of measured qubits.
    make_measurements: bool (optional)
        If True, include measurement gates at the end of the circuit.

    Returns
    -------
    circuit: quantum circuit
        Circuit (including quantum and classical registers) with the gates needed to perform the measurements.
    """
    # Create the quantum circuit
    qr = QuantumRegister(n_qubits)

    if not measure_all:
        n_measures = 0
        classical_registers = []
        for measure in measurements:
            if measure[0] != 0:  # If the operation is not the identity
                classical_registers.append(ClassicalRegister(len(measure[1])))
                n_measures += len(measure[1])

        circuit = QuantumCircuit(qr, *classical_registers)
    else:
        circuit = QuantumCircuit(qr, ClassicalRegister(n_qubits))

    counter = 0  # Index for the classical register
    for measure in measurements:  # Iterate over all the measurements
        measure_label, qubits = measure  # Extract the index of the measurement and the measured qubits

        qubits = list(np.abs(np.array(qubits) - n_qubits + 1))[::-1]  # Goes to the qiskit convention

        if measure_label == 0:
            # No measurement
            if measure_all:
                pass
            else:
                continue
        elif measure_label == 1:
            # X Circuit
            circuit.h(qubits)
        elif measure_label == 2:
            # Y Circuit
            circuit.sdg(qubits)
            circuit.h(qubits)
        elif measure_label == 3:
            # Z Circuit
            pass
        elif measure_label == 4:
            # Bell Circuit
            circuit.cnot(qubits[0], qubits[1])
            circuit.h(qubits[0])
        elif measure_label == 5:
            # Omega xx Circuit
            circuit.s(qubits)
            circuit.h(qubits[0])
            circuit.cnot(qubits[0], qubits[1])
            circuit.h(qubits[0])
        elif measure_label == 6:
            # Omega yy Circuit
            circuit.h(qubits[0])
            circuit.cnot(qubits[0], qubits[1])
            circuit.h(qubits[0])
        elif measure_label == 7:
            # Omega zz Circuit
            circuit.s(qubits[0])
            circuit.cnot(qubits[0], qubits[1])
            circuit.h(qubits[0])
        elif measure_label == 8:
            # Chi Circuit
            circuit.u2(np.pi / 2, np.pi, qubits[0])
            circuit.cnot(qubits[0], qubits[1])
            circuit.h(qubits[0])
        elif measure_label == 9:
            # Chi_prime Circuit
            circuit.u2(0, np.pi / 2, qubits[0])
            circuit.cnot(qubits[0], qubits[1])
            circuit.h(qubits[0])

        if make_measurements:
            if measure_all:
                circuit.measure(qubits, qubits)
            else:
                circuit.measure(qubits, classical_registers[counter])
        counter += 1

    return circuit


def probability2expected(Pauli_weights, Pauli_labels, Groups, Measurements, shift=True, print_progress=False,
                         progress_diagonal=False):
    """
    Compute the prefactors for computing the expected value of a given Hamiltonian with the probabilities measured based
    on some grouping of measurements.

    Parameters
    ----------
    Pauli_weights: list (complex)
        Weights of each pauli string in the Hamiltonian.
    Pauli_labels: list (str)
        Pauli string in the str convention.
    Groups: list(list(int))
        List in which each element is represented the indices of pauli string that are measured simultaneously.
    Measurements: list(list(int, list(int)))
        List with all the measurements. Each measurement is a list in which the first index is an int encoding the
        measurement, and the second element is another list with the indices of the measured qubits. The convention for
        the indices of the qubits is opposite to the qiskit convention. Here the qubit with higher weight is named as
        q_0.
    shift: (optional) bool
        Change between qubits numbering conventions.

    Return
    ------
    diagonal_factor_all: list(array(int))
        Diagonal factors for all the tensor products. Each element in the list represents the diagonal factors for a
        given group of measurements.
    """
    n_qubits = len(Pauli_labels[0])
    diagonal_factors_all = []  # Initialize the list with all the diagonal factors
    factors = []

    pbar = None
    if print_progress:
        pbar = tqdm(total=len(Groups), desc='Computing diagonal factors')

    for measurements, group in zip(Measurements, Groups):  # Iterate over all the measurements
        # Pauli weights and string in each measurement
        factors.append([Pauli_weights[i] for i in group])
        # pauli_weights = [Pauli_weights[i] for i in group]

        pauli_labels = [Pauli_labels[i] for i in group]
        diagonal_factors_temp = []  # Initialize the diagonal factors for the given group

        for i in range(len(pauli_labels)):  # Iterate over all the measured pauli strings
            diagonal_factors = []  # Initialize the diagonal factors for one pauli string
            for j in range(len(measurements)):  # Iterate over all the measurements in a given group
                index_measure, qubits = measurements[j]
                if 0 < index_measure <= 3:  # Single qubit measurement
                    if pauli_labels[i][qubits[0]] == 'I':  # If the identity is grouped with another measurement
                        diagonal_factors.append(factors_list[0][1])
                    else:  # If a Pauli matrix is measured
                        diagonal_factors.append(factors_list[0][0])
                elif index_measure > 3:  # Entangled qubits measurement
                    # Obtain the tensor product of pauli matrices measured
                    measure_string = pauli_labels[i][qubits[0]] + pauli_labels[i][qubits[1]]
                    map_basis = maps[index_measure - 4]  # Map of tensor products of the entangled basis
                    diagonal_factors.append(factors_list[index_measure - 3][map_basis[measure_string]])
                else:
                    if shift:
                        diagonal_factors.append(factors_list[0][1])

            # Generate the product tensor of all the diagonal factors
            diagonal_factors = generate_diagonal_factors(*diagonal_factors, print_progress=progress_diagonal)
            if shift:
                chain_qubits = []
                for j in range(len(measurements)):  # Iterate over all the measurements in a given group
                    index_measure, qubits = measurements[len(measurements) - 1 - j]
                    qubits = np.abs(np.array(qubits) - n_qubits + 1)

                    for qubit in qubits:
                        chain_qubits.append(qubit)

                permutations = swaps(chain_qubits)
                for permutation in permutations:
                    diagonal_factors = permute_indices(diagonal_factors, permutation[0], permutation[1], n_qubits)

            diagonal_factors_temp.append(
                diagonal_factors)  # diagonal_factors_temp.append(diagonal_factors * pauli_weights[i])

        if print_progress:
            pbar.update()

        diagonal_factors_all.append(np.array(diagonal_factors_temp,
                                             dtype='int8'))  # diagonal_factors_all.append(np.array(diagonal_factors_temp))

    if print_progress:
        pbar.close()

    return [diagonal_factors_all, factors]  # return diagonal_factors_all


def probability2expected_binary(Pauli_weights, Pauli_labels, Groups, Measurements, shift=True, print_progress=False,
                                progress_diagonal=False):
    """
    Compute the prefactors for computing the expected value of a given Hamiltonian with the probabilities measured based
    on some grouping of measurements.

    Parameters
    ----------
    Pauli_weights: list (complex)
        Weights of each pauli string in the Hamiltonian.
    Pauli_labels: list (str)
        Pauli string in the str convention.
    Groups: list(list(int))
        List in which each element is represented the indices of pauli string that are measured simultaneously.
    Measurements: list(list(int, list(int)))
        List with all the measurements. Each measurement is a list in which the first index is an int encoding the
        measurement, and the second element is another list with the indices of the measured qubits. The convention for
        the indices of the qubits is opposite to the qiskit convention. Here the qubit with higher weight is named as
        q_0.
    shift: (optional) bool
        Change between qubits numbering conventions.

    Return
    ------
    diagonal_factor_all: list(array(int))
        Diagonal factors for all the tensor products. Each element in the list represents the diagonal factors for a
        given group of measurements.
    """
    n_qubits = len(Pauli_labels[0])
    diagonal_factors_all = []  # Initialize the list with all the diagonal factors
    factors = []

    pbar = None
    if print_progress:
        pbar = tqdm(total=len(Groups), desc='Computing diagonal factors')

    for measurements, group in zip(Measurements, Groups):  # Iterate over all the measurements
        # Pauli weights and string in each measurement
        factors.append([Pauli_weights[i] for i in group])
        # pauli_weights = [Pauli_weights[i] for i in group]

        pauli_labels = [Pauli_labels[i] for i in group]
        diagonal_factors_temp = []  # Initialize the diagonal factors for the given group

        for i in range(len(pauli_labels)):  # Iterate over all the measured pauli strings
            diagonal_factors = []  # Initialize the diagonal factors for one pauli string
            for j in range(len(measurements)):  # Iterate over all the measurements in a given group
                index_measure, qubits = measurements[j]
                if 0 < index_measure <= 3:  # Single qubit measurement
                    if pauli_labels[i][qubits[0]] == 'I':  # If the identity is grouped with another measurement
                        diagonal_factors.append(factors_list_bool[0][1])
                    else:  # If a Pauli matrix is measured
                        diagonal_factors.append(factors_list_bool[0][0])
                elif index_measure > 3:  # Entangled qubits measurement
                    # Obtain the tensor product of pauli matrices measured
                    measure_string = pauli_labels[i][qubits[0]] + pauli_labels[i][qubits[1]]
                    map_basis = maps[index_measure - 4]  # Map of tensor products of the entangled basis
                    diagonal_factors.append(factors_list_bool[index_measure - 3][map_basis[measure_string]])
                else:
                    if shift:
                        diagonal_factors.append(factors_list_bool[0][1])

            # Generate the product tensor of all the diagonal factors
            diagonal_factors = generate_diagonal_factors_binary(*diagonal_factors, print_progress=progress_diagonal)
            if shift:
                chain_qubits = []
                for j in range(len(measurements)):  # Iterate over all the measurements in a given group
                    index_measure, qubits = measurements[len(measurements) - 1 - j]
                    qubits = np.abs(np.array(qubits) - n_qubits + 1)

                    for qubit in qubits:
                        chain_qubits.append(qubit)

                permutations = swaps(chain_qubits)
                for permutation in permutations:
                    diagonal_factors = permute_indices(diagonal_factors, permutation[0], permutation[1], n_qubits)

            diagonal_factors_temp.append(
                diagonal_factors)  # diagonal_factors_temp.append(diagonal_factors * pauli_weights[i])

        if print_progress:
            pbar.update()

        diagonal_factors_all.append(np.array(diagonal_factors_temp,
                                             dtype='bool'))  # diagonal_factors_all.append(np.array(diagonal_factors_temp))

    if print_progress:
        pbar.close()

    return [diagonal_factors_all, factors]


def probability2expected_parallel(n_jobs, Pauli_weights, Pauli_labels, Groups, Measurements, shift=True,
                                  print_progress=False, binary=True):
    if n_jobs == -1:
        n_jobs = os.cpu_count()

    if binary:
        p2e_fun = probability2expected_binary
    else:
        p2e_fun = probability2expected

    n_jobs = min(n_jobs, len(Groups))

    Groups_batchs = [Groups[i::n_jobs] for i in range(n_jobs)]
    Measurements_batchs = [Measurements[i::n_jobs] for i in range(n_jobs)]

    if print_progress:
        with tqdm_joblib(tqdm(total=n_jobs, desc='Computing diagonal factors')) as _:
            results = Parallel(n_jobs=n_jobs)(
                delayed(p2e_fun)(Pauli_weights, Pauli_labels, Groups_batchs[j], Measurements_batchs[j], shift=shift) for
                j in range(n_jobs))
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(p2e_fun)(Pauli_weights, Pauli_labels, Groups_batchs[j], Measurements_batchs[j], shift=shift) for j
            in range(n_jobs))

    results_sorted = []
    counter = 0
    while len(results_sorted) < len(Groups):
        for result in results:
            try:
                results_sorted.append(
                    [result[0][counter], result[1][counter]])  # results_sorted.append(result[counter])
            except IndexError:
                pass
        counter += 1

    return results_sorted
