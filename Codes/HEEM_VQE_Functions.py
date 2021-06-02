import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, assemble

# Maps for the order of measurements in each basis
maps = [np.array(['XX', 'YY', 'ZZ', 'II']),  # Bell
        np.array(['XX', 'YZ', 'ZY', 'II']),  # Omega xx
        np.array(['YY', 'XZ', 'ZX', 'II']),  # Omega yy
        np.array(['ZZ', 'XY', 'YX', 'II']),  # Omega zz
        np.array(['XY', 'YZ', 'ZX', 'II']),  # Chi
        np.array(['YX', 'ZY', 'XZ', 'II'])]  # Chi_prime

# Factors for expected value of one qubit, and two qubits (in the correct order for each basis)
factors_list = [[np.array([1, -1]), np.array([1, 1])],  # One qubit
                [np.array([1, -1, 1, -1]), np.array([-1, 1, 1, -1]), np.array([1, 1, -1, -1]), np.array([1, 1, 1, 1])],
                # Bell
                [np.array([1, -1, -1, 1]), np.array([-1, 1, -1, 1]), np.array([-1, -1, 1, 1]), np.array([1, 1, 1, 1])],
                # Omega xx
                [np.array([1, -1, -1, 1]), np.array([1, -1, 1, -1]), np.array([1, 1, -1, -1]), np.array([1, 1, 1, 1])],
                # Omega yy
                [np.array([1, 1, -1, -1]), np.array([-1, 1, -1, 1]), np.array([-1, 1, 1, -1]), np.array([1, 1, 1, 1])],
                # Omega zz
                [np.array([1, -1, 1, -1]), np.array([-1, 1, 1, -1]), np.array([1, 1, -1, -1]), np.array([1, 1, 1, 1])],
                # Chi
                [np.array([-1, 1, 1, -1]), np.array([1, 1, -1, -1]), np.array([1, -1, 1, -1]),
                 np.array([1, 1, 1, 1])]]  # Chi_prime


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


def generate_diagonal_factors(*factors):
    """
    Generate the diagonal part of the tensor product of matrices that represent the basis in which each qubit (or pair
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
    for i in range(1, len(factors)):  # Run over all the indices, except the first one
        temp = np.array([])  # Temporary variable to create the new diagonal factors
        for j in range(len(diagonal_factor)):  # Run over all the elements of the current diagonal factors
            temp = np.hstack([temp, diagonal_factor[j] * factors[i]])  # Append a_jj * B
        diagonal_factor = temp[:]  # Save the computed diagonal factors

    return diagonal_factor


def measure_circuit_factor(measurements, n_qubits):
    """
    Function to create the circuit needed to obtain a given group of measurements. Each measurement will be save in an
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

    Returns
    -------
    circuit: quantum circuit
        Circuit (including quantum and classical registers) with the gates needed to perform the measurements.
    n_measures: int
        Number of measured qubits
    """
    # Initialize the number of measured qubits to 0 and a list with the classical registers for each measurement
    n_measures = 0
    classical_registers = []
    for measure in measurements:
        if measure[0] != 0:  # If the operation is not the identity
            classical_registers.append(ClassicalRegister(len(measure[1])))
            n_measures += len(measure[1])

    # Create the quantum circuit
    qr = QuantumRegister(n_qubits)
    circuit = QuantumCircuit(qr, *classical_registers)

    counter = 0  # Index for the classical register
    for measure in measurements:  # Iterate over all the measurements
        measure_label, qubits = measure  # Extract the index of the measurement and the measured qubits
        qubits = np.abs(np.array(qubits) - n_qubits + 1)  # Goes to the qiskit convention
        qubits = sorted(qubits)  # Ensure the order of the qubits of entangled measurements
        if measure_label == 0:
            # No measurement
            continue
        elif measure_label == 1:
            # X Circuit
            circuit.h(qubits)
        elif measure_label == 2:
            # Y Circuit
            circuit.sdg(qubits)
            circuit.h(qubits)
            pass
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

        circuit.measure(qubits, classical_registers[counter])
        counter += 1

    return circuit, n_measures


def probability2expected(Pauli_weights, Pauli_labels, Groups, Measurements):
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
        List with all the measurements. Each measured is a list in which the first index in the int encoding the
        measurement, and the second element is another list with the indices of the measured qubits. The convention for
        the indices of the qubits is opposite to the qiskit convention. Here the qubit with higher weight is named as
        q_0.

    Return
    ------
    diagonal_factor_all: list(array(int))
        Diagonal factors for all the tensor products. Each element in the list represents the diagonal factors for a
        given group of measurements.
    """
    diagonal_factors_all = []  # Initialize the list with all the diagonal factors

    for measurements, group in zip(Measurements, Groups):  # Iterate over all the measurements
        # Pauli weights and string in each measurement
        pauli_weights = [Pauli_weights[i] for i in group]
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
                    index = np.where(map_basis == measure_string)[0][0]  # Index in the map
                    diagonal_factors.append(factors_list[index_measure - 3][index])

            # Generate the product tensor of all the diagonal factors
            diagonal_factors = generate_diagonal_factors(*diagonal_factors)
            diagonal_factors_temp.append(diagonal_factors * pauli_weights[i])

        diagonal_factors_all.append(np.array(diagonal_factors_temp))

    return diagonal_factors_all


def from_string_to_numbers(pauli_labels):
    """
    Function that transform a set of pauli string from the str convention ('IXYZ'), to the number convention (0123).

    Parameter
    ---------
    pauli_labels: list
        List with the pauli string written as a string.

    Return
    ------
    PS: array
        Pauli strings in the number convention.
    """
    map_str_int = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}  # Map between str and int conventions
    PS = []  # Initialize the Pauli string for int convention

    for label in pauli_labels:  # Iterate over all the given pauli strings
        temp = []
        for letter in label:  # Map each element of a Pauli string
            temp.append(map_str_int[letter])

        PS.append(np.array(temp))
    return np.array(PS)
