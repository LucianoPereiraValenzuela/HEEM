import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from tqdm.auto import tqdm
from typing import Optional, Dict, Tuple, List, Union

PartialMeasurement = Tuple[int, List[int]]

_MAPS = [{'XX': 0, 'YY': 1, 'ZZ': 2, 'II': 3},  # Bell
         {'XX': 0, 'YZ': 1, 'ZY': 2, 'II': 3},  # Omega xx
         {'YY': 0, 'XZ': 1, 'ZX': 2, 'II': 3},  # Omega yy
         {'ZZ': 0, 'XY': 1, 'YX': 2, 'II': 3},  # Omega zz
         {'XY': 0, 'YZ': 1, 'ZX': 2, 'II': 3},  # Chi
         {'YX': 0, 'ZY': 1, 'XZ': 2, 'II': 3}]  # Chi_prime

# Diagonal factors for expected value of one qubit, and two qubits measurements
_FACTORS_LIST = [[np.array([1, 1], dtype='int8'), np.array([1, -1], dtype='int8')],  # One qubit
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

# Write the diagonal factors using bool with the encoding (1 -> False, -1 -> True).
_FACTORS_LIST_BOOL = []
for factors_temp in _FACTORS_LIST:
    _FACTORS_LIST_BOOL.append([])
    for factor_temp in factors_temp:
        temp = ~np.array((factor_temp + 1), dtype=bool)

        _FACTORS_LIST_BOOL[-1].append(temp)


def _post_process_results(result: Dict[str, int], indices_dtype: Optional[str] = 'I',
                          counts_dtype: Optional[str] = 'H') -> Tuple[np.ndarray, np.ndarray]:
    """
    Write the results of a quantum circuit simulation/experiment from the qiskit convention in a dic, to a sparse one,
    where the measured indices are in decimal. Only those indices that have been measured at least once are included in
    the return.

    Notes
    -----
    For the indices we are using an unsigned 32-bit integer. If more than 32 qubits are simulated, change it to a 64-bit
    one. In the same way, for the counts we use 16-bit unsigned ints, so the maximum number of shots is 2^16. If more
    are used, then change the data type.

    Parameters
    ----------
    result: dict[str, int]
        Results in the qiskit convention, where the key is a string with teh index in binary, and the values as the
        number of counts for each index.
    indices_dtype: str (optional, default='I')
        Data type for the indices.
    counts_dtype: str (optional, default='H')
        Data type for the counts.

    Returns
    -------
    indices: ndarray
        Indices (in decimal) where at least one count is measured.
    counts: ndarray
        Counts for each index.
    """

    # Initialize list for the results and the counts
    labels = []
    counts = []
    for key in result.keys():
        labels.append(key.replace(" ", ""))  # Join all the classical register in one single string with no spaces
        counts.append(result[key])

    indices = np.zeros(len(labels), dtype=indices_dtype)

    for j in range(len(labels)):
        indices[j] = int(labels[j], 2)

    return indices, np.array(counts, dtype=counts_dtype)


def generate_diagonal_factors(*factors: List, print_progress: Optional[bool] = False) -> np.ndarray:
    """
    Generate the diagonal part of the tensor product of matrices that represent the basis in which each qubit (or a pair
    of qubits) has been measured. Originally we have to compute the Kronecker product of 1's and -1's. Obtaining the
    following truth table (left). However, the same can be obtained with boolean variables, under a sum in Z2 (right)

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
    print_progress: bool (optional, default=False)
        If True, print the progress of the calculation.

    Return
    ------
    diagonal_factor: ndarray(int)
        Diagonal elements of the tensor product
    """
    factors = list(factors)
    diagonal_factor = factors.pop(0)  # Initialize the diagonal factors as the diagonal of the first matrix

    pbar = tqdm(total=len(factors), desc='Computing Kronecker product', disable=not print_progress)

    while len(factors) != 0:  # Run over all the indices, except the first one
        v2 = factors.pop(0)
        diagonal_factor = v2[0] ^ diagonal_factor

        n = len(diagonal_factor)
        for j in range(1, len(v2)):
            if v2[j] == v2[0]:
                diagonal_factor = np.hstack([diagonal_factor, diagonal_factor[:n]])
            else:
                diagonal_factor = np.hstack([diagonal_factor, ~diagonal_factor[:n]])
            pbar.update()
        pbar.close()
    return diagonal_factor


def _measure_circuit_factor(measurements: List[PartialMeasurement], n_qubits: int,
                            make_measurements: Optional[bool] = True,
                            measure_all: Optional[bool] = False) -> QuantumCircuit:
    # Create the quantum circuit
    qr = QuantumRegister(n_qubits)

    if not measure_all:  # If not all qubits are measured, then each partial measurement is in a separate classical reg.
        classical_registers = []
        for measure in measurements:
            if measure[0] != 0:  # If the operation is not the identity
                classical_registers.append(ClassicalRegister(len(measure[1])))
        circuit = QuantumCircuit(qr, *classical_registers)
    else:
        circuit = QuantumCircuit(qr, ClassicalRegister(n_qubits))

    count = 0
    for measure in measurements:  # Iterate over all the measurements
        measure_label, qubits = measure  # Extract the index of the measurement and the measured qubits

        qubits = list(np.abs(np.array(qubits) - n_qubits + 1))[::-1]  # Goes to the qiskit convention

        if measure_label == 0:
            # No measurement
            if measure_all:
                pass
            else:
                continue
        elif measure_label == 1:  # X
            circuit.h(qubits)
        elif measure_label == 2:  # Y
            circuit.sdg(qubits)
            circuit.h(qubits)
        elif measure_label == 3:  # Z
            pass
        elif measure_label == 4:  # Bell
            circuit.cnot(qubits[0], qubits[1])
            circuit.h(qubits[0])
        elif measure_label == 5:  # Omega xx
            circuit.s(qubits)
            circuit.h(qubits[0])
            circuit.cnot(qubits[0], qubits[1])
            circuit.h(qubits[0])
        elif measure_label == 6:  # Omega yy
            circuit.h(qubits[0])
            circuit.cnot(qubits[0], qubits[1])
            circuit.h(qubits[0])
        elif measure_label == 7:  # Omega zz
            circuit.s(qubits[0])
            circuit.cnot(qubits[0], qubits[1])
            circuit.h(qubits[0])
        elif measure_label == 8:  # Chi
            circuit.u2(np.pi / 2, np.pi, qubits[0])
            circuit.cnot(qubits[0], qubits[1])
            circuit.h(qubits[0])
        elif measure_label == 9:  # Chi_prime
            circuit.u2(0, np.pi / 2, qubits[0])
            circuit.cnot(qubits[0], qubits[1])
            circuit.h(qubits[0])

        if make_measurements:
            if measure_all:
                circuit.measure(qubits, qubits)
            else:
                circuit.measure(qubits, classical_registers[count])
        count += 1

    return circuit


def create_circuits(Measurements: Union[List[PartialMeasurement], List[List[PartialMeasurement]]], n_qubits: int,
                    make_measurements: Optional[bool] = True, measure_all: Optional[bool] = False,
                    initial_state: Optional[QuantumCircuit] = None) -> Union[QuantumCircuit, List[QuantumCircuit]]:
    """
    Function to create the circuit needed to obtain a given group of measurements. Each measurement will be saved in an
    individual classical register. Each measurement is coded with an int number. The available measurements are:
    0 -> I, 1 -> X, 2 -> Y, 3 -> Z, 4 -> Bell, 5 -> Omega_xx, 6 -> Omega_yy, 7 -> Omega_zz, 8 -> Chi, 9 -> Chi_prime

    To ensure the correct functionality of this function each qubit can only be in one of the measurements, so it's only
    measured once. If a qubit is not provided, then it is not measured.

    Parameters
    ----------
    Measurements: list(_PartialMeasurement)
        List with all the measurements in a given grouping. The measurement is a tuple in which the first index is the
        int encoding the measurement, and the second element is another list with the indices of the measured qubits.
        The convention for the indices of the qubits is opposite to the qiskit convention. Here the qubit with higher
        weight is named as q_0.
    n_qubits: int
        Total number of qubits in the circuit. This values does not have to coincide with the number of measured qubits.
    make_measurements: bool (optional, default=True)
        If True, include measurement gates at the end of the circuit.
    measure_all: bool (optional, default=False)
        If True, measure all the qubits, so we can perform measurement error mitigation.
    initial_state: QuantumCircuit (optional, default=None)
        If provided, append the same initial state to all the circuits

    Returns
    -------
    circuits: QuantumCircuit or list(QuantumCircuit)
        Circuit (including quantum and classical registers) with the gates needed to perform the measurements.
    """
    initial_list = True
    if type(Measurements[0]) is tuple:
        Measurements = [Measurements]
        initial_list = False

    circuits = [
        _measure_circuit_factor(measurement, n_qubits, measure_all=measure_all, make_measurements=make_measurements) for
        measurement in Measurements]

    if initial_state is not None:
        circuits = [circuit.compose(initial_state, front=True) for circuit in circuits]

    if not initial_list:
        circuits = circuits[0]
    return circuits


def probability2expected(Pauli_labels: List[str], Pauli_weights: List[complex],
                         Groups: Union[List[List[int]], List[int]],
                         Measurements: Union[List[List[PartialMeasurement]], List[PartialMeasurement]],
                         swap: Optional[bool] = False, print_progress: Optional[bool] = False,
                         progress_diagonal: Optional[bool] = False) -> Union[
    Tuple[List[np.ndarray], List[List[complex]]], Tuple[np.ndarray, List[complex]]]:
    """
    Compute the prefactors for computing the expected value of a given Hamiltonian with the probabilities measured based
    on some grouping of measurements.

    Note: If swap=True, the code slows down due to the sorting and swapping are not optimal.

    Parameters
    ----------
    Pauli_labels: list (str)
        Pauli string in the str convention.
    Pauli_weights: list (complex)
        Weights of each Pauli label
    Groups: list(list(int))
        List in which each element is represented the indices of pauli string that are measured simultaneously.
    Measurements: list(list(_PartialMeasurement))
        List with all the measurements. Each measurement is a list in which the first index is an int encoding the
        measurement, and the second element is another list with the indices of the measured qubits. The convention for
        the indices of the qubits is opposite to the qiskit convention. Here the qubit with higher weight is named as
        q_0.
    swap: bool (optional, default=False)
        Change between qubits numbering conventions.
    print_progress: bool (optional, default=False)
        If True, print the progress bar for as the diagonal factors for each group is computed.
    progress_diagonal: bool (optional, default=False)
        If True, print the progress bar for the build of each diagonal factors.
    Return
    ------
    diagonal_factor_all: list(ndarray(int))
        Diagonal factors for all the tensor products. Each element in the list represents the diagonal factors for a
        given group of measurements.
    weights: list(ndarray)
        Weights of each
    """

    initial_list = True
    if type(Measurements[0]) is tuple:
        Measurements = [Measurements]
        Groups = [Groups]
        initial_list = False

    n_qubits = len(Pauli_labels[0])
    diagonal_factors_all = []  # Initialize the list with all the diagonal factors
    factors = []

    total = 0
    for group in Groups:
        total += len(group)

    pbar = tqdm(total=total, desc='Computing diagonal factors', disable=not print_progress)

    for measurements, group in zip(Measurements, Groups):  # Iterate over all the measurements
        pauli_labels = [Pauli_labels[i] for i in group]
        factors.append([Pauli_weights[i] for i in group])
        diagonal_factors_temp = []  # Initialize the diagonal factors for the given group

        for i in range(len(pauli_labels)):  # Iterate over all the measured pauli strings
            diagonal_factors = []  # Initialize the diagonal factors for one pauli string
            for j in range(len(measurements)):  # Iterate over all the measurements in a given group
                index_measure, qubits = measurements[j]
                if 0 < index_measure <= 3:  # Single qubit measurement
                    if pauli_labels[i][qubits[0]] == 'I':  # If the identity is grouped with another measurement
                        diagonal_factors.append(_FACTORS_LIST_BOOL[0][0])
                    else:  # If a Pauli matrix is measured
                        diagonal_factors.append(_FACTORS_LIST_BOOL[0][1])
                elif index_measure > 3:  # Entangled qubits measurement
                    # Obtain the tensor product of pauli matrices measured
                    measure_string = pauli_labels[i][qubits[0]] + pauli_labels[i][qubits[1]]
                    map_basis = _MAPS[index_measure - 4]  # Map of tensor products of the entangled basis
                    diagonal_factors.append(_FACTORS_LIST_BOOL[index_measure - 3][map_basis[measure_string]])
                else:
                    if swap:
                        diagonal_factors.append(_FACTORS_LIST_BOOL[0][0])

            # Generate the product tensor of all the diagonal factors
            diagonal_factors = generate_diagonal_factors(*diagonal_factors, print_progress=progress_diagonal)

            if swap:
                chain_qubits = []
                for j in range(len(measurements)):  # Iterate over all the measurements in a given group
                    index_measure, qubits = measurements[len(measurements) - 1 - j]
                    qubits = np.abs(np.array(qubits) - n_qubits + 1)

                    for qubit in qubits:
                        chain_qubits.append(qubit)

                permutations = _swaps(chain_qubits)
                for permutation in permutations:
                    diagonal_factors = _permute_indices(diagonal_factors, permutation[0], permutation[1], n_qubits)

            diagonal_factors_temp.append(diagonal_factors)
            pbar.update()

        diagonal_factors_all.append(np.array(diagonal_factors_temp))
    pbar.close()

    if not initial_list:
        diagonal_factors_all = diagonal_factors_all[0]
        factors = factors[0]

    return diagonal_factors_all, factors


def compute_energy(counts: List[Dict[str, int]], labels: List[str], coeffs: List[complex], Groups: List[List[int]],
                   Measurements: List[List[PartialMeasurement]], shots: int, progress_bar: Optional[bool] = True,
                   **kwargs) -> float:
    """
    Use the data obtained from the simulation/experiment and compute the obtained Hamiltonian expected value, i.e,
    the energy.

    Parameters
    ----------
    counts: list(dic(str, int))
        Counts of the simulation/experiment in the qiskit convention. Each count represent one circuit. The keys
        represent the index of the measurement in binary, and the value the number of times it its measured.
    labels: list(str)
        Pauli labels of the Hamiltonian
    coeffs: list(complex)
        Coefficients for each Pauli label
    Groups: list(list(int))
        Indices of the grouped Pauli labels.
    Measurements: list(list(_PartialMeasurement))
        List with the partial measurements for each group.
    shots: int
        Number of shots (same for all circuits)
    progress_bar: bool (optional, default=True)
        If True, print a progress bar for the calculation of partial energies of each grouping.
    kwargs: dict (optional)
        Optional arguments for the probability2expected functions such as progress bars and shift.

    Return
    ------
    energy: float
        Expected value of the Hamiltonian.
    """

    pbar = tqdm(range(len(counts)), desc='Computing energy', disable=not progress_bar)

    energy = 0
    for i in pbar:
        indices, values = _post_process_results(counts[i])

        diagonals, factors = probability2expected(labels, coeffs, Groups[i], Measurements[i], **kwargs)
        diagonals = [(~diagonal * 2 - 1).astype('int8') for diagonal in diagonals[:, indices]]

        energy += np.sum((diagonals * np.array(factors)[:, None]) * values[None, :]) / shots
    return energy.real


def _swap_positions(str_variable: str, pos1: int, pos2: int) -> str:
    """
    Swap the position of two indices of a given string.

    Parameters
    ----------
    str_variable: str
        String to interchange the indices. The length must be >= max(pos1, pos2)
    pos1: int
        Index of the first element to swap
    pos2: int
        Index of the second element to swap

    Return
    ------
    Reformat string with the given swaps
    """

    list_variable = list(str_variable)
    list_variable[pos1], list_variable[pos2] = list_variable[pos2], list_variable[pos1]
    return ''.join(list_variable)


def _permute_indices(diagonal_factors: np.ndarray, qubit0: int, qubit1: int, n_qubits: int) -> np.ndarray:
    """
    Permute the diagonal factors indices by the interchange of qubit_0 <---> qubit_1, maintaining all other indices the
    same.

    Parameters
    ----------
    diagonal_factors: ndarray (2 ** n_qubits)
        Diagonal factors for the computation of the expected energy
    qubit0: int
        Index of the first qubit to swap
    qubit1: int
        Index of the second qubit to swap
    n_qubits: int
        Number of qubits in the circuit

    Return
    ------
    temp: ndarray (2 ** n_qubits)
        Refactor diagonal factors
    """
    diagonals_permuted = np.zeros_like(diagonal_factors)

    # Iterate over all the possible outputs of the circuit
    for i in range(len(diagonal_factors)):
        new = bin(i)[2:]  # New index in binary
        if len(new) < n_qubits:  # Complete with 0's if the index is not of the correct size
            new = ''.join(['0']) * (n_qubits - len(new)) + new
        old = _swap_positions(new, qubit0, qubit1)  # Swap the indices of qubit_0 and qubit_1

        # Copy the old diagonal factor in the new position
        diagonals_permuted[int(new, 2)] = diagonal_factors[int(old, 2)]

    return diagonals_permuted


def _swaps(arr: List[int], reverse: Optional[bool] = True) -> np.ndarray:
    """
    Compute the needed swaps of two elements to sort a given array in descending (or ascending) order.

    Parameters
    ----------
    arr: list
        Original array with unsorted numbers [0, 1, ...., len(arr) - 1]. A given element can not appear twice.
    reverse: bool (optional, default=True)
        If True, sort in descending order. If False, then sort in ascending order

    Returns
    -------
    swaps: ndarray (n, 2)
        Array containing the indices needed to perform a total of n swaps. Each swap corresponds to a given row. The
        swaps must be performed in the correct order, starting from swaps[0], and finish in swaps[-1].
    """
    # If descending order, reverse the order of the original array
    if reverse:
        arr = arr[::-1]

    n = len(arr)  # Number of elements
    swaps_list = []  # List with the swaps

    i = 0
    while i < n:
        if arr[i] != i:  # If the element is not in the correct locations
            swaps_list.append(np.array([i, arr[i]]))

            # Interchange the element with the correct element in a given location
            arr[arr[i]], arr[i] = arr[i], arr[arr[i]]
        else:
            i += 1

    swaps_list = np.array(swaps_list)

    # If descending order, transform the indices in each swap. E.g. if N = 3: 0 --> |0 - 3 + 1| = 2, 1 -> 1 and 2 -> 0
    if reverse:
        swaps_list = np.abs(swaps_list - n + 1)

    return swaps_list
