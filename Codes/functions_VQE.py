import numpy as np
from qiskit import *  # Optimize this imports

# Maps for the order of measurements in each basis
maps = [np.array(['XX', 'YY', 'ZZ']),  # Bell
        np.array(['XX', 'YZ', 'ZY']),  # Omega xx
        np.array(['YY', 'XZ', 'ZX']),  # Omega yy
        np.array(['ZZ', 'XY', 'YX']),  # Omega zz
        np.array(['XY', 'YZ', 'ZX']),  # Chi
        np.array(['YX', 'ZY', 'XZ'])]  # Pi

# Factors for expected value of one qubit, and two qubits (in the correct order for each basis)
factors_list = [np.array([1, -1]),  # One qubit
                [np.array([1, -1, 1, -1]), np.array([-1, 1, 1, -1]), np.array([1, 1, -1, -1])],  # Bell
                [np.array([1, -1, -1, 1]), np.array([-1, 1, -1, 1]), np.array([-1, -1, 1, 1])],  # Omega xx
                [np.array([1, -1, -1, 1]), np.array([1, -1, 1, -1]), np.array([1, 1, -1, -1])],  # Omega yy
                [np.array([1, 1, -1, -1]), np.array([-1, 1, -1, 1]), np.array([-1, 1, 1, -1])],  # Omega zz
                [np.array([1, -1, 1, -1]), np.array([-1, 1, 1, -1]), np.array([1, 1, -1, -1])],  # Chi
                [np.array([-1, 1, 1, -1]), np.array([1, 1, -1, -1]), np.array([1, -1, 1, -1])]]  # Pi


def post_process_results(result, n_q, NUM_SHOTS):
    labels = []
    counts = []
    for key in result.keys():
        labels.append(key.replace(' ', ''))
        counts.append(result[key])

    probs = np.zeros(2 ** n_q)
    for j in range(len(labels)):
        probs[int(labels[j], 2)] += counts[j] / NUM_SHOTS

    return probs


def generate_diagonal_factors(*factors):
    factors = factors[::-1]  # We may invert the order of the factors
    diagonal_factor = factors[0]
    for i in range(1, len(factors)):
        temp = np.array([])
        for j in range(len(diagonal_factor)):
            temp = np.hstack([temp, factors[i] * diagonal_factor[j]])
        diagonal_factor = temp[:]

    return diagonal_factor


def measure_circuit_factor(measurements, n_qubits):
    n_measures = 0
    classical_registers = []
    for measure in measurements:
        if measure[0] != 0:
            classical_registers.append(ClassicalRegister(len(measure[1])))
            n_measures += len(measure[1])

    qr = QuantumRegister(n_qubits)
    circuit = QuantumCircuit(qr, *classical_registers)

    counter = 0
    for measure in measurements:
        measure_label, qubits = measure
        qubits = np.abs(np.array(qubits) - n_qubits + 1)
        qubits = sorted(qubits)  # Ensure the order of the qubits
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
            # Pi Circuit
            circuit.u2(0, np.pi / 2, qubits[0])
            circuit.cnot(qubits[0], qubits[1])
            circuit.h(qubits[0])

        circuit.measure(qubits, classical_registers[counter])
        counter += 1

    return circuit, n_measures


def objective_function(params, Pauli_weights, Pauli_labels, Groups, Measurements, variation_func, backend, NUM_SHOTS):
    energy = 0
    n_qubits = len(Pauli_labels[0])

    for measurements, group in zip(Measurements, Groups):
        pauli_weights = [Pauli_weights[i] for i in group]
        pauli_labels = [Pauli_labels[i] for i in group]

        mc, n_measures = measure_circuit_factor(measurements, n_qubits)
        qc = variation_func(params, mc.qregs, mc.cregs)
        qc_final = qc.compose(mc)

        t_qc = transpile(qc_final, backend)
        q_obj = assemble(t_qc, shots=NUM_SHOTS)
        counts = backend.run(q_obj).result().get_counts(qc_final)

        probabilities = post_process_results(counts, n_measures, NUM_SHOTS)

        for i in range(len(pauli_labels)):
            diagonal_factors = []
            for j in range(len(measurements)):
                index_measure, qubits = measurements[j]
                if 0 < index_measure <= 3:  # Single qubit measurement
                    diagonal_factors.append(factors_list[0])
                elif index_measure > 3:
                    measure_string = pauli_labels[i][qubits[0]] + pauli_labels[i][qubits[1]]
                    map = maps[index_measure - 4]
                    index = np.where(map == measure_string)[0][0]
                    diagonal_factors.append(factors_list[index_measure - 3][index])

            diagonal_factors = generate_diagonal_factors(*diagonal_factors)
            energy += np.sum(probabilities * diagonal_factors) * pauli_weights[i]

    print('Energy: {}'.format(energy))
    return energy


def from_string_to_numbers(pauli_labels):
    map = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
    PS = []

    for label in pauli_labels:
        temp = []
        for letter in label:
            temp.append(map[letter])

        PS.append(np.array(temp))
    return np.array(PS)
