from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
from functions_VQE import objective_function, from_string_to_numbers
from qiskit import *
from GroupingAlgorithms import grouping
from itertools import permutations
from qiskit.circuit.library import EfficientSU2
from qiskit.aqua.operators.legacy import WeightedPauliOperator
from qiskit.quantum_info import Pauli
from qiskit.aqua.algorithms import NumPyEigensolver, VQE
from qiskit.aqua.components.optimizers import COBYLA


def get_var_form(params, qr, cr):
    """
    Generate the variational circuit using a efficientSU2 circuit, with linear entanglements between qubits (the cnot
    gates are only applied to neighbours qubits). Each repetition of the circuit is formed by a R_y and R_z to all the
    qubits, and then the cnot gates.

    Parameters
    ----------
    param (numpy.array): Array with the values of the angles of each one qubit rotation gate
    reps (Optional, int): Number of repetitions for the variational circuit

    Return
    ------
    (qiskit.circuit) of N qubits and N classical registers, with the variational form
    """
    qc = QuantumCircuit(*qr,
                        *cr)  # Create a quantum circuit with classical registers, so later we can concatenate circuits

    # Create variational circuit (without classical register), and substitute the parameters of each gate
    qc_temp = EfficientSU2(num_qubits, entanglement='full', reps=n_rep).bind_parameters(params)

    return qc.compose(qc_temp)  # Add a  classical register to the circuit


def create_pauli_qiskit(weights, labels):
    """
    Transform a Pauli Weight in form of dictionary into a qiskit operator. To preforms this transformation we construct
    a list, in which each elements is another list of the form [weight, Pauli].

    Parameters
    ----------
    pauli (dic{'str': complex}): Each keys is a string representing the Pauli chains, e.g., 'IXXZ', and the values are
                               given by the weight of each Pauli string.

    Returns
    -------
    (WeightedPauliOperator) qiskit operator

    """
    temp = []  # List in which save the weights and the Pauli strings
    for label, weight in zip(labels, weights):  # Iterate over all the Pauli strings
        temp.append([weight, Pauli(label)])
    return WeightedPauliOperator(temp)  # Transform the list into a qiskit operator


initial_state = np.array([0, -1j * np.sqrt(5) / 2, -1 / 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
initial_state = initial_state / np.linalg.norm(initial_state)

backend = Aer.get_backend("qasm_simulator")  # Backend for simulation
NUM_SHOTS = 100000  # Number of shots for each circuit
classical_optimizer = 'COBYLA'
basis = [4, 6, 7, 8, 9, 5, 3, 2, 1]

num_qubits = 2
n_rep = 4
num_vars = num_qubits * 2 * (
        n_rep + 1)  # Number of parameters of the variational circuit, one parameters for each R_z and R_y
WC = list(permutations(list(range(num_qubits)), 2))

# lower and upper bound for variables
bounds = [[-2 * np.pi, 2 * np.pi]] * num_vars  # From 0 to 2 pi ??

# construct the bounds in the form of constraints
cons = []
for factor in range(len(bounds)):
    lower, upper = bounds[factor]
    l = {'type': 'ineq',
         'fun': lambda x, lb=lower, i=factor: x[i] - lb}
    u = {'type': 'ineq',
         'fun': lambda x, ub=upper, i=factor: ub - x[i]}
    cons.append(l)
    cons.append(u)

Pauli_weights = [1, 5, -1]
Pauli_labels = ['XX', 'YY', 'XZ']

pauli_qiskit = create_pauli_qiskit(Pauli_weights, Pauli_labels)
exact_result = NumPyEigensolver(pauli_qiskit).run()
exact_energy = np.real(exact_result.eigenvalues)[0]
print('The exact energy is {:.3f}'.format(exact_energy))

PS = from_string_to_numbers(Pauli_labels)
Groups, Measurements = grouping(PS, basis, WC)

# optimizer = COBYLA(maxiter=1000, disp=True)
# vqe = VQE(pauli_qiskit, EfficientSU2(num_qubits, entanglement='full', reps=n_rep), optimizer)
# results = vqe.run(backend)
# vqe_result = np.real(results['eigenvalue'])
# print(vqe_result)

initial_params = np.random.rand(num_vars)  # Initialize the parameters for the variational circuit with random values
result = minimize(objective_function, initial_params,
                  args=(Pauli_weights, Pauli_labels, Groups, Measurements, get_var_form, backend, NUM_SHOTS),
                  options={'maxiter': 1000, 'disp': True}, constraints=cons, method=classical_optimizer)
