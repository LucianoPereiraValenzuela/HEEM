import numpy as np
from typing import Union, List, Optional

from qiskit.circuit.library import EfficientSU2
from qiskit.compiler import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.algorithms.optimizers import Optimizer
from qiskit.providers import Backend

from grouping import Grouping
from measurements import create_circuits, compute_energy


class VQE:
    def __init__(self, grouping: Grouping, optimizer: Optimizer, params: Union[List, np.ndarray], backend: Backend,
                 circuit_init: Optional[QuantumCircuit] = None, shots: Optional[int] = 2 ** 10):
        self._grouping = grouping
        self._grouping.check_grouping()

        self._optimizer = optimizer
        self._params = params
        self._backend = backend
        self._circuit_init = circuit_init
        self._shots = shots

        self._num_params = 0
        self._variational_circuit = None
        self._circuits = []
        self._num_qubits = len(self._grouping.T)

        self._generate_circuits()

    def compute_minimum_eigenvalue(self):
        results = self._optimizer.optimize(self._num_params, self.energy_evaluation, initial_point=self._params)
        return results

    def energy_evaluation(self, params: List[float]) -> float:
        circuits_tp = transpile([circuit.assign_parameters(params) for circuit in self._circuits],
                                backend=self._backend)
        counts = self._backend.run(circuits_tp, shots=self._shots,
                                   initial_layout=self._grouping.T).result().get_counts()

        return compute_energy(counts, self._grouping.labels_string(), self._grouping.coeffs, self._grouping.groups,
                              self._grouping.measurements, self._shots, progress_bar=False)

    def _generate_circuits(self):
        self._variational_circuit = self._hardware_efficient_circuit()

        self._circuits = [
            transpile(create_circuits(measure, self._num_qubits), initial_layout=self._grouping.T).compose(
                self._variational_circuit, front=True) for measure in self._grouping.measurements]

    def _hardware_efficient_circuit(self):
        WC = [indx for indx in self._grouping.connectivity if indx[0] < self._num_qubits and indx[1] < self._num_qubits]

        variational_circuit = EfficientSU2(self._num_qubits, reps=1, entanglement=WC)

        if self._circuit_init is not None:
            variational_circuit = self._circuit_init.compose(variational_circuit)
        self._num_params = variational_circuit.num_parameters
        return variational_circuit
