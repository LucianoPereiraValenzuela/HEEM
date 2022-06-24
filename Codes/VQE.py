"""
Variational Quantum Eigensolver algorithm with Hardware Efficient Entangled measurements.
"""

from typing import Optional, List, Callable, Union, Dict
import logging
import time
import numpy as np
from GroupingAlgorithm import groupingWithOrder, TPBgrouping
from itertools import permutations
from HEEM_VQE_Functions import measure_circuit_factor, probability2expected, post_process_results
from utils import Label2Chain
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.opflow import OperatorBase, StateFn
from qiskit.providers import Backend, BaseBackend
from qiskit.providers.aer import AerSimulator
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.opflow.gradients import GradientBase
from qiskit.circuit.library import RealAmplitudes
from qiskit.algorithms.optimizers import Optimizer, SLSQP
from qiskit.algorithms.variational_algorithm import VariationalResult
from qiskit.algorithms.minimum_eigen_solvers.minimum_eigen_solver import MinimumEigensolver, MinimumEigensolverResult
from qiskit.algorithms.exceptions import AlgorithmError
import networkx as nx
from typing import List

logger = logging.getLogger(__name__)


class VQE(MinimumEigensolver):
    """
    Class of the Variational Quantum Eigensolver.
    """

    def __init__(self, ansatz: Optional[QuantumCircuit] = None, optimizer: Optional[Optimizer] = None,
                 initial_point: Optional[np.ndarray] = None, gradient: Optional[Union[GradientBase, Callable]] = None,
                 grouping: Optional[str] = 'TPB', order: Optional[np.ndarray] = np.array([4, 3, 2, 1]),
                 connectivity: Optional[list] = None,
                 callback: Optional[Callable[[int, np.ndarray, float, None], None]] = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = AerSimulator(
                    method='statevector'),
                    Measurements: Optional[list] = None, 
                    layout: Optional[Union[list]] = None, 
                    Groups: Optional[Union[List[List[int]]]] = None) -> None:
        """
        Parameters
        ----------
        ansatz: Qiskit circuit (optional)
            A parameterized circuit used as Ansatz for the wave function. By default ansatz = RealAmplitudes().
        optimizer: Classical optimizer (optional).
            By default optimizer = SLSQP().
        initial_point: ndarray (optional)
            An optional initial point (i.e. initial parameter values) for the optimizer. If ``None`` then VQE will look
            to the ansatz for a preferred point and if not will simply compute a random one.
        gradient: Gradient Function (optional)
            An optional gradient function or operator for optimizer. (Not implemented yet)
        grouping: str (optional)
            Method for grouping the local observables of the Hamiltonian into compatible measurements. Two methods are
            available:
                tensor product basis ('TPB')
                entangled measurements ('Entangled')
        order: list(int) (optional)
            Priority of the bases when grouping local operators in entangled measurements.  grouping = 'Entangled'
            is required. The available measurements are:
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
        connectivity: list(tuples) or list(list) (optional)
            The inter-qubit connectivity allowed for entangled measurements. grouping = 'Entangled' is required.
            As example, consider the 4-qubits device
                               0--1--2
                                  |
                                  3
            For this case we have: connectivity = [(0,1),(1,0),(1,2),(1,3),(2,1),(3,1)]. By default connection between
            all qubits is used.
        callback: Callable(int, np.ndarray) (optional)
            A callback that can access the intermediate data during the optimization.  The inputs are the number of
            evaluations, the parameters of the given iteration, and the evaluated energy.
        quantum_instance: Quantum Instance or Backend.
        """

        if ansatz is None:
            self._ansatz = RealAmplitudes()
        else:
            self._ansatz = ansatz

        if optimizer is None:
            self._optimizer = SLSQP()
        else:
            self._optimizer = optimizer

        # set the initial point to the preferred parameters of the ansatz
        if initial_point is None and hasattr(ansatz, 'preferred_init_points'):
            self._initial_point = ansatz.preferred_init_points
        else:
            self._initial_point = initial_point

        if isinstance(quantum_instance, Backend) or isinstance(quantum_instance, BaseBackend):
            quantum_instance = QuantumInstance(backend=quantum_instance)

        self._order = order
        self._connectivity = connectivity
        self._grouping = grouping
        self._Groups = Groups
        self._Measurements = Measurements
        self._layout = layout

        self._eval_count = 0
        self._callback = callback
        self.energies = []
        self._total_time = 0
        self._quantum_instance = quantum_instance
        self._gradient = gradient
        self._cost_fn = self._energy_evaluation
        self._ansatz_params = sorted(ansatz.parameters, key=lambda p: p.name)
        logger.info(self.print_settings())
        self._coeff = []
        self._label = []
        self._diagonals = []
        self._factors = []
        self._expect_op = []
        self._ret = None
        self._eval_time = None

    @property
    def setting(self):
        """Prepare the setting of VQE as a string."""
        ret = "Algorithm: {}\n".format(self.__class__.__name__)
        params = ""
        for key, value in self.__dict__.items():
            if key[0] == "_":
                if "initial_point" in key and value is None:
                    params += "-- {}: {}\n".format(key[1:], "Random seed")
                else:
                    params += "-- {}: {}\n".format(key[1:], value)
        ret += "{}".format(params)
        return ret

    def print_settings(self):
        """
        Preparing the setting of VQE into a string.

        Return
        ------
        str: the formatted setting of VQE
        """
        ret = "\n"
        ret += "==================== Setting of {} ============================\n".format(self.__class__.__name__)
        ret += "{}".format(self.setting)
        ret += "===============================================================\n"
        if hasattr(self._ansatz, 'setting'):
            ret += "{}".format(self._ansatz.setting)
        elif hasattr(self._ansatz, 'print_settings'):
            ret += "{}".format(self._ansatz.print_settings())
        elif isinstance(self._ansatz, QuantumCircuit):
            ret += "ansatz is a custom circuit"
        else:
            ret += "ansatz has not been set"
        ret += "===============================================================\n"
        ret += "{}".format(self._optimizer.setting)
        ret += "===============================================================\n"
        return ret

    # Checks
    def _check_operator_varform(self, operator: OperatorBase):
        """Check that the number of qubits of operator and ansatz match."""
        if operator is not None and self._ansatz is not None:
            if operator.num_qubits != self._ansatz.num_qubits:
                # try to set the number of qubits on the ansatz, if possible
                try:
                    self._ansatz.num_qubits = operator.num_qubits
                    self._ansatz_params = sorted(self._ansatz.parameters, key=lambda p: p.name)
                except AttributeError as ex:
                    raise AlgorithmError("The number of qubits of the ansatz does not match the "
                                         "operator, and the ansatz does not allow setting the "
                                         "number of qubits using `num_qubits`.") from ex

    def _check_operator(self, operator: OperatorBase) -> OperatorBase:
        """ set operator """
        self._expect_op = None
        self._check_operator_varform(operator)
        return operator

    def construct_expectation(self, parameter, operator):
        """
        Generate the ansatz circuit and expectation value measurement, and return their
        runnable composition.

        Parameters
        ----------
        parameter: np.array
            Parameters for the ansatz circuit.
        operator: Qubit operator
            Operator of the observable.

        Return
        ------
        circuits: list(qiskit circuits)
            List with the qiskit circuits to evaluate the energy.
        """

        if isinstance(self._ansatz, QuantumCircuit):
            param_dict = dict(zip(self._ansatz_params, parameter))  # type: Dict
            wave_function = self._ansatz.assign_parameters(param_dict)
        else:
            wave_function = self._ansatz.construct_circuit(parameter)

        paulis, self._coeff, self._label = Label2Chain(operator)
        num_qubits = operator.num_qubits

        if self._Groups is None and self._Measurements is None:
            if self._grouping.lower() == 'entangled':
                if self._connectivity is None:
                    self._connectivity = list(permutations(list(range(num_qubits)), 2))

                G = nx.Graph()
                G.add_nodes_from(range(num_qubits))
                G.add_edges_from(self._connectivity)
                #             self._Groups, self._Measurements = grouping(paulis, self._order, self._connectivity)
                self._Groups, self._Measurements, self._layout = groupingWithOrder(paulis, G)

            elif self._grouping.lower() == 'tpb':
                _, self._Groups, self._Measurements = TPBgrouping(paulis)
                self._layout = None
            else:
                raise Exception('Grouping algorithm not implemented. Available groupings: Entangled, TPB.')
        elif self._Groups is None or self._Measurements is None:
            raise Exception(
                'When introducing a given grouping, needs for both grouping and measurements. If you want to'
                ' automatically compute the grouping, do not introduce neither variable.')

        if self._grouping.lower() == 'entangled' and self._layout is None:
            raise Exception('When using a pre-calculated grouping with order needs the teo-phys maps between qubits.')

        if self._layout is not None:
            self._layout = self._layout[::-1]

        self._quantum_instance.set_config(initial_layout=self._layout)

        circuits = [(measure_circuit_factor(measure, num_qubits).compose(self._ansatz, front=True)) for measure in
                    self._Measurements]

        self._diagonals, self._factors = probability2expected(self._coeff, self._label, self._Groups,
                                                              self._Measurements)

        return circuits

    def number_cnots(self, operator=None):
        try:
            self._expect_op[0]
        except Exception:
            self._expect_op = self.construct_expectation(self._ansatz_params, operator)

        circuits = self._expect_op
        circuits = self._quantum_instance.transpile(circuits)

        n_cnots = []
        for circuit in circuits:
            try:
                n_cnots.append(circuit.count_ops()['cx'])
            except KeyError:
                n_cnots.append(0)

        return n_cnots, circuits

    def number_circuits(self, operator=None):
        try:
            self._expect_op[0]
        except Exception:
            self._expect_op = self.construct_expectation(self._ansatz_params, operator)

        circuits = self._expect_op

        return len(circuits)

    def _circuit_sampler(self, expected_op, params):
        """
        Execute the circuits to evaluate the expected value of the Hamiltonian.

        Parameters
        ----------
        expected_op: list(qiskit circuits)
            Circuits to evaluate the energy (wave function+measurements).
        params: ndarray
            Parameters of the circuits.

        Return
        ------
        ExpectedValue: float
            Expected value.
        """
        start = time.time()
        expected_op = [qci.assign_parameters(params) for qci in expected_op]

        counts = self._quantum_instance.execute(expected_op).get_counts()

        probabilities = [
            post_process_results(counts[j], expected_op[j].num_clbits, self._quantum_instance.run_config.shots) for j in
            range(len(counts))]

        ExpectedValue = 0
        for j in range(len(probabilities)):
            ExpectedValue += np.sum((self._prob2Exp[j] * self._factors[j][:, None]) @ probabilities[j])

        self.energies.append(ExpectedValue)

        if self._callback is not None:
            self._callback(self._eval_count, params, ExpectedValue, None)

        self._total_time += time.time() - start

        return ExpectedValue

    def _energy_evaluation(self, parameters):
        """
        Evaluate energy at given parameters for the ansatz.
        This is the objective function to be passed to the optimizer that is used for evaluation..

        Parameter
        ---------
        parameters: ndarray
            Parameters of the circuits.

        Return
        ------
        means: float
            Expected value.
        """

        num_parameters = self._ansatz.num_parameters
        if num_parameters == 0:
            raise RuntimeError('The ansatz cannot have 0 parameters.')

        start_time = time.time()

        means = self._circuit_sampler(self._expect_op, params=parameters)

        self._eval_count += 1

        end_time = time.time()
        logger.info('Energy evaluation returned %s - %.5f (ms), eval count: %s', means, (end_time - start_time) * 1000,
                    self._eval_count)

        return means

    def compute_minimum_eigenvalue(self, operator: OperatorBase, aux_operators: Optional[
        List[Optional[OperatorBase]]] = None) -> MinimumEigensolverResult:
        """
        Execute the VQE for a given Hamiltonian.

        Parameter
        ---------
        operator: OperatorBase
            Hamiltonian.

        Return
        ------
        self._ret : Qiskit Results.
                    Results of the VQE. It includes:
                    'aux_operator_eigenvalues'
                    'cost_function_evals'
                    'eigenstate'
                    'eigenvalue'
                    'optimal_point'
                    'optimal_value'
                    'optimizer_evals'
                    'optimizer_time'
        """

        operator = self._check_operator(operator)

        self._expect_op = self.construct_expectation(self._ansatz_params, operator)

        # Convert the gradient operator into a callable function that is compatible with the
        # optimization routine.
        if self._gradient:
            if isinstance(self._gradient, GradientBase):
                self._gradient = self._gradient.gradient_wrapper(~StateFn(operator) @ StateFn(self._ansatz),
                                                                 bind_params=self._ansatz_params,
                                                                 backend=self._quantum_instance)

        vqresult = self.find_minimum(initial_point=self._initial_point, ansatz=self._ansatz,
                                     cost_fn=self._energy_evaluation, gradient_fn=self._gradient,
                                     optimizer=self._optimizer)

        self._ret = VQEResult()
        self._ret.combine(vqresult)

        if vqresult.optimizer_evals is not None and self._eval_count >= vqresult.optimizer_evals:
            self._eval_count = vqresult.optimizer_evals
        self._eval_time = vqresult.optimizer_time
        logger.info('Optimization complete in %s seconds.\nFound opt_params %s in %s evals', self._eval_time,
                    vqresult.optimal_point, self._eval_count)

        self._ret.eigenvalue = vqresult.optimal_value + 0j
        self._ret.eigenstate = self.get_optimal_vector()
        self._ret.eigenvalue = self.get_optimal_cost()
        self._ret.cost_function_evals = self._eval_count

        return self._ret

    def find_minimum(self, initial_point: Optional[np.ndarray] = None, ansatz: Optional[QuantumCircuit] = None,
                     cost_fn: Optional[Callable] = None, optimizer: Optional[Optimizer] = None,
                     gradient_fn: Optional[Callable] = None, ) -> "VariationalResult":
        """Optimize to find the minimum cost value.

        Args:
            initial_point: If not `None` will be used instead of any initial point supplied via
                constructor. If `None` and `None` was supplied to constructor then a random
                point will be used if the optimizer requires an initial point.
            ansatz: If not `None` will be used instead of any ansatz supplied via constructor.
            cost_fn: If not `None` will be used instead of any cost_fn supplied via
                constructor.
            optimizer: If not `None` will be used instead of any optimizer supplied via
                constructor.
            gradient_fn: Optional gradient function for optimizer

        Returns:
            dict: Optimized variational parameters, and corresponding minimum cost value.

        Raises:
            ValueError: invalid input

        """
        initial_point = initial_point if initial_point is not None else self._initial_point
        ansatz = ansatz if ansatz is not None else self._ansatz
        cost_fn = cost_fn if cost_fn is not None else self._cost_fn
        optimizer = optimizer if optimizer is not None else self._optimizer

        if ansatz is None:
            raise ValueError("Ansatz neither supplied to constructor nor find minimum.")
        if cost_fn is None:
            raise ValueError("Cost function neither supplied to constructor nor find minimum.")
        if optimizer is None:
            raise ValueError("Optimizer neither supplied to constructor nor find minimum.")

        nparms = ansatz.num_parameters

        if hasattr(ansatz, "parameter_bounds") and ansatz.parameter_bounds is not None:
            bounds = ansatz.parameter_bounds
        else:
            bounds = [(None, None)] * nparms

        if initial_point is not None and len(initial_point) != nparms:
            raise ValueError("Initial point size {} and parameter size {} mismatch".format(len(initial_point), nparms))
        if len(bounds) != nparms:
            raise ValueError("Ansatz bounds size does not match parameter size")
        # If *any* value is *equal* in bounds array to None then the problem does *not* have bounds
        problem_has_bounds = not np.any(np.equal(bounds, None))
        # Check capabilities of the optimizer
        if problem_has_bounds:
            if not optimizer.is_bounds_supported:
                raise ValueError("Problem has bounds but optimizer does not support bounds")
        else:
            if optimizer.is_bounds_required:
                raise ValueError("Problem does not have bounds but optimizer requires bounds")
        if initial_point is not None:
            if not optimizer.is_initial_point_supported:
                raise ValueError("Optimizer does not support initial point")
        else:
            if optimizer.is_initial_point_required:
                if hasattr(ansatz, "preferred_init_points"):
                    # Note: default implementation returns None, hence check again after below
                    initial_point = ansatz.preferred_init_points

                if initial_point is None:  # If still None use a random generated point
                    low = [(l if l is not None else -2 * np.pi) for (l, u) in bounds]
                    high = [(u if u is not None else 2 * np.pi) for (l, u) in bounds]
                    initial_point = algorithm_globals.random.uniform(low, high)

        start = time.time()
        if not optimizer.is_gradient_supported:  # ignore the passed gradient function
            gradient_fn = None
        else:
            if not gradient_fn:
                gradient_fn = self._gradient

        logger.info("Starting optimizer.\nbounds=%s\ninitial point=%s", bounds, initial_point)
        opt_params, opt_val, num_optimizer_evals = optimizer.optimize(nparms, cost_fn, variable_bounds=bounds,
                                                                      initial_point=initial_point,
                                                                      gradient_function=gradient_fn, )
        eval_time = time.time() - start

        result = VariationalResult()
        result.optimizer_evals = num_optimizer_evals
        result.optimizer_time = eval_time
        result.optimal_value = opt_val
        result.optimal_point = opt_params
        result.optimal_parameters = dict(zip(self._ansatz_params, opt_params))

        return result

    # Get Optimal
    def get_optimal_cost(self) -> float:
        """Get the minimal cost or energy found by the VQE."""
        if self._ret.optimal_point is None:
            raise AlgorithmError("Cannot return optimal cost before running the "
                                 "algorithm to find optimal params.")
        return self._ret.optimal_value

    def get_optimal_circuit(self) -> QuantumCircuit:
        """Get the circuit with the optimal parameters."""
        if self._ret.optimal_point is None:
            raise AlgorithmError("Cannot find optimal circuit before running the "
                                 "algorithm to find optimal params.")
        return self._ansatz.assign_parameters(self._ret.optimal_parameters)

    def get_optimal_vector(self) -> Union[List[float], Dict[str, int]]:
        """Get the simulation outcome of the optimal circuit. """
        from qiskit.utils.run_circuits import find_regs_by_name

        if self._ret.optimal_point is None:
            raise AlgorithmError("Cannot find optimal vector before running the "
                                 "algorithm to find optimal params.")
        qc = self.get_optimal_circuit()
        if self._quantum_instance.is_statevector:
            ret = self._quantum_instance.execute(qc)
            min_vector = ret.get_statevector(qc)
        else:
            c = ClassicalRegister(qc.width(), name='c')
            q = find_regs_by_name(qc, 'q')
            qc.add_register(c)
            qc.barrier(q)
            qc.measure(q, c)
            ret = self._quantum_instance.execute(qc)
            counts = ret.get_counts(qc)
            # normalize, just as done in CircuitSampler.sample_circuits
            shots = self._quantum_instance._run_config.shots
            min_vector = {b: (v / shots) ** 0.5 for (b, v) in counts.items()}
        return min_vector

    @property
    def optimal_params(self) -> List[float]:
        """The optimal parameters for the ansatz."""
        if self._ret.optimal_point is None:
            raise AlgorithmError("Cannot find optimal params before running the algorithm.")
        return self._ret.optimal_point

    def total_time(self):
        return self._total_time


class VQEResult(VariationalResult, MinimumEigensolverResult):
    """ VQE Result."""

    def __init__(self) -> None:
        super().__init__()
        self._cost_function_evals = None

    @property
    def cost_function_evals(self) -> Optional[int]:
        """ Returns number of cost optimizer evaluations """
        return self._cost_function_evals

    @cost_function_evals.setter
    def cost_function_evals(self, value: int) -> None:
        """ Sets number of cost function evaluations """
        self._cost_function_evals = value
