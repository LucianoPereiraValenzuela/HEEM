"""
Variational Quantum Eigensolver algorithm with Hardware Efficient Entangled measurements.
"""

from typing import Optional, List, Callable, Union, Dict
import logging
from time import time
import numpy as np
from GroupingAlgorithm import *
from HEEM_VQE_Functions import *
from utils import Label2Chain
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.opflow import OperatorBase, StateFn, CircuitStateFn, ListOp, I
from qiskit.providers import Backend, BaseBackend
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import Parameter
from qiskit.utils.quantum_instance import QuantumInstance
from qiskit.opflow.gradients import GradientBase
from qiskit.circuit.library import RealAmplitudes
from qiskit.algorithms.optimizers import Optimizer, SLSQP
from qiskit.algorithms.variational_algorithm import VariationalAlgorithm, VariationalResult
from qiskit.algorithms.minimum_eigen_solvers.minimum_eigen_solver import MinimumEigensolver, MinimumEigensolverResult
from qiskit.algorithms.exceptions import AlgorithmError
logger = logging.getLogger(__name__)

class VQE(VariationalAlgorithm, MinimumEigensolver):
    """
    Class of the Variationa Quantum Eigensolver.
    """
    
    
    def __init__(self,
                 ansatz        : Optional[QuantumCircuit] = None,
                 optimizer     : Optional[Optimizer] = None,
                 initial_point : Optional[np.ndarray] = None,
                 gradient      : Optional[Union[GradientBase, Callable]] = None,
                 grouping      : Optional[str] = 'TPB',
                 order         : Optional[np.ndarray] = [4,3,2,1],
                 conectivity   : Optional[list] = None,
                 callback      : Optional[Callable[[int, np.ndarray], None]] = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = AerSimulator(method="statevector") ) -> None:
        """
        Parameters.
        ---------------------------
        ansatz           : A parameterized circuit used as Ansatz for the wave function. 
                           By default ansatz = RealAmplitudes(). 
        optimizer        : A classical optimizer. 
                           By default optimizer = SLSQP().
        initial_point    : An optional initial point (i.e. initial parameter values)
                           for the optimizer. If ``None`` then VQE will look to the ansatz for a preferred
                           point and if not will simply compute a random one.
        gradient         : An optional gradient function or operator for optimizer. (Not tested)
        grouping         : str
                           Method for grouping the local observables of the Hamiltonian into compatible measurements. 
                           Two methods are available: tensor product basis ('TPB') and entangled measurements ('Entangled').
        order            : list( int )
                           Priority of the bases when grouping local operators in entangled measurements. 
                           grouping = 'Entangled' is required. The available measurements are:
                            0 -> Identity
                            1 -> X
                            2 -> Y
                            3 -> Z
                            4 -> Bell
                            5 -> Omega_xx
                            6 -> Omega_yy
                            7 -> Omega_zz
                            8 -> Chi
                            9 -> Pi
        conectivity      : list( tuples ) or list( list )
                           The inter-qubit connectivity allowed for entangled measurements. 
                           grouping = 'Entangled' is required.
                           As example, consider the 4-qubits device
                               0--1--2
                                  |
                                  3
                           For this case we have: conectivity = [(0,1),(1,0),(1,2),(1,3),(2,1),(3,1)].
                           By default connection between all qubits is used. 
        callback         : Callable( int, np.ndarray )
                           A callback that can access the intermediate data during the optimization. 
                           The inputs are the number of evaluations and the evaluated energy.
        quantum_instance : Quantum Instance or Backend.
        """
        
        if ansatz is None:
            self.ansatz = RealAmplitudes() 
        else:
            self.ansatz = ansatz
        
        if optimizer is None:
            self.optimizer = SLSQP()
        
        # set the initial point to the preferred parameters of the ansatz
        if initial_point is None and hasattr(ansatz, 'preferred_init_points'):
            self.initial_point = ansatz.preferred_init_points
        
        super().__init__(ansatz    = ansatz,
                         optimizer = optimizer,
                         cost_fn   = self._energy_evaluation,
                         gradient  = gradient,
                         initial_point    = initial_point,
                         quantum_instance = quantum_instance)
        
        self._order      = order
        self._conectiviy = conectivity
        self._grouping   = grouping
        self._eval_count = 0
        self._callback   = callback
        logger.info(self.print_settings())
          
############################
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

        Returns:
            str: the formatted setting of VQE
        """
        ret = "\n"
        ret += "==================== Setting of {} ============================\n".format(
            self.__class__.__name__)
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
    
###### Checks #########
    def _check_operator_varform(self,
                                operator: OperatorBase):
        """Check that the number of qubits of operator and ansatz match."""
        if operator is not None and self.ansatz is not None:
            if operator.num_qubits != self.ansatz.num_qubits:
                # try to set the number of qubits on the ansatz, if possible
                try:
                    self.ansatz.num_qubits = operator.num_qubits
                    self._ansatz_params = sorted(self.ansatz.parameters, key=lambda p: p.name)
                except AttributeError as ex:
                    raise AlgorithmError("The number of qubits of the ansatz does not match the "
                                         "operator, and the ansatz does not allow setting the "
                                         "number of qubits using `num_qubits`.") from ex
    
    def _check_operator(self, operator: OperatorBase) -> OperatorBase:
        """ set operator """
        self._expect_op = None
        self._check_operator_varform(operator)
        return operator
#######################    
    
    
    def construct_expectation( self,
                             parameter, 
                             operator
                             ):
        """
        Generate the ansatz circuit and expectation value measurement, and return their
        runnable composition. 
        
        parameters.
        ---------------------------
        parameter : Parameters for the ansatz circuit.
        operator  : Qubit operator of the Observable.
        
        Returns
        -----------------------
        circuits  : qiskit circuits to evaluate the energy.
        """
        
        if isinstance(self.ansatz, QuantumCircuit):
            param_dict = dict(zip(self._ansatz_params, parameter))  # type: Dict
            wave_function = self.ansatz.assign_parameters(param_dict)
        else:
            wave_function = self.ansatz.construct_circuit(parameter)
            
        paulis, self._coeff, self._label = Label2Chain(operator)
        num_qubits = operator.num_qubits
        
        if self._grouping == 'Entangled':
            if self._conectiviy is None :
                self._conectiviy = list(permutations(list(range(num_qubits)),2))
            self._Groups, self._Measurements = grouping( paulis, self._order, self._conectiviy )
            
        elif self._grouping == 'TPB':
            _, self._Groups, self._Measurements = TPBgrouping( paulis )    
    
        circuits  = []
        n_measure = []
        for measure in self._Measurements :
            circuits_temp, n_measure_temp = measure_circuit_factor( measure , num_qubits )
            circuits.append( circuits_temp.compose( self.ansatz, front=True ) )
            n_measure.append( n_measure_temp )
            
        self._prob2Exp =  probability2expected( self._coeff, self._label, self._Groups, self._Measurements )
        
        return circuits
        
    def _circuit_sampler( self, expected_op, params ):

        expected_op = [ qci.assign_parameters(params) for qci in expected_op ] # Esto de deberóa poder hacer más eficiente!
#         t_qc   = transpile( expected_op, self.quantum_instance )
#         qc_obj = assemble( t_qc, shots=NUM_SHOTS)

        counts = self.quantum_instance.execute(expected_op).get_counts()
        
        probabilities = [ post_process_results(counts[j], expected_op[j].num_clbits, self.quantum_instance.backend.options.shots) for j in range(len(counts))]

        ExpectedValue = 0
        for j in range(len(probabilities)) :
            ExpectedValue += np.sum( self._prob2Exp[j]@probabilities[j] )

        if self._callback is not None :
            self._callback( ExpectedValue, params )
            
        return ExpectedValue
        
    def _energy_evaluation( self, parameters ) :

        num_parameters = self.ansatz.num_parameters
        if num_parameters == 0:
            raise RuntimeError('The ansatz cannot have 0 parameters.')

        start_time = time()

        means = self._circuit_sampler( self._expect_op, params=parameters )
        
        self._eval_count += 1
        
        end_time = time()
        logger.info('Energy evaluation returned %s - %.5f (ms), eval count: %s', means, (end_time - start_time) * 1000, self._eval_count)
        
        return means
        

    def compute_minimum_eigenvalue(
                                    self,
                                    operator: OperatorBase,
                                    ) -> MinimumEigensolverResult:
        
        
        operator = self._check_operator(operator)
        
        self._expect_op = self.construct_expectation(self._ansatz_params, operator)
        
        # Convert the gradient operator into a callable function that is compatible with the
        # optimization routine.
        if self._gradient:
            if isinstance(self._gradient, GradientBase):
                self._gradient = self._gradient.gradient_wrapper(
                    ~StateFn(operator) @ StateFn(self._ansatz),
                    bind_params=self._ansatz_params,
                    backend=self._quantum_instance)
        
        vqresult = self.find_minimum(initial_point=self.initial_point,
                                     ansatz=self.ansatz,
                                     cost_fn=self._energy_evaluation,
                                     gradient_fn=self._gradient,
                                     optimizer=self.optimizer)
        
        self._ret = VQEResult()
        self._ret.combine(vqresult)
    
        if vqresult.optimizer_evals is not None and \
                self._eval_count >= vqresult.optimizer_evals:
            self._eval_count = vqresult.optimizer_evals
        self._eval_time = vqresult.optimizer_time
        logger.info('Optimization complete in %s seconds.\nFound opt_params %s in %s evals',
                    self._eval_time, vqresult.optimal_point, self._eval_count)

        self._ret.eigenvalue = vqresult.optimal_value + 0j
        self._ret.eigenstate = self.get_optimal_vector()
        self._ret.eigenvalue = self.get_optimal_cost()
        self._ret.cost_function_evals = self._eval_count

        return self._ret
        
    
######### Get Optimal #################    
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
        return self.ansatz.assign_parameters(self._ret.optimal_parameters)


    def get_optimal_vector(self) -> Union[List[float], Dict[str, int]]:
        """Get the simulation outcome of the optimal circuit. """
        from qiskit.utils.run_circuits import find_regs_by_name

        if self._ret.optimal_point is None:
            raise AlgorithmError("Cannot find optimal vector before running the "
                                 "algorithm to find optimal params.")
        qc = self.get_optimal_circuit()
        min_vector = {}
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
        
##################################################
        
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
        
    
#     ###################################################################
    





















