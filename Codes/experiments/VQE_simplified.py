import sys 
sys.path.append('../') 
import numpy as np 
from utils import * 
from GroupingAlgorithm import grouping, groupingWithOrder 
from HEEM_VQE_Functions import * 
from qiskit.circuit.library import EfficientSU2 
from qiskit.compiler import transpile 

class VQE :

    def __init__( self, 
                    circuit_init,
                    optimizer,
                    params,
                    backend,
                    shots = 2**10,
                    grouping = 'HEEM',
                    conectivity = None,
                    back = None ):

        self.circuit_init = circuit_init
        #self._num_params = variational_circuit.num_parameters 
        self._optimizer = optimizer
        self._params = params 
        self._grouping = grouping
        self._backend = backend
        self._shots = shots

        if conectivity is None :
            conectivity = get_backend_connectivity( backend )
        self._conectivity = conectivity
        self._back = back


    def compute_minimum_eigenvalue( self, H ):

        self.circuits( H )

        results = self._optimizer.optimize( self._num_params, self.energy_evaluation, initial_point=self._params)

        return results

    def energy_evaluation( self, params ):
        circuits_tp = [ circuit.assign_parameters(params) for circuit in self._circuits ]
        circuits_tp = transpile( circuits_tp, backend=self._backend )
        counts = self._backend.run( circuits_tp, shots=self._shots ).result().get_counts()
        probs = [ post_process_results( count, self._num_qubits, self._shots ) for count in counts ]
        ExpVal = 0
        for j in range(len(probs)):
            ExpVal += np.sum(self._prob2Exp[j]@probs[j])

        if self._back is not None:
            self._back( ExpVal, params ) 

        return ExpVal

    def grouping( self, H ):

        self._num_qubits = H.num_qubits 
        paulis, coeff, labels = Label2Chain( H )

        if self._grouping == 'TPB' :
            Color, self._Groups, self._Measurements = TPBgrouping( paulis )
            self._layaout = range(self._num_qubits)
        elif self._grouping == 'EM' :
            self._Groups, self._Measurements, self._layaout = groupingWithOrder( paulis )
            self._layaout = self._layaout[::-1]
        elif self._grouping == 'HEEM' :
            self._Groups, self._Measurements, self._layaout = groupingWithOrder( paulis, G=self._conectivity )
            self._layaout = self._layaout[::-1]

        self._prob2Exp = probability2expected( coeff, labels, self._Groups, self._Measurements)



    def circuits( self, H ):

        self.grouping( H )

        self._variational_circuit = self.hardware_efficient_circuit()

        self._circuits = [ transpile( measure_circuit_factor( measure , self._num_qubits 
                            ), initial_layout=self._layaout ).compose( self._variational_circuit, front=True  )
                            for measure in self._Measurements ]  


    def hardware_efficient_circuit( self ):
        
        WC = [ indx for indx in self._conectivity if indx[0]<indx[1] ]
        variational_circuit = self.circuit_init.compose( EfficientSU2( self._num_qubits, reps=1, entanglement=WC ) ) 
        self._num_params = variational_circuit.num_parameters 
        return variational_circuit







