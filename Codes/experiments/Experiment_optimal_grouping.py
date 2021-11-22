# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 02:39:13 2021

@author: lucia
"""

import sys

sys.path.append('../')

import numpy as np

from VQE import VQE
from utils import number2SummedOp

from qiskit import IBMQ
from qiskit import Aer
from qiskit.utils.quantum_instance import QuantumInstance
from qiskit.algorithms.optimizers import SPSA
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from qiskit.circuit.library import EfficientSU2

def callback(evals, params, mean, deviation):
    stream = getattr(sys, "stdout")
    #print("{}, {}".format(evals, mean), file=stream)
    stream.flush()
    
    
def run_VQE(solver, qubitOp, seed, nmax=100):
    np.random.seed(seed)

    solution = False
    while not solution and nmax > 0:
        try:
            solver.compute_minimum_eigenvalue(qubitOp)
            solution = True
        except Exception:
            print('Trying again...')
            nmax -= 1

    if not solution:
        return None

    return solver.energies

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-csic', group='internal', project='iff-csic')
name_backend = 'ibmq_montreal'
backend_device = provider.get_backend(name_backend)
backend_sim = Aer.get_backend('aer_simulator')
coupling_map = backend_device.configuration().coupling_map

qi = QuantumInstance(backend=backend_sim,
                    shots=2**13,
                    measurement_error_mitigation_cls=CompleteMeasFitter)

Molecules = [ 'H20', 'BeH2', 'LiH' ]

for molecule in Molecules:
    
    data = np.load('../data/optimal_grouping_' + molecule + '_' + name_backend + '.npy', allow_pickle=True).item()
    qubit_op = number2SummedOp(data['optimal_labels'], data['optimal_coeffs'])

    init_state = data['init_state']

    num_qubits = qubit_op.num_qubits
    ansatz = init_state.compose(EfficientSU2(num_qubits, ['ry', 'rz'], entanglement='linear', reps=1))
    num_var = ansatz.num_parameters
    initial_params = [0.1] * num_var

    maxiter = 100
    optimizer = SPSA(maxiter=maxiter, last_avg=1)

    solvers_TPB  = VQE(ansatz, optimizer, initial_params, grouping='TPB',
                        quantum_instance=qi, callback=callback) 

    solvers_EM   = VQE(ansatz, optimizer, initial_params, grouping='Entangled',
                        quantum_instance=qi, callback=callback) 

    solvers_HEEM = VQE(ansatz, optimizer, initial_params, grouping='Entangled',
                        quantum_instance=qi, connectivity=coupling_map, callback=callback) 

    results_TPB  = run_VQE(solvers_TPB, qubit_op, None)
    
    results_EM   = run_VQE(solvers_EM, qubit_op, None)
    
    results_HEEM = run_VQE(solvers_HEEM, qubit_op, None)
    
    np.save( 'results_'+molecule, np.array( [results_TPB, 
                                              results_EM,
                                              results_HEEM] ) )
    












    
    