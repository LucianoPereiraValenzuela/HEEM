import numpy as np
import matplotlib.pyplot as plt
from VQE import VQE
from GroupingAlgorithms import *
# Importing standard Qiskit libraries
from qiskit import IBMQ
from qiskit.providers.aer import AerSimulator
from qiskit.circuit.library import EfficientSU2
from qiskit.utils.quantum_instance import QuantumInstance
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit_nature.circuit.library import HartreeFock
from qiskit_nature.transformers import FreezeCoreTransformer
from qiskit_nature.drivers import PyQuanteDriver
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.mappers.second_quantization import ParityMapper
from qiskit_nature.converters.second_quantization.qubit_converter import QubitConverter
from qiskit.algorithms.optimizers import SPSA

from joblib import Parallel, delayed
from palettable.cartocolors.sequential import BluGrn_7, OrYel_7, Magenta_7
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

#%%

def molecule(d, quantum_instance, conectivity):
    
    E = np.zeros(4)
    
    molecule = 'Li 0.0 0.0 0.0; H 0.0 0.0 ' + str(d)
    driver = PyQuanteDriver(molecule)
    qmolecule = driver.run()
    freezeCoreTransfomer = FreezeCoreTransformer( freeze_core=True, remove_orbitals= [3,4] )
    problem = ElectronicStructureProblem(driver,q_molecule_transformers=[freezeCoreTransfomer])
    
    # Generate the second-quantized operators
    second_q_ops = problem.second_q_ops()
    
    # Hamiltonian
    main_op = second_q_ops[0]
    
    # Setup the mapper and qubit converter
    mapper_type = 'ParityMapper'
    mapper = ParityMapper()
    converter = QubitConverter( mapper=mapper, two_qubit_reduction=True) #1] 
    
    # The fermionic operators are mapped to qubit operators
    num_particles = (problem.molecule_data_transformed.num_alpha, problem.molecule_data_transformed.num_beta)
    num_spin_orbitals = 2 * problem.molecule_data_transformed.num_molecular_orbitals
    qubit_op = converter.convert(main_op, num_particles=num_particles)
    num_qubits = qubit_op.num_qubits

    init_state = HartreeFock(num_spin_orbitals, num_particles, converter)
    
    # Exact energy
    E[0] = NumPyMinimumEigensolver().compute_minimum_eigenvalue(qubit_op).eigenvalue.real 
    
    # VQE
    optimizer = SPSA( maxiter=100, last_avg=1 )    
    ansatz = init_state.compose( EfficientSU2(num_qubits, ['ry','rz'], entanglement = 'linear', reps=1 ) )
    num_var = ansatz.num_parameters
    initial_params = [0.01] * num_var
    
    solver_TPB  = VQE( ansatz , optimizer, initial_params, grouping = 'TPB', quantum_instance = quantum_instance )
    solver_ENT  = VQE( ansatz , optimizer, initial_params, grouping = 'Entangled', quantum_instance = quantum_instance)
    solver_HEEM = VQE( ansatz , optimizer, initial_params, grouping = 'Entangled', conectivity = conectivity, quantum_instance = quantum_instance )
    
    E[1]  = solver_TPB.compute_minimum_eigenvalue(qubit_op).eigenvalue.real
    E[2]  = solver_ENT.compute_minimum_eigenvalue(qubit_op).eigenvalue.real
    E[3]= solver_HEEM.compute_minimum_eigenvalue(qubit_op).eigenvalue.real
    
    return E


#%%
def molecule_potentials_comparison(distances, quantum_instance, conectivity, file_name_out):
    
    E_EXACT = np.zeros_like( distances )
    E_TPB   = np.zeros_like( distances )
    E_ENT   = np.zeros_like( distances )
    E_HEEM  = np.zeros_like( distances )
    
    results = Parallel( n_jobs = len(distances) )(delayed(molecule)(d, quantum_instance, conectivity) for d in distances)
    for i in range(len(distances)):
        E_EXACT[i] = results[i][0]
        E_TPB[i]   = results[i][1]
        E_ENT[i]   = results[i][2]
        E_HEEM[i]  = results[i][3]

    # Save results
    np.savez(file_name_out, distances = distances, E_EXACT = E_EXACT, E_TPB = E_TPB, E_ENT = E_ENT, E_HEEM = E_HEEM) 
    
    return None

#%% Test - molecule_potentials_comparison
IBMQ.load_account()
provider      = IBMQ.get_provider(hub='ibm-q', group='open', project='main') 
backend_santiago = provider.get_backend('ibmq_santiago')

distances     = [0.5, 1.5, 5]#np.linspace(0.5,5,3)
backend_noise = AerSimulator.from_backend(backend_santiago)
backend_sim   = AerSimulator(method="statevector") # Backend for simulation
conectivity   = get_backend_conectivity(backend_santiago)
NUM_SHOTS     = 2**13  # Number of shots for each circuit
qi            = QuantumInstance( backend_sim, shots = NUM_SHOTS )

file_name_out = 'data\LiH_' + qi.backend_name + '_NUM_SHOTS=' + str(NUM_SHOTS) + '_dist=' + str(distances[0]) + '_to_' + str(distances[-1]) 

molecule_potentials_comparison(distances, qi, conectivity, file_name_out)

#%% Load and print data
data = np.load(file_name_out + '.npz')

E_EXACT = data['E_EXACT']
E_TPB   = data['E_TPB'  ]
E_ENT   = data['E_ENT'  ]
E_HEEM  = data['E_HEEM' ]

fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(6, 4.5))
ax.set_xlabel(r'$d$ ')
ax.set_ylabel(r'$E$ ')

ax.plot(distances,E_TPB , color = OrYel_7.mpl_colors[-3] )
ax.plot(distances,E_ENT , color = Magenta_7.mpl_colors[-3])
ax.plot(distances,E_HEEM , color = BluGrn_7.mpl_colors[-3])
ax.plot(distances,E_EXACT , color = 'black', linestyle = '--')

ax.legend(['$TPB$','$ENT$','$HEEM$'])
    
plt.savefig("Figures\E vs d.pdf")
