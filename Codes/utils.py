import numpy as np
from qiskit import IBMQ
from qiskit.opflow.primitive_ops import PauliOp
from qiskit.opflow.list_ops import SummedOp
from qiskit.quantum_info import Pauli
from qiskit.opflow.primitive_ops.pauli_sum_op import PauliSumOp
from qiskit.opflow.primitive_ops.tapered_pauli_sum_op import TaperedPauliSumOp

def HeisenbergHamiltonian( J=1, H=1, num_qubits=2, neighbours=[(0,1)] ):
    """
    Qiskit operator of the 3-D Heisemberg Hamiltonian of a lattice of spins.
    
    H = - J Σ_j ( X_j X_{j+1} + Y_j Y_{j+1} + Z_j Z_{j+1} ) - H Σ_j Z_j
    
    input:
        J          : Real. Coupling constant.
        H          : Real. External magnetic field.
        num_qubits : Integer. Number of qubits.
        neighbours : List of tuples. Coupling between the spins.
    output:
        Hamiltonian : SummedOp of Qiskit. Heisenberg Hamiltonian of the system.
    """
    num_op = num_qubits + 3*len(neighbours)
    Hamiltonian_op_x = []    
    Hamiltonian_op_z = []  
    Hamiltonian_coef = num_qubits*[-H] + num_op*[-J]
    
    for idx in range(num_qubits):
        op_x = np.zeros( num_qubits )
        op_z = np.zeros( num_qubits )
        op_z[idx] = 1
        Hamiltonian_op_x.append( op_x.copy() )
        Hamiltonian_op_z.append( op_z.copy() )        
    
    for idx in neighbours:
        op_x = np.zeros( num_qubits )
        op_z = np.zeros( num_qubits )
        op_x[idx[0]] = 1
        op_x[idx[1]] = 1
        Hamiltonian_op_x.append( op_x.copy() )
        Hamiltonian_op_z.append( op_z.copy() )
        op_z[idx[0]] = 1
        op_z[idx[1]] = 1
        Hamiltonian_op_x.append( op_x.copy() )
        Hamiltonian_op_z.append( op_z.copy() )        
        op_x[idx[0]] = 0
        op_x[idx[1]] = 0
        Hamiltonian_op_x.append( op_x.copy() )
        Hamiltonian_op_z.append( op_z.copy() )        
#     Hamiltonian = WeightedPauliOperator( 
#         [ [Hamiltonian_coef[j], Pauli( ( Hamiltonian_op_z[j], Hamiltonian_op_x[j] )) ] 
#          for j in range(num_op) ] )

    Hamiltonian = SummedOp( [ PauliOp(Pauli( ( Hamiltonian_op_z[j], Hamiltonian_op_x[j] )),Hamiltonian_coef[j]) for j in range(num_op) ] )

    return Hamiltonian


def RandomHamiltonian( num_qubits=2, num_paulis=4 ):
    
    idxs = np.random.randint(2, size=(2,num_qubits,num_paulis) )

    Hamiltonian = SummedOp( [ PauliOp(Pauli( ( idxs[0,:,j], idxs[1,:,j] )),1) for j in range(num_paulis) ] )
    
    return Hamiltonian




def Label2Chain(QubitOp):
    """
    Transform a string of Pauli matrices into a numpy array.
    'I' --> 0
    'X' --> 1
    'Y' --> 2
    'Z' --> 3
    
    input:
        QubitOp : SummedOp of Qiskit.
    output:
        ops     : narray of the Pauli operators.
                  ops.shape = ( number_of_operators, number_of_qubits )
        coef    : coefficient of each Pauli operator.
    """
    Dict = {'I' : 0,
            'X' : 1,
            'Y' : 2,
            'Z' : 3}
    
    if type( QubitOp ) == PauliSumOp or type( QubitOp ) == TaperedPauliSumOp:
        QubitOp = QubitOp.to_pauli_op()
        
    label = []
    ops   = []
    coef  = []
    
    for idx in QubitOp.oplist :
        label_temp = idx.primitive.to_label() 
        label.append(label_temp)
        ops.append( [ Dict.get(idx) for idx in label_temp ])
        coef.append(idx.coeff)      
    
    return np.array(ops), coef, label

def get_backend_conectivity(backend):
    """
    Get the conected qubit of q backend. Has to be a quantum computer.
    """
    defaults = backend.defaults()
    conexions = [ indx for indx in defaults.instruction_schedule_map.qubits_with_instruction('cx') ]
    return conexions

