import numpy as np
import networkx as nx
from itertools import permutations
from qiskit import IBMQ
from qiskit.opflow.primitive_ops import PauliOp
from qiskit.opflow.list_ops import SummedOp
from qiskit.quantum_info import Pauli
from qiskit.opflow.primitive_ops.pauli_sum_op import PauliSumOp
from qiskit.opflow.primitive_ops.tapered_pauli_sum_op import TaperedPauliSumOp


Comp=[]
Comp.append([])#This empty list is just to fix notation
Comp.append([[0],[1]])
Comp.append([[0],[2]])
Comp.append([[0],[3]])
Comp.append([[0,0],[1,1],[2,2],[3,3]])
Comp.append([[0,0],[1,1],[2,3],[3,2]])
Comp.append([[0,0],[2,2],[1,3],[3,1]])
Comp.append([[0,0],[3,3],[1,2],[2,1]])
Comp.append([[0,0],[1,2],[2,3],[3,1]])
Comp.append([[0,0],[2,1],[3,2],[1,3]])
length=[]
length.append([])#This empty list is just to fix notation
length.append(1)
length.append(1)
length.append(1)
length.append(2)
length.append(2)
length.append(2)
length.append(2)
length.append(2)
length.append(2)

def PauliGraph(PS):#PS==Pauli Strings. AM=Admisible Measurements. WC==Well Connected Qubits. 
    #    If we want to group n Pauli arrays of size N, PS should be a matrix of n rows and N columns,
    # each row representing a Pauli string.
    n=np.size(PS[:,0])
    N=np.size(PS[0,:])
    G = nx.Graph()
    G.add_nodes_from(np.arange(n))
    for i in range(n):
        v_i=PS[i,:]
        for j in range(i+1,n):
            v_j=PS[j,:]
            qubits=np.arange(N)
            noncommonqubits=np.delete(qubits,np.argwhere(v_i==v_j))
            vi=v_i[noncommonqubits]
            vj=v_j[noncommonqubits]
            if (vi*vj!=0).any():
                G.add_edges_from([(i,j)])
    return G


def LDFC(PG):
    SV=sorted(PG.degree, key=lambda x: x[1], reverse=True)#Sorted Vertices by decreasing degree
    n=PG.number_of_nodes()
    aux=list(np.arange(n))
    Color=n*np.ones(n)
    for i in range(n):
        IV=list(list(PG.neighbors(SV[i][0])))#Vertices that are Incompatible with vertex SV[i][0]
        IC=Color[IV]#Colors that are assigned to vertices that are incompatible with vertex SV[i]
        AC=[ elem for elem in aux if elem not in IC]#Available colors for vertex SV[i]
        Color[SV[i][0]]=min(AC)
    MC=int(max(Color))
    Groups=[]
    for i in range(MC+1):
        Groups.append(list(np.argwhere(Color==i)))
    return Color, Groups #Color is an array whose i entry has the color assigned to the i Pauli String.
    #Groups is a list of lists, where the i list comprenhends the arrays assigned to the color i.
        
def TPBgrouping(PS): #PS==Pauli Strings. AM=Admisible Measurements. WC==Well Connected Qubits. 
    #    If we want to group n Pauli arrays of size N, PS should be a matrix of n rows and N columns,
    # each row representing a Pauli string.
    PG=PauliGraph(PS)
    Color, Groups=LDFC(PG)
    N=np.size(PS[0,:])
    Measurements=[]#The list of measurements. Each element will be the total measurement for a certain group. That measurement 
    #will be encoded as an N-array of {0,1,3,4}. 0 will appear in the position k if in the qubit k we can measure with any 
    # basis (will only happen if the k factor of every element of the group is I), 0 will appear in the position k if in the qubit k
    #we can measure with TPBX,...
    for i in range(len(Groups)):
        Mi=[]
        for k in range(N):
            Mi.append(max(PS[Groups[i],k]))
        Measurements.append(Mi)
    return Color, Groups, Measurements


def MeasurementAssignment(Vi,Vj,Mi,AM,WC):#This program is the Algorithm 2 of https://arxiv.org/abs/1909.09119. Syntax can
    #be looked in 'grouping(PS,AM,WC)'
    
    # Let's first check for compatibility of Vj with the current assigment of Mi.
    # Mi is a list of local measurement. Each local measurement is encoded as list of two elements. The first one 
    # are the qubits where the local measurement acts and the second is the type of local measurement. For example,
    # if Mi contains {4,(1,2)} it would mean that Mi has the Bell measurement (nº4) as the local measurement acting on 
    # the qubits (1,2).
    N=np.size(Vi)
    U=list(np.arange(N))
    S=0
    for LM in Mi:
        if list(Vj[LM[1]]) not in Comp[LM[0]]:
            return Mi, S
        else:
            for s in LM[1]:### SEGURO QUE HAY UNA FORMA MÁS RÁPIDA DE ELIMINAR VARIOS VALORES A LA VEZ DE LA LISTA
                U.remove(s)
    commonfactors=np.argwhere(Vi==Vj)
    for k in commonfactors:
        if k in U:
            U.remove(k)
    PMi=Mi[:] #I create a potential Mi.
    while len(U)!=0:
        for Eps in AM:
            if len(U)>=length[Eps]:
                perm=list(permutations(U,length[Eps])) #length of each local measurement will be manually programmed
                perm=list({*map(tuple, map(sorted, perm))}) #This is a code for eliminating the permutations that
                #are equal up to order for perm. This would reduce the iterations (I believe) without affecting the algorithm,
                #because the WC array will contain all possible permutations, even those that are equal with disitinct order.
                #and if the qubits (l,k) of Vi and Vj are compatible with a certain measurement, the qubits (k,l) of Vi and 
                #Vj will be compatible with other measurement. I should explain this better. 
                for per in perm:
                    if per in WC: 
                    #This raises an error, so here I have to check the code.
                        if (list(Vi[[per]]) in Comp[Eps]) and (list(Vj[[per]]) in Comp[Eps]):
                            PMi.append([Eps,list(per)])
                            for s in per:
                                U.remove(s)
                            break
                else:
                    continue
                break
        else:
            return Mi, S
    S=1
    return PMi, S            

def grouping(PS, AM, WC): #PS==Pauli Strings. AM=Admisible Measurements. WC==Well Connected Qubits. 
    #    If we want to group n Pauli arrays of size N, PS should be a matrix of n rows and N columns,
    # each row representing a Pauli string. 
    #    AM should be a vector containing the admisible measurements in the order of prefered assignenment. 
    #    WC should be a vector containing the pairs of qubits with good connectivity.
    PG=PauliGraph(PS)
    SV=sorted(PG.degree, key=lambda x: x[1], reverse=True)#Sorted Vertices by decreasing degree
    n=np.size(PS[:,0])
    N=np.size(PS[0,:])
    AS=[]#list of strings with assigned measurement
    Groups=[]#list of groups
    Measurements=[]#list of total measurements Mi
    for k in range(n):
        i=SV[k][0]
        if i not in AS:
            Mi=[]#Mi will be the total measurement. It will be a list of local measurements. Each local measurement
            #will appear as a list of two elements. The first will correspond with the local measurement and the second
            # to the qubits. For example, if Mi contains {4,(1,2)} it would mean that Mi has the Bell measurement (nº4)
            #as the local measurement acting on the qubits (1,2)
            GroupMi=[i]
            AS.append(i)
            for l in range(n):
                j=SV[l][0]
                if j not in AS:
                    Mi, S=MeasurementAssignment(PS[i,:],PS[j,:],Mi,AM,WC)#S is the success variable. If Mi is compatible with
                    #Vj S=1 otherwise S=0
                    if S==1:
                        AS.append(j)
                        GroupMi.append(j)
            QWM=list(np.arange(N))#Qubits Without a Measurement assigned by Mi. There, all factors 
            # of the group will be equal or the identity, so we will have to use a TPB measurement.
            for LM in Mi:
                for s in LM[1]:
                    QWM.remove(s)
            for q in QWM:
                TPBq=max(PS[GroupMi,q])
                Mi.append([TPBq,[q]])
            Groups.append(GroupMi)
            Measurements.append(Mi)
            
    return Groups, Measurements


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
    
    if type( QubitOp ) == PauliSumOp or type( QubitOp) == TaperedPauliSumOp:
        QubitOp = qubit_op.to_pauli_op()
        
    ops = [[ Dict.get(idx2) for idx2 in idx.primitive.to_label()] for idx in QubitOp.oplist ]
    coef = [ idx.coeff for idx in QubitOp.oplist ]        
    
    return np.array(ops), coef