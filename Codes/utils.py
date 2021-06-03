import numpy as np
from qiskit.opflow.primitive_ops import PauliOp
from qiskit.opflow.list_ops import SummedOp
from qiskit.quantum_info import Pauli
from qiskit.opflow.primitive_ops.pauli_sum_op import PauliSumOp
from qiskit.opflow.primitive_ops.tapered_pauli_sum_op import TaperedPauliSumOp
from qiskit_nature.circuit.library import HartreeFock
from qiskit_nature.transformers import FreezeCoreTransformer
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.mappers.second_quantization import ParityMapper, JordanWignerMapper, BravyiKitaevMapper
from qiskit_nature.converters.second_quantization.qubit_converter import QubitConverter
from qiskit_nature.drivers import PySCFDriver


def HeisenbergHamiltonian(J=1, H=1, num_qubits=2, neighbours=[(0, 1)]):
	"""
	Qiskit operator of the 3-D Heisemberg Hamiltonian of a lattice of spins.

	H = - J Σ_j ( X_j X_{j+1} + Y_j Y_{j+1} + Z_j Z_{j+1} ) - H Σ_j Z_j

	Parameters
	----------
	J: float
		Coupling constant.
	H: float
		External magnetic field.
	num_qubits: int.
		Number of qubits.
	neighbours: list(tuples).
		Coupling between the spins.

	Return
	------
	Hamiltonian: SummedOp
		Heisenberg Hamiltonian of the system.
	"""
	num_op = num_qubits + 3 * len(neighbours)
	Hamiltonian_op_x = []
	Hamiltonian_op_z = []
	Hamiltonian_coef = num_qubits * [-H] + num_op * [-J]

	for idx in range(num_qubits):
		op_x = np.zeros(num_qubits)
		op_z = np.zeros(num_qubits)
		op_z[idx] = 1
		Hamiltonian_op_x.append(op_x.copy())
		Hamiltonian_op_z.append(op_z.copy())

	for idx in neighbours:
		op_x = np.zeros(num_qubits)
		op_z = np.zeros(num_qubits)
		op_x[idx[0]] = 1
		op_x[idx[1]] = 1
		Hamiltonian_op_x.append(op_x.copy())
		Hamiltonian_op_z.append(op_z.copy())
		op_z[idx[0]] = 1
		op_z[idx[1]] = 1
		Hamiltonian_op_x.append(op_x.copy())
		Hamiltonian_op_z.append(op_z.copy())
		op_x[idx[0]] = 0
		op_x[idx[1]] = 0
		Hamiltonian_op_x.append(op_x.copy())
		Hamiltonian_op_z.append(op_z.copy())

	Hamiltonian = SummedOp(
		[PauliOp(Pauli((Hamiltonian_op_z[j], Hamiltonian_op_x[j])), Hamiltonian_coef[j]) for j in range(num_op)])

	return Hamiltonian


def RandomHamiltonian(num_qubits=2, num_paulis=4):
	idxs = np.random.randint(2, size=(2, num_qubits, num_paulis))

	Hamiltonian = SummedOp([PauliOp(Pauli((idxs[0, :, j], idxs[1, :, j])), 1) for j in range(num_paulis)])

	return Hamiltonian


def Label2Chain(QubitOp):
	"""
	Transform a string of Pauli matrices into a numpy array.
	'I' --> 0
	'X' --> 1
	'Y' --> 2
	'Z' --> 3

	Parameters
	----------
	QubitOp: SummedOp.

	Returns
	-------
	ops: ndarray(Pauli operators) (number_of_operators, number_of_qubits)
	coef: list(float)
		Coefficients of each Pauli operator.
	label: list(str)
		Pauli strings
	"""
	Dict = {'I': 0,
	        'X': 1,
	        'Y': 2,
	        'Z': 3}

	if type(QubitOp) == PauliSumOp or type(QubitOp) == TaperedPauliSumOp:
		QubitOp = QubitOp.to_pauli_op()

	label = []
	ops = []
	coef = []

	for idx in QubitOp.oplist:
		label_temp = idx.primitive.to_label()
		label.append(label_temp)
		ops.append([Dict.get(idx) for idx in label_temp])
		coef.append(idx.coeff)

	return np.array(ops), coef, label


def from_string_to_numbers(pauli_labels):
	"""
	Function that transform a set of pauli string from the str convention ('IXYZ'), to the number convention (0123).

	Parameter
	---------
	pauli_labels: list
		List with the pauli string written as a string.

	Return
	------
	PS: array
		Pauli strings in the number convention.
	"""
	map_str_int = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}  # Map between str and int conventions
	PS = []  # Initialize the Pauli string for int convention

	for label in pauli_labels:  # Iterate over all the given pauli strings
		temp = []
		for letter in label:  # Map each element of a Pauli string
			temp.append(map_str_int[letter])

		PS.append(np.array(temp))
	return np.array(PS)


def get_backend_conectivity(backend):
	"""
	Get the connected qubit of q backend. Has to be a quantum computer.

	Parameters
	----------
	backend: qiskit.backend

	Return
	------
	connexions: (list)
		List with the connected qubits
	"""
	defaults = backend.defaults()
	connexions = [indx for indx in defaults.instruction_schedule_map.qubits_with_instruction('cx')]
	return connexions


def LiH(distance=1.5474, freeze_core=True, remove_orbitals=True, initial_state=False, mapper_type='ParityMapper'):
	"""
	Qiskit operator of the LiH

	Parameters
	----------
	distance: float (optional)
		Distance between atoms of Li and H
	freeze_core: Bool (optional)
		If freeze some cores that do highly impact in the energy
	remove_orbitals: Bool (optional)
		Remove some orbitals that do no impact in the energy
	initial_state: Bool (optional)
		Return the initial Hartree Fock state
	mapper_type: str (optional)
		Type of mapping between orbitals and qubits. Available options:
			'ParityMapper'
			'JordanWignerMapper'
			'BravyiKitaevMapper'

	Returns
	-------
	qubit_op: SummedOp
		Pauli strings and coefficients for the Hamiltonian
	init_state: QuantumCircuit (if initial_state=True)
		Quantum Circuit with the initial state given by Hartree Fock
	"""

	molecule = 'Li 0.0 0.0 0.0; H 0.0 0.0 ' + str(distance)

	try:
		driver = PySCFDriver(molecule)
	except:
		from qiskit_nature.drivers import PyQuanteDriver
		driver = PyQuanteDriver(molecule)

	qmolecule = driver.run()

	if remove_orbitals is True:
		freezeCoreTransfomer = FreezeCoreTransformer(freeze_core=freeze_core, remove_orbitals=[3, 4])
	else:
		freezeCoreTransfomer = FreezeCoreTransformer(freeze_core=freeze_core)

	problem = ElectronicStructureProblem(driver, q_molecule_transformers=[freezeCoreTransfomer])

	# Generate the second-quantized operators
	second_q_ops = problem.second_q_ops()

	# Hamiltonian
	main_op = second_q_ops[0]

	# Setup the mapper and qubit converter
	if mapper_type == 'ParityMapper':
		mapper = ParityMapper()
	elif mapper_type == 'JordanWignerMapper':
		mapper = JordanWignerMapper()
	elif mapper_type == 'BravyiKitaevMapper':
		mapper = BravyiKitaevMapper()

	# The fermionic operators are mapped
	converter = QubitConverter(mapper=mapper, two_qubit_reduction=True, z2symmetry_reduction='auto')

	# The fermionic operators are mapped to qubit operators
	num_particles = (problem.molecule_data_transformed.num_alpha, problem.molecule_data_transformed.num_beta)

	num_spin_orbitals = 2 * problem.molecule_data_transformed.num_molecular_orbitals

	qubit_op = converter.convert(main_op, num_particles=num_particles)

	if initial_state is False:
		return qubit_op
	else:
		init_state = HartreeFock(num_spin_orbitals, num_particles, converter)
		return qubit_op, init_state


def BeH2(distance=1.339, freeze_core=True, remove_orbitals=True, initial_state=False, mapper_type='ParityMapper'):
	"""
	Qiskit operator of the BeH2

	Parameters
	----------
	distance: float (optional)
		Distance between atoms of Be and H
	freeze_core: Bool (optional)
		If freeze some cores that do highly impact in the energy
	remove_orbitals: Bool (optional)
		Remove some orbitals that do no impact in the energy
	initial_state: Bool (optional)
		Return the initial Hartree Fock state
	mapper_type: str (optional)
		Type of mapping between orbitals and qubits. Available options:
			'ParityMapper'
			'JordanWignerMapper'
			'BravyiKitaevMapper'

	Returns
	-------
	qubit_op: SummedOp
		Pauli strings and coefficients for the Hamiltonian
	init_state: QuantumCircuit (if initial_state=True)
		Quantum Circuit with the initial state given by Hartree Fock
	"""

	molecule = 'H 0.0 0.0 -' + str(distance) + '; Be 0.0 0.0 0.0; H 0.0 0.0 ' + str(distance)

	try:
		driver = PySCFDriver(molecule)
	except:
		from qiskit_nature.drivers import PyQuanteDriver
		driver = PyQuanteDriver(molecule)

	qmolecule = driver.run()

	if remove_orbitals is True:
		freezeCoreTransfomer = FreezeCoreTransformer(freeze_core=freeze_core, remove_orbitals=[-1, -3])
	else:
		freezeCoreTransfomer = FreezeCoreTransformer(freeze_core=freeze_core)

	problem = ElectronicStructureProblem(driver, q_molecule_transformers=[freezeCoreTransfomer])

	# Generate the second-quantized operators
	second_q_ops = problem.second_q_ops()

	# Hamiltonian
	main_op = second_q_ops[0]

	# Setup the mapper and qubit converter
	if mapper_type == 'ParityMapper':
		mapper = ParityMapper()
	elif mapper_type == 'JordanWignerMapper':
		mapper = JordanWignerMapper()
	elif mapper_type == 'BravyiKitaevMapper':
		mapper = BravyiKitaevMapper()

	# The fermionic operators are mapped
	converter = QubitConverter(mapper=mapper, two_qubit_reduction=True, z2symmetry_reduction=[1, -1, 1])

	# The fermionic operators are mapped to qubit operators
	num_particles = (problem.molecule_data_transformed.num_alpha, problem.molecule_data_transformed.num_beta)

	num_spin_orbitals = 2 * problem.molecule_data_transformed.num_molecular_orbitals

	qubit_op = converter.convert(main_op, num_particles=num_particles)

	if initial_state is False:
		return qubit_op
	else:
		init_state = HartreeFock(num_spin_orbitals, num_particles, converter)
		return qubit_op, init_state


def unpack_functions(pack):
	"""
	Unpack the list where the first element is the index of the async execution, the second index in the function to
	run, the third index are the function variables, and the last index (if provided) are the optional arguments.

	Parameter
	---------
	pack: list
		List with all the data

	 Return
	 ------
	 Result of the function
	"""
	if len(pack) < 4:  # If no optional arguments are provided
		pack.append({})
	return [pack[0], pack[1](*pack[2], **pack[3])]


def sort_solution(data):
	"""
	Function to sort the data obtained for a parallel computation

	Parameter
	---------
	data: list
		List in which each entry represents one solution of the parallel computation. The elements are
		also list which contains in the first element the index and in the second one the result of the computation.

	Return
	------
	List with the data sorted
	"""
	n = len(data)  # Extract the number of computations done
	sorted_sol = [None] * n  # Empty list with the correct number of elements
	for i in range(n):  # Iterate over all the elements
		index = data[i][0]  # Obtain the index of the result
		temp = data[i][1]  # Obtain the result
		sorted_sol[index] = temp  # Save the result in the correct element

	return sorted_sol
