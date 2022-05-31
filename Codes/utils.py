import numpy as np
import matplotlib.pyplot as plt
import os
import os.path
import copy
import sys
from networkx import is_connected
from datetime import datetime, timedelta
from itertools import permutations
import contextlib
import joblib
import webbrowser
import zipfile
import time
import shutil
import json
from tqdm.auto import tqdm

from qiskit import transpile, execute
from qiskit.quantum_info import Pauli
from qiskit.opflow.primitive_ops import PauliOp
from qiskit.opflow.list_ops import SummedOp
from qiskit.opflow.primitive_ops.pauli_sum_op import PauliSumOp
from qiskit.opflow.primitive_ops.tapered_pauli_sum_op import TaperedPauliSumOp
from qiskit_nature.circuit.library import HartreeFock
from qiskit_nature.transformers.second_quantization.electronic import FreezeCoreTransformer
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.mappers.second_quantization import ParityMapper, JordanWignerMapper, BravyiKitaevMapper
from qiskit_nature.converters.second_quantization.qubit_converter import QubitConverter
from qiskit_nature.drivers.second_quantization import PySCFDriver
from qiskit.providers import JobStatus
from qiskit.providers.ibmq.job import exceptions

from GroupingAlgorithm import groupingWithOrder, TPBgrouping, grouping


def HeisenbergHamiltonian(J=1, H=1, num_qubits=2, neighbours=None):
    """
    Qiskit operator of the 3-D Heisenberg Hamiltonian of a lattice of spins.

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

    if neighbours is None:
        neighbours = [(0, 1)]

    num_op = num_qubits + 3 * len(neighbours)
    Hamiltonian_op_x = []
    Hamiltonian_op_z = []
    Hamiltonian_coeff = num_qubits * [-H] + num_op * [-J]

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
        [PauliOp(Pauli((Hamiltonian_op_z[j], Hamiltonian_op_x[j])), Hamiltonian_coeff[j]) for j in range(num_op)])

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
    coeff: list(float)
        Coefficients of each Pauli operator.
    label: list(str)
        Pauli strings
    """
    Dict = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}

    if type(QubitOp) == PauliSumOp or type(QubitOp) == TaperedPauliSumOp:
        QubitOp = QubitOp.to_pauli_op()

    label = []
    ops = []
    coeff = []

    for idx in QubitOp.oplist:
        label_temp = idx.primitive.to_label()
        label.append(label_temp)
        ops.append([Dict.get(s) for s in label_temp])
        coeff.append(idx.coeff)

    return np.array(ops), coeff, label


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


def get_backend_connectivity(backend):
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


def H2(distance=None, freeze_core=True, remove_orbitals=False, initial_state=False, operator=True,
       mapper_type='ParityMapper'):
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
    operator: Bool (optional)

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

    if distance is None:
        distance = .761

    molecule = 'H .0 .0 .0; H .0 .0 ' + str(distance)

    try:
        driver = PySCFDriver(molecule)
    except Exception:
        from qiskit_nature.drivers.second_quantization.pyquanted import PyQuanteDriver
        driver = PyQuanteDriver(molecule)

    # qmolecule = driver.run()

    if remove_orbitals is False:
        Transformer = FreezeCoreTransformer(freeze_core=freeze_core)
    else:
        Transformer = FreezeCoreTransformer(freeze_core=freeze_core, remove_orbitals=remove_orbitals)

    problem = ElectronicStructureProblem(driver, transformers=[Transformer])

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
    else:
        # TODO: Raise an error
        return None

    # The fermionic operators are mapped
    converter = QubitConverter(mapper=mapper, two_qubit_reduction=True)

    if operator is False:
        return converter, problem
    else:
        particle_number = problem.grouped_property_transformed.get_property("ParticleNumber")
        num_particles = (particle_number.num_alpha, particle_number.num_beta)
        num_spin_orbitals = particle_number.num_spin_orbitals
        qubit_op = converter.convert(main_op, num_particles=num_particles)
        if initial_state is False:
            return qubit_op
        else:
            init_state = HartreeFock(num_spin_orbitals, num_particles, converter)
            return qubit_op, init_state


def LiH(distance=None, freeze_core=True, remove_orbitals=None, initial_state=False, operator=True,
        mapper_type='ParityMapper'):
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
    operator: Bool (optional)

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

    if distance is None:
        distance = 1.5474

    if remove_orbitals is None:
        remove_orbitals = [3, 4]

    molecule = 'Li 0.0 0.0 0.0; H 0.0 0.0 ' + str(distance)

    try:
        driver = PySCFDriver(molecule)
    except Exception:
        from qiskit_nature.drivers.second_quantization.pyquanted import PyQuanteDriver
        driver = PyQuanteDriver(molecule)

    # qmolecule = driver.run()

    if remove_orbitals is False:
        Transformer = FreezeCoreTransformer(freeze_core=freeze_core)
    else:
        Transformer = FreezeCoreTransformer(freeze_core=freeze_core, remove_orbitals=remove_orbitals)

    problem = ElectronicStructureProblem(driver, transformers=[Transformer])

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
    else:
        return None

    # The fermionic operators are mapped
    converter = QubitConverter(mapper=mapper, two_qubit_reduction=True)

    if operator is False:
        return converter, problem
    else:
        # # The fermionic operators are mapped to qubit operators
        # num_particles = (problem.grouped_property_transformed.get_property("ParticleNumber").num_alpha,
        #                  problem.grouped_property_transformed.get_property("ParticleNumber").num_beta)
        # num_spin_orbitals = 2 * problem.molecule_data_transformed.num_molecular_orbitals

        particle_number = problem.grouped_property_transformed.get_property("ParticleNumber")
        num_particles = (particle_number.num_alpha, particle_number.num_beta)
        num_spin_orbitals = particle_number.num_spin_orbitals
        qubit_op = converter.convert(main_op, num_particles=num_particles)
        if initial_state is False:
            return qubit_op
        else:
            init_state = HartreeFock(num_spin_orbitals, num_particles, converter)
            return qubit_op, init_state


def BeH2(distance=None, freeze_core=True, remove_orbitals=None, initial_state=False, operator=True,
         mapper_type='ParityMapper'):
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
    operator: Bool (optional)

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

    if distance is None:
        distance = 1.339

    if remove_orbitals is None:
        remove_orbitals = [3, 6]

    molecule = 'H 0.0 0.0 -' + str(distance) + '; Be 0.0 0.0 0.0; H 0.0 0.0 ' + str(distance)

    try:
        driver = PySCFDriver(molecule)
    except Exception:
        from qiskit_nature.drivers.second_quantization.pyquanted import PyQuanteDriver
        driver = PyQuanteDriver(molecule)

    # qmolecule = driver.run()
    if remove_orbitals is False:
        Transformer = FreezeCoreTransformer(freeze_core=freeze_core)
    else:
        Transformer = FreezeCoreTransformer(freeze_core=freeze_core, remove_orbitals=remove_orbitals)

    problem = ElectronicStructureProblem(driver, transformers=[Transformer])

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
    else:
        return None

    # The fermionic operators are mapped
    converter = QubitConverter(mapper=mapper, two_qubit_reduction=True)

    if operator is False:
        return converter, problem
    else:
        # num_particles = (problem.grouped_property_transformed.get_property("ParticleNumber").num_alpha,
        #                  problem.grouped_property_transformed.get_property("ParticleNumber").num_beta)

        particle_number = problem.grouped_property_transformed.get_property("ParticleNumber")
        num_particles = (particle_number.num_alpha, particle_number.num_beta)
        num_spin_orbitals = particle_number.num_spin_orbitals
        qubit_op = converter.convert(main_op, num_particles=num_particles)
        if initial_state is False:
            return qubit_op
        else:
            init_state = HartreeFock(num_spin_orbitals, num_particles, converter)
            return qubit_op, init_state


def H2O(distance=None, freeze_core=True, remove_orbitals=None, initial_state=False, operator=True,
        mapper_type='ParityMapper'):
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
    operator: Bool (optional)

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

    if distance is None:
        distance = 0.9584

    if remove_orbitals is None:
        remove_orbitals = [4]

    x = distance * np.sin(np.deg2rad(104.45 / 2))
    y = distance * np.cos(np.deg2rad(104.45 / 2))

    molecule = 'O 0.0 0.0 0.0; H ' + str(x) + ' ' + str(y) + ' 0.0; H -' + str(x) + ' ' + str(y) + ' 0.0'

    try:
        driver = PySCFDriver(molecule)
    except Exception:
        from qiskit_nature.drivers.second_quantization.pyquanted import PyQuanteDriver
        driver = PyQuanteDriver(molecule)

    # qmolecule = driver.run()
    if remove_orbitals is False:
        Transformer = FreezeCoreTransformer(freeze_core=freeze_core)
    else:
        Transformer = FreezeCoreTransformer(freeze_core=freeze_core, remove_orbitals=remove_orbitals)

    problem = ElectronicStructureProblem(driver, transformers=[Transformer])

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
    else:
        return None

    # The fermionic operators are mapped
    converter = QubitConverter(mapper=mapper, two_qubit_reduction=True)

    if not operator:
        return converter, problem
    else:
        # num_particles = (problem.grouped_property_transformed.get_property("ParticleNumber").num_alpha,
        #                 problem.grouped_property_transformed.get_property("ParticleNumber").num_beta)

        particle_number = problem.grouped_property_transformed.get_property("ParticleNumber")
        num_particles = (particle_number.num_alpha, particle_number.num_beta)
        num_spin_orbitals = particle_number.num_spin_orbitals
        qubit_op = converter.convert(main_op, num_particles=num_particles)
        if initial_state is False:
            return qubit_op
        else:
            init_state = HartreeFock(num_spin_orbitals, num_particles, converter)
            return qubit_op, init_state


def CH4(distance=None, freeze_core=True, remove_orbitals=None, initial_state=False, operator=True,
        mapper_type='ParityMapper'):
    """
    Qiskit operator of the CH4

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
    operator: Bool (optional)

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

    if distance is None:
        distance = 0.9573

    if remove_orbitals is None:
        remove_orbitals = [7, 8]

    #          H(1)
    #          O
    #   H(2)      H(3)   H(4)

    theta = 109.5
    r_inf = distance * np.cos(np.deg2rad(theta - 90))
    height_low = distance * np.sin(np.deg2rad(theta - 90))

    H1 = np.array([0, 0, distance])
    H2 = np.array([r_inf, 0, -height_low])
    H3 = np.array([-r_inf * np.cos(np.pi / 3), r_inf * np.sin(np.pi / 3), -height_low])
    H4 = np.array([-r_inf * np.cos(np.pi / 3), -r_inf * np.sin(np.pi / 3), -height_low])

    molecule = 'O 0 0 0; H {}; H {}; H {}; H {}'.format(str(H1)[1:-1], str(H2)[1:-1], str(H3)[1:-1], str(H4)[1:-1])

    try:
        driver = PySCFDriver(molecule)
    except Exception:
        from qiskit_nature.drivers.second_quantization.pyquanted import PyQuanteDriver
        driver = PyQuanteDriver(molecule)

    if remove_orbitals is False:
        Transformer = FreezeCoreTransformer(freeze_core=freeze_core)
    else:
        Transformer = FreezeCoreTransformer(freeze_core=freeze_core, remove_orbitals=remove_orbitals)

    problem = ElectronicStructureProblem(driver, transformers=[Transformer])

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
    else:
        return None

    # The fermionic operators are mapped
    converter = QubitConverter(mapper=mapper, two_qubit_reduction=True)

    if not operator:
        return converter, problem
    else:
        particle_number = problem.grouped_property_transformed.get_property("ParticleNumber")
        num_particles = (particle_number.num_alpha, particle_number.num_beta)
        num_spin_orbitals = particle_number.num_spin_orbitals
        qubit_op = converter.convert(main_op, num_particles=num_particles)
        if initial_state is False:
            return qubit_op
        else:
            init_state = HartreeFock(num_spin_orbitals, num_particles, converter)
            return qubit_op, init_state


def C2H2(distance=None, freeze_core=True, remove_orbitals=None, initial_state=False, operator=True,
         mapper_type='ParityMapper'):
    """
    Qiskit operator of the C2H2

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
    operator: Bool (optional)

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

    if distance is None:
        distance = [1.2, 1.06]

    if remove_orbitals is None:
        remove_orbitals = [11]

    #   H(1)  C(1)  C(2)  H(2)

    H1 = str(np.array([0, 0, 0]))[1:-1]
    C1 = str(np.array([0, 0, distance[1]]))[1:-1]
    C2 = str(np.array([0, 0, distance[1] + distance[0]]))[1:-1]
    H2 = str(np.array([0, 0, 2 * distance[1] + distance[0]]))[1:-1]

    molecule = 'H {}; C {}; C {}; H {}'.format(H1, C1, C2, H2)

    try:
        driver = PySCFDriver(molecule)
    except Exception:
        from qiskit_nature.drivers.second_quantization.pyquanted import PyQuanteDriver
        driver = PyQuanteDriver(molecule)

    if remove_orbitals is False:
        Transformer = FreezeCoreTransformer(freeze_core=freeze_core)
    else:
        Transformer = FreezeCoreTransformer(freeze_core=freeze_core, remove_orbitals=remove_orbitals)

    problem = ElectronicStructureProblem(driver, transformers=[Transformer])

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
    else:
        return None

    # The fermionic operators are mapped
    converter = QubitConverter(mapper=mapper, two_qubit_reduction=True)

    if not operator:
        return converter, problem
    else:
        particle_number = problem.grouped_property_transformed.get_property("ParticleNumber")
        num_particles = (particle_number.num_alpha, particle_number.num_beta)
        num_spin_orbitals = particle_number.num_spin_orbitals
        qubit_op = converter.convert(main_op, num_particles=num_particles)
        if initial_state is False:
            return qubit_op
        else:
            init_state = HartreeFock(num_spin_orbitals, num_particles, converter)
            return qubit_op, init_state


def molecules(molecule_name, distance=None, freeze_core=True, remove_orbitals=None, operator=True, initial_state=False,
              mapper_type='ParityMapper', load=True):
    if load:
        try:
            qubit_op = np.load('../data/molecules_qubitop.npy', allow_pickle=True).item()[molecule_name]
            print('Molecule loaded')
            return qubit_op
        except KeyError:
            print('Computing molecule')

    molecule_name = molecule_name.lower()
    if molecule_name == 'h2':
        return H2(distance=distance, freeze_core=freeze_core, remove_orbitals=remove_orbitals,
                  initial_state=initial_state, operator=operator, mapper_type=mapper_type)
    elif molecule_name == 'lih':
        return LiH(distance=distance, freeze_core=freeze_core, remove_orbitals=remove_orbitals,
                   initial_state=initial_state, operator=operator, mapper_type=mapper_type)
    elif molecule_name == 'beh2':
        return BeH2(distance=distance, freeze_core=freeze_core, remove_orbitals=remove_orbitals,
                    initial_state=initial_state, operator=operator, mapper_type=mapper_type)
    elif molecule_name == 'h2o':
        return H2O(distance=distance, freeze_core=freeze_core, remove_orbitals=remove_orbitals,
                   initial_state=initial_state, operator=operator, mapper_type=mapper_type)
    elif molecule_name == 'ch4':
        return CH4(distance=distance, freeze_core=freeze_core, remove_orbitals=remove_orbitals,
                   initial_state=initial_state, operator=operator, mapper_type=mapper_type)
    elif molecule_name == 'c2h2':
        return C2H2(distance=distance, freeze_core=freeze_core, remove_orbitals=remove_orbitals,
                    initial_state=initial_state, operator=operator, mapper_type=mapper_type)
    else:
        raise Exception('The molecule {} is not implemented.'.format(molecule_name))


# DEPRECATED
# def unpack_functions(pack):
# 	"""
# 	Unpack the list where the first element is the index of the async execution, the second index in the function to
# 	run, the third index are the function variables, and the last index (if provided) are the optional arguments.
#
# 	Parameter
# 	---------
# 	pack: list
# 		List with all the data
#
# 	 Return
# 	 ------
# 	 Result of the function
# 	"""
# 	if len(pack) < 4:  # If no optional arguments are provided
# 		pack.append({})
# 	return [pack[0], pack[1](*pack[2], **pack[3])]
#
#
# def sort_solution(data):
# 	"""
# 	Function to sort the data obtained for a parallel computation
#
# 	Parameter
# 	---------
# 	data: list
# 		List in which each entry represents one solution of the parallel computation. The elements are
# 		also list which contains in the first element the index and in the second one the result of the computation.
#
# 	Return
# 	------
# 	List with the data sorted
# 	"""
# 	n = len(data)  # Extract the number of computations done
# 	sorted_sol = [None] * n  # Empty list with the correct number of elements
# 	for i in range(n):  # Iterate over all the elements
# 		index = data[i][0]  # Obtain the index of the result
# 		temp = data[i][1]  # Obtain the result
# 		sorted_sol[index] = temp  # Save the result in the correct element
#
# 	return sorted_sol
#
#
# def isnotebook():
# 	"""
# 	Check if the script is been running in a jupyter notebook instance
#
# 	Return
# 	------
# 	True is the instance is a Jupyter notebook, false in other cases
# 	"""
# 	try:
# 		shell = get_ipython().__class__.__name__
# 		if shell == 'ZMQInteractiveShell':
# 			return True  # Jupyter notebook or qtconsole
# 		elif shell == 'TerminalInteractiveShell':
# 			return False  # Terminal running IPython
# 		else:
# 			return False  # Other type (?)
# 	except NameError:
# 		return False  # Probably standard Python interpreter
#

def permute_indices(diagonal_factors, qubit0, qubit1, n_qubits):
    """
    Permute the diagonal factors indices by the interchange of qubit_0 <---> qubit_1, maintaining all other indices the
    same.

    Parameters
    ----------
    diagonal_factors: ndarray (2 ** n_qubits)
        Diagonal factors for the computation of the expected energy
    qubit0: int
        Index of the first qubit to swap
    qubit1: int
        Index of the second qubit to swap
    n_qubits: int
        Number of qubits in the circuit

    Return
    ------
    temp: ndarray (2 ** n_qubits)
        Refactor diagonal factors
    """
    temp = np.zeros(2 ** n_qubits, dtype=diagonal_factors.dtype)

    # Iterate over all the possible outputs of the circuit
    for i in range(len(temp)):
        new = bin(i)[2:]  # New index in binary
        if len(new) < n_qubits:  # Complete with 0's if the index is not of the correct size
            new = ''.join(['0']) * (n_qubits - len(new)) + new
        old = swapPositions(new, qubit0, qubit1)  # Swap the indices of qubit_0 and qubit_1
        temp[int(new, 2)] = diagonal_factors[int(old, 2)]  # Copy the old diagonal factor in the new position

    return temp


def swapPositions(str_variable, pos1, pos2):
    """
    Swap the position of two indices of a given string.

    Parameters
    ----------
    str_variable: str
        String to interchange the indices. The length must be >= max(pos1, pos2)
    pos1: int
        Index of the first element to swap
    pos2: int
        Index of the second element to swap

    Return
    ------
    Reformat string with the given swaps
    """
    list_variable = list(str_variable)
    list_variable[pos1], list_variable[pos2] = list_variable[pos2], list_variable[pos1]
    return ''.join(list_variable)


def swaps(arr, reverse=True):
    """
    Compute the needed swaps of two elements to sort a given array in descending (or ascending) order.

    Parameters
    ----------
    arr: list
        Original array with unsorted numbers [0, 1, ...., len(arr) - 1]. A given element can not appear twice.
    reverse: bool (optional, default=True)
        If reverse=True, sort in descending order, if reverse=False, sort in ascending order

    Returns
    -------
    swaps: ndarray (n, 2)
        Array containing the indices needed to perform a total of n swaps. Each swap corresponds to a given row. The
        swaps must be performed in the correct order, starting from swaps[0], and finish in swaps[-1].
    """
    # If descending order, reverse the order of the original array
    if reverse:
        arr = arr[::-1]
    n = len(arr)  # Number of elements
    swaps_list = []  # List with the swaps

    # Start the algorithm
    i = 0
    while i < n:
        if arr[i] != i:  # If the element is not in the correct locations
            swaps_list.append(np.array([i, arr[i]]))
            # Interchange the element with the correct element in a given location
            arr[arr[i]], arr[i] = arr[i], arr[arr[i]]
        else:
            i += 1

    swaps_list = np.array(swaps_list)

    # If descending order, transform the indices in each swap. E.g. if N = 3: 0 --> |0 - 3 + 1| = 2, 1 -> 1 and 2 -> 0
    if reverse:
        swaps_list = np.abs(swaps_list - n + 1)

    return swaps_list


def change_order_qubitop(qubit_op, order_paulis, order_qubits):
    """

    Parameters
    ----------
    qubit_op
    order_paulis
    order_qubits

    Returns
    -------

    """
    paulis, coeffs, labels = Label2Chain(qubit_op)

    operators = []
    for i in range(len(labels)):
        coeff = coeffs[order_paulis[i]]
        label = labels[order_paulis[i]]

        temp = ''
        for j in range(len(order_qubits)):
            temp += label[order_qubits[j]]

        operators.append(PauliOp(Pauli(temp), coeff))

    new_qubit_op = SummedOp(operators)
    return new_qubit_op


def number2SummedOp(labels, coeffs):
    operators = []
    for label, coeff in zip(labels, coeffs):
        operators.append(PauliOp(Pauli(label), coeff))

    new_qubit_op = SummedOp(operators)
    return new_qubit_op


def question_overwrite(name):
    """
    Make a question if you want to overwrite a certain file that already exists in the directory. There is only two
    possible answers y -> yes (true) or	n -> no (False). If the answer is non of this two the question is repeated until
    a good answer is given.

    Parameter
    ----------
    name: (Str)
        Name of the file to overwrite

    Return
    ------
    (Bool)
        Answer given by the user
    """

    temp = input('Do you want to overwrite the file ({})?  [y]/n: '.format(name))  # Ask for an answer by keyword input

    if temp == 'y' or temp == '':
        return True
    elif temp == 'n':
        return False
    else:  # If the answer is not correct
        print('I didn\'t understand your answer.')
        return question_overwrite(name)  # The function will repeat until a correct answer if provided


def save_figure(fig, file_dic):
    fig.savefig(file_dic, bbox_inches="tight",
                dpi=600)  # Save the figure with the corresponding file direction and the correct extension


def save_data(data, file_dic):
    np.save(file_dic, data)


def save_object(object_save, name, overwrite=None, extension=None, dic=None, prefix='', back=0, temp=False, index=0,
                ask=True, extent=False, silent=False):
    """
    Save a given figure or date. We must introduce the name with which we want to save the file. If the file already
    exist, then we will be asked if we want to overwrite it. We can also change the	extension used to save the image.
    This function has a protection for not overwriting and save a temp file if the overwriting question is not asked.

    Parameters
    ----------
    object_save: (fig or dic)
        Matplotlib figure or dictionary with the data to save
    name: (str)
        String with the name of the file in which save the data
    overwrite: Optional (bool)
        Condition to overwrite or not the data. If a value is not given then the function will ask by default
    extension: Optional (str)
        Extension of the save folder. By default extension='npy' for data, and extension='pdf' for figures
    dic: Optional (str)
        Directory to save the data. By default dic='data/
    prefix: Optional (str)
        Prefix for the folder 'data/'. This must be used if the function is called from a sub folder or is the target
        folder have a complex name. By default prefix=''
    back: Optional (int)
        Number of times to go back from the script that call this function until reach the parent folder of the target
        directory. By default back=0
    temp: (bool)
        If a temp file is saved. Only works with overwrite=None
    index: (int)
        Index to include in the file name is the previous onw is already occupied
    ask: (bool)
        Condition than controls if the question to overwrite is done.
    """

    if str(type(object_save)) == "<class 'matplotlib.figure.Figure'>":
        object_type = 'figure'
        save_function = save_figure
    else:
        object_type = 'data'
        save_function = save_data

    if dic is None:
        dic = '../data/'

    dic = '../' * back + dic

    if extension is None:
        if object_type == 'figure':
            extension = 'pdf'
        else:
            extension = 'npy'

    file_dic = prefix + dic + name  # Complete the directory of the file including its extension

    if object_type == 'data':
        if index != 0:  # If an index is specified
            file_dic += ' (' + str(index) + ')'  # Include the index in the same

    if overwrite is None:  # If the user does not give a preference for the overwriting
        if os.path.isfile(file_dic + '.' + extension):  # If the file exists in the folder
            if temp:
                save_function(object_save, file_dic + '_temp' + '.' + extension)
            if ask:  # The function will ask if the user want to overwrite the file
                overwrite = question_overwrite(file_dic + '.' + extension)
            else:
                overwrite = False
        else:
            overwrite = True  # If the file does not exist, them the figure will be saved

    if overwrite:  # Depending on the answer of the user
        save_function(object_save, file_dic + '.' + extension)
        if not silent:
            print(object_type, 'saved as', file_dic + '.' + extension)
    elif extent:
        # If the user does not want to overwrite, a copy is saved
        # The copy will include the typical (1) at the end of the name, if this already exist then (2) without asking.
        # If the file also exist then (3)
        # and so on until an empty number is reached.
        save_object(object_save, name, index=index + 1, ask=False, temp=False)

    if os.path.isfile(file_dic + '_temp.npy'):
        os.remove(file_dic + '_temp.npy')  # If the file is correctly saved the temp file is removed


def n_groups_shuffle(paulis, G, seed, shuffle_paulis=True, shuffle_qubits=True, x=1, n_max=10000, n_delete=0,
                     connected=False, full_output=False, grouping_method='Entangled', order=True, AM=None,
                     minimal_output=False):
    num_qubits = len(paulis[0])
    G_new = copy.deepcopy(G)

    if x < 1 or n_delete > 0:
        edges = list(G_new.edges())

        if x < 1:
            n_delete = int((1 - x) * len(edges))

        indices_delete = np.random.default_rng().choice(len(edges), size=n_delete, replace=False)
        for index in indices_delete:
            G_new.remove_edge(*edges[index])

        if not is_connected(G_new):
            if n_max == 0:
                return np.nan, None, None, G_new
            else:
                return n_groups_shuffle(paulis, G, seed, shuffle_paulis=shuffle_paulis, shuffle_qubits=shuffle_qubits,
                                        x=x, n_max=n_max - 1, n_delete=n_delete)

    np.random.seed(seed)
    order_paulis = np.arange(len(paulis))
    order_qubits = np.arange(num_qubits)
    if shuffle_paulis:
        np.random.shuffle(order_paulis)

    if shuffle_qubits:
        np.random.shuffle(order_qubits)

        temp = copy.deepcopy(paulis)
        for i in range(len(order_qubits)):
            paulis[:, i] = temp[:, order_qubits[i]]

    T = None
    if grouping_method.lower() == 'entangled':
        if order:
            Groups, Measurements, T = groupingWithOrder(paulis[order_paulis], G_new, connected=connected)

        else:
            if G_new is None:
                WC = None
            else:
                WC = G_new.edges

            Groups, Measurements = grouping(paulis[order_paulis], AM=AM, WC=WC)

    elif grouping_method.lower() == 'tpb':
        _, Groups, Measurements = TPBgrouping(paulis)
    else:
        raise Exception('Invalid grouping method.')

    if minimal_output:
        return len(Groups)

    output = [len(Groups), order_paulis, order_qubits, G_new]
    if full_output:
        output.append(Groups)
        output.append(Measurements)
        output.append(T)

    return output


def unconnected_measurements(WC, Measurements, T=None):
    if T is None:
        T = np.arange(np.max(WC) + 1)

    counter = 0
    for Groups in Measurements:
        for measurement in Groups:
            if measurement[0] > 3:
                qubit_1_teo, qubit_2_teo = measurement[1]
                qubit_1_phys = T[qubit_1_teo]
                qubit_2_phys = T[qubit_2_teo]
                if (qubit_1_phys, qubit_2_phys) not in WC:
                    counter += 1

    return counter


def number_cnots(circuits, qi):
    circuits = qi.transpile(circuits)

    n_cnots = 0
    for circuit in circuits:
        try:
            n_cnots += circuit.count_ops()['cx']
        except KeyError:
            pass

    return n_cnots


def number_cnots_raw(measurements, num_qubits, WC=None, T=None):
    from HEEM_VQE_Functions import measure_circuit_factor

    if WC is None:
        WC = list(permutations(list(range(num_qubits)), 2))

    if type(WC[0]) == tuple:
        WC = [list(x) for x in WC]

    if T is None:
        T = list(np.arange(num_qubits))

    circuits = [measure_circuit_factor(measure, num_qubits, make_measurements=False) for measure in measurements]
    circuits_transpiled = transpile(circuits, coupling_map=WC, initial_layout=T[::-1])

    n_cnots = 0
    for circuit in circuits_transpiled:
        try:
            n_cnots += circuit.count_ops()['cx']
        except KeyError:
            pass

    return n_cnots


def current_time():
    now = datetime.now()
    return now.strftime("%d/%m/%Y, %H:%M:%S")


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def flatten_groups(groups):
    groups_list = []
    for group in groups:
        groups_list += group
        groups_list.append(-1)
    return groups_list


def recover_groups(groups_list):
    groups = []

    start = 0
    for i in range(len(groups_list)):
        if groups_list[i] == -1:
            end = i

            groups.append(groups_list[start:end])
            start = end + 1

    return groups


def flatten_measurements(measurements):
    measure_index = []
    qubits = []
    for group in measurements:
        for measure in group:
            measure_index.append(measure[0])
            qubits += measure[-1]

        measure_index.append(-1)

    return measure_index, qubits


def recover_measurements(measurements_indices, qubits):
    measurements = [[]]

    counter_group = 0
    for i in measurements_indices[:-1]:
        if i == -1:
            measurements.append([])
            counter_group += 1
        else:
            if i < 4:
                step = 1
            else:
                step = 2
            temp = [i, [qubits.pop(0) for _ in range(step)]]
            measurements[counter_group].append(temp)

    return measurements


def load_grouping_data(molecule_name, method):
    file = '../data/groupings/' + molecule_name + '/'

    try:
        np.load(file + 'Measurements_qubits_' + method + '.npy')
        data_computed = True  # print('Data loaded')
    except FileNotFoundError:
        print('Data does not exist.')
        data_computed = False

    if data_computed:
        Groups = recover_groups(np.load(file + 'Groups_' + method + '.npy'))

        Measurements = recover_measurements(np.load(file + 'Measurements_' + method + '.npy'),
                                            np.load(file + 'Measurements_qubits_' + method + '.npy').tolist())

        returns = [Groups, Measurements]

        try:
            layout = np.load(file + 'layout_' + method + '.npy', allow_pickle=True).tolist()
            returns.append(layout)
        except FileNotFoundError:
            pass

        return returns

    else:
        return None


def download_data_IBMQ(job_id, download_path):
    """
    Load data from simulation of experiment from IBMQ. This is to avoid the slow job.result() methods. The data is
    downloaded using the job is via the browser. If the download fails, it is repeated until successful. The data is
    download as a .zip file in the download folder.

    Parameters
    ----------
    job_id: str
        IBMQ job id
    """
    total_wait = 0
    max_wait = 120
    wait = 5

    prefix = 'https://quantum-computing.ibm.com/api/jobs/download?id='

    webbrowser.open(prefix + job_id)
    while True:
        if len([name for name in os.listdir(download_path) if job_id in name]) > 0:
            break
        else:
            time.sleep(wait)
            total_wait += wait
        if total_wait > max_wait:
            download_data_IBMQ(job_id, download_path)


def load_data_IBMQ(job_id, download_path=None, verbose=False):
    """
    Load the data, already downloaded, and extract the probability vector. After this, all the data in the HDD is
    removed.
    Parameters
    ----------
    job_id: str
        IBMQ job id
    """

    if download_path is None:
        # download_path = '/mnt/d/david/Downloads/'
        download_path = 'D:/david/Downloads/'

    download_data_IBMQ(job_id, download_path)
    if verbose:
        print('data downloaded')

    name_file = download_path + 'job-' + job_id + '.zip'

    with zipfile.ZipFile(name_file, 'r') as zip_ref:
        zip_ref.extractall(download_path + job_id)

    os.remove(name_file)

    if verbose:
        print('Loading data...')
    data = json.load(open(download_path + job_id + '/' + job_id + '-output.json'))['results']

    shutil.rmtree(download_path + job_id)

    return [np.linalg.norm(np.array(data[i]['data']['snapshots']['statevector']['snapshot'][0]), axis=1) ** 2 for i in
            range(len(data))]


def send_job_backend(backend, circuits_batch, index, n_batches, kwards_run, job_tag, verbose=True, output_file=None):
    if verbose:
        print('Sending Job {}/{} to IBMQ'.format(index + 1, n_batches))

    job = execute(circuits_batch, backend, **kwards_run)
    if job_tag is not None:
        if type(job_tag) is not list:
            job_tag = [job_tag]

        job.update_tags(additional_tags=job_tag)

    job.update_name('{}/{}'.format(index + 1, n_batches))
    if verbose:
        print('Job {}/{} sent to IBMQ'.format(index + 1, n_batches))
    if output_file is not None:
        with open(output_file, 'w') as f:
            f.write('{}/{}\n'.format(index + 1, n_batches))

    return job


def check_status_jobs(running, n_batches, verbose=False):
    """
    Check is some job is done in order to send the next one, or if some job has raised an error.
    """
    # print('Checking...')
    for j, (index, job) in enumerate(running):
        # stop = datetime.now() + timedelta(seconds=5)  # Check status for a maximum of 5"
        if verbose:
            print(f'Checking job {index + 1}/{n_batches}')

        status = job.status()
        # while datetime.now() < stop:
        if status == JobStatus.DONE:
            if verbose:
                print(f'Job {index + 1}/{n_batches} done')
            return True, j, index
        elif status == JobStatus.CANCELLED:
            if verbose:
                print(f'Job {index + 1}/{n_batches} canceled')
            return False, j, index
        elif status == JobStatus.ERROR:
            if verbose:
                print(f'Job {index + 1}/{n_batches} failed')
            return False, j, index
    return None


def send_ibmq_parallel(backend, batch_size, circuits, job_tag=None, kwards_run=None, verbose=True, output_file=None,
                       progress_bar=False, n_jobs_parallel=5, waiting_time = 20):
    if kwards_run is None:
        kwards_run = {}

    n_circuits = len(circuits)
    n_batches = int(np.ceil(n_circuits / batch_size))

    if n_jobs_parallel == -1:
        n_jobs_parallel = n_batches

    if progress_bar:
        pbar = tqdm(total=n_batches, desc='Jobs completed')
    else:
        pbar = None

    # Indices for the first and last circuit in each batch
    indices = []
    for i in range(n_batches):
        initial = i * batch_size
        final = min(initial + batch_size, n_circuits)

        indices.append([initial, final])

    job_done = [None] * n_batches
    running_jobs = []

    # Send the initial jobs
    for i, index in enumerate(indices[:n_jobs_parallel]):
        job = send_job_backend(backend, circuits[index[0]:index[1]], i, n_batches, kwards_run, job_tag, verbose,
                               output_file)

        running_jobs.append([i, job])

    max_id_sent = n_jobs_parallel - 1

    # Iterate until all jobs have been done
    while len(running_jobs) > 0:
        temp = check_status_jobs(running_jobs, n_batches, verbose=verbose)
        if verbose:
            print('Finish checking')

        if temp is not None:  # Some job has been completed or cancelled
            done, running_id, job_id = temp
            if done:
                if progress_bar:
                    pbar.update()

                job_done[job_id] = running_jobs.pop(running_id)[1]

                if max_id_sent + 1 < n_batches:
                    max_id_sent += 1
                    job = send_job_backend(backend, circuits[indices[max_id_sent][0]:indices[max_id_sent][1]],
                                           max_id_sent, n_batches, kwards_run, job_tag, verbose, output_file)

                    running_jobs.append([max_id_sent, job])
            else:
                running_jobs.pop(running_id)

                job = send_job_backend(backend, circuits[indices[job_id][0]:indices[job_id][1]], job_id, n_batches,
                                       kwards_run, job_tag, verbose, output_file)
                running_jobs.append([job_id, job])
        else:
            time.sleep(waiting_time)

    if progress_bar:
        pbar.close()

    return job_done
