from typing import Union, List, Optional, Tuple
import pickle

from qiskit_nature.drivers.second_quantization import PySCFDriver
from qiskit_nature.transformers.second_quantization.electronic import FreezeCoreTransformer
from qiskit_nature.circuit.library import HartreeFock
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.mappers.second_quantization import ParityMapper, JordanWignerMapper, BravyiKitaevMapper
from qiskit_nature.converters.second_quantization.qubit_converter import QubitConverter
from qiskit.opflow.primitive_ops import TaperedPauliSumOp, PauliSumOp

MoleculeType = Union[PauliSumOp, TaperedPauliSumOp]


def extract_Paulis(qubit_op: MoleculeType) -> Tuple[List[str], List[complex]]:
    """
    Extract the Pauli labels, and the coefficients for each one from a given qubit operator.
    Parameters
    ----------
    qubit_op: TaperedPauliSumOp:
        Qubit operator to which extract the information

    Returns
    -------
    labels: list[str]
        Pauli labels in the string convention
    coeffs: list[complex]
        Coefficient for each of the Pauli strings
    """

    qubit_op_data = qubit_op.primitive.to_list()
    labels = [data[0] for data in qubit_op_data]
    coeffs = [data[1] for data in qubit_op_data]

    return labels, coeffs


def general_molecule(molecule: str, freeze_core: bool, orbitals_remove: Union[None, List[int]],
                     mapper_type: Union[None, str]) -> Union[Tuple[MoleculeType, HartreeFock], Exception]:
    """
    Compute the qubit operator and the initial Hartree Fock state of a given molecule. By default, the library
    PySCFDriver is used. If not possible to load, then  use PyQuanteDriver.
    Parameters
    ----------
    molecule: str
        Molecule in string format. Each atom is divided with ';'. The style for each atom is '[name] [x] [y] [z]'.
    freeze_core: bool (optional, default=True)
        Freeze some cores that do not highly impact in the energy of the molecule.
    orbitals_remove: bool (optional, default=None)
        Orbitals to remove that do not impact in the energy. The indices of the orbitals to remove are given by a list
        of integers. If not provided, no orbitals are removed.
    mapper_type: str (optional, default=None)
        Type of mapping between orbitals and qubits. Available options:
            'ParityMapper' (used by default if not provided)
            'JordanWignerMapper'
            'BravyiKitaevMapper'

    Returns
    -------
    qubit_op: TaperedPauliSumOp
        Qubit operator for the molecule
    init_state: HartreeFock
        Quantum circuit with the initial state given by Hartree Fock.
    """
    if mapper_type is None:
        mapper_type = 'ParityMapper'

    try:
        driver = PySCFDriver(molecule)
    except Exception:
        from qiskit_nature.drivers.second_quantization.pyquanted import PyQuanteDriver
        driver = PyQuanteDriver(molecule)

    Transformer = FreezeCoreTransformer(freeze_core=freeze_core, remove_orbitals=orbitals_remove)
    problem = ElectronicStructureProblem(driver, transformers=[Transformer])

    # Generate the second-quantized operators
    second_q_ops = problem.second_q_ops()

    # Hamiltonian
    main_op = second_q_ops[0]

    # Set up the mapper and qubit converter
    if mapper_type == 'ParityMapper':
        mapper = ParityMapper()
    elif mapper_type == 'JordanWignerMapper':
        mapper = JordanWignerMapper()
    elif mapper_type == 'BravyiKitaevMapper':
        mapper = BravyiKitaevMapper()
    else:
        return Exception(
            'Mapping not implemented. Try with one between: ParityMapper, JordanWignerMapper or BravyiKitaevMapper')

    # The fermionic operators are mapped
    converter = QubitConverter(mapper=mapper, two_qubit_reduction=True)

    particle_number = problem.grouped_property_transformed.get_property("ParticleNumber")
    num_particles = (particle_number.num_alpha, particle_number.num_beta)
    qubit_op = converter.convert(main_op, num_particles=num_particles)
    num_spin_orbitals = particle_number.num_spin_orbitals

    init_state = HartreeFock(num_spin_orbitals, num_particles, converter)

    return qubit_op, init_state


def H2(distance: Optional[float] = None, freeze_core: Optional[bool] = True,
       orbitals_remove: Optional[List[int]] = None, initial_state: Optional[bool] = False,
       mapper_type: Optional[str] = None) -> Union[MoleculeType, Tuple[MoleculeType, HartreeFock]]:
    """
    Qiskit operator for the H2 molecule.

    Parameters
    ----------
    distance: float (optional, default=None)
        Distance between atoms of hydrogen. If not, .761 is assumed.
    freeze_core
    orbitals_remove
    initial_state
    mapper_type

    (For more info about other arguments, see at the documentation of general_molecule() )

    Returns
    -------
    qubit_op: TaperedPauliSumOp
        Qubit operator for the molecule
    init_state: HartreeFock (only if initial_state=True)
        Quantum circuit with the initial state given by Hartree Fock.
    """

    if distance is None:
        distance = .761

    molecule = 'H .0 .0 .0; H .0 .0 ' + str(distance)

    qubit_op, init_state = general_molecule(molecule, freeze_core, orbitals_remove, mapper_type)

    if initial_state:
        return qubit_op, init_state
    else:
        return qubit_op


def LiH(distance: Optional[float] = None, freeze_core: Optional[bool] = True,
        orbitals_remove: Optional[List[int]] = None, initial_state: Optional[bool] = False,
        mapper_type: Optional[str] = None) -> Union[MoleculeType, Tuple[MoleculeType, HartreeFock]]:
    """
    Qiskit operator for the LiH molecule.

    Parameters
    ----------
    distance: float (optional, default=None)
        Distance between atoms of hydrogen. If not, 1.339 is assumed.
    freeze_core
    orbitals_remove
    initial_state
    mapper_type

    (For more info about other arguments, see at the documentation of general_molecule() )

    Returns
    -------
    qubit_op: TaperedPauliSumOp
        Qubit operator for the molecule
    init_state: HartreeFock (only if initial_state=True)
        Quantum circuit with the initial state given by Hartree Fock.
    """

    if distance is None:
        distance = 1.339

    if orbitals_remove is None:
        orbitals_remove = [3, 6]

    molecule = 'H 0.0 0.0 -' + str(distance) + '; Be 0.0 0.0 0.0; H 0.0 0.0 ' + str(distance)

    qubit_op, init_state = general_molecule(molecule, freeze_core, orbitals_remove, mapper_type)

    if initial_state:
        return qubit_op, init_state
    else:
        return qubit_op


def compute_molecule(molecule_name: str, distance: Optional[float] = None, freeze_core: Optional[bool] = True,
                     orbitals_remove: Optional[List[str]] = None, initial_state: Optional[bool] = False,
                     mapper_type: Optional[str] = None, load: Optional[bool] = False) -> Union[
    MoleculeType, Tuple[MoleculeType, HartreeFock]]:
    if load:
        try:
            file_name = '../data/molecules/' + molecule_name + '.npy'
            with open(file_name, 'rb') as file:
                qubit_op = pickle.load(file)
            print('Molecule loaded')
            if initial_state:
                file_name = '../data/molecules/' + molecule_name + '_initial_state' + '.npy'
                with open(file_name, 'rb') as file:
                    initial_state = pickle.load(file)

                return qubit_op, initial_state
            else:
                return qubit_op
        except KeyError:
            print('Molecule not found, computing it ...')

        molecule_name = molecule_name.lower()
        # TODO: Include molecules: [BeH2, H2O, CH4, C2H2, CH3OH, C2H6]
        if molecule_name == 'h2':
            return H2(distance=distance, freeze_core=freeze_core, orbitals_remove=orbitals_remove,
                      initial_state=initial_state, mapper_type=mapper_type)
        elif molecule_name == 'lih':
            return LiH(distance=distance, freeze_core=freeze_core, orbitals_remove=orbitals_remove,
                       initial_state=initial_state, mapper_type=mapper_type)
        else:
            raise Exception('The molecule {} is not implemented.'.format(molecule_name))
