from typing import Union, List, Optional, Tuple
import pickle

from qiskit_nature.drivers.second_quantization import PySCFDriver
from qiskit_nature.transformers.second_quantization.electronic import FreezeCoreTransformer
from qiskit_nature.circuit.library import HartreeFock
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.mappers.second_quantization import ParityMapper, JordanWignerMapper, BravyiKitaevMapper
from qiskit_nature.converters.second_quantization.qubit_converter import QubitConverter
from qiskit.opflow.primitive_ops import TaperedPauliSumOp, PauliSumOp
from qiskit.exceptions import MissingOptionalLibraryError

MoleculeType = Union[PauliSumOp, TaperedPauliSumOp]


def extract_paulis(qubit_op: MoleculeType) -> Tuple[List[str], List[complex]]:
    """
    Extract the Pauli labels and the coefficients from a given qubit operator.
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
    except MissingOptionalLibraryError:
        from qiskit_nature.drivers.second_quantization.pyquanted import PyQuanteDriver
        driver = PyQuanteDriver(molecule)

    Transformer = FreezeCoreTransformer(freeze_core=freeze_core, remove_orbitals=orbitals_remove)
    problem = ElectronicStructureProblem(driver, transformers=[Transformer])

    # Generate the second-quantized operators
    second_q_ops = problem.second_q_ops()

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


def BeH2(distance: Optional[float] = None, freeze_core: Optional[bool] = True,
         orbitals_remove: Optional[List[int]] = None, initial_state: Optional[bool] = False,
         mapper_type: Optional[str] = None) -> Union[MoleculeType, Tuple[MoleculeType, HartreeFock]]:
    """
    Qiskit operator for the BeH2 molecule.

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


def LiH(distance: Optional[float] = None, freeze_core: Optional[bool] = True,
         orbitals_remove: Optional[List[int]] = None, initial_state: Optional[bool] = False,
         mapper_type: Optional[str] = None) -> Union[MoleculeType, Tuple[MoleculeType, HartreeFock]]:
   """
    Qiskit operator for the LiH molecule.

    Parameters
    ----------
    distance: float (optional, default=None)
        If not, 1.5474 is assumed.
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
        distance = 1.5474

    if remove_orbitals is None:
        remove_orbitals = [3, 4]

    molecule = 'Li 0.0 0.0 0.0; H 0.0 0.0 ' + str(distance)

    qubit_op, init_state = general_molecule(molecule, freeze_core, orbitals_remove, mapper_type)

    if initial_state:
        return qubit_op, init_state
    else:
        return qubit_op


def H2O(distance: Optional[float] = None, freeze_core: Optional[bool] = True,
         orbitals_remove: Optional[List[int]] = None, initial_state: Optional[bool] = False,
         mapper_type: Optional[str] = None) -> Union[MoleculeType, Tuple[MoleculeType, HartreeFock]]:
   """
    Qiskit operator for the H2O molecule.

    Parameters
    ----------
    distance: float (optional, default=None)
        If not, 0.9584 is assumed.
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
        distance = 0.9584

    if remove_orbitals is None:
        remove_orbitals = [4]

    x = distance * np.sin(np.deg2rad(104.45 / 2))
    y = distance * np.cos(np.deg2rad(104.45 / 2))

    molecule = 'O 0.0 0.0 0.0; H ' + str(x) + ' ' + str(y) + ' 0.0; H -' + str(x) + ' ' + str(y) + ' 0.0'

    qubit_op, init_state = general_molecule(molecule, freeze_core, orbitals_remove, mapper_type)

    if initial_state:
        return qubit_op, init_state
    else:
        return qubit_op

    
def CH4(distance: Optional[float] = None, freeze_core: Optional[bool] = True,
         orbitals_remove: Optional[List[int]] = None, initial_state: Optional[bool] = False,
         mapper_type: Optional[str] = None) -> Union[MoleculeType, Tuple[MoleculeType, HartreeFock]]:
   """
    Qiskit operator for the CH4 molecule.

    Parameters
    ----------
    distance: float (optional, default=None)
        If not, 0.9573 is assumed.
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
        distance = 0.9573

    if remove_orbitals is None:
        remove_orbitals = [7,8]

    theta = 109.5
    r_inf = distance * np.cos(np.deg2rad(theta - 90))
    height_low = distance * np.sin(np.deg2rad(theta - 90))

    H1 = np.array([0, 0, distance])
    H2 = np.array([r_inf, 0, -height_low])
    H3 = np.array([-r_inf * np.cos(np.pi / 3), r_inf * np.sin(np.pi / 3), -height_low])
    H4 = np.array([-r_inf * np.cos(np.pi / 3), -r_inf * np.sin(np.pi / 3), -height_low])

    molecule = 'O 0 0 0; H {}; H {}; H {}; H {}'.format(str(H1)[1:-1], str(H2)[1:-1], str(H3)[1:-1], str(H4)[1:-1])

    qubit_op, init_state = general_molecule(molecule, freeze_core, orbitals_remove, mapper_type)

    if initial_state:
        return qubit_op, init_state
    else:
        return qubit_op

    
def C2H2(distance: Optional[float] = None, freeze_core: Optional[bool] = True,
         orbitals_remove: Optional[List[int]] = None, initial_state: Optional[bool] = False,
         mapper_type: Optional[str] = None) -> Union[MoleculeType, Tuple[MoleculeType, HartreeFock]]:
   """
    Qiskit operator for the C2H2 molecule.

    Parameters
    ----------
    distance: float (optional, default=None)
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
        distance = [1.2, 1.06]

    if remove_orbitals is None:
        remove_orbitals = [11]

    #   H(1)  C(1)  C(2)  H(2)

    H1 = str(np.array([0, 0, 0]))[1:-1]
    C1 = str(np.array([0, 0, distance[1]]))[1:-1]
    C2 = str(np.array([0, 0, distance[1] + distance[0]]))[1:-1]
    H2 = str(np.array([0, 0, 2 * distance[1] + distance[0]]))[1:-1]

    molecule = 'H {}; C {}; C {}; H {}'.format(H1, C1, C2, H2)

    qubit_op, init_state = general_molecule(molecule, freeze_core, orbitals_remove, mapper_type)

    if initial_state:
        return qubit_op, init_state
    else:
        return qubit_op


    
def compute_molecule(molecule_name: str, distance: Optional[float] = None, freeze_core: Optional[bool] = True,
                     orbitals_remove: Optional[List[str]] = None, initial_state: Optional[bool] = False,
                     mapper_type: Optional[str] = None, load: Optional[bool] = False) -> Union[
    MoleculeType, Tuple[MoleculeType, HartreeFock]]:
    """
    Compute the molecule qubit operator. If previously compute, it can also be loaded. If desired, the initial Hartree
    Fock circuit can be returned.
    Parameters
    ----------
    molecule_name: str
        Molecule's name. This argument is case-insensitive.
    distance: float (optional, default=None)
        Distance between molecules. This parameter is only used in the most simple molecules. If not provided, the
        equilibrium distance is used.
    freeze_core
    orbitals_remove
    initial_state
    mapper_type
    load: bool (default=False)
        If True, search for the precomputed molecule data at ../data/molecules/[molecule_name].pickle. If the data is
        not found, them compute the molecule.

    (For more info about other arguments, see at the documentation of general_molecule() )

    Returns
    -------
    qubit_op: TaperedPauliSumOp
        Qubit operator for the molecule
    init_state: HartreeFock (only if initial_state=True)
        Quantum circuit with the initial state given by Hartree Fock.
    """
    if load:
        try:
            file_name = '../data/molecules/' + molecule_name + '.pickle'
            with open(file_name, 'rb') as file:
                qubit_op = pickle.load(file)
            print('Molecule loaded')
            if initial_state:
                file_name = '../data/molecules/' + molecule_name + '_initial_state' + '.pickle'
                with open(file_name, 'rb') as file:
                    initial_state = pickle.load(file)
                return qubit_op, initial_state
            else:
                return qubit_op
        except KeyError:
            print('Molecule not found, computing it ...')

    molecule_name = molecule_name.lower()
    # TODO: Include molecules: [LiH, H2O, CH4, C2H2, CH3OH, C2H6]
    if molecule_name == 'h2':
        return H2(distance=distance, freeze_core=freeze_core, orbitals_remove=orbitals_remove,
                  initial_state=initial_state, mapper_type=mapper_type)
    elif molecule_name == 'beh2':
        return BeH2(distance=distance, freeze_core=freeze_core, orbitals_remove=orbitals_remove,
                    initial_state=initial_state, mapper_type=mapper_type)
    else:
        raise Exception('The molecule {} is not implemented.'.format(molecule_name))
