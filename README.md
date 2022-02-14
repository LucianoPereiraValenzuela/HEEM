![Entanglement Qubits - Apaisado](https://user-images.githubusercontent.com/11279156/120717557-0ae1b700-c4c8-11eb-92b0-54f718282f7d.png)
# Hardware-efficient-variational-quantum-eigensolver-with-entangled-measurements

[![DOI](https://zenodo.org/badge/360313020.svg)](https://zenodo.org/badge/latestdoi/360313020)

The Variational Quantum Eigensolver (VQE) is a hybrid quantum-classical algorithm whose aim is to find the ground state of a Quantum Hamiltonian. 
Implementing VQE in NISQ devices is challenging. The two main obstacles are: 

1) Deep circuits are implemented with low quality [1].
2) Many circuits are required to evaluate the energy [2,3].

Both difficulties have been studied independently. ¿Can they be tackled simultaneously?

We propose to use Hardware-Efficient Entangled Measurements (HEEM), which are measurements that take advantage of entanglement between neighboring qubits according to the device’s architecture. The usage of HEEM reduces the number of measurements needed to perform VQE without compromising the quality of the experiment.

In this github we implement the VQE algorithm using HEEM to estimate the energy. Cite to the doi: 10.5281/zenodo.6074767 

## Dependences

This package needs the following packages:

```bash
pip install qiskit
pip install qiskit_nature
pip install pyscf
pip install networkx
pip install tqdm
pip install joblib
pip install SciencePlots
```

## Folders
This github is organizated ad follows:

In [Articles](https://github.com/LucianoPereiraValenzuela/Hardware-efficient-variational-quantum-eigensolver-with-entangled-measurements/tree/main/Articles) you can find the main refenreces for this project. In [Figures](https://github.com/LucianoPereiraValenzuela/Hardware-efficient-variational-quantum-eigensolver-with-entangled-measurements/tree/main/Figures) we include all the plots and scketches used, while in [Report](https://github.com/LucianoPereiraValenzuela/Hardware-efficient-variational-quantum-eigensolver-with-entangled-measurements/tree/main/Report) there is a [PDF](https://github.com/LucianoPereiraValenzuela/Hardware-efficient-variational-quantum-eigensolver-with-entangled-measurements/blob/main/Report/HEEM_VQE%20Report.pdf) with all the technical details of the algoritmhs and the main results of the project, together with the LaTeX files to compile the mentioned report. The main folder is [Codes](https://github.com/LucianoPereiraValenzuela/Hardware-efficient-variational-quantum-eigensolver-with-entangled-measurements/tree/main/Codes), with all the python scrips and jupyter notebooks needed to reproduce the results.

In this last folder you can find the new VQE class in `VQE.py` than implement the HEEM functions of `HEEM_VQE_Functions.py` and the grouping from `GroupingAlgorithm.py`. On `utils.py` we include differents functions of general purpose that will be needed along the project. In [data](https://github.com/LucianoPereiraValenzuela/Hardware-efficient-variational-quantum-eigensolver-with-entangled-measurements/tree/main/Codes/data) we save all the simulations and experiments. In the folder [deprecated](https://github.com/LucianoPereiraValenzuela/Hardware-efficient-variational-quantum-eigensolver-with-entangled-measurements/tree/main/Codes/deprecated) there are all versions of the codes, which are no longer in use. In [experiments](https://github.com/LucianoPereiraValenzuela/Hardware-efficient-variational-quantum-eigensolver-with-entangled-measurements/tree/main/Codes/experiments) is the jupyter notebook used to perform an experiment in a real devide. Finaly, in the folder [tests](https://github.com/LucianoPereiraValenzuela/Hardware-efficient-variational-quantum-eigensolver-with-entangled-measurements/tree/main/Codes/tests) and [examples](https://github.com/LucianoPereiraValenzuela/Hardware-efficient-variational-quantum-eigensolver-with-entangled-measurements/tree/main/Codes/examples) there are several notebooks checking the correct funtionality of the different functions and shows some simple examples for their use.

## Usage
Here we provide a minimal example for the calculation of the minimum energy at a given distance for the LiH molecule using HEEM (NOTE: This example is designed to be executed in a jupyter notebook instance).

1. To use HEEM VQE we first have to import the modified VQE class.
``` python
from VQE import VQE
```

2. In order to use HEEM VQE to simulate molecules, we have created some functions that provided qiskit quantum operators for LiH and BeH2 molecules
``` python
from utils import LiH

qubit_op, init_state = LiH(initial_state=True)
num_qubits = qubit_op.num_qubits
```

3. Import the qiskit quantum instante for the simulation, and functions needed to simulate a real device
``` python
from qiskit import Aer
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.test.mock import FakeVigo
from qiskit.utils.quantum_instance import QuantumInstance
  
backend = Aer.get_backend('qasm_simulator')
device_backend = FakeVigo()
device = QasmSimulator.from_backend(device_backend)
coupling_map = device.configuration().coupling_map
noise_model = NoiseModel.from_backend(device)
basis_gates = noise_model.basis_gates
qi = QuantumInstance(backend=backend, coupling_map=coupling_map,
                     noise_model=noise_model, basis_gates=basis_gates)

```

4. Choose a classical optimizer and and ansatz for the variational circuit
``` python
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import EfficientSU2

optimizer = SPSA(maxiter=150, last_avg=1)
ansatz = init_state.compose(EfficientSU2(num_qubits, ['ry', 'rz'],
                                         entanglement='linear', reps=1))
initial_params = [0.1] * ansatz.num_parameters
```

5. Create a callback to verify the progress of VQE
``` python
from IPython.display import display, clear_output
def callback(evals, params, mean, deviation):  
    display("{}, {}".format(evals, mean))
    clear_output(wait=True)
```

6. Initialize and run the algorithm
``` python
solver = VQE(ansatz, optimizer, grouping='Entangled',
             quantum_instance=qi, conectivity=coupling_map,
             callback=callback)
result = solver.compute_minimum_eigenvalue(qubit_op).eigenvalue.real
```

7. Extract the energy computed in each iteration and plot it
``` python
import matplotlib.pyplot as plt
energies = solver.energies
plt.plot(energies)
```

![test](https://user-images.githubusercontent.com/11279156/120759606-78ff9b80-c513-11eb-93d5-257487a6c91d.jpeg)


For more examples, look at the [Examples](https://github.com/LucianoPereiraValenzuela/Hardware-efficient-variational-quantum-eigensolver-with-entangled-measurements/tree/main/Codes/examples) and [Test](https://github.com/LucianoPereiraValenzuela/Hardware-efficient-variational-quantum-eigensolver-with-entangled-measurements/tree/main/Codes/tests) folders.

## References
[1] Nature 549, 242 (2017).   
[2] Npj Quantum Information 6, 56 (2020).  
[3] Phys. Rev. A 101, 062322 (2020).
[4] arXiv:1907.13623 (2019)
