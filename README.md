![Entanglement Qubits - Apaisado](https://user-images.githubusercontent.com/11279156/120717557-0ae1b700-c4c8-11eb-92b0-54f718282f7d.png)
# Hardware efficient entangled measurements (HEEM)

[![DOI](https://zenodo.org/badge/360313020.svg)](https://zenodo.org/badge/latestdoi/360313020)

The Variational Quantum Eigensolver (VQE) is a hybrid quantum-classical algorithm whose aim is to find the ground state of a Quantum Hamiltonian. 
Implementing VQE in NISQ devices is challenging. The two main obstacles are: 

1) Deep circuits are implemented with low quality [1].
2) Many circuits are required to evaluate the energy [2, 3].

Both difficulties have been studied independently. Can they be tackled simultaneously?

We propose to use Hardware-Efficient Entangled Measurements (HEEM), which are measurements that take advantage of entanglement between neighboring qubits according to the device’s architecture. The usage of HEEM reduces the number of measurements needed to perform VQE without compromising the quality of the experiment.

In this repo we implement the VQE algorithm using HEEM to estimate the energy of some simple molecules such as $BeH_2$ or $H_2O$. 

If you find this repository useful, please consider citing the article [arXiv: 2202.06979](https://arxiv.org/abs/2202.06979).

## Dependencies
The required packages, together with their versions, are located in [`requirements.txt`](https://github.com/LucianoPereiraValenzuela/HEEM/blob/main/requirements.txt). In order to create a conda enviroment with all the packages use the following command:
```
conda create --name <env> --file <this file>
```

## Folders
The main scrips are located in the root folder. Inside [`examples`](https://github.com/LucianoPereiraValenzuela/HEEM/tree/main/examples) there are several jupyter notebooks that show the main features and how to use it. Finally, in  [`benchmarks`](https://github.com/LucianoPereiraValenzuela/HEEM/tree/main/benchmarks) there are two jupyter notebooks to calculate and compare the different methods for the grouping of molecules, and also random Pauli strings.

## Usage
### Define the Hamiltonian
First you need to define the Pauli string of interest. This can be easily done using `compute_molecule(<molecule_name>)`, located inside [`molecules.py`](https://github.com/LucianoPereiraValenzuela/HEEM/blob/main/molecules.py). Here we have predefined a total of eigth molecules
- H2
- LiH
- BeH2
- H2O
- CH4
- C2H2
- CH3OH
- C2H6

Other molecules (or different Hamiltonians) can be included by providing the Pauli string as a `TaperedPauliSumOp` or a `PauliSumOp`. Also, the user can define the Pauli string as two lists, one with the strings definind the operators over each qubit, and the second one which contain the weights of each operator.

### Obtain the grouping
Once the Pauli string is obtained, you have to chose an algorithm for the grouping of the different terms in the Hamiltonian in order to reduce the number of circuits needed to measure the expected energy. All the needed functions for this porpuse are located inside [`grouping.py`](https://github.com/LucianoPereiraValenzuela/HEEM/blob/main/grouping.py). The easiest way to obtain the grouping is to use the `Group()` class. As an input, you must provide the labels of the Pauli string, the connectivity of the targe device, and the desired grouping algorithm. By default the choosen algorithm is HEEM. To perform the actual grouping you must use `Grouping.group()`.

In order to obtain the results, you can use
- `Grouping.groups`: Return the indices of the different terms that have been grouped
- `Grouping.measurements`: Return the measurements need, which must be applied to the quantum circuit
- `Grouping.n_cnots()`: Return the number of cnots needed to measure the energy

Other methods have been included to visualize the grouping and the transpiled measurements such as `Grouping.draw_entangled_measurements()` or `Group.draw_transpiled_chip`. You can look at the source code for more info.

### Measure the energy
After performing the grouping fot he Pauli string, we need to obtain the actual `QuantumCircuits` that must be simulated/send to IBMQ, which can be done with the `create_circuits()` function located inside [`measurements.py`](https://github.com/LucianoPereiraValenzuela/HEEM/blob/main/measurements.py). The number of circuits that must be simulated can be quite large, so we have created `send_ibmq_parallel()` inside [`utils.py`](https://github.com/LucianoPereiraValenzuela/HEEM/blob/main/utils.py) to send all the circuits to IBMQ, which are then distributed in different batches and send in parallel for simulating then inside the HPC IBM backed.

Once you have the counts of the different quantum circuits, you must use the function `compute_energy()` inside [`measurements.py`](https://github.com/LucianoPereiraValenzuela/HEEM/blob/main/measurements.py) to postprocress the result and obtain the expected energy of the provided Hamiltonian.

## References
[1] A. Kandala, et. al, Nature **549**, 242-246 (2017)   
[2] I. Hamamura, et. al, npj Quantum Inf **6**, 56 (2020)  
[3] A. Zhao, et. al, Phys. Rev. A **101**, 062322 (2020)

