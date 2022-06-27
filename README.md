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

## Folders

## Usage

## References
[1] A. Kandala, et. al, Nature **549**, 242-246 (2017)   
[2] I. Hamamura, et. al, npj Quantum Inf **6**, 56 (2020)  
[3] A. Zhao, et. al, Phys. Rev. A **101**, 062322 (2020)

