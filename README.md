![Entanglement Qubits - Apaisado](https://user-images.githubusercontent.com/11279156/120717557-0ae1b700-c4c8-11eb-92b0-54f718282f7d.png)
# Hardware-efficient-variational-quantum-eigensolver-with-entangled-measurements

The Variational Quantum Eigensolver (VQE) is a hybrid quantum-classical algorithm whose aim is to find the ground state of a Quantum Hamiltonian. 
Implementing VQE in NISQ devices is challenging. The two main obstacles are: 

1) Deep circuits are implemented with low quality. Hardware-efficient circuits implement gates with low error. [1].
2) Many circuits are required to evaluate the energy. Entanglement allows reducing the number of circuits [2, 3].

Both difficulties have been studied independently. ¿Can they be tackled simultaneously?

Hardware-efficient entangled measurements (HEEM) are measurements that take advantage of entanglement between neighboring qubits according to device’s architecture.

This is a library to implement the VQE using HEEM to estimate the energy.

## References
[1] Nature 549, 242 (2017).   
[2] Npj Quantum Information 6, 56 (2020).  
[3] Phys. Rev. A 101, 062322 (2020).
