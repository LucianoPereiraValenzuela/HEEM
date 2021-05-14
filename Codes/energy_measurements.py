from qiskit import *
import numpy as np

sim = Aer.get_backend('qasm_simulator')

# circuit for the state Tri1
Tri1 = QuantumCircuit(2, 2)
Tri1.h(0)
Tri1.cx(0, 1)

# circuit for the state Tri2
Tri2 = QuantumCircuit(2, 2)
Tri2.x(0)
Tri2.h(0)
Tri2.cx(0, 1)

# circuit for the state Tri3
Tri3 = QuantumCircuit(2, 2)
Tri3.h(0)
Tri3.x(1)
Tri3.cx(0, 1)

# circuit for the state Sing
Sing = QuantumCircuit(2, 2)
Sing.x((0, 1))
Sing.h(0)
Sing.cx(0, 1)

# <ZZ>
measure_ZZ = QuantumCircuit(2, 2)
measure_ZZ.measure((0, 1), (0, 1))

# <XX>
measure_XX = QuantumCircuit(2, 2)
measure_XX.h((0, 1))
measure_XX.measure((0, 1), (0, 1))

# <YY>
measure_YY = QuantumCircuit(2, 2)
measure_YY.sdg((0, 1))
measure_YY.h((0, 1))
measure_YY.measure((0, 1), (0, 1))

shots = 2 ** 14  # number of samples used for statistics

A = 1.47e-6  # unit of A is eV
E_sim = []
for state_init in [Tri1, Tri2, Tri3, Sing]:
	Energy_meas = []
	for measure_circuit in [measure_XX, measure_YY, measure_ZZ]:
		# run the circuit with a the selected measurement and get the number of samples that output each bit value
		qc = state_init.compose(measure_circuit)
		qobj = assemble(qc, shots=shots)
		counts = sim.run(qobj).result().get_counts(qc)

		# calculate the probabilities for each computational basis
		probs = {}
		for output in ['00', '01', '10', '11']:
			if output in counts:
				probs[output] = counts[output] / shots
			else:
				probs[output] = 0

		Energy_meas.append(probs['00'] - probs['01'] - probs['10'] + probs['11'])

		print(qc)

	E_sim.append(A * np.sum(np.array(Energy_meas)))

# Run this cell to print out your results
print('Energy expection value of the state Tri1 : {:.3e} eV'.format(E_sim[0]))
print('Energy expection value of the state Tri2 : {:.3e} eV'.format(E_sim[1]))
print('Energy expection value of the state Tri3 : {:.3e} eV'.format(E_sim[2]))
print('Energy expection value of the state Sing : {:.3e} eV'.format(E_sim[3]))
