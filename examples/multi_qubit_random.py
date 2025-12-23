from pygridsynth.multi_qubit_unitary_approximation import (
    approximate_multi_qubit_unitary,
)
from pygridsynth.mymath import random_su

# Generate a random SU(2^n) unitary
num_qubits = 2
U = random_su(num_qubits)
epsilon = "1e-10"

# Approximate with high precision
circuit, U_approx = approximate_multi_qubit_unitary(U, num_qubits, epsilon)
