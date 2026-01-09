from pygridsynth.multi_qubit_unitary_approximation import (
    approximate_multi_qubit_unitary,
)
from pygridsynth.mymath import random_su

# Generate a random SU(2^n) unitary
num_qubits = 2
U = random_su(num_qubits)
epsilon = "1e-10"

# Return DOmegaMatrix instead of mpmath.matrix for more efficient representation
circuit, U_domega = approximate_multi_qubit_unitary(
    U, num_qubits, epsilon, return_domega_matrix=True
)

# Convert to complex matrix if needed
U_complex = U_domega.to_complex_matrix
