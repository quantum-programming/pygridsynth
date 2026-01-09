import mpmath

from pygridsynth.multi_qubit_unitary_approximation import (
    approximate_multi_qubit_unitary,
)

# Define a target unitary matrix (example: 2-qubit identity)
num_qubits = 2
U = mpmath.eye(2**num_qubits)  # 4x4 identity matrix
epsilon = "1e-10"

# Approximate the unitary
circuit, U_approx = approximate_multi_qubit_unitary(U, num_qubits, epsilon)

print(f"Circuit length: {len(circuit)}")
print(f"Circuit: {str(circuit)}")
