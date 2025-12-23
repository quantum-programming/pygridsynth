import mpmath

from pygridsynth.mixed_synthesis import mixed_synthesis_parallel

# Generate a random SU(2^n) unitary matrix
num_qubits = 2
unitary = mpmath.eye(2**num_qubits)

# Parameters
eps = 1e-4  # Error tolerance
M = 64  # Number of Hermitian operators for perturbation
seed = 123  # Random seed for reproducibility

# For faster computation with multiple cores
result = mixed_synthesis_parallel(unitary, num_qubits, eps, M, seed=seed)
