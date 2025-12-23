from pygridsynth.mixed_synthesis import (
    compute_diamond_norm_error,
    mixed_synthesis_sequential,
)
from pygridsynth.mymath import random_su

# Generate a random SU(2^n) unitary matrix
num_qubits = 2
unitary = random_su(num_qubits)

# Parameters
eps = 1e-4  # Error tolerance
M = 64  # Number of Hermitian operators for perturbation
seed = 123  # Random seed for reproducibility

# Compute mixed synthesis (sequential version)
result = mixed_synthesis_sequential(unitary, num_qubits, eps, M, seed=seed)

if result is not None:
    circuit_list, eu_np_list, probs_gptm, u_choi, u_choi_opt = result
    print(f"Number of circuits: {len(circuit_list)}")
    print(f"Mixing probabilities: {probs_gptm}")
    error = compute_diamond_norm_error(u_choi, u_choi_opt, eps)
    print(f"error: {error}")
