"""
Sequential execution version of mixed synthesis.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import cvxpy as cp
import numpy as np

from .mixed_synthesis import (
    compute_optimal_mixing_probabilities,
    process_unitary_approximation_sequential,
)
from .mymath import diamond_norm_error_from_choi, random_su

if TYPE_CHECKING:
    from .quantum_circuit import QuantumCircuit

warnings.filterwarnings("ignore", category=UserWarning)


def main() -> list[
    tuple[
        int,
        float,
        list[QuantumCircuit],
        list[np.ndarray],
        np.ndarray | None,
        float | None,
    ]
]:
    """
    Main processing function.

    Returns:
        List of results as tuples of
            (num_qubits, eps, circuits, eu_np_list, probs_gptm, error).
        - num_qubits: Number of qubits.
        - eps: Error tolerance parameter.
        - circuits: List of QuantumCircuits for perturbed unitaries.
        - eu_np_list: List of approximated unitary matrices (numpy arrays).
        - probs_gptm: Array of mixing probabilities, or None on failure.
        - error: Diamond norm error, or None on failure.
    """
    # Parameter settings
    eps_list = [1e-3, 1e-4]
    num_qubits_list = [1, 2]
    M_list = [16, 64]
    num_trial = 2

    # Initialize GUROBI solver
    if "GUROBI" in cp.installed_solvers():
        cp.Problem(cp.Minimize(cp.Variable(1)), []).solve(
            solver=cp.GUROBI, verbose=False
        )

    final_results: list[
        tuple[
            int,
            float,
            list[QuantumCircuit],
            list[np.ndarray],
            np.ndarray | None,
            float | None,
        ]
    ] = []

    for num_qubits, M in zip(num_qubits_list, M_list):
        unitaries = [random_su(num_qubits) for _ in range(num_trial)]

        # Process each combination of eps and unitary
        for eps in eps_list:
            for unitary in unitaries:
                # Process unitary approximation sequentially
                u_gptm, u_choi, circuits, eu_np_list, eu_gptm_list, eu_choi_list = (
                    process_unitary_approximation_sequential(
                        unitary, num_qubits, eps, M, seed=123
                    )
                )
                # Compute optimal mixing probabilities
                result = compute_optimal_mixing_probabilities(
                    u_gptm, num_qubits, eu_gptm_list, eu_choi_list, eps
                )
                if result is None:
                    final_results.append(
                        (num_qubits, eps, circuits, eu_np_list, None, None)
                    )
                else:
                    probs_gptm, u_choi_opt = result
                    error = diamond_norm_error_from_choi(
                        u_choi, u_choi_opt, eps, mixed_synthesis=True
                    )
                    final_results.append(
                        (num_qubits, eps, circuits, eu_np_list, probs_gptm, error)
                    )

    return final_results


if __name__ == "__main__":
    results = main()
    for result in results:
        num_qubits, eps, circuits, eu_np_list, probs_gptm, error = result
        print(f"num_qubits: {num_qubits}")
        print(f"eps: {eps}")
        print(f"error: {error:.4e}")
        print("--------------------------------")
