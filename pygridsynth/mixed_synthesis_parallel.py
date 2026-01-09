"""
Parallel execution version of mixed synthesis.
"""

from __future__ import annotations

import os
import warnings
from multiprocessing import Pool
from typing import TYPE_CHECKING

import cvxpy as cp
import numpy as np

from .mixed_synthesis import (
    compute_optimal_mixing_probabilities,
    process_unitary_approximation_parallel,
)
from .mymath import diamond_norm_error_from_choi, random_su

if TYPE_CHECKING:
    from .quantum_circuit import QuantumCircuit

warnings.filterwarnings("ignore", category=UserWarning)


def my_task(
    args: tuple[
        int, np.ndarray, np.ndarray, int, list[np.ndarray], list[np.ndarray], float
    ],
) -> tuple[int, np.ndarray | None, float | None]:
    """
    Compute optimal mixing probabilities and diamond norm error for a task.

    Args:
        args: Tuple of
                (idx, u_gptm, u_choi, num_qubits, eu_gptm_list, eu_choi_list, eps).
            - idx: Task index.
            - u_gptm: GPTM representation of target unitary.
            - u_choi: Choi representation of target unitary.
            - num_qubits: Number of qubits.
            - eu_gptm_list: List of GPTM representations of perturbed unitaries.
            - eu_choi_list: List of Choi representations of perturbed unitaries.
            - eps: Error tolerance parameter.

    Returns:
        Tuple of (idx, probs_gptm, error).
        - idx: Task index.
        - probs_gptm: Array of mixing probabilities, or None on failure.
        - error: Diamond norm error, or None on failure.
    """
    idx, u_gptm, u_choi, num_qubits, eu_gptm_list, eu_choi_list, eps = args
    result = compute_optimal_mixing_probabilities(
        u_gptm, num_qubits, eu_gptm_list, eu_choi_list, eps
    )
    if result is None:
        return (idx, None, None)
    probs_gptm, u_choi_opt = result
    error = diamond_norm_error_from_choi(u_choi, u_choi_opt, eps, mixed_synthesis=True)
    return (idx, probs_gptm, error)


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
    eps_list = [1e-3, 1e-4, 1e-5, 1e-6]
    num_qubits_list = [1, 2]
    M_list = [16, 64]
    num_trial = 2

    # Initialize GUROBI solver
    cp.Problem(cp.Minimize(cp.Variable(1)), []).solve(solver=cp.GUROBI, verbose=False)

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
    tasks = []
    idx = 0

    for num_qubits, M in zip(num_qubits_list, M_list):
        unitaries = [random_su(num_qubits) for _ in range(num_trial)]

        # Process all unitary approximations first

        for eps in eps_list:
            for unitary in unitaries:
                # Process unitary approximation in parallel
                u_gptm, u_choi, circuits, eu_np_list, eu_gptm_list, eu_choi_list = (
                    process_unitary_approximation_parallel(
                        unitary, num_qubits, eps, M, seed=123
                    )
                )
                final_results.append(
                    (num_qubits, eps, circuits, eu_np_list, None, None)
                )
                tasks.append(
                    (idx, u_gptm, u_choi, num_qubits, eu_gptm_list, eu_choi_list, eps)
                )
                idx += 1

    # Compute optimal mixing probabilities in parallel
    num_workers = os.cpu_count()
    with Pool(processes=num_workers) as pool:
        results = pool.map(my_task, tasks)
    for result in results:
        idx, probs_gptm, error = result
        num_qubits, eps, circuits, eu_np_list, _, _ = final_results[idx]
        final_results[idx] = (
            num_qubits,
            eps,
            circuits,
            eu_np_list,
            probs_gptm,
            error,
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
