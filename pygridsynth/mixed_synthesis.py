"""
Mixed synthesis module.
"""

from __future__ import annotations

import os
from multiprocessing import Pool

import cvxpy as cp
import mpmath
import numpy as np
import scipy

from .mixed_synthesis_utils import (
    get_random_hermitian_operator,
    unitary_to_choi,
    unitary_to_gptm,
)
from .multi_qubit_unitary_approximation import approximate_multi_qubit_unitary
from .mymath import dps_for_epsilon
from .quantum_circuit import QuantumCircuit
from .quantum_gate import CxGate, HGate, SGate, SXGate, TGate, WGate  # noqa: F401


def _get_default_solver() -> str | None:
    """
    Get the default solver with priority order.

    Returns:
        Available solver name, or None if no solver is available.
    """
    available_solvers = cp.installed_solvers()

    preferred_solvers = ["GUROBI", "SCS", "ECOS", "OSQP"]

    for solver_name in preferred_solvers:
        if solver_name in available_solvers:
            return solver_name

    if available_solvers:
        return available_solvers[0]

    return None


def solve_LP(
    A: np.ndarray,
    b: np.ndarray,
    eps: float,
    scale: float = 1.0,
    solver: str | None = None,
) -> np.ndarray | None:
    """
    Solve a linear programming problem for optimal mixing probabilities.

    Args:
        A: Constraint matrix.
        b: Constraint vector.
        eps: Error tolerance parameter.
        scale: Scale factor.
        solver: Solver to use (None for auto-selection).

    Returns:
        Optimal solution vector, or None on failure.
    """
    n, m = A.shape
    x = cp.Variable(m)
    r = cp.Variable(n)

    x.value = np.ones(m) / (eps * m)
    objective = cp.Minimize(cp.sum(r))
    constraints = [
        cp.sum(x) * eps == 1,
        x >= 0,
        r >= 0,
        A @ x - b / eps <= r / scale,
        -(A @ x - b / eps) <= r / scale,
    ]
    prob = cp.Problem(objective, constraints)

    if solver is None:
        solver = _get_default_solver()
        if solver is None:
            return None

    try:
        prob.solve(solver=solver, verbose=False)
    except Exception:
        return None
    if x.value is None:
        return None
    else:
        x = x.value / np.sum(x.value)
        return x


def approximate_unitary_task(
    eu: list[list[complex]], num_qubits: int, eps: float, dps: int = -1
) -> tuple[float, list[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute approximation task for a perturbed unitary matrix.

    Args:
        eu: Target unitary matrix perturbed by a Hermitian operator (as a list).
        num_qubits: Number of qubits.
        eps: Error tolerance parameter.
        dps: Decimal precision (default: -1 for auto).

    Returns:
        Tuple of (phase, gates, eu_np, eu_gptm, eu_choi).
        - phase: Phase of the circuit.
        - gates: List of gate strings in the circuit.
        - eu_np: Approximated unitary matrix (numpy array).
        - eu_gptm: GPTM representation of approximated unitary.
        - eu_choi: Choi representation of approximated unitary.
    """
    if dps == -1:
        dps = dps_for_epsilon(eps)
    with mpmath.workdps(dps):
        circuit, U_approx = approximate_multi_qubit_unitary(
            mpmath.matrix(eu), num_qubits, mpmath.mpf(eps), return_domega_matrix=False
        )
    eu_np = np.array(U_approx.tolist(), dtype=complex)
    eu_gptm = unitary_to_gptm(eu_np)
    eu_choi = unitary_to_choi(eu_np)
    return float(circuit.phase), [str(g) for g in circuit], eu_np, eu_gptm, eu_choi


def compute_optimal_mixing_probabilities(
    u_gptm: np.ndarray,
    num_qubits: int,
    eu_gptm_list: list[np.ndarray],
    eu_choi_list: list[np.ndarray],
    eps: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Compute optimal mixing probabilities
    from GPTM representation of perturbed unitaries.

    Args:
        u_gptm: GPTM representation of target unitary.
        num_qubits: Number of qubits.
        eu_gptm_list: List of GPTM representations of perturbed unitaries.
        eu_choi_list: List of Choi representations of perturbed unitaries.
        eps: Error tolerance parameter.

    Returns:
        Tuple of (probs_gptm, u_choi_opt), or None on failure.
        - probs_gptm: Array of mixing probabilities.
        - u_choi_opt: Optimal Choi matrix of the mixed unitary.
    """
    A = np.array([eu_gptm.reshape(eu_gptm.size) for eu_gptm in eu_gptm_list]).T.real
    b = u_gptm.reshape(2 ** (4 * num_qubits)).real

    probs_gptm = solve_LP(A, b, eps=eps, scale=1e-2 / eps)
    if probs_gptm is None:
        return None
    else:
        u_choi_opt = np.einsum("i,ijk->jk", probs_gptm, eu_choi_list)

    return probs_gptm, u_choi_opt


def process_unitary_approximation_parallel(
    unitary: mpmath.matrix,
    num_qubits: int,
    eps: float,
    M: int,
    seed: int = 123,
    dps: int = -1,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[QuantumCircuit],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
]:
    """
    Process unitary approximation in parallel.

    Args:
        unitary: Target unitary matrix.
        num_qubits: Number of qubits.
        eps: Error tolerance parameter.
        M: Number of Hermitian operators for perturbation.
        seed: Random seed.
        dps: Decimal precision (default: -1 for auto).

    Returns:
        Tuple of (u_gptm, u_choi, circuit_list, eu_np_list, eu_gptm_list, eu_choi_list).
        - u_gptm: GPTM representation of target unitary.
        - u_choi: Choi representation of target unitary.
        - circuit_list: List of QuantumCircuit objects for perturbed unitaries.
        - eu_np_list: List of target unitary matrices perturbed by Hermitian operators.
        - eu_gptm_list: List of GPTM representations of perturbed unitaries.
        - eu_choi_list: List of Choi representations of perturbed unitaries.
    """
    herm_list = get_random_hermitian_operator(2**num_qubits, M, seed=seed)
    unitary_np = np.array(unitary.tolist(), dtype=complex)

    u_choi = unitary_to_choi(unitary_np)
    u_gptm = unitary_to_gptm(unitary_np)
    eu_list = [
        (unitary @ mpmath.matrix(scipy.linalg.expm(1j * _herm * eps))).tolist()
        for _herm in herm_list
    ]

    num_workers = os.cpu_count()
    with Pool(processes=num_workers) as pool:
        results = pool.starmap(
            approximate_unitary_task,
            [(eu, num_qubits, eps, dps) for eu in eu_list],
        )
    circuit_list = [
        QuantumCircuit(phase=result[0], args=[eval(g) for g in result[1]])
        for result in results
    ]
    eu_np_list = [result[2] for result in results]
    eu_gptm_list = [result[3] for result in results]
    eu_choi_list = [result[4] for result in results]
    return (u_gptm, u_choi, circuit_list, eu_np_list, eu_gptm_list, eu_choi_list)


def process_unitary_approximation_sequential(
    unitary: mpmath.matrix,
    num_qubits: int,
    eps: float,
    M: int,
    seed: int = 123,
    dps: int = -1,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[QuantumCircuit],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
]:
    """
    Process unitary approximation sequentially.

    Args:
        unitary: Target unitary matrix.
        num_qubits: Number of qubits.
        eps: Error tolerance parameter.
        M: Number of Hermitian operators for perturbation.
        seed: Random seed.
        dps: Decimal precision (default: -1 for auto).

    Returns:
        Tuple of (u_gptm, u_choi, circuit_list, eu_np_list, eu_gptm_list, eu_choi_list).
        - u_gptm: GPTM representation of target unitary.
        - u_choi: Choi representation of target unitary.
        - circuit_list: List of QuantumCircuit objects for perturbed unitaries.
        - eu_np_list: List of target unitary matrices perturbed by Hermitian operators.
        - eu_gptm_list: List of GPTM representations of perturbed unitaries.
        - eu_choi_list: List of Choi representations of perturbed unitaries.
    """
    herm_list = get_random_hermitian_operator(2**num_qubits, M, seed=seed)
    unitary_np = np.array(unitary.tolist(), dtype=complex)
    u_choi = unitary_to_choi(unitary_np)
    u_gptm = unitary_to_gptm(unitary_np)
    eu_list = [
        (unitary @ mpmath.matrix(scipy.linalg.expm(1j * _herm * eps))).tolist()
        for _herm in herm_list
    ]

    circuit_list = []
    eu_np_list = []
    eu_gptm_list = []
    eu_choi_list = []
    for eu in eu_list:
        result = approximate_unitary_task(eu, num_qubits, eps, dps=dps)
        circuit = QuantumCircuit(phase=result[0], args=[eval(g) for g in result[1]])
        circuit_list.append(circuit)
        eu_np_list.append(result[2])
        eu_gptm_list.append(result[3])
        eu_choi_list.append(result[4])
    return (u_gptm, u_choi, circuit_list, eu_np_list, eu_gptm_list, eu_choi_list)


def mixed_synthesis_parallel(
    unitary: mpmath.matrix | np.ndarray,
    num_qubits: int,
    eps: float,
    M: int,
    seed: int = 123,
    dps: int = -1,
) -> (
    tuple[list[QuantumCircuit], list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]
    | None
):
    """
    Compute mixed probabilities for mixed unitary synthesis (parallel version).

    Args:
        unitary: Target unitary matrix.
        num_qubits: Number of qubits.
        eps: Error tolerance parameter.
        M: Number of Hermitian operators for perturbation.
        seed: Random seed.
        dps: Decimal precision (default: -1 for auto).

    Returns:
        Tuple of (circuit_list, eu_np_list, probs_gptm, u_choi, u_choi_opt),
            or None on failure.
        - circuit_list: List of QuantumCircuits for perturbed unitaries.
        - eu_np_list: List of target unitary matrices perturbed by Hermitian operators.
        - probs_gptm: Array of mixing probabilities.
        - u_choi: Choi representation of target unitary.
        - u_choi_opt: Optimal mixed Choi matrix.
    """
    if not isinstance(unitary, mpmath.matrix):
        unitary = mpmath.matrix(unitary)

    u_gptm, u_choi, circuit_list, eu_np_list, eu_gptm_list, eu_choi_list = (
        process_unitary_approximation_parallel(
            unitary, num_qubits, eps, M, seed=seed, dps=dps
        )
    )
    result = compute_optimal_mixing_probabilities(
        u_gptm, num_qubits, eu_gptm_list, eu_choi_list, eps
    )
    if result is None:
        return None

    probs_gptm, u_choi_opt = result

    return circuit_list, eu_np_list, probs_gptm, u_choi, u_choi_opt


def mixed_synthesis_sequential(
    unitary: mpmath.matrix | np.ndarray,
    num_qubits: int,
    eps: float,
    M: int,
    seed: int = 123,
    dps: int = -1,
) -> (
    tuple[list[QuantumCircuit], list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]
    | None
):
    """
    Compute mixed probabilities for mixed unitary synthesis (sequential version).

    Args:
        unitary: Target unitary matrix.
        num_qubits: Number of qubits.
        eps: Error tolerance parameter.
        M: Number of Hermitian operators for perturbation.
        seed: Random seed.
        dps: Decimal precision (default: -1 for auto).

    Returns:
        Tuple of (circuit_list, eu_np_list, probs_gptm, u_choi, u_choi_opt),
            or None on failure.
        - circuit_list: List of QuantumCircuits for perturbed unitaries.
        - eu_np_list: List of target unitary matrices perturbed by Hermitian operators.
        - probs_gptm: Array of mixing probabilities.
        - u_choi: Choi representation of target unitary.
        - u_choi_opt: Optimal mixed Choi matrix.
    """
    if not isinstance(unitary, mpmath.matrix):
        unitary = mpmath.matrix(unitary)

    u_gptm, u_choi, circuit_list, eu_np_list, eu_gptm_list, eu_choi_list = (
        process_unitary_approximation_sequential(
            unitary, num_qubits, eps, M, seed=seed, dps=dps
        )
    )
    result = compute_optimal_mixing_probabilities(
        u_gptm, num_qubits, eu_gptm_list, eu_choi_list, eps
    )
    if result is None:
        return None

    probs_gptm, u_choi_opt = result

    return circuit_list, eu_np_list, probs_gptm, u_choi, u_choi_opt
