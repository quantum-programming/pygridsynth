"""
Utility functions for mixed synthesis.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
from numba import njit


@njit
def vector_norm(x: np.ndarray) -> float:
    """Compute Euclidean norm of a 1D array."""
    s = 0.0
    for i in range(x.shape[0]):
        s += x[i] * x[i]
    return np.sqrt(s)


@njit
def repulsive_update_numba(
    points: np.ndarray, step: float, epsilon: float
) -> np.ndarray:
    """
    Update points based on repulsive forces between them.
    For each point i, compute repulsive force from all other points:
      force = Σ_{j≠i} (x_i - x_j) / (||x_i - x_j||^2 + epsilon)
    Project to tangent space, move by step, and project back to sphere.
    """
    M, d = points.shape
    new_points = np.empty_like(points)
    for i in range(M):
        force = np.zeros(d)
        for j in range(M):
            if i == j:
                continue
            diff = points[i] - points[j]
            dist_sq = 0.0
            for k in range(d):
                dist_sq += diff[k] * diff[k]
            for k in range(d):
                force[k] += diff[k] / (dist_sq + epsilon)
        # Project to tangent space: subtract (force・x_i) x_i from force
        dot = 0.0
        for k in range(d):
            dot += force[k] * points[i][k]
        tangent_force = np.empty(d)
        for k in range(d):
            tangent_force[k] = force[k] - dot * points[i][k]
        # Update point and compute new position
        x_new = np.empty(d)
        for k in range(d):
            x_new[k] = points[i][k] + step * tangent_force[k]
        norm_val = vector_norm(x_new)
        for k in range(d):
            new_points[i][k] = x_new[k] / norm_val
    return new_points


def relax_points_numba(
    points: np.ndarray,
    iterations: int = 100,
    step: float = 0.001,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Relax point cloud using Numba-compiled repulsive_update_numba.

    Args:
        points: Point cloud to relax.
        iterations: Number of relaxation iterations.
        step: Step size for updates.
        epsilon: Small constant for numerical stability.

    Returns:
        Relaxed point cloud.
    """
    for _ in range(iterations):
        points = repulsive_update_numba(points, step, epsilon)
    return points


def initial_points(d: int, M: int, seed: int | None = None) -> np.ndarray:
    """
    Uniformly sample M points from the unit sphere S^(d-1) in d dimensions.

    Args:
        d: Dimension of the space.
        M: Number of points to sample.
        seed: Random seed (optional).

    Returns:
        Array of shape (M, d) containing the sampled points.
    """
    points = np.empty((M, d))
    if seed is not None:
        np.random.seed(seed)
    for i in range(M):
        x = np.random.randn(d)
        norm_val = np.linalg.norm(x)
        points[i] = x / norm_val
    return points


def vector_to_upper_triangular(vector: np.ndarray, d: int) -> np.ndarray:
    """
    Convert a vector to an upper triangular matrix.

    Args:
        vector: Vector to convert.
        d: Dimension of the square matrix.

    Returns:
        Upper triangular matrix (excluding diagonal).
    """
    matrix = np.zeros((d, d))
    indices = np.triu_indices(d, k=1)
    matrix[indices] = vector
    return matrix


def points_to_hermitian_matrix(points: np.ndarray) -> np.ndarray:
    """
    Convert points to a Hermitian matrix.

    Args:
        points: Points array.

    Returns:
        Hermitian matrix.
    """
    hilbert_dim = int(np.sqrt(len(points) + 1))
    assert hilbert_dim**2 - 1 == len(points)

    non_diag_real = points[: (hilbert_dim * (hilbert_dim - 1)) // 2]
    non_diag_imag = points[
        (hilbert_dim * (hilbert_dim - 1)) // 2 : hilbert_dim * (hilbert_dim - 1)
    ]
    diag = list(points[hilbert_dim * (hilbert_dim - 1) :]) + [
        -sum(points[(hilbert_dim * (hilbert_dim - 1)) :])
    ]

    mat = vector_to_upper_triangular(
        non_diag_real, hilbert_dim
    ) + 1j * vector_to_upper_triangular(non_diag_imag, hilbert_dim)
    mat = mat + mat.conj().T
    mat += np.diag(diag)

    return mat


def operate_M_tensor(vector: np.ndarray, M: np.ndarray) -> None:
    r"""
    In-place Fast Walsh-Hadamard Transform-like operation M^\otimes n @ vec.

    Args:
        vector: Vector to transform (modified in-place).
        M: Transformation matrix.
    """
    h = 1
    d = M.shape[0]

    while h < len(vector):
        # perform FWHT
        for i in range(0, len(vector), h * d):
            for j in range(i, i + h):
                if d == 2:
                    x = vector[j % len(vector)]
                    y = vector[(j + h) % len(vector)]
                    vector[j % len(vector)] = M[0, 0] * x + M[0, 1] * y
                    vector[(j + h) % len(vector)] = M[1, 0] * x + M[1, 1] * y
                elif d == 4:
                    x = vector[j % len(vector)]
                    y = vector[(j + h) % len(vector)]
                    z = vector[(j + 2 * h) % len(vector)]
                    w = vector[(j + 3 * h) % len(vector)]

                    vector[j % len(vector)] = (
                        M[0, 0] * x + M[0, 1] * y + M[0, 2] * z + M[0, 3] * w
                    )
                    vector[(j + h) % len(vector)] = (
                        M[1, 0] * x + M[1, 1] * y + M[1, 2] * z + M[1, 3] * w
                    )
                    vector[(j + 2 * h) % len(vector)] = (
                        M[2, 0] * x + M[2, 1] * y + M[2, 2] * z + M[2, 3] * w
                    )
                    vector[(j + 3 * h) % len(vector)] = (
                        M[3, 0] * x + M[3, 1] * y + M[3, 2] * z + M[3, 3] * w
                    )

        # normalize and increment
        h *= d


def permutate_qubit_vector(vector: np.ndarray, new_order: list[int]) -> np.ndarray:
    """
    Change qubit order (replacement for qulacs permutate_qubit).

    Args:
        vector: Vector of dimension 2^(n_qubit).
        new_order: New qubit order (e.g., [0, 2, 1, 3] for 2 qubits).

    Returns:
        Vector with reordered qubits.
    """
    n_qubits = len(new_order)
    dim = 2**n_qubits

    tensor = vector.reshape([2] * n_qubits)
    transposed = np.transpose(tensor, new_order)
    return transposed.reshape(dim)


def pauli_vec_to_state(pauli_vec: np.ndarray) -> np.ndarray:
    """
    Convert Pauli vector to density matrix (implementation without qulacs).

    Args:
        pauli_vec: Pauli vector.

    Returns:
        Density matrix.
    """
    n_qubit = (pauli_vec.shape[0].bit_length() - 1) // 2
    M_inv = np.array([[1, 0, 0, 1], [0, 1, -1j, 0], [0, 1, 1j, 0], [1, 0, 0, -1]])
    new_order = [2 * i for i in range(n_qubit)] + [2 * i + 1 for i in range(n_qubit)]

    rho_vec_tmp = pauli_vec.copy()
    operate_M_tensor(rho_vec_tmp, M_inv)

    # Change qubit order (replacement for qulacs permutate_qubit)
    rho_vec = permutate_qubit_vector(rho_vec_tmp, new_order)

    # Reshape to density matrix
    rho = rho_vec.reshape(2**n_qubit, 2**n_qubit)
    return rho


def get_random_hermitian_operator(
    hilbert_dim: int,
    num_herm: int = 1,
    iterations: int = 100,
    step: float = 0.001,
    seed: int | None = None,
) -> list[np.ndarray]:
    """
    Generate random Hermitian operators.

    Args:
        hilbert_dim: Hilbert space dimension.
        num_herm: Number of Hermitian operators to generate.
        iterations: Number of relaxation iterations.
        step: Step size for relaxation.
        seed: Random seed (optional).

    Returns:
        List of Hermitian matrices.
    """
    d = hilbert_dim**2 - 1
    # Generate initial point cloud
    points = initial_points(d, num_herm, seed=seed)

    # Lightweight relaxation using Numba
    points_relaxed = relax_points_numba(points, iterations=iterations, step=step)

    herm_list = []
    for _points in points_relaxed:
        if hilbert_dim == 2 ** (hilbert_dim.bit_length() - 1):
            tmp = pauli_vec_to_state(np.concatenate(([0j], _points)))
            herm = tmp / np.sqrt(2)
        else:
            tmp = points_to_hermitian_matrix(_points)
            herm = tmp / np.sqrt(hilbert_dim)
        herm_list.append(herm)
    return herm_list


def hermitian_generalized_pauli_operators(d: int) -> list[np.ndarray]:
    r"""
    Construct Hermitian basis {H_k} (k=0,...,d^2-1) for d-dimensional Hilbert space
    from Weyl-Heisenberg generalized Pauli operators Q_{a,b} = X^a Z^b, a,b = 0,...,d-1.

    Note: Standard Q_{a,b} are not Hermitian for d>2, so we construct Hermitian
    basis from pairs Q_{a,b} and Q_{a,b}^\dagger:
        H^{(R)}_{a,b} = (Q_{a,b}+Q_{a,b}^\dagger)/√2
        H^{(I)}_{a,b} = -i (Q_{a,b}-Q_{a,b}^\dagger)/√2

    For self-adjoint cases (2a≡0, 2b≡0 mod d), we use phase correction:
        H = exp(-iπ a b/d) Q_{a,b}

    Args:
        d: Hilbert space dimension.

    Returns:
        List of d^2 Hermitian (d,d) matrices (numpy arrays).
    """
    H_ops = []
    omega = np.exp(2j * np.pi / d)

    # Define shift operator X
    X = np.zeros((d, d), dtype=complex)
    for j in range(d):
        X[j, (j + 1) % d] = 1

    # Define phase operator Z
    Z = np.diag([omega**j for j in range(d)])

    # Set for duplicate checking
    processed = set()

    for a in range(d):
        for b in range(d):
            # Key for (a,b) (considering adjoint (a',b') for order independence)
            key = (a, b)
            # Adjoint corresponds to exponents (-a mod d, -b mod d)
            a_p, b_p = (-a) % d, (-b) % d
            key_partner = (a_p, b_p)

            # Skip if already processed
            if key in processed or key_partner in processed:
                continue

            # Q_{a,b} = X^a Z^b
            Q = np.linalg.matrix_power(X, a) @ np.linalg.matrix_power(Z, b)
            Q_dag = Q.conj().T  # Q^\dagger

            # Self-adjoint case (up to phase): 2a≡0 and 2b≡0 (mod d)
            if ((2 * a) % d == 0) and ((2 * b) % d == 0):
                # Q_dag = ω^{-ab} Q,
                # so phase correction φ = ω^{-ab/2} makes it Hermitian
                phase = np.exp(-1j * np.pi * a * b / d)
                H = phase * Q
                H_ops.append(H)
                processed.add(key)
            else:
                # Create Hermitian combinations from pairs
                # Note: Q_dag = ω^{-ab} Q_{a_p,b_p},
                # but we use Q and Q_dag directly here
                H_R = (Q + Q_dag) / np.sqrt(2)
                H_I = -1j * (Q - Q_dag) / np.sqrt(2)
                H_ops.append(H_R)
                H_ops.append(H_I)
                processed.add(key)
                processed.add(key_partner)

    # Number of operators should be d^2
    if len(H_ops) != d**2:
        raise ValueError(
            f"Generated {len(H_ops)} Hermitian operators, but should be d^2 = {d**2}."
        )

    return H_ops


def unitary_to_gptm(U: np.ndarray) -> np.ndarray:
    """
    Compute GPTM (Generalized Pauli Transfer Matrix) for unitary channel Λ(ρ) = U ρ U†
    using Hermitian generalized Pauli basis {H_k}.

    Definition:
        M_{ij} = (1/d) Tr[ H_j  U H_i U† ]

    Args:
        U: Unitary matrix of dimension (d,d).

    Returns:
        GPTM matrix of size (d^2, d^2).
    """
    d = U.shape[0]
    H_ops = hermitian_generalized_pauli_operators(d)
    num_ops = len(H_ops)
    M = np.zeros((num_ops, num_ops), dtype=complex)
    U_dag = U.conj().T

    for i, H_i in enumerate(H_ops):
        transformed = U @ H_i @ U_dag
        for j, H_j in enumerate(H_ops):
            M[j, i] = np.trace(H_j @ transformed) / d
    return M


def unitary_to_choi(unitary: np.ndarray) -> np.ndarray:
    """
    Convert unitary matrix to Choi matrix.

    Args:
        unitary: Unitary matrix.

    Returns:
        Choi matrix.
    """
    dim = unitary.shape[0]
    max_entangled_vec = np.identity(dim).reshape(dim * dim)

    large_unitary = np.kron(np.identity(dim), unitary)
    vec = large_unitary @ max_entangled_vec

    return np.outer(vec, vec.conj())


def choi_to_unitary(choi: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """
    Convert Choi matrix to unitary matrix.

    Args:
        choi: Choi matrix.
        tol: Tolerance for rank check.

    Returns:
        Unitary matrix.

    Raises:
        AssertionError: If input does not correspond to a unitary.
    """
    vals, vecs = np.linalg.eigh(choi)
    assert (
        np.linalg.matrix_rank(choi, tol=tol) == 1
    ), "input does not correspond to unitary."
    n_qubit = (len(vals).bit_length() - 1) // 2

    u_ret = vecs[:, np.argmax(vals)].reshape(2**n_qubit, 2**n_qubit).T * np.sqrt(
        np.max(vals)
    )
    u_ret /= u_ret[0, 0] / (np.abs(u_ret[0, 0]))
    return u_ret


def diamond_norm_choi(
    choi1: np.ndarray,
    choi2: np.ndarray | None = None,
    scale: float = 1,
    solver: str | None = None,
) -> float:
    """
    Compute the diamond norm of the difference channel whose Choi matrix is J_delta.

    Args:
        choi1: Choi matrix of the first channel, shape (d*d, d*d).
        choi2: Choi matrix of the second channel, shape (d*d, d*d) (optional).
        scale: Scaling factor.
        solver: The CVXPY solver to use (None for default: cp.SCS).

    Returns:
        The computed diamond norm.
    """
    if choi2 is None:
        dim = int(np.sqrt(choi1.shape[0]))
        choi2 = unitary_to_choi(np.diag(np.ones(dim)))

    if solver is None:
        solver = cp.SCS

    J_delta = (choi1 - choi2) * scale

    d = int(np.sqrt(J_delta.shape[0]))
    n = d * d
    # Variable for the lifted operator in the SDP: an n x n Hermitian matrix.
    X = cp.Variable((n, n), hermitian=True)
    # Scalar variable t that will bound the operator norm of the partial trace.
    t = cp.Variable(nonneg=True)

    constraints = []
    # X must be positive semidefinite and also dominate J_delta.
    constraints += [X >> 0, X - J_delta >> 0]

    # We now define the partial trace of X over the second subsystem.
    # For a block-structured matrix X (with blocks of size d x d),
    # the (i,j) entry of Y = Tr_B(X) is given by
    # summing the (i*d+k, j*d+k) entries of X.
    Y = cp.Variable((d, d), hermitian=True)
    for i in range(d):
        for j in range(d):
            # Sum over the appropriate indices to get (Y)_{ij}
            constraints.append(
                Y[i, j] == cp.sum([X[i * d + k, j * d + k] for k in range(d)])
            )

    # Impose the operator norm constraint: ||Y||_∞ ≤ t.
    # This is equivalent to: t*I - Y >= 0 and t*I + Y >= 0.
    constraints.append(t * np.eye(d) - Y >> 0)
    constraints.append(t * np.eye(d) + Y >> 0)

    # Set up the SDP: minimize t subject to the constraints.
    prob = cp.Problem(cp.Minimize(t), constraints)
    prob.solve(solver=solver)

    return prob.value * 2 / scale
