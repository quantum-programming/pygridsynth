import warnings
from itertools import accumulate
from typing import TypeAlias

import mpmath
import numpy as np

from .mixed_synthesis_utils import diamond_norm_choi, unitary_to_choi

RealNum: TypeAlias = int | float | mpmath.mpf
MPFConvertible: TypeAlias = RealNum | mpmath.mpf


def dps_for_epsilon(epsilon: MPFConvertible) -> int:
    e = mpmath.mpf(epsilon)
    k = -mpmath.log10(e)
    return int(15 + 2.5 * int(mpmath.ceil(k)))  # used in newsynth


def convert_theta_and_epsilon(
    theta: MPFConvertible, epsilon: MPFConvertible, dps: int
) -> tuple[mpmath.mpf, mpmath.mpf]:
    if isinstance(theta, float):
        warnings.warn(
            (
                f"pygridsynth is synthesizing the angle {theta}. "
                "Please verify that this is the intended value. "
                "Using float may introduce precision errors; "
                "consider using mpmath.mpf for exact precision."
            ),
            UserWarning,
            stacklevel=2,
        )

    if isinstance(epsilon, float):
        warnings.warn(
            (
                f"pygridsynth is using epsilon={epsilon} as the tolerance. "
                "Please verify that this is the intended value. "
                "Using float may introduce precision errors; "
                "consider using mpmath.mpf for exact precision."
            ),
            UserWarning,
            stacklevel=2,
        )

    with mpmath.workdps(dps):
        theta_mpf = mpmath.mpf(theta)
        epsilon_mpf = mpmath.mpf(epsilon)

    return theta_mpf, epsilon_mpf


def SQRT2() -> mpmath.mpf:
    return mpmath.sqrt(2)


def ntz(n: int) -> int:
    return 0 if n == 0 else ((n & -n) - 1).bit_count()


def floor(x: int | float | mpmath.mpf) -> int:
    return int(mpmath.floor(x, prec=0))


def ceil(x: int | float | mpmath.mpf) -> int:
    return int(mpmath.ceil(x, prec=0))


def sqrt(x: int | float | mpmath.mpf) -> mpmath.mpf:
    return mpmath.sqrt(x)


def log(x: int | float | mpmath.mpf) -> mpmath.mpf:
    return mpmath.log(x)


def sign(x: int | float | mpmath.mpf) -> int:
    return 1 if x > 0 else -1 if x < 0 else 0


def floorsqrt(x: int | float | mpmath.mpf) -> int:
    if x < 0:
        raise ValueError
    ok = 0
    ng = ceil(x) + 1
    while ng - ok > 1:
        mid = ok + (ng - ok) // 2
        if mid * mid <= x:
            ok = mid
        else:
            ng = mid
    return ok


def rounddiv(x: int, y: int) -> int:
    return (x + y // 2) // y if y > 0 else (x - (-y) // 2) // y


def pow_sqrt2(k: int) -> mpmath.mpf:
    k_div_2, k_mod_2 = k >> 1, k & 1
    return (1 << k_div_2) * SQRT2() if k_mod_2 else 1 << k_div_2


def floorlog(
    x: int | float | mpmath.mpf, y: int | float | mpmath.mpf
) -> tuple[int, float | mpmath.mpf]:
    if x <= 0:
        raise ValueError("math domain error")

    tmp = y
    m = 0
    while x >= tmp or x * tmp < 1:
        tmp *= tmp
        m += 1

    pow_y = reversed(list(accumulate([0] * (m - 1), lambda x0, x1: x0 * x0, initial=y)))
    n, r = (0, x) if x >= 1 else (-1, x * tmp)
    for p in pow_y:
        n <<= 1
        if r > p:
            r /= p
            n += 1
    return n, r


def solve_quadratic(
    a: int | float | mpmath.mpf,
    b: int | float | mpmath.mpf,
    c: int | float | mpmath.mpf,
) -> tuple[mpmath.mpf, mpmath.mpf] | None:
    if a < 0:
        a, b, c = -a, -b, -c
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return None
    s1 = -b - sqrt(discriminant)
    s2 = -b + sqrt(discriminant)
    if b >= 0:
        return (s1 / (2 * a), s2 / (2 * a))
    else:
        if c == 0:
            return (0, -b / a)
        else:
            return ((2 * c) / s2, (2 * c) / s1)


def mpmath_matrix_to_numpy(M: mpmath.matrix) -> np.ndarray:
    return np.array(M.tolist(), dtype=complex)


def numpy_matrix_to_mpmath(M: np.ndarray) -> mpmath.matrix:
    return mpmath.matrix(M.tolist())


def trace(M: mpmath.matrix) -> mpmath.mpf:
    return sum(M[i, i] for i in range(min(M.rows, M.cols)))


def kron(A: mpmath.matrix, B: mpmath.matrix) -> mpmath.matrix:
    return mpmath.matrix(np.kron(A.tolist(), B.tolist()))


def all_close(
    A: mpmath.matrix, B: mpmath.matrix, tol: mpmath.mpf = mpmath.mpf("1e-5")
) -> bool:
    return mpmath.norm(mpmath.matrix(A - B), p="inf") < tol


def einsum(subscripts: str, *operands: np.ndarray) -> np.ndarray:
    return np.einsum(subscripts, *operands)


def from_matrix_to_tensor(mat: list[list], n) -> np.ndarray:
    return np.array(mat, dtype=object).reshape([2] * n * 2)


def from_tensor_to_matrix(mat: np.ndarray, n) -> list[list]:
    return mat.reshape((2**n, 2**n)).tolist()


def random_su(n: int) -> mpmath.matrix:
    """
    Generate a random SU(n) unitary matrix.

    Args:
        n: Number of qubits.

    Returns:
        Random SU(2^n) unitary matrix.
    """
    dim = 2**n
    z = mpmath.matrix(np.random.random_sample((dim, dim))) + 1j * mpmath.matrix(
        np.random.random_sample((dim, dim))
    ) / mpmath.sqrt(2)
    q, _ = mpmath.qr(z)
    q /= mpmath.det(q) ** (1 / dim)
    return q


def diamond_norm_error(
    u: np.ndarray | mpmath.matrix,
    u_opt: np.ndarray | mpmath.matrix,
    eps: MPFConvertible,
) -> float:
    """
    Compute error using diamond norm.

    Args:
        u: Target unitary.
        u_opt: Mixed unitary.
        eps: Error tolerance parameter.

    Returns:
        Diamond norm error between the target and mixed unitaries.
    """
    if isinstance(u, mpmath.matrix):
        u = mpmath_matrix_to_numpy(u)
    if isinstance(u_opt, mpmath.matrix):
        u_opt = mpmath_matrix_to_numpy(u_opt)
    if isinstance(eps, mpmath.mpf):
        eps = float(eps)
    u_choi = unitary_to_choi(u)
    u_choi_opt = unitary_to_choi(u_opt)
    return diamond_norm_error_from_choi(u_choi, u_choi_opt, eps, mixed_synthesis=False)


def diamond_norm_error_from_choi(
    u_choi: np.ndarray,
    u_choi_opt: np.ndarray,
    eps: float,
    mixed_synthesis: bool = False,
) -> float:
    """
    Compute error using diamond norm.

    Args:
        u_choi: Choi representation of target unitary.
        u_choi_opt: Choi representation of mixed unitary.
        eps: Error tolerance parameter.
        mixed_synthesis: Whether the error is for mixed synthesis.

    Returns:
        Diamond norm error between the target and mixed unitaries.
    """
    scale = 1e-2 / eps**2 if mixed_synthesis else 1e-2 / eps
    return diamond_norm_choi(u_choi, u_choi_opt, scale=scale)
