import time
import warnings
from typing import Literal, overload

import mpmath

from .config import GridsynthConfig
from .domega_unitary import DOmegaMatrix
from .gridsynth import gridsynth_circuit
from .mymath import (
    MPFConvertible,
    all_close,
    convert_theta_and_epsilon,
    dps_for_epsilon,
    kron,
)
from .quantum_circuit import QuantumCircuit
from .quantum_gate import CxGate, HGate, QuantumGate, RzGate
from .two_qubit_unitary_approximation import approximate_two_qubit_unitary
from .unitary_approximation import approximate_one_qubit_unitary


def _blockZXZ(
    U: mpmath.matrix, num_qubits: int, verbose: int = 0
) -> tuple[mpmath.matrix, mpmath.matrix, mpmath.matrix, mpmath.matrix]:
    n = 2**num_qubits
    n_half = 2 ** (num_qubits - 1)
    I = mpmath.eye(n_half)

    # upper left block
    X = U[0:n_half, 0:n_half]
    # lower left block
    U12 = U[0:n_half, n_half:n]

    # svd: M = W*diag(S)*V.H
    # X = W11*diag(S11)*Vh11
    # X = Sx*Ux
    W11, S11, Vh11 = mpmath.svd_c(X)
    Sx = W11 @ mpmath.diag(S11) @ W11.H
    Ux = W11 @ Vh11

    # svd: U12 = W12*diag(S12)*Vh12
    # Y = Sy*Uy
    W12, S12, Vh12 = mpmath.svd_c(U12)
    Sy = W12 @ mpmath.diag(S12) @ W12.H
    Uy = W12 @ Vh12

    A = (Sx + 1j * Sy) @ Ux
    C = -1j * Ux.H @ Uy
    B = U[n_half:n, 0:n_half] + U[n_half:n, n_half:n] @ C.H
    Z = 2 * A.H @ X - I

    if verbose >= 1:
        IC = mpmath.zeros(n)
        IC[:n_half, :n_half] = I
        IC[n_half:, n_half:] = C

        AB = mpmath.zeros(n)
        AB[:n_half, :n_half] = A
        AB[n_half:n, n_half:n] = B

        IZ = mpmath.zeros(n)
        IZ[:n_half, :n_half] = I + Z
        IZ[n_half:, n_half:] = I + Z
        IZ[n_half:, :n_half] = I - Z
        IZ[:n_half, n_half:] = I - Z

        print(
            "Block-ZXZ correct for matrix of shape: ",
            U.rows,
            U.cols,
            " ? ",
            all_close(0.5 * AB @ IZ @ IC, U),
        )

    return A, B, Z, C


def _demultiplex(
    M1: mpmath.matrix, M2: mpmath.matrix, verbose: int = 0
) -> tuple[mpmath.matrix, list[mpmath.mpf], mpmath.matrix]:
    eigenvalues, V = mpmath.eig(M1 @ M2.H)
    D_sqrt = [mpmath.sqrt(eigenvalue) for eigenvalue in eigenvalues]
    W = mpmath.diag(D_sqrt) @ V.H @ M2
    if verbose >= 1:
        print(
            "Demultiplexing correct? ",
            all_close(V @ mpmath.diag(D_sqrt) @ W, M1),
            all_close(V @ mpmath.diag(D_sqrt).conjugate() @ W, M2),
        )
    return V, D_sqrt, W


def _bitwise_inner_product(a: int, b: int) -> int:
    i = a & b
    # number of set bits in i
    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    i = (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24
    return i % 2


# returns M^k = (-1)^(b_(i-1)*g_(i-1)),
# where * is bitwise inner product, g = binary gray code, b = binary code.
def _genMk(n: int) -> mpmath.matrix:
    Mk = mpmath.matrix(n)
    for i in range(0, n):
        for j in range(0, n):
            Mk[i, j] = (-1) ** (_bitwise_inner_product((i), ((j) ^ (j) >> 1)))
    return Mk


# input D as (one dimensional) array
def _decompose_cz(
    D: mpmath.matrix,
    control_qubits: list[int],
    target_qubit: int,
    inverse: bool = False,
) -> QuantumCircuit:
    n = len(D)
    control_qubits.reverse()
    D_arg = [-2 * mpmath.arg(d) for d in D]
    ar = mpmath.lu_solve(_genMk(n), D_arg)
    circuit = QuantumCircuit()

    if n != 2 ** len(control_qubits):
        print(
            "Warning: shape mismatch for controlled Z between"
            "# control bits and length of D"
        )

    if inverse:
        for i in range(n - 1, -1, -1):
            idx = (i ^ (i >> 1) ^ (i + 1) ^ ((i + 1) >> 1)).bit_length() - 1
            if idx == len(control_qubits):
                posc = control_qubits[-1]
            else:
                posc = control_qubits[idx]
            circuit.append(CxGate(posc, target_qubit))
            circuit.append(RzGate(ar[i].real, target_qubit))

    else:
        for i in range(0, n):
            idx = (i ^ (i >> 1) ^ (i + 1) ^ ((i + 1) >> 1)).bit_length() - 1
            if idx == len(control_qubits):
                posc = control_qubits[-1]
            else:
                posc = control_qubits[idx]
            circuit.append(RzGate(ar[i].real, target_qubit))
            circuit.append(CxGate(posc, target_qubit))

    return circuit


def _decompose_recursively(
    U: mpmath.matrix, wires: list[int], verbose: int = 0
) -> QuantumCircuit:
    num_qubits = len(wires)
    if num_qubits == 1 or num_qubits == 2:
        return QuantumCircuit.from_list([QuantumGate(U, wires)])

    n_half = 2 ** (num_qubits - 1)
    I = mpmath.eye(n_half)
    circuit = QuantumCircuit()

    A1, A2, B, C = _blockZXZ(U, num_qubits, verbose=verbose)
    V_a, D_a, W_a = _demultiplex(A1, A2, verbose=verbose)
    V_c, D_c, W_c = _demultiplex(I, C, verbose=verbose)

    V_a_circ = _decompose_recursively(V_a, wires[1:], verbose=verbose)
    circuit += V_a_circ

    D_a_circ = _decompose_cz(D_a, wires[1:], wires[0])
    circuit += D_a_circ[:-1]
    circuit.append(HGate(wires[0]))

    new_middle_ul = W_a @ V_c
    Sz_I = kron(mpmath.matrix([[1, 0], [0, -1]]), mpmath.eye(2 ** (num_qubits - 2)))
    new_middle_lr = Sz_I @ W_a @ B @ V_c @ Sz_I
    V_m, D_sqrt_m, W_m = _demultiplex(new_middle_ul, new_middle_lr, verbose=verbose)

    V_m_circ = _decompose_recursively(V_m, wires[1:], verbose=verbose)
    circuit += V_m_circ

    D_sqrt_m_circ = _decompose_cz(D_sqrt_m, wires[1:], wires[0])
    circuit += D_sqrt_m_circ

    W_m_circ = _decompose_recursively(W_m, wires[1:], verbose=verbose)
    circuit += W_m_circ

    circuit.append(HGate(wires[0]))

    D_c_circ = _decompose_cz(D_c, wires[1:], wires[0], inverse=True)
    circuit += D_c_circ[1:]

    circ_W_c = _decompose_recursively(W_c, wires[1:], verbose=verbose)
    circuit += circ_W_c

    return circuit


@overload
def approximate_multi_qubit_unitary(
    U: mpmath.matrix,
    num_qubits: int,
    epsilon: MPFConvertible,
    cfg: GridsynthConfig | None = None,
    return_domega_matrix: Literal[True] = True,
    scale_epsilon: bool = True,
    **kwargs,
) -> tuple[QuantumCircuit, DOmegaMatrix]: ...


@overload
def approximate_multi_qubit_unitary(
    U: mpmath.matrix,
    num_qubits: int,
    epsilon: MPFConvertible,
    cfg: GridsynthConfig | None = None,
    return_domega_matrix: Literal[False] = False,
    scale_epsilon: bool = True,
    **kwargs,
) -> tuple[QuantumCircuit, mpmath.matrix]: ...


def approximate_multi_qubit_unitary(
    U: mpmath.matrix,
    num_qubits: int,
    epsilon: MPFConvertible,
    cfg: GridsynthConfig | None = None,
    return_domega_matrix: bool = False,
    scale_epsilon: bool = True,
    **kwargs,
) -> tuple[QuantumCircuit, DOmegaMatrix | mpmath.matrix]:
    if cfg is None:
        cfg = GridsynthConfig(**kwargs)
    elif kwargs:
        warnings.warn("When 'cfg' is provided, 'kwargs' are ignored.", stacklevel=2)

    if cfg.dps is None:
        cfg.dps = dps_for_epsilon(epsilon)
    cfg.up_to_phase = True

    if num_qubits == 1:
        return approximate_one_qubit_unitary(
            U,
            epsilon,
            wires=[0],
            decompose_partially=False,
            return_domega_matrix=return_domega_matrix,
            scale_epsilon=scale_epsilon,
            cfg=cfg,
        )
    elif num_qubits == 2:
        return approximate_two_qubit_unitary(
            U,
            epsilon,
            wires=[0, 1],
            decompose_partially=False,
            return_domega_matrix=return_domega_matrix,
            scale_epsilon=scale_epsilon,
            cfg=cfg,
        )

    with mpmath.workdps(cfg.dps):
        _, epsilon = convert_theta_and_epsilon("0", epsilon, dps=cfg.dps)
        if scale_epsilon:
            epsilon /= 18 * 4 ** (num_qubits - 2) - 3 * 2 ** (num_qubits - 1) + 3
        wires = list(range(num_qubits))

        start = time.time() if cfg.measure_time else 0.0
        decomposed_circuit = _decompose_recursively(U, wires=wires, verbose=cfg.verbose)
        if cfg.measure_time:
            print("--------------------------------")
            print(f"time of _decompose_recursively: {(time.time() - start) * 1000} ms")
            print("--------------------------------")
        diag = [1] * (2**2)
        circuit = QuantumCircuit()

        U_approx = DOmegaMatrix.identity(wires=wires)
        for i, c in enumerate(decomposed_circuit):
            if isinstance(c, RzGate):
                circuit_rz = gridsynth_circuit(c.theta, epsilon, wires=c.wires, cfg=cfg)
                circuit += circuit_rz
                U_approx = U_approx @ DOmegaMatrix.from_single_qubit_circuit(
                    circuit_rz, wires=c.wires
                )
            elif isinstance(c, CxGate):
                circuit.append(c)
                U_approx = U_approx @ DOmegaMatrix.from_cx_gate(c)
            elif isinstance(c, HGate):
                circuit.append(c)
                U_approx = U_approx @ DOmegaMatrix.from_single_qubit_gate(c)
            elif i == len(decomposed_circuit) - 1:
                c.matrix = mpmath.diag(diag) @ c.matrix
                scale = mpmath.det(c.matrix) ** (-1 / 4)
                circuit.phase -= mpmath.arg(scale)
                U_approx.phase -= mpmath.arg(scale)
                c.matrix *= scale
                tmp_circuit, tmp_unitary = approximate_two_qubit_unitary(
                    c.matrix,
                    epsilon,
                    wires=c.wires,
                    decompose_partially=False,
                    return_domega_matrix=True,
                    cfg=cfg,
                )
                circuit += tmp_circuit
                U_approx = U_approx @ tmp_unitary
            else:
                c.matrix = mpmath.diag(diag) @ c.matrix
                scale = mpmath.det(c.matrix) ** (-1 / 4)
                circuit.phase -= mpmath.arg(scale)
                U_approx.phase -= mpmath.arg(scale)
                c.matrix *= scale
                tmp_circuit, tmp_diag, tmp_unitary = approximate_two_qubit_unitary(
                    c.matrix,
                    epsilon,
                    wires=c.wires,
                    decompose_partially=True,
                    return_domega_matrix=True,
                    scale_epsilon=False,
                    cfg=cfg,
                )
                U_approx = U_approx @ tmp_unitary
                circuit += tmp_circuit
                diag = tmp_diag

        if cfg.measure_time:
            print("--------------------------------")
            print(
                "time of approximate_multi_qubit_unitary: "
                f"{(time.time() - start) * 1000} ms"
            )
            print("--------------------------------")
        if return_domega_matrix:
            return circuit, U_approx
        else:
            return circuit, U_approx.to_complex_matrix
