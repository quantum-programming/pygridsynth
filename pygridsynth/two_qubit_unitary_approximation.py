import time
import warnings
from typing import Literal, overload

import mpmath
import numpy as np

from .config import GridsynthConfig
from .domega_unitary import DOmegaMatrix
from .gridsynth import gridsynth_circuit
from .mymath import (
    MPFConvertible,
    all_close,
    convert_theta_and_epsilon,
    dps_for_epsilon,
    kron,
    sqrt,
    trace,
)
from .quantum_circuit import QuantumCircuit
from .quantum_gate import CxGate, HGate, Rx, RxGate, Rz, RzGate, SingleQubitGate, cnot01
from .synthesis_of_cliffordT import decompose_domega_unitary
from .unitary_approximation import (
    approximate_one_qubit_unitary,
    euler_decompose,
    magnitude_approximate,
)


def _SU2SU2_to_tensor_products(U: mpmath.matrix) -> tuple[mpmath.matrix, mpmath.matrix]:
    U_reshaped = mpmath.matrix(
        np.array(U).reshape((2, 2, 2, 2)).transpose(0, 2, 1, 3).reshape(4, 4)
    )
    u, s, vh = mpmath.svd_c(U_reshaped)
    A = sqrt(s[0]) * mpmath.matrix(np.array(u[:, 0].tolist()).reshape((2, 2)))
    B = sqrt(s[0]) * mpmath.matrix(np.array(vh[0, :].tolist()).reshape((2, 2)))
    angle = mpmath.arg(mpmath.det(A)) / 2
    A *= mpmath.exp(-1.0j * angle)
    B *= mpmath.exp(1.0j * angle)
    return A, B


def _extract_SU2SU2_prefactors(
    U: mpmath.matrix, V: mpmath.matrix
) -> tuple[mpmath.matrix, mpmath.matrix, mpmath.matrix, mpmath.matrix]:
    u = E.H @ U @ E
    v = E.H @ V @ E
    uuT = u @ u.T
    vvT = v @ v.T
    uuT_real_plus_imag = mpmath.matrix(
        [
            [mpmath.re(uuT[i, j]) + mpmath.im(uuT[i, j]) for j in range(4)]
            for i in range(4)
        ]
    )
    vvT_real_plus_imag = mpmath.matrix(
        [
            [mpmath.re(vvT[i, j]) + mpmath.im(vvT[i, j]) for j in range(4)]
            for i in range(4)
        ]
    )
    _, p = mpmath.eigsy(uuT_real_plus_imag)
    _, q = mpmath.eigsy(vvT_real_plus_imag)
    p = p @ mpmath.diag([1, 1, 1, mpmath.sign(mpmath.det(p))])
    q = q @ mpmath.diag([1, 1, 1, mpmath.sign(mpmath.det(q))])

    G = p @ q.T
    H = v.H @ G.T @ u
    AB = E @ G @ E.H
    A, B = _SU2SU2_to_tensor_products(AB)
    CD = E @ H @ E.H
    C, D = _SU2SU2_to_tensor_products(CD)

    return A, B, C, D


Sy_Sy = mpmath.matrix([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]])
I_Sz = mpmath.matrix([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
E = (
    1
    / sqrt(2)
    * mpmath.matrix(
        [[1, 1.0j, 0, 0], [0, 0, 1.0j, 1], [0, 0, 1.0j, -1], [1, -1.0j, 0, 0]]
    )
)


def _gamma(u: mpmath.matrix) -> mpmath.matrix:
    return u @ Sy_Sy @ u.T @ Sy_Sy


def _decompose_with_2_cnots(
    U: mpmath.matrix, wires: list[int], verbose: int = 0
) -> QuantumCircuit:
    m = _gamma(U)

    eig_vals, _ = mpmath.eig(m)
    angle_eig_vals = sorted([mpmath.arg(eigval) for eigval in eig_vals])
    r = angle_eig_vals[0]
    s = angle_eig_vals[1]
    t = angle_eig_vals[2]
    diff_r_s = mpmath.fabs(mpmath.exp(1.0j * r) + mpmath.exp(1.0j * s))
    diff_r_t = mpmath.fabs(mpmath.exp(1.0j * r) + mpmath.exp(1.0j * t))
    if diff_r_s < diff_r_t:
        s, t = t, s
    r = mpmath.fabs(r)
    s = mpmath.fabs(s)
    theta = (r + s) / 2
    phi = (r - s) / 2

    V = cnot01() @ kron(Rx(theta), Rz(phi)) @ cnot01()
    A, B, C, D = _extract_SU2SU2_prefactors(U, V)

    if verbose >= 1:
        print(
            "_decompose_with_2_cnots correct?",
            all_close(U, kron(A, B) @ V @ kron(C, D)),
        )

    gates = [
        SingleQubitGate(A, wires[0]),
        SingleQubitGate(B, wires[1]),
        CxGate(wires[0], wires[1]),
        RxGate(theta, wires[0]),
        RzGate(phi, wires[1]),
        CxGate(wires[0], wires[1]),
        SingleQubitGate(C, wires[0]),
        SingleQubitGate(D, wires[1]),
    ]
    circuit = QuantumCircuit.from_list(gates)
    return circuit


@overload
def _decompose_two_qubit_unitary(
    U: mpmath.matrix,
    wires: list[int] = [0, 1],
    decompose_partially: Literal[True] = True,
    verbose: int = 0,
) -> tuple[QuantumCircuit, list[mpmath.mpf]]: ...


@overload
def _decompose_two_qubit_unitary(
    U: mpmath.matrix,
    wires: list[int] = [0, 1],
    decompose_partially: Literal[False] = False,
    verbose: int = 0,
) -> QuantumCircuit: ...


def _decompose_two_qubit_unitary(
    U: mpmath.matrix,
    wires: list[int] = [0, 1],
    decompose_partially: bool = False,
    verbose: int = 0,
) -> tuple[QuantumCircuit, list[mpmath.mpf]] | QuantumCircuit:
    if decompose_partially:
        gamma_U = _gamma(U.T)

        psi = mpmath.atan2(mpmath.im(trace(gamma_U)), mpmath.re(trace(I_Sz @ gamma_U)))
        Delta = mpmath.diag(
            [
                mpmath.exp(-1.0j * psi / 2),
                mpmath.exp(1.0j * psi / 2),
                mpmath.exp(1.0j * psi / 2),
                mpmath.exp(-1.0j * psi / 2),
            ]
        )
        Delta_inv_diag = [
            mpmath.exp(1.0j * psi / 2),
            mpmath.exp(-1.0j * psi / 2),
            mpmath.exp(-1.0j * psi / 2),
            mpmath.exp(1.0j * psi / 2),
        ]

        circuit = _decompose_with_2_cnots(U @ Delta, wires=wires)

        if verbose >= 1:
            print(
                "_decompose_two_qubit_unitary correct?",
                all_close(
                    U,
                    mpmath.exp(1.0j * circuit.phase)
                    * U
                    @ Delta
                    @ mpmath.diag(Delta_inv_diag),
                ),
            )
        return circuit, Delta_inv_diag
    else:
        U2 = mpmath.exp(0.25j * mpmath.mp.pi) * U @ cnot01()
        gamma_U = _gamma(U2.T)

        psi = mpmath.atan2(mpmath.im(trace(gamma_U)), mpmath.re(trace(I_Sz @ gamma_U)))

        Delta = cnot01() @ kron(mpmath.eye(2), Rz(psi)) @ cnot01()
        circuit = _decompose_with_2_cnots(U2 @ Delta, wires=wires)

        circuit.phase += -0.25 * mpmath.mp.pi
        circuit += [CxGate(wires[0], wires[1]), RzGate(-psi, wires[1])]
        if verbose >= 1:
            print(
                "_decompose_two_qubit_unitary correct?",
                all_close(
                    U,
                    mpmath.exp(1.0j * circuit.phase)
                    * kron(circuit[0].matrix, circuit[1].matrix)
                    @ cnot01()
                    @ kron(circuit[3].matrix, circuit[4].matrix)
                    @ cnot01()
                    @ kron(circuit[6].matrix, circuit[7].matrix)
                    @ cnot01()
                    @ kron(mpmath.eye(2), circuit[9].matrix),
                ),
            )
        return circuit


@overload
def approximate_two_qubit_unitary(
    U: mpmath.matrix,
    epsilon: MPFConvertible,
    wires: list[int] = [0, 1],
    cfg: GridsynthConfig | None = None,
    decompose_partially: Literal[True] = True,
    return_domega_matrix: Literal[True] = True,
    scale_epsilon: bool = True,
    **kwargs,
) -> tuple[QuantumCircuit, list[mpmath.mpf], DOmegaMatrix]: ...


@overload
def approximate_two_qubit_unitary(
    U: mpmath.matrix,
    epsilon: MPFConvertible,
    wires: list[int] = [0, 1],
    cfg: GridsynthConfig | None = None,
    decompose_partially: Literal[True] = True,
    return_domega_matrix: Literal[False] = False,
    scale_epsilon: bool = True,
    **kwargs,
) -> tuple[QuantumCircuit, list[mpmath.mpf], mpmath.matrix]: ...


@overload
def approximate_two_qubit_unitary(
    U: mpmath.matrix,
    epsilon: MPFConvertible,
    wires: list[int] = [0, 1],
    cfg: GridsynthConfig | None = None,
    decompose_partially: Literal[False] = False,
    return_domega_matrix: Literal[True] = True,
    scale_epsilon: bool = True,
    **kwargs,
) -> tuple[QuantumCircuit, DOmegaMatrix]: ...


@overload
def approximate_two_qubit_unitary(
    U: mpmath.matrix,
    epsilon: MPFConvertible,
    wires: list[int] = [0, 1],
    cfg: GridsynthConfig | None = None,
    decompose_partially: Literal[False] = False,
    return_domega_matrix: Literal[False] = False,
    scale_epsilon: bool = True,
    **kwargs,
) -> tuple[QuantumCircuit, mpmath.matrix]: ...


@overload
def approximate_two_qubit_unitary(
    U: mpmath.matrix,
    epsilon: MPFConvertible,
    wires: list[int] = [0, 1],
    cfg: GridsynthConfig | None = None,
    decompose_partially: Literal[False] = False,
    return_domega_matrix: bool = False,
    scale_epsilon: bool = True,
    **kwargs,
) -> tuple[QuantumCircuit, DOmegaMatrix | mpmath.matrix]: ...


def approximate_two_qubit_unitary(
    U: mpmath.matrix,
    epsilon: MPFConvertible,
    wires: list[int] = [0, 1],
    cfg: GridsynthConfig | None = None,
    decompose_partially: bool = False,
    return_domega_matrix: bool = False,
    scale_epsilon: bool = True,
    **kwargs,
) -> (
    tuple[QuantumCircuit, list[mpmath.mpf], DOmegaMatrix | mpmath.matrix]
    | tuple[QuantumCircuit, DOmegaMatrix | mpmath.matrix]
):
    if cfg is None:
        cfg = GridsynthConfig(**kwargs)
    elif kwargs:
        warnings.warn("When 'cfg' is provided, 'kwargs' are ignored.", stacklevel=2)

    if cfg.dps is None:
        cfg.dps = dps_for_epsilon(epsilon)
    cfg.up_to_phase |= decompose_partially

    with mpmath.workdps(cfg.dps):
        _, epsilon = convert_theta_and_epsilon("0", epsilon, dps=cfg.dps)
        if scale_epsilon:
            if decompose_partially:
                epsilon /= 12
            else:
                epsilon /= 15

        start = time.time() if cfg.measure_time else 0.0
        if decompose_partially:
            decomposed_circuit, diag = _decompose_two_qubit_unitary(
                U,
                wires=wires,
                decompose_partially=decompose_partially,
                verbose=cfg.verbose,
            )
        else:
            decomposed_circuit = _decompose_two_qubit_unitary(
                U,
                wires=wires,
                decompose_partially=decompose_partially,
                verbose=cfg.verbose,
            )
        if cfg.measure_time:
            print("--------------------------------")
            print(
                "time of _decompose_two_qubit_unitary: "
                f"{(time.time() - start) * 1000} ms"
            )
            print("--------------------------------")
        circuit = QuantumCircuit(phase=decomposed_circuit.phase)
        rx_theta_approx = magnitude_approximate(
            decomposed_circuit[3].theta, epsilon, cfg=cfg
        )
        phase_mag_rx_theta, phi1_mag_rx_theta, theta_mag_rx_theta, phi2_mag_rx_theta = (
            euler_decompose(rx_theta_approx.to_complex_matrix)
        )
        circuit_rx_theta_approx = decompose_domega_unitary(
            rx_theta_approx, wires=decomposed_circuit[3].wires, up_to_phase=True
        )
        rz_phi_approx = magnitude_approximate(
            decomposed_circuit[4].theta, epsilon, cfg=cfg
        )
        phase_mag_rz_phi, phi1_mag_rz_phi, theta_mag_rz_phi, phi2_mag_rz_phi = (
            euler_decompose(rz_phi_approx.to_complex_matrix)
        )
        circuit_rz_phi_approx = decompose_domega_unitary(
            rz_phi_approx, wires=decomposed_circuit[4].wires, up_to_phase=True
        )
        circuit_rz_phi_approx = (
            [HGate(target_qubit=decomposed_circuit[4].target_qubit)]
            + circuit_rz_phi_approx
            + [HGate(target_qubit=decomposed_circuit[4].target_qubit)]
        )

        circuit.phase -= phase_mag_rx_theta
        circuit.phase -= phase_mag_rz_phi
        decomposed_circuit[0].matrix = decomposed_circuit[0].matrix @ Rz(
            -phi1_mag_rx_theta
        )
        decomposed_circuit[1].matrix = decomposed_circuit[1].matrix @ Rx(
            -phi1_mag_rz_phi
        )
        decomposed_circuit[6].matrix = (
            Rz(-phi2_mag_rx_theta) @ decomposed_circuit[6].matrix
        )
        decomposed_circuit[7].matrix = (
            Rx(-phi2_mag_rz_phi) @ decomposed_circuit[7].matrix
        )

        U_approx = mpmath.eye(2**2) * mpmath.exp(1.0j * circuit.phase)
        U_approx = DOmegaMatrix.identity(wires=wires)
        U_approx.phase = circuit.phase
        for i in range(len(decomposed_circuit)):
            if isinstance(decomposed_circuit[i], CxGate):
                cx_gate = CxGate(
                    control_qubit=decomposed_circuit[i].control_qubit,
                    target_qubit=decomposed_circuit[i].target_qubit,
                )
                circuit.append(cx_gate)
                U_approx = U_approx @ DOmegaMatrix.from_cx_gate(cx_gate)
            else:
                if i == 3:
                    circuit += circuit_rx_theta_approx
                    U_approx = U_approx @ DOmegaMatrix.from_single_qubit_circuit(
                        circuit_rx_theta_approx, [decomposed_circuit[i].target_qubit]
                    )
                elif i == 4:
                    circuit += circuit_rz_phi_approx
                    U_approx = U_approx @ DOmegaMatrix.from_single_qubit_circuit(
                        circuit_rz_phi_approx, [decomposed_circuit[i].target_qubit]
                    )
                elif decompose_partially and i in [6, 7]:
                    tmp_circuit, tmp_rz, tmp_unitary = approximate_one_qubit_unitary(
                        decomposed_circuit[i].matrix,
                        epsilon,
                        wires=decomposed_circuit[i].wires,
                        decompose_partially=decompose_partially,
                        return_domega_matrix=True,
                        scale_epsilon=False,
                        cfg=cfg,
                    )
                    circuit += tmp_circuit
                    U_approx = U_approx @ tmp_unitary

                    if decomposed_circuit[i].target_qubit == wires[0]:
                        diag[0] *= mpmath.exp(-1.0j * tmp_rz.theta / 2)
                        diag[1] *= mpmath.exp(-1.0j * tmp_rz.theta / 2)
                        diag[2] *= mpmath.exp(1.0j * tmp_rz.theta / 2)
                        diag[3] *= mpmath.exp(1.0j * tmp_rz.theta / 2)
                    elif decomposed_circuit[i].target_qubit == wires[1]:
                        diag[0] *= mpmath.exp(-1.0j * tmp_rz.theta / 2)
                        diag[1] *= mpmath.exp(1.0j * tmp_rz.theta / 2)
                        diag[2] *= mpmath.exp(-1.0j * tmp_rz.theta / 2)
                        diag[3] *= mpmath.exp(1.0j * tmp_rz.theta / 2)
                elif i == 9:
                    circuit_rz_psi = gridsynth_circuit(
                        decomposed_circuit[i].theta,
                        epsilon,
                        wires=decomposed_circuit[i].wires,
                        cfg=cfg,
                    )
                    circuit += circuit_rz_psi
                    U_approx = U_approx @ DOmegaMatrix.from_single_qubit_circuit(
                        circuit_rz_psi, [decomposed_circuit[i].target_qubit]
                    )

                else:
                    tmp_circuit, tmp_unitary = approximate_one_qubit_unitary(
                        decomposed_circuit[i].matrix,
                        epsilon,
                        wires=decomposed_circuit[i].wires,
                        decompose_partially=False,
                        return_domega_matrix=True,
                        scale_epsilon=False,
                        cfg=cfg,
                    )
                    circuit += tmp_circuit
                    U_approx = U_approx @ tmp_unitary

        if not cfg.up_to_phase:
            circuit.decompose_phase_gate()
        if cfg.measure_time:
            print("--------------------------------")
            print(
                "time of approximate_two_qubit_unitary: "
                f"{(time.time() - start) * 1000} ms"
            )
            print("--------------------------------")
        if decompose_partially:
            if return_domega_matrix:
                return circuit, diag, U_approx
            else:
                return circuit, diag, U_approx.to_complex_matrix
        else:
            if return_domega_matrix:
                return circuit, U_approx
            else:
                return circuit, U_approx.to_complex_matrix
