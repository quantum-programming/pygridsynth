import time
import warnings
from typing import Literal, overload

import mpmath

from .config import GridsynthConfig
from .diophantine import Result, diophantine_dyadic
from .domega_unitary import DOmegaMatrix, DOmegaUnitary
from .gridsynth import gridsynth_circuit
from .mymath import MPFConvertible, convert_theta_and_epsilon, dps_for_epsilon, sqrt
from .normal_form import NormalForm
from .odgp import solve_scaled_ODGP
from .quantum_circuit import QuantumCircuit
from .quantum_gate import RzGate
from .region import Interval
from .ring import DRootTwo
from .synthesis_of_cliffordT import decompose_domega_unitary


def euler_decompose(
    unitary: mpmath.matrix,
) -> tuple[mpmath.mpf, mpmath.mpf, mpmath.mpf, mpmath.mpf]:
    # unitary = e^{i phase} e^{- i phi1 / 2 Z} e^{- i theta / 2 X} e^{- i phi2 / 2 Z}
    # 0 <= theta <= pi, - pi <= phase < pi, - pi <= phi1 < pi, - pi <= phi2 < pi

    det_arg = mpmath.arg(mpmath.det(unitary))
    psi1 = mpmath.arg(unitary[1, 1])
    psi2 = mpmath.arg(unitary[1, 0])

    phase = det_arg / 2
    theta = 2 * mpmath.atan2(mpmath.fabs(unitary[1, 0]), mpmath.fabs(unitary[1, 1]))
    phi1 = psi1 + psi2 - det_arg + mpmath.mp.pi / 2
    phi2 = psi1 - psi2 - mpmath.mp.pi / 2

    if phi1 >= mpmath.mp.pi:
        phi1 -= 2 * mpmath.mp.pi
        phase += mpmath.mp.pi
    if phi1 < -mpmath.mp.pi:
        phi1 += 2 * mpmath.mp.pi
        phase += mpmath.mp.pi
    if phi2 >= mpmath.mp.pi:
        phi2 -= 2 * mpmath.mp.pi
        phase += mpmath.mp.pi
    if phi2 <= -mpmath.mp.pi:
        phi2 += 2 * mpmath.mp.pi
        phase += mpmath.mp.pi
    if phase >= mpmath.mp.pi:
        phase -= 2 * mpmath.mp.pi

    return (phase, phi1, theta, phi2)


def _generate_epsilon_interval(theta: mpmath.mpf, epsilon: mpmath.mpf) -> Interval:
    l = mpmath.cos(-theta / 2 - epsilon / 2) ** 2
    r = mpmath.cos(-theta / 2 + epsilon / 2) ** 2
    if l > r:
        l, r = r, l
    if -epsilon <= theta <= epsilon:
        r = 1
    if mpmath.mp.pi - epsilon / 2 <= theta / 2 <= mpmath.mp.pi + epsilon / 2:
        r = 1
    if -mpmath.mp.pi - epsilon / 2 <= theta / 2 <= -mpmath.mp.pi + epsilon / 2:
        r = 1
    if mpmath.mp.pi / 2 - epsilon / 2 <= theta / 2 <= mpmath.mp.pi / 2 + epsilon / 2:
        l = 0

    return Interval(l, r)


def unitary_diamond_distance(u1: mpmath.matrix, u2: mpmath.matrix) -> mpmath.mpf:
    mat = u1.H @ u2
    eig_vals, _ = mpmath.eig(mat)
    phases = [mpmath.arg(eig_val) for eig_val in eig_vals]
    d = mpmath.fabs(mpmath.cos((phases[0] - phases[1]) / 2))
    return 2 * sqrt(1 - d**2)


def check(u_target: mpmath.matrix, gates: str) -> None:
    t_count = gates.count("T")
    h_count = gates.count("H")
    u_approx = DOmegaUnitary.from_gates(gates)
    e = unitary_diamond_distance(u_target, u_approx.to_complex_matrix)
    print(f"{gates=}")
    print(f"{t_count=}, {h_count=}")
    print(f"u_approx={u_approx.to_matrix}")
    print(f"{e=}")


def _magnitude_approximate_with_fixed_k(
    odgp_sets: tuple[Interval, Interval],
    k: int,
    cfg: GridsynthConfig,
) -> tuple[DOmegaUnitary | None, float, float]:
    start = time.time() if cfg.measure_time else 0.0
    sol = solve_scaled_ODGP(*odgp_sets, k)
    time_of_solve_ODGP = time.time() - start if cfg.measure_time else 0.0

    start = time.time() if cfg.measure_time else 0.0
    time_of_diophantine_dyadic = 0.0
    u_approx = None
    for m in sol:
        if m.parity == 0:
            continue
        z = diophantine_dyadic(m, seed=cfg.seed, loop_controller=cfg.loop_controller)
        if not isinstance(z, Result):
            if (z * z.conj).residue == 0:
                continue
            xi = 1 - DRootTwo.fromDOmega(z.conj * z)
            w = diophantine_dyadic(
                xi, seed=cfg.seed, loop_controller=cfg.loop_controller
            )
            if not isinstance(w, Result):
                z = z.reduce_denomexp()
                w = w.reduce_denomexp()
                if z.k > w.k:
                    w = w.renew_denomexp(z.k)
                elif z.k < w.k:
                    z = z.renew_denomexp(w.k)

                k1 = (z + w).reduce_denomexp().k
                k2 = (z + w.mul_by_omega()).reduce_denomexp().k
                if k1 <= k2:
                    u_approx = DOmegaUnitary(z, w, 0)
                else:
                    u_approx = DOmegaUnitary(z, w.mul_by_omega(), 0)
                break

    time_of_diophantine_dyadic += time.time() - start if cfg.measure_time else 0.0
    return u_approx, time_of_solve_ODGP, time_of_diophantine_dyadic


def magnitude_approximate(
    theta: MPFConvertible,
    epsilon: MPFConvertible,
    cfg: GridsynthConfig | None = None,
    **kwargs,
) -> DOmegaUnitary:
    if cfg is None:
        cfg = GridsynthConfig(**kwargs)
    elif kwargs:
        warnings.warn(
            "When 'cfg' is provided, 'kwargs' are ignored.",
            stacklevel=2,
        )

    if cfg.dps is None:
        cfg.dps = dps_for_epsilon(epsilon)

    with mpmath.workdps(cfg.dps):
        theta, epsilon = convert_theta_and_epsilon(theta, epsilon, dps=cfg.dps)

        epsilon_interval = _generate_epsilon_interval(theta, epsilon)
        unit_interval = Interval(0, 1)
        odgp_sets = (epsilon_interval, unit_interval)

        u_approx = None
        k = 0
        time_of_solve_ODGP = 0.0
        time_of_diophantine_dyadic = 0.0
        while True:
            u_approx, time1, time2 = _magnitude_approximate_with_fixed_k(
                odgp_sets, k, cfg=cfg
            )
            time_of_solve_ODGP += time1
            time_of_diophantine_dyadic += time2
            if u_approx is not None:
                break

            k += 1

        if cfg.measure_time:
            print(f"time of solve_ODGP: {time_of_solve_ODGP * 1000} ms")
            print(
                "time of diophantine_dyadic: " f"{time_of_diophantine_dyadic * 1000} ms"
            )
        if cfg.verbose >= 2:
            print(f"{u_approx=}")
        return u_approx


@overload
def approximate_one_qubit_unitary(
    unitary: mpmath.matrix,
    epsilon: MPFConvertible,
    wires: list[int] = [0],
    decompose_partially: Literal[True] = True,
    return_domega_matrix: Literal[True] = True,
    scale_epsilon: bool = True,
    cfg: GridsynthConfig | None = None,
    **kwargs,
) -> tuple[QuantumCircuit, RzGate, DOmegaMatrix]: ...


@overload
def approximate_one_qubit_unitary(
    unitary: mpmath.matrix,
    epsilon: MPFConvertible,
    wires: list[int] = [0],
    decompose_partially: Literal[True] = True,
    return_domega_matrix: Literal[False] = False,
    scale_epsilon: bool = True,
    cfg: GridsynthConfig | None = None,
    **kwargs,
) -> tuple[QuantumCircuit, RzGate, mpmath.matrix]: ...


@overload
def approximate_one_qubit_unitary(
    unitary: mpmath.matrix,
    epsilon: MPFConvertible,
    wires: list[int] = [0],
    decompose_partially: Literal[False] = False,
    return_domega_matrix: Literal[True] = True,
    scale_epsilon: bool = True,
    cfg: GridsynthConfig | None = None,
    **kwargs,
) -> tuple[QuantumCircuit, DOmegaMatrix]: ...


@overload
def approximate_one_qubit_unitary(
    unitary: mpmath.matrix,
    epsilon: MPFConvertible,
    wires: list[int] = [0],
    decompose_partially: Literal[False] = False,
    return_domega_matrix: Literal[False] = False,
    scale_epsilon: bool = True,
    cfg: GridsynthConfig | None = None,
    **kwargs,
) -> tuple[QuantumCircuit, mpmath.matrix]: ...


@overload
def approximate_one_qubit_unitary(
    unitary: mpmath.matrix,
    epsilon: MPFConvertible,
    wires: list[int] = [0],
    decompose_partially: Literal[False] = False,
    return_domega_matrix: bool = False,
    scale_epsilon: bool = True,
    cfg: GridsynthConfig | None = None,
    **kwargs,
) -> tuple[QuantumCircuit, DOmegaMatrix | mpmath.matrix]: ...


def approximate_one_qubit_unitary(
    unitary: mpmath.matrix,
    epsilon: MPFConvertible,
    wires: list[int] = [0],
    decompose_partially: bool = False,
    return_domega_matrix: bool = False,
    scale_epsilon: bool = True,
    cfg: GridsynthConfig | None = None,
    **kwargs,
) -> (
    tuple[QuantumCircuit, RzGate, DOmegaMatrix | mpmath.matrix]
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
                epsilon /= 2
            else:
                epsilon /= 3

        phase, phi1, theta, phi2 = euler_decompose(unitary)
        rx_approx = magnitude_approximate(theta, epsilon, cfg=cfg)
        phase_mag, phi1_mag, theta_mag, phi2_mag = euler_decompose(
            rx_approx.to_complex_matrix
        )
        phase -= phase_mag
        phi1 -= phi1_mag
        phi2 -= phi2_mag

        circuit_rz_left = gridsynth_circuit(phi1, epsilon, wires=wires, cfg=cfg)
        circuit_rx_approx = decompose_domega_unitary(
            rx_approx, wires=wires, up_to_phase=cfg.up_to_phase
        )
        circuit = circuit_rz_left + circuit_rx_approx
        circuit.phase += phase
        Rz_right = RzGate(phi2, wires[0])

        if decompose_partially:
            circuit = NormalForm.from_circuit(circuit).to_circuit(wires=wires)
            U_approx = DOmegaMatrix.from_single_qubit_circuit(circuit, wires=wires)
            if return_domega_matrix:
                return circuit, Rz_right, U_approx
            else:
                return circuit, Rz_right, U_approx.to_complex_matrix

        else:
            circuit_rz_right = gridsynth_circuit(
                Rz_right.theta, epsilon, wires=wires, cfg=cfg
            )
            circuit += circuit_rz_right

            if not cfg.up_to_phase:
                circuit.decompose_phase_gate()
            circuit = NormalForm.from_circuit(circuit).to_circuit(wires=wires)
            U_approx = DOmegaMatrix.from_single_qubit_circuit(circuit, wires=wires)
            if return_domega_matrix:
                return circuit, U_approx
            else:
                return circuit, U_approx.to_complex_matrix
