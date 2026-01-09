import time
import warnings

import mpmath

from .config import GridsynthConfig
from .diophantine import Result, diophantine_dyadic
from .domega_unitary import DOmegaUnitary
from .grid_op import GridOp
from .mymath import (
    MPFConvertible,
    RealNum,
    convert_theta_and_epsilon,
    dps_for_epsilon,
    solve_quadratic,
    sqrt,
)
from .quantum_circuit import QuantumCircuit
from .quantum_gate import Rz
from .region import ConvexSet, Ellipse, Rectangle
from .ring import DOmega, DRootTwo, ZOmega, ZRootTwo
from .synthesis_of_cliffordT import decompose_domega_unitary
from .tdgp import solve_TDGP
from .to_upright import to_upright_ellipse_pair, to_upright_set_pair


class EpsilonRegion(ConvexSet):
    def __init__(
        self, theta: RealNum, epsilon: RealNum, scale: ZRootTwo = ZRootTwo(1, 0)
    ) -> None:
        self._theta: mpmath.mpf = mpmath.mpf(theta)
        self._epsilon: mpmath.mpf = mpmath.mpf(epsilon)
        self._scale: ZRootTwo = scale
        self._d: mpmath.mpf = sqrt(1 - epsilon**2 / 4) * sqrt(scale.to_real)
        self._z_x: mpmath.mpf = mpmath.cos(-theta / 2)
        self._z_y: mpmath.mpf = mpmath.sin(-theta / 2)
        D_1 = mpmath.matrix([[self._z_x, -self._z_y], [self._z_y, self._z_x]])
        D_2 = (
            mpmath.matrix([[64 * (1 / epsilon) ** 4, 0], [0, 4 * (1 / epsilon) ** 2]])
            / scale.to_real
        )
        D_3 = mpmath.matrix([[self._z_x, self._z_y], [-self._z_y, self._z_x]])
        p = mpmath.matrix([self._d * self._z_x, self._d * self._z_y])
        ellipse = Ellipse(D_1 @ D_2 @ D_3, p)
        super().__init__(ellipse)

    @property
    def theta(self) -> mpmath.mpf:
        return self._theta

    @property
    def epsilon(self) -> mpmath.mpf:
        return self._epsilon

    def inside(self, u: DOmega) -> bool:
        cos_similarity = self._z_x * u.real + self._z_y * u.imag
        return (
            DRootTwo.fromDOmega(u.conj * u) <= self._scale and cos_similarity >= self._d
        )

    def intersect(self, u0: DOmega, v: DOmega) -> tuple[mpmath.mpf, mpmath.mpf] | None:
        a = v.conj * v
        b = 2 * v.conj * u0
        c = u0.conj * u0 - self._scale

        vz = self._z_x * v.real + self._z_y * v.imag
        rhs = self._d - self._z_x * u0.real - self._z_y * u0.imag
        t = solve_quadratic(a.real, b.real, c.real)
        if t is None:
            return None
        t0, t1 = t
        if vz > 0:
            t2 = rhs / vz
            return (t0, t1) if t0 > t2 else (t2, t1)
        elif vz < 0:
            t2 = rhs / vz
            return (t0, t1) if t1 < t2 else (t0, t2)
        else:
            return (t0, t1) if rhs <= 0 else None


class UnitDisk(ConvexSet):
    def __init__(self, scale: ZRootTwo = ZRootTwo(1, 0)) -> None:
        self._scale: ZRootTwo = scale
        s_inv = 1 / scale.to_real
        ellipse = Ellipse(
            mpmath.matrix([[s_inv, 0], [0, s_inv]]), mpmath.matrix([0, 0])
        )
        super().__init__(ellipse)

    def inside(self, u: DOmega) -> bool:
        return DRootTwo.fromDOmega(u.conj * u) <= self._scale

    def intersect(self, u0: DOmega, v: DOmega) -> tuple[mpmath.mpf, mpmath.mpf] | None:
        a = v.conj * v
        b = 2 * v.conj * u0
        c = u0.conj * u0 - self._scale
        return solve_quadratic(a.real, b.real, c.real)


def generate_complex_unitary(sol: tuple[DOmega, DOmega]) -> mpmath.matrix:
    u, t = sol
    return mpmath.matrix(
        [[u.to_complex, -t.conj.to_complex], [t.to_complex, u.conj.to_complex]]
    )


def error(
    theta: MPFConvertible,
    gates: str,
    phase: RealNum = 0,
    epsilon: MPFConvertible | None = None,
    dps: int | None = None,
) -> mpmath.mpf:
    if dps is None:
        if epsilon is None:
            raise ValueError(
                "Either `dps` or `epsilon` must be provided to determine precision."
            )
        else:
            dps = dps_for_epsilon(epsilon)
    with mpmath.workdps(dps):
        theta = mpmath.mpf(theta)
        phase = mpmath.mpf(phase)
        u_target = Rz(theta)
        u_approx = (
            mpmath.exp(1.0j * phase) * DOmegaUnitary.from_gates(gates).to_complex_matrix
        )
        return 2 * sqrt(
            mpmath.re(1 - mpmath.re(mpmath.conj(u_target[0, 0]) * u_approx[0, 0]) ** 2)
        )


def check(
    theta: MPFConvertible,
    gates: str,
    phase: RealNum = 0,
    epsilon: MPFConvertible | None = None,
    dps: int | None = None,
) -> None:
    t_count = gates.count("T")
    h_count = gates.count("H")
    u_approx = DOmegaUnitary.from_gates(gates)
    e = error(theta, gates, phase=phase, epsilon=epsilon, dps=dps)
    print(f"{gates=}")
    print(f"{t_count=}, {h_count=}")
    print(f"u_approx={u_approx.to_matrix}")
    print(f"{e=}")


def get_synthesized_unitary(
    gates: str, epsilon: MPFConvertible | None = None, dps: int | None = None
) -> mpmath.matrix:
    if dps is None:
        if epsilon is None:
            raise ValueError(
                "Either `dps` or `epsilon` must be provided to determine precision."
            )
        else:
            dps = dps_for_epsilon(epsilon)
    with mpmath.workdps(dps):
        return DOmegaUnitary.from_gates(gates).to_complex_matrix


def _gridsynth_up_to_phase_with_fixed_k(
    tdgp_sets: tuple[
        ConvexSet, ConvexSet, GridOp, Ellipse, Ellipse, Rectangle, Rectangle
    ],
    k: int,
    has_phase: bool,
    cfg: GridsynthConfig,
) -> tuple[DOmegaUnitary | None, float, float]:
    start = time.time() if cfg.measure_time else 0.0
    sol = solve_TDGP(
        *tdgp_sets,
        k,
        verbose=cfg.verbose,
        show_graph=cfg.show_graph,
    )
    time_of_solve_TDGP = time.time() - start if cfg.measure_time else 0.0

    start = time.time() if cfg.measure_time else 0.0
    time_of_diophantine_dyadic = 0.0
    u_approx = None
    for z in sol:
        if (z * z.conj).residue == 0:
            continue
        if has_phase:
            z *= DOmega(ZOmega(0, -1, 1, 0), 1)
        xi = 1 - DRootTwo.fromDOmega(z.conj * z)
        w = diophantine_dyadic(xi, seed=cfg.seed, loop_controller=cfg.loop_controller)
        if not isinstance(w, Result):
            z = z.reduce_denomexp()
            w = w.reduce_denomexp()
            if z.k > w.k:
                w = w.renew_denomexp(z.k)
            elif z.k < w.k:
                z = z.renew_denomexp(w.k)

            k1 = (z + w).reduce_denomexp().k
            k2 = (z + w.mul_by_omega()).reduce_denomexp().k
            k3 = (z + w.mul_by_omega_inv()).reduce_denomexp().k
            if has_phase:
                if k1 <= k2 and k1 <= k3:
                    u_approx = DOmegaUnitary(z, w, -1)
                else:
                    u_approx = DOmegaUnitary(z, w.mul_by_omega_inv(), -1)
            else:
                if k1 <= k2:
                    u_approx = DOmegaUnitary(z, w, 0)
                else:
                    u_approx = DOmegaUnitary(z, w.mul_by_omega(), 0)
            break

    time_of_diophantine_dyadic += time.time() - start if cfg.measure_time else 0.0
    return u_approx, time_of_solve_TDGP, time_of_diophantine_dyadic


def _gridsynth_exact(
    theta: mpmath.mpf, epsilon: mpmath.mpf, cfg: GridsynthConfig
) -> DOmegaUnitary:
    with mpmath.workdps(cfg.dps):
        epsilon_region = EpsilonRegion(theta, epsilon)
        unit_disk = UnitDisk()

        start = time.time() if cfg.measure_time else 0.0
        transformed = to_upright_set_pair(
            epsilon_region,
            unit_disk,
            verbose=cfg.verbose,
            show_graph=cfg.show_graph,
        )

        tdgp_sets = (epsilon_region, unit_disk, *transformed)

        if cfg.measure_time:
            print(f"time of to_upright_set_pair: {(time.time() - start) * 1000} ms")
        if cfg.verbose >= 2:
            print("------------------")

        u_approx = None
        k = 0
        time_of_solve_TDGP = 0.0
        time_of_diophantine_dyadic = 0.0
        while True:
            u_approx, time1, time2 = _gridsynth_up_to_phase_with_fixed_k(
                tdgp_sets,
                k,
                has_phase=False,
                cfg=cfg,
            )
            time_of_solve_TDGP += time1
            time_of_diophantine_dyadic += time2
            if u_approx is not None:
                break

            k += 1

        if cfg.measure_time:
            print(f"time of solve_TDGP: {time_of_solve_TDGP * 1000} ms")
            print(
                "time of diophantine_dyadic: " f"{time_of_diophantine_dyadic * 1000} ms"
            )
        if cfg.verbose >= 2:
            print(f"{u_approx=}")
            print("------------------")
        return u_approx


def _gridsynth_up_to_phase(
    theta: mpmath.mpf, epsilon: mpmath.mpf, cfg: GridsynthConfig
) -> DOmegaUnitary:
    with mpmath.workdps(cfg.dps):
        epsilon_region0 = EpsilonRegion(theta, epsilon)
        unit_disk0 = UnitDisk()
        epsilon_region1 = EpsilonRegion(theta, epsilon, scale=ZRootTwo(2, 1))
        unit_disk1 = UnitDisk(scale=ZRootTwo(2, -1))

        start = time.time() if cfg.measure_time else 0.0
        opG = to_upright_ellipse_pair(
            epsilon_region0.ellipse, unit_disk0.ellipse, verbose=cfg.verbose
        )
        transformed0 = to_upright_set_pair(
            epsilon_region0,
            unit_disk0,
            opG=opG,
            verbose=cfg.verbose,
            show_graph=cfg.show_graph,
        )
        transformed1 = to_upright_set_pair(
            epsilon_region1,
            unit_disk1,
            opG=opG,
            verbose=cfg.verbose,
            show_graph=cfg.show_graph,
        )
        tdgp_sets0 = (epsilon_region0, unit_disk0, *transformed0)
        tdgp_sets1 = (epsilon_region1, unit_disk1, *transformed1)

        if cfg.measure_time:
            print(f"time of to_upright_set_pair: {(time.time() - start) * 1000} ms")
        if cfg.verbose >= 2:
            print("------------------")

        u_approx = None
        k = 0
        has_phase = False
        time_of_solve_TDGP = 0.0
        time_of_diophantine_dyadic = 0.0
        while True:
            u_approx, time1, time2 = _gridsynth_up_to_phase_with_fixed_k(
                tdgp_sets1 if has_phase else tdgp_sets0,
                k,
                has_phase=has_phase,
                cfg=cfg,
            )
            time_of_solve_TDGP += time1
            time_of_diophantine_dyadic += time2
            if u_approx is not None:
                break

            if k >= 2:
                has_phase = not has_phase
                if has_phase:
                    k += 1
            else:
                if k == 0 and not has_phase:
                    k, has_phase = 1, False
                elif k == 1 and not has_phase:
                    k, has_phase = 0, True
                elif k == 0 and has_phase:
                    k, has_phase = 1, True
                elif k == 1 and has_phase:
                    k, has_phase = 2, False

        if cfg.measure_time:
            print(f"time of solve_TDGP: {time_of_solve_TDGP * 1000} ms")
            print(
                "time of diophantine_dyadic: " f"{time_of_diophantine_dyadic * 1000} ms"
            )
        if cfg.verbose >= 2:
            print(f"{u_approx=}")
            print("------------------")
        return u_approx


def gridsynth(
    theta: MPFConvertible,
    epsilon: MPFConvertible,
    cfg: GridsynthConfig | None = None,
    **kwargs,
) -> DOmegaUnitary:
    if cfg is None:
        cfg = GridsynthConfig(**kwargs)
    elif kwargs:
        warnings.warn("When 'cfg' is provided, 'kwargs' are ignored.", stacklevel=2)

    if cfg.dps is None:
        cfg.dps = dps_for_epsilon(epsilon)

    with mpmath.workdps(cfg.dps):
        theta, epsilon = convert_theta_and_epsilon(theta, epsilon, dps=cfg.dps)

        if cfg.up_to_phase:
            return _gridsynth_up_to_phase(theta, epsilon, cfg=cfg)
        else:
            return _gridsynth_exact(theta, epsilon, cfg=cfg)


def gridsynth_circuit(
    theta: MPFConvertible,
    epsilon: MPFConvertible,
    wires: list[int] = [0],
    cfg: GridsynthConfig | None = None,
    **kwargs,
) -> QuantumCircuit:
    if cfg is None:
        cfg = GridsynthConfig(**kwargs)
    elif kwargs:
        warnings.warn("When 'cfg' is provided, 'kwargs' are ignored.", stacklevel=2)

    if cfg.dps is None:
        cfg.dps = dps_for_epsilon(epsilon)

    with mpmath.workdps(cfg.dps):
        start_total = time.time() if cfg.measure_time else 0.0
        u_approx = gridsynth(theta=theta, epsilon=epsilon, cfg=cfg)

        start = time.time() if cfg.measure_time else 0.0
        circuit = decompose_domega_unitary(
            u_approx, wires=wires, up_to_phase=cfg.up_to_phase
        )
        if u_approx.n != 0:
            circuit.phase = (circuit.phase + mpmath.mp.pi / 8) % (2 * mpmath.mp.pi)
        if cfg.measure_time:
            print(
                f"time of decompose_domega_unitary: {(time.time() - start) * 1000} ms"
            )
            print(f"time of gridsynth_circuit: {(time.time() - start_total) * 1000} ms")

        return circuit


def gridsynth_gates(
    theta: MPFConvertible,
    epsilon: MPFConvertible,
    cfg: GridsynthConfig | None = None,
    **kwargs,
) -> str:
    if cfg is None:
        cfg = GridsynthConfig(**kwargs)
    elif kwargs:
        warnings.warn("When 'cfg' is provided, 'kwargs' are ignored.", stacklevel=2)

    if cfg.dps is None:
        cfg.dps = dps_for_epsilon(epsilon)

    with mpmath.workdps(cfg.dps):
        circuit = gridsynth_circuit(
            theta=theta,
            epsilon=epsilon,
            wires=[0],
            cfg=cfg,
        )
        return circuit.to_simple_str()
