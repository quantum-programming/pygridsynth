import time
import warnings

import mpmath

from .diophantine import NO_SOLUTION, diophantine_dyadic, set_random_seed
from .loop_controller import LoopController
from .mymath import solve_quadratic, sqrt
from .quantum_gate import Rz
from .region import ConvexSet, Ellipse
from .ring import DRootTwo, ZRootTwo, DOmega
from .synthesis_of_cliffordT import decompose_domega_unitary
from .tdgp import solve_TDGP
from .to_upright import to_upright_set_pair
from .unitary import DOmegaUnitary


class EpsilonRegion(ConvexSet):
    def __init__(self, theta, epsilon):
        self._theta = theta
        self._epsilon = epsilon
        self._d = sqrt(1 - epsilon**2 / 4)
        self._z_x = mpmath.cos(-theta / 2)
        self._z_y = mpmath.sin(-theta / 2)
        D_1 = mpmath.matrix([[self._z_x, -self._z_y], [self._z_y, self._z_x]])
        D_2 = mpmath.matrix([[64 * (1 / epsilon) ** 4, 0], [0, 4 * (1 / epsilon) ** 2]])
        D_3 = mpmath.matrix([[self._z_x, self._z_y], [-self._z_y, self._z_x]])
        p = mpmath.matrix([self._d * self._z_x, self._d * self._z_y])
        ellipse = Ellipse(D_1 @ D_2 @ D_3, p)
        super().__init__(ellipse)

    @property
    def theta(self):
        return self._theta

    @property
    def epsilon(self):
        return self._epsilon

    def inside(self, u):
        cos_similarity = self._z_x * u.real + self._z_y * u.imag
        return DRootTwo.fromDOmega(u.conj * u) <= 1 and cos_similarity >= self._d

    def intersect(self, u0, v):
        a = v.conj * v
        b = 2 * v.conj * u0
        c = u0.conj * u0 - 1

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


# Scale epsilon region by 1/(2 + sqrt(2))
class EpsilonRegionScaled(ConvexSet):
    scale = mpmath.mpmathify(2) + sqrt(mpmath.mpmathify(2))

    def __init__(self, theta, epsilon):
        scale = EpsilonRegionScaled.scale
        self._theta = theta
        self._epsilon = epsilon
        self._d = sqrt(1 - epsilon**2 / 4) * mpmath.sqrt(scale)
        self._z_x = mpmath.cos(-theta / 2)
        self._z_y = mpmath.sin(-theta / 2)
        D_1 = mpmath.matrix([[self._z_x, -self._z_y], [self._z_y, self._z_x]])
        D_2 = mpmath.matrix([[64 * (1 / epsilon) ** 4 / scale, 0], [0, 4 * (1 / epsilon) ** 2 / scale]])
        D_3 = mpmath.matrix([[self._z_x, self._z_y], [-self._z_y, self._z_x]])
        p = mpmath.matrix([self._d * self._z_x, self._d * self._z_y])
        ellipse = Ellipse(D_1 @ D_2 @ D_3, p)
        super().__init__(ellipse)

    @property
    def theta(self):
        return self._theta

    @property
    def epsilon(self):
        return self._epsilon

    def inside(self, u):
        cos_similarity = self._z_x * u.real + self._z_y * u.imag
        return DRootTwo.fromDOmega(u.conj * u) <= EpsilonRegionScaled.scale and cos_similarity >= self._d

    def intersect(self, u0, v):
        a = v.conj * v
        b = 2 * v.conj * u0
        scale = ZRootTwo(2, 1)
        c = (u0.conj * u0) - DOmega.from_zroottwo(scale)

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
    def __init__(self):
        ellipse = Ellipse(mpmath.matrix([[1, 0], [0, 1]]), mpmath.matrix([0, 0]))
        super().__init__(ellipse)

    def inside(self, u):
        return DRootTwo.fromDOmega(u.conj * u) <= 1

    def intersect(self, u0, v):
        a = v.conj * v
        b = 2 * v.conj * u0
        c = u0.conj * u0 - 1
        return solve_quadratic(a.real, b.real, c.real)


# Scaled unit disk with hard-coded scale factor
class UnitDiskScaled(ConvexSet):

    scale = mpmath.mpmathify(2) - sqrt(mpmath.mpmathify(2))

    def __init__(self):
        sinv = 1 / UnitDiskScaled.scale
        ellipse = Ellipse(mpmath.matrix([[sinv, 0], [0, sinv]]), mpmath.matrix([0, 0]))
        super().__init__(ellipse)


    def inside(self, u):
        scale = ZRootTwo(2, -1) # 2 - sqrt(2)
        return DRootTwo.fromDOmega(u.conj * u) <= DOmega.from_zroottwo(scale)
#        return DRootTwo.fromDOmega(u.conj * u) <= UnitDiskScaled.scale

    def intersect(self, u0, v):
        a = v.conj * v
        b = 2 * v.conj * u0
        scale = ZRootTwo(2, -1) # 2 - sqrt(2)
        c = u0.conj * u0 - DOmega.from_zroottwo(scale)
        return solve_quadratic(a.real, b.real, c.real)

def generate_complex_unitary(sol):
    u, t = sol
    return mpmath.matrix(
        [[u.to_complex, -t.conj.to_complex], [t.to_complex, u.conj.to_complex]]
    )


def error(theta, gates):
    tcount = gates.count("T")
    epsilon = 2 ** (-tcount // 3)
    dps = _dps_for_epsilon(epsilon)
    with mpmath.workdps(dps):
        u_target = Rz(theta)
        u_approx = DOmegaUnitary.from_gates(gates).to_complex_matrix
        return 2 * sqrt(
            mpmath.re(1 - mpmath.re(mpmath.conj(u_target[0, 0]) * u_approx[0, 0]) ** 2)
        )


def check(theta, gates):
    t_count = gates.count("T")
    h_count = gates.count("H")
    u_approx = DOmegaUnitary.from_gates(gates)
    e = error(theta, gates)
    print(f"{gates=}")
    print(f"{t_count=}, {h_count=}")
    print(f"u_approx={u_approx.to_matrix}")
    print(f"{e=}")


def get_synthesized_unitary(gates):
    tcount = gates.count("T")
    epsilon = 2 ** (-tcount // 3)
    dps = _dps_for_epsilon(epsilon)
    with mpmath.workdps(dps):
        return DOmegaUnitary.from_gates(gates).to_complex_matrix


def gridsynth(
    theta,
    epsilon,
    dps=None,
    loop_controller=None,
    phase=False,
    verbose=False,
    measure_time=False,
    show_graph=False,
):
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

    if dps is None:
        dps = _dps_for_epsilon(epsilon)
    with mpmath.workdps(dps):
        theta = mpmath.mpmathify(theta)
        epsilon = mpmath.mpmathify(epsilon)
        if loop_controller is None:
            loop_controller = LoopController()

        epsilon_region = EpsilonRegion(theta, epsilon)
        unit_disk = UnitDisk()

        if phase:
            epsilon_region1 = EpsilonRegionScaled(theta, epsilon)
            unit_disk1 = UnitDiskScaled()
        else:
            epsilon_region1 = epsilon_region
            unit_disk1 = unit_disk

        k = 0

        if measure_time:
            start = time.time()
        transformed = to_upright_set_pair(
            epsilon_region, unit_disk, verbose=verbose, show_graph=show_graph
        )
        if measure_time:
            print(f"to_upright_set_pair: {time.time() - start} s")
        if verbose:
            print("------------------")

        time_of_solve_TDGP = 0
        time_of_diophantine_dyadic = 0
        while True:
            if measure_time:
                start = time.time()
            sol = solve_TDGP(
                epsilon_region1,
                unit_disk1,
                *transformed,
                k,
                verbose=verbose,
                show_graph=show_graph,
            )
            if measure_time:
                time_of_solve_TDGP += time.time() - start

            print(k)
            start = time.time()

            ct = 0
            # print(f"sol {sol}")
            # print(f"> ct {ct}")
            for z in sol:
                print(f"ct {ct}")
                ct += 1
                # if ct > 10:
                #     return
                if (z * z.conj).residue == 0:
                    continue
                xi = 1 - DRootTwo.fromDOmega(z.conj * z)
                w = diophantine_dyadic(xi, loop_controller=loop_controller)
                if w != NO_SOLUTION:
                    z = z.reduce_denomexp()
                    w = w.reduce_denomexp()
                    if z.k > w.k:
                        w = w.renew_denomexp(z.k)
                    elif z.k < w.k:
                        z = z.renew_denomexp(w.k)
                    if (z + w).reduce_denomexp().k < z.k:
                        u_approx = DOmegaUnitary(z, w, 0)
                    else:
                        u_approx = DOmegaUnitary(z, w.mul_by_omega(), 0)
                    if measure_time:
                        time_of_diophantine_dyadic += time.time() - start
                        print(f"time of solve_TDGP: {time_of_solve_TDGP * 1000} ms")
                        print(
                            "time of diophantine_dyadic: "
                            f"{time_of_diophantine_dyadic * 1000} ms"
                        )
                    if verbose:
                        print(f"{z=}, {w=}")
                        print("------------------")
                    return u_approx
            if measure_time:
                time_of_diophantine_dyadic += time.time() - start
            k += 1


def gridsynth_circuit(
    theta,
    epsilon,
    wires=[0],
    decompose_phase_gate=True,
    dps=None,
    loop_controller=None,
    phase=False,
    verbose=False,
    measure_time=False,
    show_graph=False,
):
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

    if dps is None:
        dps = _dps_for_epsilon(epsilon)
    with mpmath.workdps(dps):
        theta = mpmath.mpmathify(theta)
        epsilon = mpmath.mpmathify(epsilon)
        start_total = time.time() if measure_time else 0.0
        u_approx = gridsynth(
            theta=theta,
            epsilon=epsilon,
            dps=dps,
            loop_controller=loop_controller,
            phase=phase,
            verbose=verbose,
            measure_time=measure_time,
            show_graph=show_graph,
        )

        start = time.time() if measure_time else 0.0
        circuit = decompose_domega_unitary(
            u_approx, wires=wires, decompose_phase_gate=decompose_phase_gate
        )
        if measure_time:
            print(
                f"time of decompose_domega_unitary: {(time.time() - start) * 1000} ms"
            )
            print(f"total time: {(time.time() - start_total) * 1000} ms")

        return circuit


def gridsynth_gates(
    theta,
    epsilon,
    dps=None,
    dtimeout=None,
    ftimeout=None,
    dloop=10,
    floop=10,
    seed=0,
    phase=False,
    verbose=False,
    measure_time=False,
    show_graph=False,
    decompose_phase_gate=True,
):
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

    print(f"PHASE: {phase}")
    set_random_seed(seed)

    loop_controller = LoopController(
        dloop=dloop, floop=floop, dtimeout=dtimeout, ftimeout=ftimeout
    )

    if dps is None:
        dps = _dps_for_epsilon(epsilon)
    with mpmath.workdps(dps):
        theta = mpmath.mpmathify(theta)
        epsilon = mpmath.mpmathify(epsilon)
        circuit = gridsynth_circuit(
            theta=theta,
            epsilon=epsilon,
            wires=[0],
            decompose_phase_gate=decompose_phase_gate,
            dps=dps,
            loop_controller=loop_controller,
            phase=phase,
            verbose=verbose,
            measure_time=measure_time,
            show_graph=show_graph,
        )
        return circuit.to_simple_str()


def _dps_for_epsilon(epsilon) -> int:
    e = mpmath.mpmathify(epsilon)
    k = -mpmath.log10(e)
    return int(15 + 2.5 * int(mpmath.ceil(k)))  # used in newsynth
