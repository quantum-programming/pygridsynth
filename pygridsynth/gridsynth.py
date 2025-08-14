import time

import mpmath

from .diophantine import NO_SOLUTION, diophantine_dyadic
from .loop_controller import LoopController
from .mymath import solve_quadratic, sqrt
from .quantum_gate import Rz
from .region import ConvexSet, Ellipse
from .ring import DRootTwo
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


def generate_complex_unitary(sol):
    u, t = sol
    return mpmath.matrix(
        [[u.to_complex, -t.conj.to_complex], [t.to_complex, u.conj.to_complex]]
    )


def error(theta, gates):
    u_target = Rz(theta)
    u_approx = DOmegaUnitary.from_gates(gates).to_complex_matrix
    return 2 * sqrt(
        mpmath.re(1 - mpmath.re(mpmath.conj(u_target[0, 0]) * u_approx[0, 0]) ** 2)
    )


def check(theta, gates):
    # t_count = gates.count("T")
    # h_count = gates.count("H")
    u_approx = DOmegaUnitary.from_gates(gates)
    e = error(theta, gates)
    print(f"{gates=}")
    # print(f"{t_count=}, {h_count=}")
    print(f"u_approx={u_approx.to_matrix}")
    print(f"{e=}")


def gridsynth(
    theta,
    epsilon,
    loop_controller=None,
    verbose=False,
    measure_time=False,
    show_graph=False,
):
    if loop_controller is None:
        loop_controller = LoopController()

    epsilon_region = EpsilonRegion(theta, epsilon)
    unit_disk = UnitDisk()
    k = 0

    start = time.time() if measure_time else 0.0
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
            epsilon_region,
            unit_disk,
            *transformed,
            k,
            verbose=verbose,
            show_graph=show_graph,
        )
        if measure_time:
            time_of_solve_TDGP += time.time() - start
            start = time.time()
        for z in sol:
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


def gridsynth_gates(
    theta,
    epsilon,
    wires=[0],
    decompose_phase_gate=True,
    loop_controller=None,
    verbose=False,
    measure_time=False,
    show_graph=False,
):
    start_total = time.time() if measure_time else 0.0
    u_approx = gridsynth(
        theta=theta,
        epsilon=epsilon,
        loop_controller=loop_controller,
        verbose=verbose,
        measure_time=measure_time,
        show_graph=show_graph,
    )

    start = time.time() if measure_time else 0.0
    circuit = decompose_domega_unitary(
        u_approx, wires=wires, decompose_phase_gate=decompose_phase_gate
    )
    if measure_time:
        print(f"time of decompose_domega_unitary: {(time.time() - start) * 1000} ms")
        print(f"total time: {(time.time() - start_total) * 1000} ms")

    return circuit
