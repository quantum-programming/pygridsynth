import mpmath
import time

from .mymath import sqrt, solve_quadratic
from .ring import DRootTwo
from .region import Ellipse, ConvexSet
from .to_upright import to_upright_set_pair
from .tdgp import solve_TDGP
from .diophantine import NO_SOLUTION, diophantine_dyadic
from .unitary import DOmegaUnitary
from .synthesis_of_cliffordT import decompose_domega_unitary


class EpsilonRegion(ConvexSet):
    def __init__(self, theta, epsilon):
        self._theta = theta
        self._epsilon = epsilon
        self._d = 1 - epsilon ** 2 / 2
        self._z_x = mpmath.cos(-theta / 2)
        self._z_y = mpmath.sin(-theta / 2)
        D_1 = mpmath.matrix([[self._z_x, -self._z_y], [self._z_y, self._z_x]])
        D_2 = mpmath.matrix([[4 * (1 / epsilon) ** 4, 0], [0, (1 / epsilon) ** 2]])
        D_3 = mpmath.matrix([[self._z_x, self._z_y], [-self._z_y, self._z_x]])
        p = mpmath.matrix([self._d * self._z_x, self._d * self._z_y])
        ellipse = Ellipse(D_1 * D_2 * D_3, p)
        super().__init__(ellipse)

    @property
    def theta(self):
        return self._theta

    @property
    def epsilon(self):
        return self._epsilon

    def inside(self, u):
        cos_similarity = (self._z_x * u.real + self._z_y * u.imag)
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
    return mpmath.matrix([[u.to_complex, -t.conj.to_complex],
                          [t.to_complex, u.conj.to_complex]])


def generate_target_Rz(theta):
    return mpmath.matrix([[mpmath.exp(- 1.j * theta / 2), 0],
                          [0, mpmath.exp(1.j * theta / 2)]])


def error(theta, gates):
    tcount = gates.count("T")
    epsilon = 2 ** (-tcount//3)
    dps = _dps_for_epsilon(epsilon)
    with mp_dps(dps):

        Rz = generate_target_Rz(mpmath.mpmathify(f"{theta}"))
        U = DOmegaUnitary.from_gates(gates).to_complex_matrix
        E = U - Rz
        return sqrt(mpmath.fabs(E[0, 0] * E[1, 1] - E[0, 1] * E[1, 0]))


def check(theta, gates):
    t_count = gates.count("T")
    h_count = gates.count("H")
    U_decomp = DOmegaUnitary.from_gates(gates)
    # Rz = generate_target_Rz(theta)
    # U = U_decomp.to_complex_matrix
    e = error(theta, gates)
    print(f"{gates=}")
    print(f"{t_count=}, {h_count=}")
    # print(f"{Rz=}")
    print(f"U_decomp={U_decomp.to_matrix}")
    # print(f"{U=}")
    print(f"{e=}")

def get_synthesized_unitary(gates):
    tcount = gates.count("T")
    epsilon = 2 ** (-tcount//3)
    dps = _dps_for_epsilon(epsilon)
    with mp_dps(dps):    
        return DOmegaUnitary.from_gates(gates).to_complex_matrix


def gridsynth(theta:mpmath.mpf, epsilon:mpmath.mpf,
              diophantine_timeout=1.0, factoring_timeout=1.0,
              verbose=False, measure_time=False, show_graph=False):
    dps = _dps_for_epsilon(epsilon)
    if not isinstance(epsilon, mpmath.mpf) or not isinstance(theta, mpmath.mpf):
        
        mpmath.mp.dps = dps
        epsilon = mpmath.mpmathify(f"{epsilon}")
        theta = mpmath.mpmathify(f"{theta}")

    with mp_dps(dps):
        epsilon_region = EpsilonRegion(theta, epsilon)
        unit_disk = UnitDisk()
        k = 0
        

        if measure_time:
            start = time.time()
        transformed = to_upright_set_pair(epsilon_region, unit_disk,
                                        verbose=verbose, show_graph=show_graph)
        if measure_time:
            print(f"to_upright_set_pair: {time.time() - start} s")
        if verbose:
            print("------------------")

        time_of_solve_TDGP = 0
        time_of_diophantine_dyadic = 0
        while True:
            if measure_time:
                start = time.time()
            sol = solve_TDGP(epsilon_region, unit_disk, *transformed, k,
                            verbose=verbose, show_graph=show_graph)
            if measure_time:
                time_of_solve_TDGP += time.time() - start
                start = time.time()

            for z in sol:
                if (z * z.conj).residue == 0:
                    continue
                xi = 1 - DRootTwo.fromDOmega(z.conj * z)
                w = diophantine_dyadic(xi,
                                    diophantine_timeout=diophantine_timeout,
                                    factoring_timeout=factoring_timeout)
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
                        print(f"time of diophantine_dyadic: {time_of_diophantine_dyadic * 1000} ms")
                    if verbose:
                        print(f"{z=}, {w=}")
                        print("------------------")
                    return u_approx
            if measure_time:
                time_of_diophantine_dyadic += time.time() - start
            k += 1


def gridsynth_gates(theta, epsilon,
                    diophantine_timeout= 1.0, factoring_timeout= 1.0,
                    verbose=False, measure_time=False, show_graph=False):
    dps = _dps_for_epsilon(epsilon)
    with mp_dps(dps):
        theta = mpmath.mpmathify(f"{theta}")
        epsilon = mpmath.mpmathify(f"{epsilon}")

        if measure_time:
            start_total = time.time()
        u_approx = gridsynth(theta=theta, epsilon=epsilon,
                            diophantine_timeout=diophantine_timeout,
                            factoring_timeout=factoring_timeout,
                            verbose=verbose, measure_time=measure_time, show_graph=show_graph)
        if measure_time:
            start = time.time()
        gates = decompose_domega_unitary(u_approx)
        if measure_time:
            print(f"time of decompose_domega_unitary: {(time.time() - start) * 1000} ms")
            print(f"total time: {(time.time() - start_total) * 1000} ms")
        return gates

def _dps_for_epsilon(eps_float) -> int:
    old = mpmath.mp.dps
    try:
        e = mpmath.mpmathify(f"{eps_float}")
        k = -mpmath.log10(e)
        return 15 + 2.5*int(mpmath.ceil(k)) # used in newsynth
    finally:
        mpmath.mp.dps = old

class mp_dps:
    def __init__(self, dps):
        self._old = None
        self._dps = dps
    def __enter__(self):
        self._old = mpmath.mp.dps
        mpmath.mp.dps = self._dps
    def __exit__(self, *exc):
        mpmath.mp.dps = self._old