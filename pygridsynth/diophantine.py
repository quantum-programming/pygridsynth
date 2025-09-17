import math
import random
import warnings
from enum import Enum
from typing import Literal

from .loop_controller import LoopController
from .ring import DOmega, DRootTwo, ZOmega, ZRootTwo


class Result(Enum):
    NO_SOLUTION = "NO_SOLUTION"
    UNSOLVED = "UNSOLVED"


rng = random.Random(0)


def set_random_seed(seed: int) -> None:
    global rng
    rng = random.Random(seed)


def _find_factor(n: int, loop_controller: LoopController, M: int = 128) -> int | None:
    if not (n & 1) and n > 2:
        return 2

    a = rng.randint(1, n)
    y, r, k = a, 1, 0
    L = int(10 ** (len(str(n)) / 4) * 1.1774 + 10)

    loop_controller.start_factoring()
    while True:
        x = y + n
        while k < r:
            q = 1
            y0 = y
            for _ in range(M):
                y = (y * y + a) % n
                q = q * (x - y) % n
                k += 1
                if k == r:
                    break
            g = math.gcd(q, n)
            if g != 1:
                if g == n:
                    y = y0
                    for _ in range(M):
                        y = (y * y + a) % n
                        g = math.gcd(x - y, n)
                        if g != 1:
                            break
                return None if g == n else g
            if k >= L or not loop_controller.check_factoring_continue():
                return None
        r <<= 1


def _sqrt_negative_one(p: int, L: int = 100) -> int | None:
    for _ in range(L):
        b = rng.randint(1, p - 1)
        h = pow(b, (p - 1) >> 2, p)
        r = h * h % p
        if r == p - 1:
            return h
        elif r != 1:
            return None
    return None


class F_p2:
    base: int = 0
    p: int = 0

    def __init__(self, a: int, b: int) -> None:
        if a < 0 or a >= self.__class__.p:
            a %= self.__class__.p
        if b < 0 or b >= self.__class__.p:
            b %= self.__class__.p
        self._a: int = a
        self._b: int = b

    @property
    def a(self) -> int:
        return self._a

    @property
    def b(self) -> int:
        return self._b

    def __mul__(self, other: "F_p2") -> "F_p2":
        if isinstance(other, self.__class__):
            new_a = (
                self._a * other.a
                + self._b * other.b % self.__class__.p * self.__class__.base
            )
            new_b = self._a * other.b + self._b * other.a
            return self.__class__(new_a, new_b)
        else:
            return NotImplemented

    def __pow__(self, other: int) -> "F_p2":
        if isinstance(other, int):
            if other < 0:
                return NotImplemented
            else:
                other = int(other)
                new = self.__class__(1, 0)
                tmp = self
                while other > 0:
                    if other & 1:
                        new *= tmp
                    tmp *= tmp
                    other >>= 1
                return new
        else:
            return NotImplemented


def _root_mod(x: int, p: int, L: int = 100) -> int | None:
    x = x % p
    if p == 2:
        return x
    if x == 0:
        return 0
    if not (p & 1) and p > 2:
        return None
    if pow(x, (p - 1) // 2, p) != 1:
        return None

    for _ in range(L):
        b = rng.randint(1, p - 1)
        r = pow(b, p - 1, p)
        if r != 1:
            return None

        base = (b * b + p - x) % p
        if pow(base, (p - 1) // 2, p) != 1:
            F_p2.p = p
            F_p2.base = base
            return (F_p2(b, 1) ** ((p + 1) // 2)).a
    return None


def _is_prime(n: int, L: int = 4) -> bool:
    if n < 0:
        n = -n
    if n == 0 or n == 1:
        return False
    if not (n & 1):
        return True if n == 2 else False

    r, d = 0, n - 1
    while not (d & 1):
        r += 1
        d >>= 1
    for _ in range(L):
        a = rng.randint(1, n - 1)
        a = pow(a, d, n)
        if a == 1:
            return True

        for _ in range(r):
            if a == n - 1:
                return True
            a = a * a % n
    return False


def _decompose_relatively_int_prime(
    partial_facs: list[tuple[int, int]],
) -> tuple[int, list[tuple[int, int]]]:
    u = 1
    stack = list(reversed(partial_facs))
    facs: list[tuple[int, int]] = []
    while len(stack):
        b, k_b = stack.pop()
        i = 0
        while True:
            if i >= len(facs):
                if b == 1 or b == -1:
                    if b == -1 and (k_b & 1):
                        u = -u
                else:
                    facs.append((b, k_b))
                break
            a, k_a = facs[i]
            if a == b or a == -b:
                if a == -b and (k_b & 1):
                    u = -u
                facs[i] = (a, k_a + k_b)
                break
            else:
                g = math.gcd(a, b)
                if g == 1 or g == -1:
                    i += 1
                    continue
                else:
                    partial_facs = [(a // g, k_a), (g, k_a + k_b)]
                    u_a, facs_a = _decompose_relatively_int_prime(partial_facs)
                    u *= u_a
                    facs[i] = facs_a[0]
                    facs = facs + facs_a[1:]
                    stack.append((b // g, k_b))
                    break

    return u, facs


def _adj_decompose_int_prime(p: int) -> ZOmega | Result:
    if p < 0:
        p = -p
    if p == 0 or p == 1:
        return ZOmega.from_int(p)
    if p == 2:
        return ZOmega(-1, 0, 1, 0)

    if _is_prime(p):
        if p & 0b11 == 1:
            h = _sqrt_negative_one(p)
            if h is None:
                return Result.UNSOLVED
            else:
                t = ZOmega.gcd(h + ZOmega(0, 1, 0, 0), p)
                return t if t.conj * t == p or t.conj * t == -p else Result.UNSOLVED
        elif p & 0b111 == 3:
            h = _root_mod(-2, p)
            if h is None:
                return Result.UNSOLVED
            else:
                t = ZOmega.gcd(h + ZOmega(1, 0, 1, 0), p)
                return t if t.conj * t == p or t.conj * t == -p else Result.UNSOLVED
        elif p & 0b111 == 7:
            h = _root_mod(2, p)
            if h is not None:
                return Result.NO_SOLUTION
            else:
                return Result.UNSOLVED
        else:
            return Result.UNSOLVED
    else:
        if p & 0b111 == 7:
            h = _root_mod(2, p)
            if h is not None:
                return Result.NO_SOLUTION
            else:
                return Result.UNSOLVED
        else:
            return Result.UNSOLVED


def _adj_decompose_int_prime_power(p: int, k: int) -> ZOmega | Result:
    if not (k & 1):
        return p ** (k // 2)
    else:
        t = _adj_decompose_int_prime(p)
        if isinstance(t, Result):
            return t
        else:
            return t**k


def _adj_decompose_int(
    n: int, loop_controller: LoopController
) -> ZOmega | Literal[Result.NO_SOLUTION]:
    if n < 0:
        n = -n
    facs = [(n, 1)]
    t = ZOmega.from_int(1)
    while len(facs):
        p, k = facs.pop()
        t_p = _adj_decompose_int_prime_power(p, k)
        if isinstance(t_p, Result):
            if t_p == Result.NO_SOLUTION:
                return Result.NO_SOLUTION
            elif t_p == Result.UNSOLVED:
                fac = _find_factor(p, loop_controller)
                if fac is None:
                    facs.append((p, k))
                    if not loop_controller.check_diophantine_continue():
                        return Result.NO_SOLUTION
                else:
                    facs.append((p // fac, k))
                    facs.append((fac, k))
                    _, facs = _decompose_relatively_int_prime(facs)
        else:
            t *= t_p
    return t


def _adj_decompose_selfassociate(
    xi: ZRootTwo, loop_controller: LoopController
) -> ZOmega | Literal[Result.NO_SOLUTION]:
    # xi \sim xi.conj_sq2
    if xi == 0:
        return ZOmega.from_int(0)

    n = math.gcd(xi.a, xi.b)
    r = xi // n
    t1 = _adj_decompose_int(n, loop_controller)
    t2 = ZOmega(0, 0, 1, 1) if r % ZRootTwo(0, 1) == 0 else 1
    if isinstance(t1, Result):
        return Result.NO_SOLUTION
    else:
        return t1 * t2


def _decompose_relatively_zomega_prime(
    partial_facs: list[tuple[ZRootTwo, int]],
) -> tuple[ZRootTwo, list[tuple[ZRootTwo, int]]]:
    u = ZRootTwo.from_int(1)
    stack = list(reversed(partial_facs))
    facs: list[tuple[ZRootTwo, int]] = []
    while len(stack):
        b, k_b = stack.pop()
        i = 0
        while True:
            if i >= len(facs):
                if ZRootTwo.sim(b, 1):
                    u *= b**k_b
                else:
                    facs.append((b, k_b))
                break
            a, k_a = facs[i]
            if ZRootTwo.sim(a, b):
                u *= (b // a) ** k_b
                facs[i] = (a, k_a + k_b)
                break
            else:
                g = ZRootTwo.gcd(a, b)
                if ZRootTwo.sim(g, 1):
                    i += 1
                    continue
                else:
                    partial_facs = [(a // g, k_a), (g, k_a + k_b)]
                    u_a, facs_a = _decompose_relatively_zomega_prime(partial_facs)
                    u *= u_a
                    facs[i] = facs_a[0]
                    facs = facs + facs_a[1:]
                    stack.append((b // g, k_b))
                    break

    return u, facs


def _adj_decompose_zomega_prime(eta: ZRootTwo) -> ZOmega | Result:
    p = eta.norm

    if p < 0:
        p = -p
    if p == 0 or p == 1:
        return ZOmega.from_int(p)
    elif p == 2:
        return ZOmega(-1, 0, 1, 0)

    if _is_prime(p):
        if p & 0b11 == 1:
            h = _sqrt_negative_one(p)
            if h is None:
                return Result.UNSOLVED
            else:
                t = ZOmega.gcd(h + ZOmega(0, 1, 0, 0), eta)
                return t if ZOmega.sim(t.conj * t, eta) else Result.UNSOLVED
        elif p & 0b111 == 3:
            h = _root_mod(-2, p)
            if h is None:
                return Result.UNSOLVED
            else:
                t = ZOmega.gcd(h + ZOmega(1, 0, 1, 0), eta)
                return t if ZOmega.sim(t.conj * t, eta) else Result.UNSOLVED
        elif p & 0b111 == 7:
            h = _root_mod(2, p)
            if h is not None:
                return Result.NO_SOLUTION
            else:
                return Result.UNSOLVED
        else:
            return Result.UNSOLVED
    else:
        if p & 0b111 == 7:
            h = _root_mod(2, p)
            if h is not None:
                return Result.NO_SOLUTION
            else:
                return Result.UNSOLVED
        else:
            return Result.UNSOLVED


def _adj_decompose_zomega_prime_power(eta: ZRootTwo, k: int) -> ZOmega | Result:
    if not (k & 1):
        return ZOmega.from_zroottwo(eta ** (k // 2))
    else:
        t = _adj_decompose_zomega_prime(eta)
        if isinstance(t, Result):
            return t
        else:
            return t**k


def _adj_decompose_selfcoprime(
    xi: ZRootTwo, loop_controller: LoopController
) -> ZOmega | Literal[Result.NO_SOLUTION]:
    # gcd(xi, xi.conj_sq2) = 1
    facs = [(xi, 1)]
    t = ZOmega.from_int(1)
    while len(facs):
        eta, k = facs.pop()
        t_eta = _adj_decompose_zomega_prime_power(eta, k)
        if isinstance(t_eta, Result):
            if t_eta == Result.NO_SOLUTION:
                return Result.NO_SOLUTION
            elif t_eta == Result.UNSOLVED:
                n = eta.norm
                if n < 0:
                    n = -n
                fac_n = _find_factor(n, loop_controller)
                if fac_n is None:
                    facs.append((eta, k))
                    if not loop_controller.check_diophantine_continue():
                        return Result.NO_SOLUTION
                else:
                    fac = ZRootTwo.gcd(xi, fac_n)
                    facs.append((eta // fac, k))
                    facs.append((fac, k))
                    _, facs = _decompose_relatively_zomega_prime(facs)
        else:
            t *= t_eta
    return t


def _adj_decompose(
    xi: ZRootTwo, loop_controller: LoopController
) -> ZOmega | Literal[Result.NO_SOLUTION]:
    if xi == 0:
        return ZOmega.from_int(0)

    d = ZRootTwo.gcd(xi, xi.conj_sq2)
    eta = xi // d
    t1 = _adj_decompose_selfassociate(d, loop_controller)
    if isinstance(t1, Result):
        return Result.NO_SOLUTION
    else:
        t2 = _adj_decompose_selfcoprime(eta, loop_controller)
        if isinstance(t2, Result):
            return Result.NO_SOLUTION
        else:
            return t1 * t2


def _diophantine(
    xi: ZRootTwo, loop_controller: LoopController
) -> ZOmega | Literal[Result.NO_SOLUTION]:
    loop_controller.start_diophantine()

    if xi == 0:
        return ZOmega.from_int(0)
    elif xi < 0 or xi.conj_sq2 < 0:
        return Result.NO_SOLUTION

    t = _adj_decompose(xi, loop_controller)
    if isinstance(t, Result):
        return Result.NO_SOLUTION
    else:
        xi_associate = ZRootTwo.from_zomega(t.conj * t)
        u = xi // xi_associate
        v = u.sqrt()
        if v is None:
            warnings.warn("cannot find square root of u")
            return Result.NO_SOLUTION
        else:
            return v * t


def diophantine_dyadic(
    xi: DRootTwo, seed: int = 0, loop_controller: LoopController | None = None
) -> DOmega | Literal[Result.NO_SOLUTION]:
    set_random_seed(seed)
    if loop_controller is None:
        loop_controller = LoopController()

    k_div_2, k_mod_2 = xi.k >> 1, xi.k & 1

    t = _diophantine(
        xi.alpha * ZRootTwo(1, 1) if k_mod_2 else xi.alpha,
        loop_controller=loop_controller,
    )
    if isinstance(t, Result):
        return Result.NO_SOLUTION
    else:
        if k_mod_2:
            t *= ZOmega(0, -1, 1, 0)
        return DOmega(t, k_div_2 + k_mod_2)
