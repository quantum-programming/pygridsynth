import warnings
import numbers
import math
import random

from .ring import ZRootTwo, ZOmega, DOmega

NO_SOLUTION = "no solution"


def _find_factor(n):
    if not (n & 1) and n > 2:
        return 2
    a = random.randint(1, n)
    f = lambda x: (x * x + a) % n
    x, y, d = 2, 2, 1
    while d == 1:
        x, y = f(x), f(f(y))
        d = math.gcd(x - y, n)
    if d != n:
        return d
    return None


def _prime_factorize(n):
    factorization = []
    p = 2
    while p * p <= n:
        if n % p == 0:
            k = 0
            while n % p == 0:
                n //= p
                k += 1
            factorization.append((p, k))
        p += 2 if p != 2 else 1
    if n != 1:
        factorization.append((n, 1))
    return factorization


def _sqrt_negative_one(p):
    while True:
        b = random.randint(1, p - 1)
        h = pow(b, (p - 1) >> 2, p)
        r = h * h % p
        if r == p - 1:
            return h
        elif r != 1:
            return None


class F_p2():
    base = 0
    p = 0

    def __init__(self, a, b):
        self._a = a % self.__class__.p
        self._b = b % self.__class__.p

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            new_a = self._a * other.a + self._b * other.b % self.__class__.p * self.__class__.base
            new_b = self._a * other.b + self._b * other.a
            return self.__class__(new_a, new_b)
        else:
            return NotImplemented

    def __pow__(self, other):
        if isinstance(other, numbers.Integral):
            if other < 0:
                return NotImplemented
            else:
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


def _root_mod(x, p):
    x = x % p
    if p == 2:
        return x
    if x == 0:
        return 0
    if not (p & 1) and p > 2:
        return None
    if pow(x, (p - 1) // 2, p) != 1:
        return None
    while True:
        b = random.randint(1, p - 1)
        r = pow(b, p - 1, p)
        if r != 1:
            return None

        base = (b * b + p - x) % p
        if pow(base, (p - 1) // 2, p) != 1:
            F_p2.p = p
            F_p2.base = base
            return (F_p2(b, 1) ** ((p + 1) // 2)).a


def _decompose_relatively_int_prime(partial_facs):
    u = 1
    stack = partial_facs
    facs = []
    while len(stack):
        b, k_b = stack.pop()
        i = 0
        while True:
            if i >= len(facs):
                if b == 1 or b == -1:
                    u *= b
                else:
                    facs.append((b, k_b))
                break
            a, k_a = facs[i]
            if a % b == 0 and b % a == 0:
                u *= b // a
                facs[i] = (a, k_a + k_b)
                break
            else:
                g = math.gcd(a, b)
                if g == 1 or g == -1:
                    i += 1
                    continue
                else:
                    partial_facs = [(g, k_a + k_b), (a // g, k_a)]
                    u_a, facs_a = _decompose_relatively_int_prime(partial_facs)
                    u *= u_a
                    facs = facs + facs_a
                    facs[i] = facs.pop()
                    stack.append((b // g, k_b))
                    break

    return u, facs


def _adj_decompose_int_prime(p):
    if p < 0:
        p = -p
    if p == 0 or p == 1:
        return ZOmega.from_int(p)
    elif p == 2:
        return ZOmega(-1, 0, 1, 0)
    elif p & 0b11 == 1:
        h = _sqrt_negative_one(p)
        if h is None:
            return None
        else:
            t = ZOmega.gcd(h + ZOmega(0, 1, 0, 0), p)
            return t if t.conj * t == p or t.conj * t == -p else None
    elif p & 0b111 == 3:
        h = _root_mod(-2, p)
        if h is None:
            return None
        else:
            t = ZOmega.gcd(h + ZOmega(1, 0, 1, 0), p)
            return t if t.conj * t == p or t.conj * t == -p else None
    elif p & 0b111 == 7:
        h = _root_mod(2, p)
        if h is not None:
            return NO_SOLUTION
        else:
            return None
    else:
        return None


def _adj_decompose_int_prime_power(p, k):
    if not (k & 1):
        return p ** (k // 2)
    else:
        t = _adj_decompose_int_prime(p)
        if t is None:
            t = _adj_decompose_int(p)

        if t == NO_SOLUTION:
            return t
        else:
            return t ** k


def _adj_decompose_int(n):
    if n < 0:
        n = -n
    if n == 0 or n == 1:
        return ZOmega.from_int(n)
    else:
        p = _find_factor(n)
        if p is None:
            t = _adj_decompose_int_prime(n)
            if t is None:
                return _adj_decompose_int(n)
            else:
                return t
        else:
            partial_facs = [(p, 1), (n // p, 1)]
            _, facs = _decompose_relatively_int_prime(partial_facs)
            t = ZOmega.from_int(1)
            for (p, k) in facs:
                t_p = _adj_decompose_int_prime_power(p, k)
                if t_p == NO_SOLUTION:
                    return NO_SOLUTION
                t *= t_p
            return t


def _adj_decompose_selfassociate(xi):
    # xi \sim xi.conj_sq2
    if xi == 0:
        return ZOmega.from_int(0)

    n = math.gcd(xi.a, xi.b)
    r = xi // n
    t1 = _adj_decompose_int(n)
    t2 = ZOmega(0, 0, 1, 1) if r % ZRootTwo(0, 1) == 0 else 1
    if t1 is None:
        return None
    elif t1 == NO_SOLUTION:
        return NO_SOLUTION
    else:
        return t1 * t2


def _decompose_relatively_zomega_prime(partial_facs):
    u = 1
    stack = partial_facs
    facs = []
    while len(stack):
        b, k_b = stack.pop()
        i = 0
        while True:
            if i >= len(facs):
                if ZRootTwo.sim(b, 1):
                    u *= b
                else:
                    facs.append((b, k_b))
                break
            a, k_a = facs[i]
            if ZRootTwo.sim(a, b):
                u *= b // a
                facs[i] = (a, k_a + k_b)
                break
            else:
                g = ZRootTwo.gcd(a, b)
                if ZRootTwo.sim(g, 1):
                    i += 1
                    continue
                else:
                    partial_facs = [(g, k_a + k_b), (a // g, k_a)]
                    u_a, facs_a = _decompose_relatively_zomega_prime(partial_facs)
                    u *= u_a
                    facs = facs + facs_a
                    facs[i] = facs.pop()
                    stack.append((b // g, k_b))
                    break

    return u, facs


def _adj_decompose_zomega_prime(eta):
    p = eta.norm

    if p < 0:
        p = -p
    if p == 0 or p == 1:
        return ZOmega.from_int(p)
    elif p == 2:
        return ZOmega(-1, 0, 1, 0)
    elif p & 0b11 == 1:
        h = _sqrt_negative_one(p)
        if h is None:
            return None
        else:
            t = ZOmega.gcd(h + ZOmega(0, 1, 0, 0), eta)
            return t if ZRootTwo.sim(t.conj * t, eta) else None
    elif p & 0b111 == 3:
        h = _root_mod(-2, p)
        if h is None:
            return None
        else:
            t = ZOmega.gcd(h + ZOmega(1, 0, 1, 0), eta)
            return t if ZRootTwo.sim(t.conj * t, eta) else None
    elif p & 0b111 == 7:
        h = _root_mod(2, p)
        if h is not None:
            return NO_SOLUTION
        else:
            return None
    else:
        return None


def _adj_decompose_zomega_prime_power(eta, k):
    if not (k & 1):
        return eta ** (k // 2)
    else:
        t = _adj_decompose_zomega_prime(eta)
        if t is None:
            t = _adj_decompose_selfcoprime(eta)

        if t == NO_SOLUTION:
            return t
        else:
            return t ** k


def _adj_decompose_selfcoprime(xi):
    # gcd(xi, xi.conj_sq2) = 1
    n = xi.norm
    if n < 0:
        n = -n
    if n == 0 or n == 1:
        return ZOmega.from_int(n)
    else:
        p = _find_factor(n)
        if p is None:
            t = _adj_decompose_zomega_prime(xi)
            if t is None:
                return _adj_decompose_selfcoprime(xi)
            else:
                return t
        else:
            eta = ZRootTwo.gcd(xi, p)
            partial_facs = [(eta, 1), (xi // eta, 1)]
            _, facs = _decompose_relatively_zomega_prime(partial_facs)
            t = ZOmega.from_int(1)
            for (eta, k) in facs:
                t_eta = _adj_decompose_zomega_prime_power(eta, k)
                if t_eta is NO_SOLUTION:
                    return NO_SOLUTION
                t *= t_eta
            return t


def _adj_decompose(xi):
    if xi == 0:
        return ZOmega.from_int(0)

    d = ZRootTwo.gcd(xi, xi.conj_sq2)
    eta = xi // d
    t1 = _adj_decompose_selfassociate(d)
    t2 = _adj_decompose_selfcoprime(eta)
    return NO_SOLUTION if t1 == NO_SOLUTION or t2 == NO_SOLUTION else t1 * t2


def _diophantine(xi):
    if xi == 0:
        return ZOmega.from_int(0)
    elif xi < 0 or xi.conj_sq2 < 0:
        return NO_SOLUTION

    t = _adj_decompose(xi)
    if t == NO_SOLUTION:
        return NO_SOLUTION
    else:
        xi_associate = ZRootTwo.from_zomega(t.conj * t)
        u = xi // xi_associate
        v = u.sqrt()
        if v is None:
            warnings.warn("cannot find square root of u")
            return NO_SOLUTION
        else:
            return v * t


def diophantine_dyadic(xi):
    k_div_2, k_mod_2 = xi.k >> 1, xi.k & 1

    t = _diophantine(xi.alpha * ZRootTwo(1, 1) if k_mod_2 else xi.alpha)
    if t == NO_SOLUTION:
        return NO_SOLUTION
    else:
        if k_mod_2:
            t *= ZOmega(0, -1, 1, 0)
        return DOmega(t, k_div_2 + k_mod_2)
