import mpmath
from itertools import accumulate


def SQRT2():
    return mpmath.sqrt(2)


def ntz(n):
    return 0 if n == 0 else ((n & -n) - 1).bit_count()


def floor(x):
    return int(mpmath.floor(x, prec=0))


def ceil(x):
    return int(mpmath.ceil(x, prec=0))


def sqrt(x):
    return mpmath.sqrt(x)


def log(x):
    return mpmath.log(x)


def sign(x):
    return 1 if x > 0 else -1 if x < 0 else 0


def floorsqrt(x):
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


def rounddiv(x, y):
    return (x + y // 2) // y if y > 0 else (x - (- y) // 2) // y


def pow_sqrt2(k):
    k_div_2, k_mod_2 = k >> 1, k & 1
    return (1 << k_div_2) * SQRT2() if k_mod_2 else 1 << k_div_2


def floorlog(x, y):
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
    return (n, r)


def solve_quadratic(a, b, c):
    if a < 0:
        a, b, c = -a, -b, -c
    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        return None
    s1 = - b - sqrt(discriminant)
    s2 = - b + sqrt(discriminant)
    if b >= 0:
        return (s1 / (2 * a), s2 / (2 * a))
    else:
        return ((2 * c) / s2, (2 * c) / s1)
