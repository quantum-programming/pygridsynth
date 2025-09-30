import itertools
from typing import Iterator

from pygridsynth.region import Interval

from .mymath import SQRT2, ceil, floor, floorlog, pow_sqrt2
from .ring import LAMBDA, DRootTwo, ZRootTwo


def _solve_ODGP_internal(I: Interval, J: Interval) -> Iterator[ZRootTwo]:
    if I.width < 0 or J.width < 0:
        return iter([])
    elif I.width > 0 and J.width <= 0:
        sol = _solve_ODGP_internal(J, I)
        return map(lambda beta: beta.conj_sq2, sol)
    else:
        (n, _) = (0, 0) if J.width <= 0 else floorlog(J.width, LAMBDA.to_real)
        if n == 0:
            a_min = ceil((I.l + J.l) / 2)
            a_max = floor((I.r + J.r) / 2)

            def gen_sol(a: int) -> Iterator[ZRootTwo]:
                b_min = ceil(SQRT2() * (a - J.r) / 2)
                b_max = floor(SQRT2() * (a - J.l) / 2)
                return map(lambda b: ZRootTwo(a, b), range(b_min, b_max + 1))

            return itertools.chain.from_iterable(map(gen_sol, range(a_min, a_max + 1)))
        else:
            lambda_n = LAMBDA**n
            lambda_inv_n = LAMBDA**-n
            lambda_conj_sq2_n = LAMBDA.conj_sq2**n
            sol = _solve_ODGP_internal(
                I * lambda_n.to_real, J * lambda_conj_sq2_n.to_real
            )
            return map(lambda beta: beta * lambda_inv_n, sol)


def solve_ODGP(I: Interval, J: Interval) -> Iterator[ZRootTwo]:
    if I.width < 0 or J.width < 0:
        return iter([])

    a = floor((I.l + J.l) / 2)
    b = floor(SQRT2() * (I.l - J.l) / 4)
    alpha = ZRootTwo(a, b)
    sol = _solve_ODGP_internal(I - alpha.to_real, J - alpha.conj_sq2.to_real)
    sol = map(lambda beta: beta + alpha, sol)
    sol = filter(
        lambda beta: I.within(beta.to_real) and J.within(beta.conj_sq2.to_real), sol
    )
    return sol


def solve_ODGP_with_parity(
    I: Interval, J: Interval, beta: ZRootTwo
) -> Iterator[ZRootTwo]:
    p = beta.parity
    sol = solve_ODGP((I - p) * SQRT2() / 2, (J - p) * (-SQRT2()) / 2)
    sol = map(lambda alpha: alpha * ZRootTwo(0, 1) + p, sol)
    return sol


def solve_scaled_ODGP(I: Interval, J: Interval, k: int) -> Iterator[DRootTwo]:
    scale = pow_sqrt2(k)
    sol = solve_ODGP(I * scale, -J * scale if k & 1 else J * scale)
    return map(lambda alpha: DRootTwo(alpha, k), sol)


def solve_scaled_ODGP_with_parity(
    I: Interval, J: Interval, k: int, beta: DRootTwo
) -> Iterator[DRootTwo]:
    if k == 0:
        sol0 = solve_ODGP_with_parity(I, J, beta.renew_denomexp(0).alpha)
        return map(lambda alpha: DRootTwo.from_zroottwo(alpha), sol0)
    else:
        p = beta.renew_denomexp(k).parity
        offset = DRootTwo.from_int(0) if p == 0 else DRootTwo.power_of_inv_sqrt2(k)
        sol1 = solve_scaled_ODGP(I - offset.to_real, J - offset.conj_sq2.to_real, k - 1)
        return map(lambda alpha: alpha + offset, sol1)
