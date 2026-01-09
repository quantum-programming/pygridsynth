import itertools
from typing import Iterator

from .grid_op import GridOp
from .myplot import plot_sol
from .odgp import solve_scaled_ODGP, solve_scaled_ODGP_with_parity
from .region import ConvexSet, Ellipse, Interval, Rectangle
from .ring import DOmega, DRootTwo


def solve_TDGP(
    setA: ConvexSet,
    setB: ConvexSet,
    opG: GridOp,
    ellipseA_upright: Ellipse,
    ellipseB_upright: Ellipse,
    bboxA: Rectangle,
    bboxB: Rectangle,
    k: int,
    verbose: int = 0,
    show_graph: bool = False,
) -> Iterator[DOmega]:
    opG_inv = opG.inv
    if opG_inv is None:
        raise ValueError(f"Cannot compute inverse: grid operator {opG} has no inverse.")

    sol_sufficient: Iterator[DOmega] = iter([])
    sol_x = solve_scaled_ODGP(bboxA.I_x, bboxB.I_x, k + 1)
    try:
        alpha0 = next(sol_x)
    except StopIteration:
        return iter([])

    sol_y = solve_scaled_ODGP(
        bboxA.I_y.fatten(bboxA.I_y.width * 1e-4),
        bboxB.I_y.fatten(bboxB.I_y.width * 1e-4),
        k + 1,
    )

    def gen_sol_sufficient(beta: DRootTwo) -> Iterator[DOmega]:
        dx = DRootTwo.power_of_inv_sqrt2(k)
        z0 = opG_inv * DOmega.from_droottwo_vector(alpha0, beta, k + 1)
        v = opG_inv * DOmega.from_droottwo_vector(dx, DRootTwo.from_int(0), k)
        t_A = setA.intersect(z0, v)
        t_B = setB.intersect(z0.conj_sq2, v.conj_sq2)
        if t_A is None or t_B is None:
            return iter([])

        parity = (beta - alpha0).mul_by_sqrt2_power_renewing_denomexp(k)
        intA, intB = Interval(*t_A), Interval(*t_B)
        dtA = 10 / max(10, (1 << k) * intB.width)
        dtB = 10 / max(10, (1 << k) * intA.width)
        intA, intB = intA.fatten(dtA), intB.fatten(dtB)
        sol_t = solve_scaled_ODGP_with_parity(intA, intB, 1, parity)
        sol_x = map(lambda alpha: alpha * dx + alpha0, sol_t)
        return map(lambda alpha: DOmega.from_droottwo_vector(alpha, beta, k), sol_x)

    sol_sufficient = itertools.chain.from_iterable(map(gen_sol_sufficient, sol_y))

    sol_transformed = map(lambda z: opG_inv * z, sol_sufficient)
    sol = filter(lambda z: setA.inside(z) and setB.inside(z.conj_sq2), sol_transformed)

    if show_graph:
        plot_sol(
            [sol_transformed, sol],
            setA.ellipse,
            setB.ellipse,
            None,
            None,
            color_list=["limegreen", "blue"],
            size_list=[5, 10],
        )

    return sol
