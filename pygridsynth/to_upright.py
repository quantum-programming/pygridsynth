from .grid_op import GridOp
from .mymath import floorsqrt, log
from .myplot import plot_sol
from .region import ConvexSet, Ellipse, EllipsePair, Rectangle
from .ring import LAMBDA, ZOmega


def _reduction(
    ellipse_pair: EllipsePair, opG_l: GridOp, opG_r: GridOp, new_opG: GridOp
) -> tuple[EllipsePair, GridOp, GridOp, bool]:
    return new_opG * ellipse_pair, opG_l, new_opG * opG_r, False


def _shift_ellipse_pair(ellipse_pair: EllipsePair, n: int) -> EllipsePair:
    lambda_n = LAMBDA**n
    lambda_inv_n = LAMBDA**-n
    ellipse_pair.A.a *= lambda_inv_n.to_real
    ellipse_pair.A.d *= lambda_n.to_real
    ellipse_pair.B.a *= lambda_n.to_real
    ellipse_pair.B.d *= lambda_inv_n.to_real
    if n & 1:
        ellipse_pair.B.b = -ellipse_pair.B.b
    return ellipse_pair


def _step_lemma(
    ellipse_pair: EllipsePair, opG_l: GridOp, opG_r: GridOp, verbose: int = 0
) -> tuple[EllipsePair, GridOp, GridOp, bool]:
    A = ellipse_pair.A
    B = ellipse_pair.B
    if verbose >= 3:
        print("-----")
        print(f"skew: {ellipse_pair.skew}, bias: {ellipse_pair.bias}")
        print(
            f"bias(A): {A.bias}, bias(B): {B.bias}, "
            + "sign(A.b):"
            + ("+" if A.b >= 0 else "-")
            + ", sign(B.b):"
            + ("+" if B.b >= 0 else "-")
        )
        print("-----")
    if B.b < 0:
        if verbose >= 3:
            print("Z")
        OP_Z = GridOp(ZOmega(0, 0, 0, 1), ZOmega(0, -1, 0, 0))
        return _reduction(ellipse_pair, opG_l, opG_r, OP_Z)
    elif A.bias * B.bias < 1:
        if verbose >= 3:
            print("X")
        OP_X = GridOp(ZOmega(0, 1, 0, 0), ZOmega(0, 0, 0, 1))
        return _reduction(ellipse_pair, opG_l, opG_r, OP_X)
    elif ellipse_pair.bias > 33.971 or ellipse_pair.bias < 0.029437:
        n = round(log(ellipse_pair.bias) / log(LAMBDA.to_real) / 8)
        OP_S = GridOp(ZOmega(-1, 0, 1, 1), ZOmega(1, -1, 1, 0))
        if verbose >= 3:
            print(f"S ({n=})")
        return _reduction(ellipse_pair, opG_l, opG_r, OP_S**n)
    elif ellipse_pair.skew <= 15:
        return ellipse_pair, opG_l, opG_r, True
    elif ellipse_pair.bias > 5.8285 or ellipse_pair.bias < 0.17157:
        n = round(log(ellipse_pair.bias) / log(LAMBDA.to_real) / 4)
        ellipse_pair = _shift_ellipse_pair(ellipse_pair, n)
        if verbose >= 3:
            print(f"sigma ({n=})")
        if n >= 0:
            OP_SIGMA_L = GridOp(ZOmega(-1, 0, 1, 1), ZOmega(0, 1, 0, 0)) ** n
            OP_SIGMA_R = GridOp(ZOmega(0, 0, 0, 1), ZOmega(1, -1, 1, 0)) ** n
        else:
            OP_SIGMA_L = GridOp(ZOmega(-1, 0, 1, -1), ZOmega(0, 1, 0, 0)) ** (-n)
            OP_SIGMA_R = GridOp(ZOmega(0, 0, 0, 1), ZOmega(1, 1, 1, 0)) ** (-n)
        return ellipse_pair, opG_l * OP_SIGMA_L, OP_SIGMA_R * opG_r, False
    elif 0.24410 <= A.bias <= 4.0968 and 0.24410 <= B.bias <= 4.0968:
        if verbose >= 3:
            print("R")
        OP_R = GridOp(ZOmega(0, 0, 1, 0), ZOmega(1, 0, 0, 0))
        return _reduction(ellipse_pair, opG_l, opG_r, OP_R)
    elif A.b >= 0 and A.bias <= 1.6969:
        if verbose >= 3:
            print("K")
        OP_K = GridOp(ZOmega(-1, -1, 0, 0), ZOmega(0, -1, 1, 0))
        return _reduction(ellipse_pair, opG_l, opG_r, OP_K)
    elif A.b >= 0 and B.bias <= 1.6969:
        if verbose >= 3:
            print("K_conj_sq2")
        OP_K_conj_sq2 = GridOp(ZOmega(1, -1, 0, 0), ZOmega(0, -1, -1, 0))
        return _reduction(ellipse_pair, opG_l, opG_r, OP_K_conj_sq2)
    elif A.b >= 0:
        n = max(1, floorsqrt(min(A.bias, B.bias) / 4))
        if verbose >= 3:
            print(f"A ({n=})")
        OP_A_n = GridOp(ZOmega(0, 0, 0, 1), ZOmega(0, 1, 0, 2 * n))
        return _reduction(ellipse_pair, opG_l, opG_r, OP_A_n)
    else:
        n = max(1, floorsqrt(min(A.bias, B.bias) / 2))
        if verbose >= 3:
            print(f"B ({n=})")
        OP_B_n = GridOp(ZOmega(0, 0, 0, 1), ZOmega(n, 1, -n, 0))
        return _reduction(ellipse_pair, opG_l, opG_r, OP_B_n)


def to_upright_ellipse_pair(
    ellipseA: Ellipse, ellipseB: Ellipse, verbose: int = 0
) -> GridOp:
    ellipseA_normalized = ellipseA.normalize()
    ellipseB_normalized = ellipseB.normalize()
    ellipse_pair = EllipsePair(ellipseA_normalized, ellipseB_normalized)
    OP_I = GridOp(ZOmega(0, 0, 0, 1), ZOmega(0, 1, 0, 0))
    opG_l, opG_r = OP_I, OP_I
    while True:
        ellipse_pair, opG_l, opG_r, end = _step_lemma(
            ellipse_pair, opG_l, opG_r, verbose=verbose
        )
        if end:
            break
    return opG_l * opG_r


def to_upright_set_pair(
    setA: ConvexSet,
    setB: ConvexSet,
    opG: GridOp | None = None,
    show_graph: bool = False,
    verbose: int = 0,
) -> tuple[GridOp, Ellipse, Ellipse, Rectangle, Rectangle]:
    if opG is None:
        opG = to_upright_ellipse_pair(setA.ellipse, setB.ellipse, verbose=verbose)
    ellipse_pair = opG * EllipsePair(setA.ellipse, setB.ellipse)
    ellipseA_upright = ellipse_pair.A
    ellipseB_upright = ellipse_pair.B
    bboxA = ellipseA_upright.bbox()
    bboxB = ellipseB_upright.bbox()
    upA = ellipseA_upright.area / bboxA.area
    upB = ellipseB_upright.area / bboxB.area
    if verbose >= 2:
        print(f"{upA=}, {upB=}")
    if show_graph:
        plot_sol([], ellipseA_upright, ellipseB_upright, bboxA, bboxB)
    return opG, ellipseA_upright, ellipseB_upright, bboxA, bboxB
