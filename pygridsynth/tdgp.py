from .ring import DRootTwo, DOmega
from .region import Interval
from .odgp import solve_scaled_ODGP, solve_scaled_ODGP_with_parity
from .myplot import plot_sol


def solve_TDGP(setA, setB, opG, ellipseA_upright, ellipseB_upright, bboxA, bboxB, k,
               verbose=False, show_graph=False):
    sol_sufficient = []
    sol_x = solve_scaled_ODGP(bboxA.I_x, bboxB.I_x, k + 1)
    sol_y = solve_scaled_ODGP(bboxA.I_y.fatten(bboxA.I_y.width * 1e-4),
                              bboxB.I_y.fatten(bboxB.I_y.width * 1e-4),
                              k + 1)
    if len(sol_x) <= 0 or len(sol_y) <= 0:
        sol_sufficient = []
    else:
        alpha0 = sol_x[0]
        for beta in sol_y:
            dx = DRootTwo.power_of_inv_sqrt2(k)
            z0 = opG.inv * DOmega.from_droottwo_vector(alpha0, beta, k + 1)
            v = opG.inv * DOmega.from_droottwo_vector(dx, DRootTwo.from_int(0), k)
            t_A = setA.intersect(z0, v)
            t_B = setB.intersect(z0.conj_sq2, v.conj_sq2)
            if t_A is None or t_B is None:
                continue

            parity = (beta - alpha0).mul_by_sqrt2_power_renewing_denomexp(k)
            intA, intB = Interval(*t_A), Interval(*t_B)
            dtA = 10 / max(10, (1 << k) * intB.width)
            dtB = 10 / max(10, (1 << k) * intA.width)
            intA, intB = intA.fatten(dtA), intB.fatten(dtB)
            sol_t = solve_scaled_ODGP_with_parity(intA, intB, 1, parity)
            sol_x = [alpha * dx + alpha0 for alpha in sol_t]
            for alpha in sol_x:
                sol_sufficient.append(DOmega.from_droottwo_vector(alpha, beta, k))
    sol_transformed = [opG.inv * z for z in sol_sufficient]
    sol = [z for z in sol_transformed if setA.inside(z) and setB.inside(z.conj_sq2)]

    if verbose and len(sol_sufficient) > 0:
        print(f"{k=}")
        print(f"size of sol_sufficient: {len(sol_sufficient)}, size of sol: {len(sol)}")
    if show_graph and len(sol_sufficient) > 0:
        plot_sol([sol_transformed, sol], setA.ellipse, setB.ellipse, None, None,
                 color_list=['limegreen', 'blue'], size_list=[5, 10])

    return sol
