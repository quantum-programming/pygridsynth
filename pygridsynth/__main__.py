import argparse

import mpmath

from .diophantine import set_random_seed
from .gridsynth import gridsynth_gates
from .loop_controller import LoopController

helps = {
    "dt": "Diophantine algorithm timeout in milliseconds",
    "ft": "Factoring algorithm timeout in milliseconds",
    "dl": "Diophantine algorithm max loop count",
    "fl": "Factoring algorithm max loop count",
    "seed": "Random seed for deterministic results",
}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("theta", type=str)
    parser.add_argument("epsilon", type=str)
    parser.add_argument("--dps", type=int, default=None)
    parser.add_argument("--dtimeout", "-dt", type=float, default=None, help=helps["dt"])
    parser.add_argument("--ftimeout", "-ft", type=float, default=None, help=helps["ft"])
    parser.add_argument("--dloop", "-dl", type=int, default=2000, help=helps["dl"])
    parser.add_argument("--floop", "-fl", type=int, default=500, help=helps["fl"])
    parser.add_argument("--seed", type=int, default=0, help=helps["seed"])
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--time", "-t", action="store_true")
    parser.add_argument("--showgraph", "-g", action="store_true")
    args = parser.parse_args()

    # Set random seed for deterministic results
    set_random_seed(args.seed)

    # Use the same heuristic as Haskell gridsynth for setting dps
    if args.dps is None:
        epsilon1 = mpmath.mpmathify(args.epsilon)
        dps_of_result = -mpmath.log10(epsilon1)
        args.dps = 15 + 2.5 * dps_of_result
    mpmath.mp.dps = args.dps
    epsilon = mpmath.mpmathify(args.epsilon)
    mpmath.mp.pretty = True
    theta = mpmath.mpmathify(args.theta)

    loop_controller = LoopController(
        dloop=args.dloop,
        floop=args.floop,
        dtimeout=args.dtimeout,
        ftimeout=args.ftimeout,
    )
    gates = gridsynth_gates(
        theta=theta,
        epsilon=epsilon,
        loop_controller=loop_controller,
        verbose=args.verbose,
        measure_time=args.time,
        show_graph=args.showgraph,
    )
    print(gates)
    return gates


if __name__ == "__main__":
    main()
