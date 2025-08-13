import argparse
import mpmath

from .gridsynth import gridsynth_gates


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('theta', type=str)
    parser.add_argument('epsilon', type=str)
    parser.add_argument('--dps', type=int, default=None)
    parser.add_argument('--dtimeout', '-dt', type=int, default=200)
    parser.add_argument('--ftimeout', '-ft', type=int, default=50)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--time', '-t', action='store_true')
    parser.add_argument('--showgraph', '-g', action='store_true')

    args = parser.parse_args()
    epsilon1 = mpmath.mpmathify(args.epsilon)
    dps_of_result = -mpmath.log10(epsilon1)
    # Use the same heuristic as Haskell gridsynth for setting dps
    if args.dps is None:
        args.dps = 15 + 2.5 * dps_of_result
    mpmath.mp.dps = args.dps
    epsilon = mpmath.mpmathify(f"{args.epsilon}")
    mpmath.mp.pretty = True
    theta = mpmath.mpmathify(f"{args.theta}")

    gates = gridsynth_gates(theta=theta, epsilon=epsilon,
                            factoring_timeout=args.ftimeout,
                            diophantine_timeout=args.dtimeout,
                            verbose=args.verbose, measure_time=args.time,
                            show_graph=args.showgraph)
    print(gates)
    return gates


if __name__ == "__main__":
    main()
