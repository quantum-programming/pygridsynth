import argparse
import mpmath

from .gridsynth import gridsynth_gates


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('theta', type=str)
    parser.add_argument('epsilon', type=str)
    parser.add_argument('--dps', type=int, default=128)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--time', '-t', action='store_true')
    parser.add_argument('--showgraph', '-g', action='store_true')

    args = parser.parse_args()
    mpmath.mp.dps = args.dps
    mpmath.mp.pretty = True
    theta = mpmath.mpmathify(args.theta)
    epsilon = mpmath.mpmathify(args.epsilon)

    gates = gridsynth_gates(theta=theta, epsilon=epsilon,
                            verbose=args.verbose, measure_time=args.time,
                            show_graph=args.showgraph)
    return gates


if __name__ == "__main__":
    main()
