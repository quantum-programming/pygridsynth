import argparse

from .gridsynth import gridsynth_gates

helps = {
    "dt": "Maximum milliseconds allowed for a single Diophantine equation solving",
    "ft": "Maximum milliseconds allowed for a single integer factoring attempt",
    "dl": (
        "Maximum number of failed integer factoring attempts "
        "allowed during Diophantine equation solving"
    ),
    "fl": (
        "Maximum number of failed integer factoring attempts "
        "allowed during the factoring process"
    ),
    "seed": "Random seed for deterministic results",
    "phase": "Ignore global phase in search candidates.",
    "strip-phase": "Remove all W gates from the output string of gates.",
}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("theta", type=str)
    parser.add_argument("epsilon", type=str)
    parser.add_argument("--dps", type=int, default=None)
    parser.add_argument("--dtimeout", "-dt", type=float, default=None, help=helps["dt"])
    parser.add_argument("--ftimeout", "-ft", type=float, default=None, help=helps["ft"])
    parser.add_argument("--dloop", "-dl", type=int, default=10, help=helps["dl"])
    parser.add_argument("--floop", "-fl", type=int, default=10, help=helps["fl"])
    parser.add_argument("--phase", "-p", action="store_true", help=helps["phase"])
    parser.add_argument("--strip-phase", "-sp", action="store_true", help=helps["strip-phase"])
    parser.add_argument("--seed", type=int, default=0, help=helps["seed"])
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--time", "-t", action="store_true")
    parser.add_argument("--showgraph", "-g", action="store_true")
    args = parser.parse_args()

    gates = gridsynth_gates(
        theta=args.theta,
        epsilon=args.epsilon,
        dps=args.dps,
        dtimeout=args.dtimeout,
        ftimeout=args.ftimeout,
        dloop=args.dloop,
        floop=args.floop,
        seed=args.seed,
        phase=args.phase,
        strip_phase=args.strip_phase,
        verbose=args.verbose,
        measure_time=args.time,
        show_graph=args.showgraph,
    )
    return gates
