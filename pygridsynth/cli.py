import argparse

from .config import GridsynthConfig
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
    "verbose": "Verbosity level (0=silent, 1=basic, 2=detailed, 3=debug)",
}


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("theta", type=str)
    parser.add_argument("epsilon", type=str)
    parser.add_argument("--dps", type=int)
    parser.add_argument("--dtimeout", "-dt", type=float, help=helps["dt"])
    parser.add_argument("--ftimeout", "-ft", type=float, help=helps["ft"])
    parser.add_argument("--dloop", "-dl", type=int, help=helps["dl"])
    parser.add_argument("--floop", "-fl", type=int, help=helps["fl"])
    parser.add_argument("--seed", type=int, help=helps["seed"])
    parser.add_argument("--verbose", "-v", type=int)
    parser.add_argument("--time", "-t", action="store_true")
    parser.add_argument("--showgraph", "-g", action="store_true")
    parser.add_argument("--up-to-phase", "-ph", action="store_true")
    args = parser.parse_args()

    cfg_args = dict()
    if args.dps is not None:
        cfg_args["dps"] = args.dps
    if args.dtimeout is not None:
        cfg_args["dtimeout"] = args.dtimeout
    if args.ftimeout is not None:
        cfg_args["ftimeout"] = args.ftimeout
    if args.dloop is not None:
        cfg_args["dloop"] = args.dloop
    if args.floop is not None:
        cfg_args["floop"] = args.floop
    if args.seed is not None:
        cfg_args["seed"] = args.seed
    if args.verbose:
        cfg_args["verbose"] = args.verbose
    if args.time:
        cfg_args["measure_time"] = args.time
    if args.showgraph:
        cfg_args["show_graph"] = args.showgraph
    if args.up_to_phase:
        cfg_args["up_to_phase"] = args.up_to_phase

    cfg = GridsynthConfig(**cfg_args)

    gates = gridsynth_gates(theta=args.theta, epsilon=args.epsilon, cfg=cfg)

    print(gates)
