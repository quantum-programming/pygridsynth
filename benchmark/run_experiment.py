import subprocess
import sys
import time
from subprocess import PIPE

import mpmath
import numpy as np
import tqdm

from pygridsynth.gridsynth import error, gridsynth_gates


def get_gateword(
    angle,
    bits=None,
    eps=None,
    digits=None,
):
    if bits is not None:
        proc = subprocess.run(
            f"gridsynth {angle} -b {bits}",
            shell=True,
            stdout=PIPE,
            stderr=PIPE,
            text=True,
        )
    if eps is not None:
        proc = subprocess.run(
            f"gridsynth {angle} -e {eps}",
            shell=True,
            stdout=PIPE,
            stderr=PIPE,
            text=True,
        )
    if digits is not None:
        proc = subprocess.run(
            f"gridsynth {angle} -d {digits}",
            shell=True,
            stdout=PIPE,
            stderr=PIPE,
            text=True,
        )

    gateword = proc.stdout
    return gateword


def run_newsynth_experiment(seed=1234, n_data=10):
    np.random.seed(
        seed,
    )
    theta_list = np.random.rand(n_data) * np.pi / 4
    eps_list = 10 ** np.linspace(-20, -1, 20)

    print("\n\nbenchmarking newsynth...")

    mpmath.mp.dps = 256
    t_newsynth_data = []
    for theta_ in tqdm.tqdm(theta_list):
        # for theta_ in theta_list:

        t_newsynth_list = []
        for eps_ in eps_list:
            theta = mpmath.mpmathify(f"{theta_}")
            eps = mpmath.mpmathify(f"{eps_}")

            t0 = time.time()
            gates = get_gateword(
                theta,
                eps=eps,
            )
            t1 = time.time()

            assert (
                error(theta, gates) < 1.5 * eps
            ), f"{eps=:.5e} not properly synthesized in gridsynth."
            t_newsynth_list.append(t1 - t0)

        t_newsynth_data.append(t_newsynth_list)

    import json
    import os
    import sys

    if not os.path.exists("./data"):
        os.makedirs("./data", exist_ok=True)

    filename = "./data/newsynth_runtime.data"
    # json.dump(t_newsynth_data, open(filename, "w"))
    results = {"runtime_data": t_newsynth_data, "eps_list": eps_list.tolist()}
    json.dump(results, open(filename, "w"))

    print(f"newsynth runtime results saved in {filename}.")


def run_pygridsynth_experiment(seed=1234, n_data=10):
    np.random.seed(seed)
    theta_list = np.random.rand(n_data) * np.pi / 4
    eps_list = 10 ** np.linspace(-20, -1, 20)

    print("\n\nbenchmarking pygridsynth...")

    t_pygridsynth_data = []
    for theta_ in tqdm.tqdm(theta_list):
        t_pygridsynth_list = []
        for eps_ in eps_list:
            # print(eps_)
            mpmath.mp.dps = 15 + int(2.5 * (-mpmath.log10(eps_)))
            theta = mpmath.mpmathify(f"{theta_}")
            eps = mpmath.mpmathify(f"{eps_}")

            t0 = time.time()
            gates = gridsynth_gates(
                theta, eps, diophantine_timeout=0.1, factoring_timeout=0.1
            )
            t1 = time.time()

            assert (
                error(theta, gates) < 1.5 * eps
            ), f"{eps=:.5e} not properly synthesized in pygridsynth."
            t_pygridsynth_list.append(t1 - t0)

        t_pygridsynth_data.append(t_pygridsynth_list)

    import json
    import os
    import sys

    if not os.path.exists("./data"):
        os.makedirs("./data", exist_ok=True)

    filename = "./data/pygridsynth_runtime.data"
    results = {"runtime_data": t_pygridsynth_data, "eps_list": eps_list.tolist()}
    json.dump(results, open(filename, "w"))

    print(f"pygridsynth runtime results saved in {filename}.")

def run_newsynth_experiment_pi_128(n_data = 3):
    eps_list = 10 ** np.linspace(-100, -10, 10)
    print("\n\nbenchmarking newsynth...")

    mpmath.mp.dps = 256
    theta = "pi/128"
    theta_mpf = mpmath.pi / 128
    t_newsynth_data = []
    for _ in range(n_data):
        t_newsynth_list = []
        for eps in eps_list:

            t0 = time.time()
            gates = get_gateword(
                theta,
                eps=eps,
            )
            t1 = time.time()

            assert (
                error(theta_mpf, gates) < 1.5 * eps
            ), f"{eps=:.5e} not properly synthesized in gridsynth."
            t_newsynth_list.append(t1 - t0)
        t_newsynth_data.append(t_newsynth_list)

    import json
    import os
    import sys

    if not os.path.exists("./data"):
        os.makedirs("./data", exist_ok=True)

    filename = "./data/newsynth_runtime_pi_128.data"
    results = {"runtime_data": t_newsynth_data, "eps_list": eps_list.tolist()}
    json.dump(results, open(filename, "w"))

    print(f"newsynth runtime results saved in {filename}.")    

def run_pygridsynth_experiment_pi_128(n_data=3):

    eps_list = 10 ** np.linspace(-100, -10, 10)

    print("\n\nbenchmarking pygridsynth...")

    t_pygridsynth_data = []
    # for theta_ in tqdm.tqdm(theta_list):
    for _ in range(n_data):
        t_pygridsynth_list = []
        for eps_ in eps_list:
            # print(eps_)
            mpmath.mp.dps = 15 + int(2.5 * (-mpmath.log10(eps_)))
            theta = mpmath.pi / 128
            eps = mpmath.mpmathify(f"{eps_}")

            t0 = time.time()
            gates = gridsynth_gates(
                theta, eps, diophantine_timeout=0.1, factoring_timeout=0.1
            )
            t1 = time.time()

            assert (
                error(theta, gates) < 1.5 * eps
            ), f"{eps=:.5e} not properly synthesized in pygridsynth."
            t_pygridsynth_list.append(t1 - t0)

        t_pygridsynth_data.append(t_pygridsynth_list)

    import json
    import os
    import sys

    if not os.path.exists("./data"):
        os.makedirs("./data", exist_ok=True)

    filename = "./data/pygridsynth_runtime_pi_128.data"
    results = {"runtime_data": t_pygridsynth_data, "eps_list": eps_list.tolist()}
    json.dump(results, open(filename, "w"))

    print(f"pygridsynth runtime results saved in {filename}.")


if __name__ == "__main__":
    seed = 1234
    n_data = 20

    print("Benchmarking random angles:")
    try:
        run_newsynth_experiment(seed, n_data)
    except:
        print("newsynth not found, skipping.\n")
    run_pygridsynth_experiment(seed, n_data)

    """
    print("\n\nBenchmarking pi/128:")
    try:
        run_newsynth_experiment_pi_128()
    except:
        print("newsynth not found, skipping.\n")
    run_pygridsynth_experiment_pi_128()
    """