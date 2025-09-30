import time

import mpmath

from pygridsynth import gridsynth_gates


def test_exec_time():
    theta = "0.24"

    for eps, TL in [(1e-30, 1), (1e-40, 1.25), (1e-50, 1.5)]:
        t0 = time.perf_counter()
        dps = 15 + int(-mpmath.log10(mpmath.mpf(eps)) * 2.5)
        mpmath.mp.dps = dps
        gates = gridsynth_gates(mpmath.mpf(theta), mpmath.mpf(eps))
        t1 = time.perf_counter()

        print(f"eps: {eps}, TL: {TL}, time: {t1 - t0:.2f} seconds")
        print(f"{gates=}")
        assert t1 - t0 < TL, f"Execution time exceeded for eps={eps}, TL={TL}"


if __name__ == "__main__":
    test_exec_time()
