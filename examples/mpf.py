import mpmath

from pygridsynth.gridsynth import gridsynth_gates

mpmath.mp.dps = 128
theta = mpmath.mpf("0.5")
epsilon = mpmath.mpf("1e-10")
gates = gridsynth_gates(theta=theta, epsilon=epsilon)
print(gates)

mpmath.mp.dps = 128
theta = mpmath.mp.pi / 8
epsilon = mpmath.mpf("1e-10")
gates = gridsynth_gates(theta=theta, epsilon=epsilon)
print(gates)
