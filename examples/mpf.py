import mpmath

from pygridsynth.gridsynth import gridsynth_gates

mpmath.mp.dps = 128
theta = mpmath.mpmathify("0.5")
epsilon = mpmath.mpmathify("1e-10")

gates = gridsynth_gates(theta=theta, epsilon=epsilon)
print(gates)
