import numpy as np

from pygridsynth.gridsynth import gridsynth_gates

# Float can be used if high precision is not required
theta = 0.5
epsilon = 1e-10
gates = gridsynth_gates(theta=theta, epsilon=epsilon)
print(gates)

theta = float(np.pi / 8)
epsilon = 1e-10
gates = gridsynth_gates(theta=theta, epsilon=epsilon)
print(gates)
