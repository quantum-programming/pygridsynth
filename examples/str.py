from pygridsynth.gridsynth import gridsynth_gates

theta = "0.5"
epsilon = "1e-10"
gates = gridsynth_gates(theta=theta, epsilon=epsilon)
print(gates)
