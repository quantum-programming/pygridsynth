from pygridsynth.mixed_synthesis import mixed_synthesis_sequential
from pygridsynth.mymath import random_su

if __name__ == "__main__":
    num_qubits = 2
    eps = 1e-4
    M = 64
    unitary = random_su(num_qubits)
    print(unitary)
    result = mixed_synthesis_sequential(unitary, num_qubits, eps, M, seed=123)
    print(result)
