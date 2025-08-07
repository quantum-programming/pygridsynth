import mpmath

CNOT01 = mpmath.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
W_PHASE = mpmath.mp.pi / 4


def Rz(theta):
    return mpmath.matrix([[mpmath.exp(- 1.j * theta / 2), 0],
                          [0, mpmath.exp(1.j * theta / 2)]])


def Rx(theta):
    return mpmath.matrix([[mpmath.cos(- theta / 2), 1.j * mpmath.sin(- theta / 2)],
                          [1.j * mpmath.sin(- theta / 2), mpmath.cos(- theta / 2)]])


class QuantumGate():
    def __init__(self, matrix, wires):
        self._matrix = matrix
        self._wires = wires

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, matrix):
        self._matrix = matrix

    @property
    def wires(self):
        return self._wires

    def to_whole_matrix(self, num_qubits):
        whole_matrix = mpmath.matrix(2 ** num_qubits)
        m = len(self._wires)
        target_qubit_sorted = sorted(self._wires, reverse=True)

        for i in range(2 ** m):
            for j in range(2 ** m):
                k = 0
                while k < 2 ** num_qubits:
                    i2 = 0
                    j2 = 0
                    for l in range(m):
                        t = num_qubits - target_qubit_sorted[l] - 1
                        if k & 1 << t:
                            k += 1 << t
                        i2 |= (i >> l & 1) << t
                        j2 |= (j >> l & 1) << t
                    if k >= 2 ** num_qubits:
                        break
                    whole_matrix[k | i2, k | j2] = self._matrix[i, j]
                    k += 1

        return whole_matrix


class SingleQubitGate(QuantumGate):
    def __init__(self, matrix, target_qubit):
        self._matrix = matrix
        self._wires = [target_qubit]

    @property
    def target_qubit(self):
        return self._wires[0]

    def to_whole_matrix(self, num_qubits):
        whole_matrix = mpmath.matrix(2 ** num_qubits)
        t = num_qubits - self.target_qubit - 1

        for i in range(2):
            for j in range(2):
                k = 0
                while k < 2 ** num_qubits:
                    if k & 1 << t:
                        k += 1 << t
                    if k >= 2 ** num_qubits:
                        break
                    whole_matrix[k | (i << t), k | (j << t)] = self._matrix[i, j]
                    k += 1

        return whole_matrix


class HGate(SingleQubitGate):
    def __init__(self, target_qubit):
        matrix = mpmath.sqrt(2) / 2 * mpmath.matrix([[1, 1], [1, -1]])
        super().__init__(matrix, target_qubit)


class TGate(SingleQubitGate):
    def __init__(self, target_qubit):
        matrix = mpmath.matrix([[1, 0], [0, mpmath.exp(1.j * mpmath.pi / 4)]])
        super().__init__(matrix, target_qubit)


class SGate(SingleQubitGate):
    def __init__(self, target_qubit):
        matrix = mpmath.matrix([[1, 0], [0, 1.j]])
        super().__init__(matrix, target_qubit)


class WGate(SingleQubitGate):
    def __init__(self):
        matrix = mpmath.exp(1.j * W_PHASE)
        super().__init__(matrix, [])

    def to_whole_matrix(self, num_qubits):
        return self._matrix


class SXGate(SingleQubitGate):
    def __init__(self, target_qubit):
        matrix = mpmath.matrix([[0, 1], [1, 0]])
        super().__init__(matrix, target_qubit)


class RzGate(SingleQubitGate):
    def __init__(self, theta, target_qubit):
        self._theta = theta
        matrix = Rz(theta)
        super().__init__(matrix, target_qubit)

    @property
    def theta(self):
        return self._theta

    def __mul__(self, other):
        if isinstance(other, Rz) and self._target_qubit == other.target_qubit:
            return RzGate(self._theta + other.theta)
        else:
            return NotImplemented


class RxGate(SingleQubitGate):
    def __init__(self, theta, target_qubit):
        self._theta = theta
        matrix = Rx(theta)
        super().__init__(matrix, target_qubit)

    @property
    def theta(self):
        return self._theta

    def __mul__(self, other):
        if isinstance(other, Rx) and self._target_qubit == other.target_qubit:
            return RxGate(self._theta + other.theta)
        else:
            return NotImplemented


class CxGate(QuantumGate):
    def __init__(self, control_qubit, target_qubit):
        matrix = CNOT01
        super().__init__(matrix, [control_qubit, target_qubit])

    @property
    def control_qubit(self):
        return self.wires[0]

    @property
    def target_qubit(self):
        return self.wires[1]

    def to_whole_matrix(self, num_qubits):
        whole_matrix = mpmath.matrix(2 ** num_qubits)
        c = num_qubits - self.control_qubit - 1
        t = num_qubits - self.target_qubit - 1
        for i0 in range(2):
            for j0 in range(2):
                for i1 in range(2):
                    for j1 in range(2):
                        k = 0
                        while k < 2 ** num_qubits:
                            if k & (1 << min(c, t)):
                                k += 1 << min(c, t)
                            if k & (1 << max(c, t)):
                                k += 1 << max(c, t)
                            if k >= 2 ** num_qubits:
                                break
                            i2 = k | (i0 << c) | (i1 << t)
                            j2 = k | (j0 << c) | (j1 << t)
                            whole_matrix[i2, j2] = self._matrix[i0 * 2 + i1, j0 * 2 + j1]
                            k += 1
        return whole_matrix


class QuantumCircuit(list):
    def __init__(self, phase=0, args=[]):
        self._phase = phase % (2 * mpmath.mp.pi)
        super().__init__(args)

    @classmethod
    def from_list(cls, gates):
        return cls(args=gates)

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase):
        self._phase = phase

    def __str__(self):
        return f"exp(1.j * {self._phase}) * \n" + "* \n".join(self)

    def __add__(self, other):
        if isinstance(other, QuantumCircuit):
            return QuantumCircuit(self.phase + other.phase, super().__add__(other))
        elif isinstance(other, list):
            return QuantumCircuit(self.phase, super().__add__(other))
        else:
            return NotImplemented

    def __radd__(self, other):
        if isinstance(other, QuantumCircuit):
            return QuantumCircuit(other.phase + self.phase, super().__radd__(other))
        elif isinstance(other, list):
            return QuantumCircuit(self.phase, other.__add__(self))
        else:
            return NotImplemented

    def __iadd__(self, other):
        if isinstance(other, QuantumCircuit):
            self.phase += other.phase
            super().__iadd__(other)
            return self
        elif isinstance(other, list):
            super().__iadd__(other)
            return self
        else:
            return NotImplemented

    def decompose_phase_gate(self):
        self._phase %= 2 * mpmath.mp.pi

        for _ in range(round(float(self._phase / W_PHASE))):
            self.append(WGate())

        self._phase = 0

    def to_matrix(self, num_qubits):
        U = mpmath.eye(2 ** num_qubits) * mpmath.exp(1.j * self._phase)
        for g in self:
            U2 = mpmath.zeros(2 ** num_qubits)
            if isinstance(g, WGate):
                U2 = U * g.matrix
            elif isinstance(g, CxGate):
                for i in range(2 ** num_qubits):
                    for j in range(2 ** num_qubits):
                        for b0 in range(2):
                            for b1 in range(2):
                                t = num_qubits - g.target_qubit - 1
                                c = num_qubits - g.control_qubit - 1
                                j2 = j ^ b0 << c ^ b1 << t
                                U2[i, j2] += U[i, j] * g.matrix[(j >> c & 1) << 1 | (j >> t & 1), (j2 >> c & 1) << 1 | (j2 >> t & 1)]
            elif isinstance(g, SingleQubitGate):
                t = num_qubits - g.target_qubit - 1
                for i in range(2 ** num_qubits):
                    for j in range(2 ** num_qubits):
                        U2[i, j] += U[i, j] * g.matrix[j >> t & 1, j >> t & 1]
                        j2 = j ^ 1 << t
                        U2[i, j2] += U[i, j] * g.matrix[j >> t & 1, j2 >> t & 1]
            else:
                raise ValueError
            U = U2

        return U
