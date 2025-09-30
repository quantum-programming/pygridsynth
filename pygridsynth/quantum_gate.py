from __future__ import annotations

from typing import Iterable

import mpmath

from .mymath import RealNum


def cnot01() -> mpmath.matrix:
    return mpmath.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


def w_phase() -> mpmath.mpf:
    return mpmath.mp.pi / 4


def Rz(theta: mpmath.mpf) -> mpmath.matrix:
    return mpmath.matrix(
        [[mpmath.exp(-1.0j * theta / 2), 0], [0, mpmath.exp(1.0j * theta / 2)]]
    )


def Rx(theta: mpmath.mpf) -> mpmath.matrix:
    return mpmath.matrix(
        [
            [mpmath.cos(-theta / 2), 1.0j * mpmath.sin(-theta / 2)],
            [1.0j * mpmath.sin(-theta / 2), mpmath.cos(-theta / 2)],
        ]
    )


class QuantumGate:
    def __init__(self, matrix: mpmath.matrix, wires: list[int]) -> None:
        self._matrix: mpmath.matrix = matrix
        self._wires: list[int] = wires

    @property
    def matrix(self) -> mpmath.matrix:
        return self._matrix

    @matrix.setter
    def matrix(self, matrix: mpmath.matrix) -> None:
        self._matrix = matrix

    @property
    def wires(self) -> list[int]:
        return self._wires

    def __str__(self) -> str:
        return f"QuantumGate({self.matrix}, {self.wires})"

    def to_whole_matrix(self, num_qubits: int) -> mpmath.matrix:
        whole_matrix = mpmath.matrix(2**num_qubits)
        m = len(self._wires)
        target_qubit_sorted = sorted(self._wires, reverse=True)

        for i in range(2**m):
            for j in range(2**m):
                k = 0
                while k < 2**num_qubits:
                    i2 = 0
                    j2 = 0
                    for l in range(m):
                        t = num_qubits - target_qubit_sorted[l] - 1
                        if k & 1 << t:
                            k += 1 << t
                        i2 |= (i >> l & 1) << t
                        j2 |= (j >> l & 1) << t
                    if k >= 2**num_qubits:
                        break
                    whole_matrix[k | i2, k | j2] = self._matrix[i, j]
                    k += 1

        return whole_matrix


class SingleQubitGate(QuantumGate):
    def __init__(self, matrix: mpmath.matrix, target_qubit: int) -> None:
        self._matrix: mpmath.matrix = matrix
        self._wires: list[int] = [target_qubit]

    @property
    def target_qubit(self) -> int:
        return self._wires[0]

    def __str__(self) -> str:
        return f"SingleQubitGate({self.matrix}, {self.target_qubit})"

    def to_whole_matrix(self, num_qubits: int) -> mpmath.matrix:
        whole_matrix = mpmath.matrix(2**num_qubits)
        t = num_qubits - self.target_qubit - 1

        for i in range(2):
            for j in range(2):
                k = 0
                while k < 2**num_qubits:
                    if k & 1 << t:
                        k += 1 << t
                    if k >= 2**num_qubits:
                        break
                    whole_matrix[k | (i << t), k | (j << t)] = self._matrix[i, j]
                    k += 1

        return whole_matrix


class HGate(SingleQubitGate):
    def __init__(self, target_qubit: int) -> None:
        matrix = mpmath.sqrt(2) / 2 * mpmath.matrix([[1, 1], [1, -1]])
        super().__init__(matrix, target_qubit)

    def __str__(self) -> str:
        return f"HGate({self.target_qubit})"

    def to_simple_str(self) -> str:
        return "H"


class TGate(SingleQubitGate):
    def __init__(self, target_qubit: int) -> None:
        matrix = mpmath.matrix([[1, 0], [0, mpmath.exp(1.0j * mpmath.pi / 4)]])
        super().__init__(matrix, target_qubit)

    def __str__(self) -> str:
        return f"TGate({self.target_qubit})"

    def to_simple_str(self) -> str:
        return "T"


class SGate(SingleQubitGate):
    def __init__(self, target_qubit: int) -> None:
        matrix = mpmath.matrix([[1, 0], [0, 1.0j]])
        super().__init__(matrix, target_qubit)

    def __str__(self) -> str:
        return f"SGate({self.target_qubit})"

    def to_simple_str(self) -> str:
        return "S"


class WGate(QuantumGate):
    def __init__(self) -> None:
        matrix = mpmath.exp(1.0j * w_phase())
        super().__init__(matrix, [])

    def __str__(self) -> str:
        return "WGate()"

    def to_simple_str(self) -> str:
        return "W"

    def to_whole_matrix(self, num_qubits: int) -> mpmath.matrix:
        return self._matrix


class SXGate(SingleQubitGate):
    def __init__(self, target_qubit: int) -> None:
        matrix = mpmath.matrix([[0, 1], [1, 0]])
        super().__init__(matrix, target_qubit)

    def __str__(self) -> str:
        return f"SXGate({self.target_qubit})"

    def to_simple_str(self) -> str:
        return "X"


class RzGate(SingleQubitGate):
    def __init__(self, theta: mpmath.mpf, target_qubit: int) -> None:
        self._theta = theta
        matrix = Rz(theta)
        super().__init__(matrix, target_qubit)

    @property
    def theta(self) -> mpmath.mpf:
        return self._theta

    def __str__(self) -> str:
        return f"RzGate({self.theta}, {self.target_qubit})"

    def __mul__(self, other: RzGate) -> RzGate:
        if isinstance(other, RzGate) and self.target_qubit == other.target_qubit:
            return RzGate(self._theta + other.theta, self.target_qubit)
        else:
            return NotImplemented


class RxGate(SingleQubitGate):
    def __init__(self, theta: mpmath.mpf, target_qubit: int) -> None:
        self._theta = theta
        matrix = Rx(theta)
        super().__init__(matrix, target_qubit)

    @property
    def theta(self) -> mpmath.mpf:
        return self._theta

    def __str__(self) -> str:
        return f"RxGate({self.theta}, {self.target_qubit})"

    def __mul__(self, other: RxGate) -> RxGate:
        if isinstance(other, RxGate) and self.target_qubit == other.target_qubit:
            return RxGate(self._theta + other.theta, self.target_qubit)
        else:
            return NotImplemented


class CxGate(QuantumGate):
    def __init__(self, control_qubit: int, target_qubit: int) -> None:
        matrix = cnot01()
        super().__init__(matrix, [control_qubit, target_qubit])

    @property
    def control_qubit(self) -> int:
        return self.wires[0]

    @property
    def target_qubit(self) -> int:
        return self.wires[1]

    def __str__(self) -> str:
        return f"CxGate({self.control_qubit}, {self.target_qubit})"

    def to_whole_matrix(self, num_qubits: int) -> mpmath.matrix:
        whole_matrix = mpmath.matrix(2**num_qubits)
        c = num_qubits - self.control_qubit - 1
        t = num_qubits - self.target_qubit - 1
        for i0 in range(2):
            for j0 in range(2):
                for i1 in range(2):
                    for j1 in range(2):
                        k = 0
                        while k < 2**num_qubits:
                            if k & (1 << min(c, t)):
                                k += 1 << min(c, t)
                            if k & (1 << max(c, t)):
                                k += 1 << max(c, t)
                            if k >= 2**num_qubits:
                                break
                            i2 = k | (i0 << c) | (i1 << t)
                            j2 = k | (j0 << c) | (j1 << t)
                            whole_matrix[i2, j2] = self._matrix[
                                i0 * 2 + i1, j0 * 2 + j1
                            ]
                            k += 1
        return whole_matrix


class QuantumCircuit(list):
    def __init__(self, phase: RealNum = 0, args: list[QuantumGate] = []) -> None:
        self._phase = mpmath.mpf(phase) % (2 * mpmath.mp.pi)
        super().__init__(args)

    @classmethod
    def from_list(cls, gates: list[QuantumGate]) -> QuantumCircuit:
        return cls(args=gates)

    @property
    def phase(self) -> mpmath.mpf:
        return self._phase

    @phase.setter
    def phase(self, phase: RealNum) -> None:
        self._phase = mpmath.mpf(phase) % (2 * mpmath.mp.pi)

    def __str__(self) -> str:
        return f"exp(1.j * {self._phase}) * " + " * ".join(str(gate) for gate in self)

    def to_simple_str(self) -> str:
        return "".join(g.to_simple_str() for g in self)

    def __add__(self, other: QuantumCircuit | list) -> QuantumCircuit:
        if isinstance(other, QuantumCircuit):
            return QuantumCircuit(self.phase + other.phase, list(self) + list(other))
        elif isinstance(other, list):
            return QuantumCircuit(self.phase, list(self) + other)
        else:
            return NotImplemented

    def __radd__(self, other: QuantumCircuit | list) -> QuantumCircuit:
        if isinstance(other, QuantumCircuit):
            return QuantumCircuit(other.phase + self.phase, list(other) + list(self))
        elif isinstance(other, list):
            return QuantumCircuit(self.phase, other + list(self))
        else:
            return NotImplemented

    def __iadd__(self, other: QuantumCircuit | Iterable) -> QuantumCircuit:
        if isinstance(other, QuantumCircuit):
            self.phase += other.phase
            super().__iadd__(list(other))
            return self
        elif isinstance(other, Iterable):
            super().__iadd__(other)
            return self
        else:
            return NotImplemented

    def decompose_phase_gate(self) -> None:
        self._phase %= 2 * mpmath.mp.pi

        for _ in range(round(float(self._phase / w_phase()))):
            self.append(WGate())

        self._phase = 0

    def to_matrix(self, num_qubits: int) -> mpmath.matrix:
        U = mpmath.eye(2**num_qubits) * mpmath.exp(1.0j * self._phase)
        for g in self:
            U2 = mpmath.zeros(2**num_qubits)
            if isinstance(g, WGate):
                U2 = U * g.matrix
            elif isinstance(g, CxGate):
                for i in range(2**num_qubits):
                    for j in range(2**num_qubits):
                        for b0 in range(2):
                            for b1 in range(2):
                                t = num_qubits - g.target_qubit - 1
                                c = num_qubits - g.control_qubit - 1
                                j2 = j ^ b0 << c ^ b1 << t
                                U2[i, j2] += (
                                    U[i, j]
                                    * g.matrix[
                                        (j >> c & 1) << 1 | (j >> t & 1),
                                        (j2 >> c & 1) << 1 | (j2 >> t & 1),
                                    ]
                                )
            elif isinstance(g, SingleQubitGate):
                t = num_qubits - g.target_qubit - 1
                for i in range(2**num_qubits):
                    for j in range(2**num_qubits):
                        U2[i, j] += U[i, j] * g.matrix[j >> t & 1, j >> t & 1]
                        j2 = j ^ 1 << t
                        U2[i, j2] += U[i, j] * g.matrix[j >> t & 1, j2 >> t & 1]
            else:
                raise ValueError
            U = U2

        return U
