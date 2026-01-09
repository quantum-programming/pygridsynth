from __future__ import annotations

import mpmath


def cnot01() -> mpmath.matrix:
    return mpmath.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


def cnot10() -> mpmath.matrix:
    return mpmath.matrix([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])


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


class SingleQubitGate(QuantumGate):
    def __init__(self, matrix: mpmath.matrix, target_qubit: int) -> None:
        self._matrix: mpmath.matrix = matrix
        self._wires: list[int] = [target_qubit]

    @property
    def target_qubit(self) -> int:
        return self._wires[0]

    def __str__(self) -> str:
        return f"SingleQubitGate({self.matrix}, {self.target_qubit})"


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
        self._theta: mpmath.mpf = theta
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
        self._theta: mpmath.mpf = theta
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
