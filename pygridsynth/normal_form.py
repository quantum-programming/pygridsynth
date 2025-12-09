from __future__ import annotations

from enum import Enum

import mpmath

from .mymath import RealNum
from .quantum_circuit import QuantumCircuit
from .quantum_gate import HGate, QuantumGate, SGate, SXGate, TGate, WGate


class Axis(Enum):
    I = 0
    H = 1
    SH = 2


class Syllable(Enum):
    I = 0
    T = 1
    HT = 2
    SHT = 3


CONJ2_TABLE = [(0, 0), (0, 0), (1, 0), (3, 2), (2, 0), (2, 4), (3, 0), (1, 6)]
CONJ3_TABLE = [
    (0, 0, 0, 0),
    (0, 0, 1, 0),
    (0, 0, 2, 0),
    (0, 0, 3, 0),
    (0, 1, 0, 0),
    (0, 1, 1, 0),
    (0, 1, 2, 0),
    (0, 1, 3, 0),
    (1, 0, 0, 0),
    (2, 0, 3, 6),
    (1, 1, 2, 2),
    (2, 1, 3, 6),
    (1, 0, 2, 0),
    (2, 1, 1, 0),
    (1, 1, 0, 6),
    (2, 0, 1, 4),
    (2, 0, 0, 0),
    (1, 1, 3, 4),
    (2, 1, 0, 0),
    (1, 0, 1, 2),
    (2, 1, 2, 2),
    (1, 1, 1, 0),
    (2, 0, 2, 6),
    (1, 0, 3, 2),
]
CINV_TABLE = [
    (0, 0, 0, 0),
    (0, 0, 3, 0),
    (0, 0, 2, 0),
    (0, 0, 1, 0),
    (0, 1, 0, 0),
    (0, 1, 1, 6),
    (0, 1, 2, 4),
    (0, 1, 3, 2),
    (2, 0, 0, 0),
    (1, 0, 1, 2),
    (2, 1, 0, 0),
    (1, 1, 3, 4),
    (2, 1, 1, 2),
    (1, 1, 1, 6),
    (2, 0, 2, 2),
    (1, 0, 3, 4),
    (1, 0, 0, 0),
    (2, 1, 3, 6),
    (1, 1, 2, 2),
    (2, 0, 3, 6),
    (1, 0, 2, 0),
    (2, 1, 1, 6),
    (1, 1, 0, 2),
    (2, 0, 1, 6),
]
TCONJ_TABLE = [
    (Axis.I, 0, 0),
    (Axis.I, 1, 7),
    (Axis.H, 3, 3),
    (Axis.H, 2, 0),
    (Axis.SH, 0, 5),
    (Axis.SH, 1, 4),
]


class Clifford:
    def __init__(self, a: int, b: int, c: int, d: int) -> None:
        if a >= 3 or a < 0:
            a %= 3
        if b >= 2 or b < 0:
            b &= 1
        if c >= 4 or c < 0:
            c &= 0b11
        if d >= 8 or d < 0:
            d &= 0b111
        self._a: int = a
        self._b: int = b
        self._c: int = c
        self._d: int = d

    @property
    def a(self) -> int:
        return self._a

    @property
    def b(self) -> int:
        return self._b

    @property
    def c(self) -> int:
        return self._c

    @property
    def d(self) -> int:
        return self._d

    def __repr__(self) -> str:
        return f"Clifford({self._a}, {self._b}, {self._c}, {self._d})"

    def __str__(self) -> str:
        return f"E^{self._a} X^{self._b} S^{self._c} ω^{self._d}"

    @classmethod
    def from_str(cls, g: str) -> Clifford:
        if g == "H":
            return CLIFFORD_H
        elif g == "S":
            return CLIFFORD_S
        elif g == "X":
            return CLIFFORD_X
        elif g == "W":
            return CLIFFORD_W
        else:
            raise ValueError

    @classmethod
    def from_gate(cls, g: QuantumGate) -> Clifford:
        if isinstance(g, HGate):
            return CLIFFORD_H
        elif isinstance(g, SGate):
            return CLIFFORD_S
        elif isinstance(g, SXGate):
            return CLIFFORD_X
        elif isinstance(g, WGate):
            return CLIFFORD_W
        else:
            raise ValueError

    def __eq__(self, other: Clifford | object) -> bool:
        if isinstance(other, Clifford):
            return (
                self._a == other.a
                and self._b == other.b
                and self._c == other.c
                and self._d == other.d
            )
        else:
            return False

    @classmethod
    def _conj2(cls, c: int, b: int) -> tuple[int, int]:
        return CONJ2_TABLE[c << 1 | b]

    @classmethod
    def _conj3(cls, b: int, c: int, a: int) -> tuple[int, int, int, int]:
        return CONJ3_TABLE[a << 3 | b << 2 | c]

    @classmethod
    def _cinv(cls, a: int, b: int, c: int) -> tuple[int, int, int, int]:
        return CINV_TABLE[a << 3 | b << 2 | c]

    @classmethod
    def _tconj(cls, a: int, b: int) -> tuple[Axis, int, int]:
        return TCONJ_TABLE[a << 1 | b]

    def __mul__(self, other: Clifford) -> Clifford:
        if isinstance(other, Clifford):
            a1, b1, c1, d1 = Clifford._conj3(self._b, self._c, other.a)
            c2, d2 = Clifford._conj2(c1, other.b)
            new_a = self._a + a1
            new_b = b1 + other.b
            new_c = c2 + other.c
            new_d = d2 + d1 + self._d + other.d
            return Clifford(new_a, new_b, new_c, new_d)
        else:
            return NotImplemented

    def inv(self) -> Clifford:
        a1, b1, c1, d1 = Clifford._cinv(self._a, self._b, self._c)
        return Clifford(a1, b1, c1, d1 - self._d)

    def decompose_coset(self) -> tuple[Axis, Clifford]:
        if self._a == 0:
            return Axis.I, self
        elif self._a == 1:
            return Axis.H, CLIFFORD_H.inv() * self
        elif self._a == 2:
            return Axis.SH, CLIFFORD_SH.inv() * self
        else:
            raise ValueError

    def decompose_tconj(self) -> tuple[Axis, Clifford]:
        axis, c1, d1 = Clifford._tconj(self._a, self._b)
        return axis, Clifford(0, self._b, c1 + self._c, d1 + self._d)

    def to_circuit(self, wires: list[int]) -> QuantumCircuit:
        axis, c = self.decompose_coset()
        circuit = QuantumCircuit()
        if axis == Axis.H:
            circuit.append(HGate(target_qubit=wires[0]))
        elif axis == Axis.SH:
            circuit.append(SGate(target_qubit=wires[0]))
            circuit.append(HGate(target_qubit=wires[0]))
        for _ in range(c.b):
            circuit.append(SXGate(target_qubit=wires[0]))
        for _ in range(c.c):
            circuit.append(SGate(target_qubit=wires[0]))
        for _ in range(c.d):
            circuit.append(WGate())
        return circuit


class NormalForm:
    def __init__(self, syllables: list[Syllable], c: Clifford, phase: RealNum = 0):
        self._syllables: list[Syllable] = syllables
        self._c: Clifford = c
        self._phase: mpmath.mpf = mpmath.mpf(phase) % (2 * mpmath.mp.pi)

    @property
    def syllables(self) -> list[Syllable]:
        return self._syllables

    @property
    def c(self) -> Clifford:
        return self._c

    @c.setter
    def c(self, c: Clifford) -> None:
        self._c = c

    @property
    def phase(self) -> mpmath.mpf:
        return self._phase

    @phase.setter
    def phase(self, phase: RealNum) -> None:
        self._phase = mpmath.mpf(phase) % (2 * mpmath.mp.pi)

    def __repr__(self) -> str:
        return (
            f"NormalForm({repr(self._syllables)}, {repr(self._c)}, {repr(self._phase)})"
        )

    def _append_gate(self, g: QuantumGate) -> None:
        if isinstance(g, TGate):
            axis, new_c = self.c.decompose_tconj()
            if axis == Axis.I:
                if len(self._syllables) == 0:
                    self._syllables.append(Syllable.T)
                elif self._syllables[-1] == Syllable.T:
                    self._syllables.pop()
                    self.c = CLIFFORD_S * new_c
                elif self._syllables[-1] == Syllable.HT:
                    self._syllables.pop()
                    self.c = CLIFFORD_HS * new_c
                elif self._syllables[-1] == Syllable.SHT:
                    self._syllables.pop()
                    self.c = CLIFFORD_SHS * new_c
            elif axis == Axis.H:
                self._syllables.append(Syllable.HT)
                self.c = new_c
            elif axis == Axis.SH:
                self._syllables.append(Syllable.SHT)
                self.c = new_c
        else:
            self.c *= Clifford.from_gate(g)

    @classmethod
    def from_circuit(cls, circuit: QuantumCircuit) -> NormalForm:
        normal_form = NormalForm([], CLIFFORD_I, phase=circuit.phase)
        for g in circuit:
            normal_form._append_gate(g)
        return normal_form

    def to_circuit(self, wires: list[int]) -> QuantumCircuit:
        circuit = QuantumCircuit(phase=self.phase)
        for syllable in self._syllables:
            if syllable == Syllable.T:
                circuit.append(TGate(target_qubit=wires[0]))
            elif syllable == Syllable.HT:
                circuit.append(HGate(target_qubit=wires[0]))
                circuit.append(TGate(target_qubit=wires[0]))
            elif syllable == Syllable.SHT:
                circuit.append(SGate(target_qubit=wires[0]))
                circuit.append(HGate(target_qubit=wires[0]))
                circuit.append(TGate(target_qubit=wires[0]))
        circuit += self._c.to_circuit(wires=wires)
        return circuit


CLIFFORD_I = Clifford(0, 0, 0, 0)
CLIFFORD_X = Clifford(0, 1, 0, 0)
CLIFFORD_H = Clifford(1, 0, 1, 5)
CLIFFORD_S = Clifford(0, 0, 1, 0)
CLIFFORD_W = Clifford(0, 0, 0, 1)
CLIFFORD_SH = CLIFFORD_S * CLIFFORD_H
CLIFFORD_HS = CLIFFORD_H * CLIFFORD_S
CLIFFORD_SHS = CLIFFORD_S * CLIFFORD_H * CLIFFORD_S
