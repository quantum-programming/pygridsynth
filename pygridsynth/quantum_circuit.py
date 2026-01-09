from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .domega_unitary import DOmegaMatrix

import functools
import operator
from typing import Iterable

import mpmath

from .mymath import RealNum
from .quantum_gate import CxGate, QuantumGate, WGate, w_phase


class QuantumCircuit(list):
    def __init__(self, phase: RealNum = 0, args: list[QuantumGate] = []) -> None:
        self._phase: mpmath.mpf = mpmath.mpf(phase) % (2 * mpmath.mp.pi)
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

    def to_domega_matrix(self, wires: list[int]) -> DOmegaMatrix:
        from .domega_unitary import DOmegaMatrix

        product = [DOmegaMatrix.identity(wires=wires)]
        product[-1].phase = self.phase
        crt_matrix = None
        for g in self:
            if isinstance(g, WGate):
                if crt_matrix is not None:
                    product.append(crt_matrix)
                crt_matrix = None
                product.append(DOmegaMatrix.from_w_gate(g))
            elif isinstance(g, CxGate):
                if crt_matrix is not None:
                    product.append(crt_matrix)
                crt_matrix = None
                product.append(DOmegaMatrix.from_cx_gate(g))
            else:
                if crt_matrix is None:
                    crt_matrix = DOmegaMatrix.from_single_qubit_gate(g)
                else:
                    if crt_matrix.wires == g.wires:
                        crt_matrix = crt_matrix @ DOmegaMatrix.from_single_qubit_gate(g)
                    else:
                        product.append(crt_matrix)
                        crt_matrix = None
                        crt_matrix = DOmegaMatrix.from_single_qubit_gate(g)
        if crt_matrix is not None:
            product.append(crt_matrix)
        return functools.reduce(operator.matmul, product)

    def to_complex_matrix(self, n: int) -> mpmath.matrix:
        return self.to_domega_matrix(wires=list(range(n))).to_complex_matrix

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
