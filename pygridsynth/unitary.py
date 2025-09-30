from __future__ import annotations

from functools import cached_property

import mpmath

from .quantum_gate import HGate, QuantumCircuit, SGate, SXGate, TGate, WGate
from .ring import DOmega


class DOmegaUnitary:
    def __init__(self, z: DOmega, w: DOmega, n: int, k: int | None = None) -> None:
        if n >= 8 or n < 0:
            n &= 0b111
        self._n: int = n
        if k is None:
            if z.k > w.k:
                w = w.renew_denomexp(z.k)
            elif z.k < w.k:
                z = z.renew_denomexp(w.k)
        else:
            z = z.renew_denomexp(k)
            w = w.renew_denomexp(k)
        self._z: DOmega = z
        self._w: DOmega = w

    @property
    def z(self) -> DOmega:
        return self._z

    @property
    def w(self) -> DOmega:
        return self._w

    @property
    def n(self) -> int:
        return self._n

    @property
    def k(self) -> int:
        return self._w.k

    @cached_property
    def to_matrix(self) -> list[list[DOmega]]:
        return [
            [self._z, -self._w.conj.mul_by_omega_power(self._n)],
            [self._w, self._z.conj.mul_by_omega_power(self._n)],
        ]

    @cached_property
    def to_complex_matrix(self) -> mpmath.matrix:
        return mpmath.matrix(
            [
                [
                    self._z.to_complex,
                    -self._w.conj.mul_by_omega_power(self._n).to_complex,
                ],
                [
                    self._w.to_complex,
                    self._z.conj.mul_by_omega_power(self._n).to_complex,
                ],
            ]
        )

    def __repr__(self) -> str:
        return f"DOmegaUnitary({repr(self._z)}, {repr(self._w)}, {self._n})"

    def __str__(self) -> str:
        return str(self.to_matrix)

    def __eq__(self, other: DOmegaUnitary | object) -> bool:
        if isinstance(other, DOmegaUnitary):
            return self._z == other.z and self._w == other.w and self._n == other.n
        else:
            return False

    def mul_by_T_from_left(self) -> DOmegaUnitary:
        return DOmegaUnitary(self._z, self._w.mul_by_omega(), self._n + 1)

    def mul_by_T_inv_from_left(self) -> DOmegaUnitary:
        return DOmegaUnitary(self._z, self._w.mul_by_omega_inv(), self._n - 1)

    def mul_by_T_power_from_left(self, m: int) -> DOmegaUnitary:
        if m >= 8 or m < 0:
            m &= 0b111
        return DOmegaUnitary(self._z, self._w.mul_by_omega_power(m), self._n + m)

    def mul_by_S_from_left(self) -> DOmegaUnitary:
        return DOmegaUnitary(self._z, self._w.mul_by_omega_power(2), self._n + 2)

    def mul_by_S_power_from_left(self, m: int) -> DOmegaUnitary:
        if m >= 4 or m < 0:
            m &= 0b11
        return DOmegaUnitary(
            self._z, self._w.mul_by_omega_power(m << 1), self._n + (m << 1)
        )

    def mul_by_H_from_left(self) -> DOmegaUnitary:
        new_z = (self._z + self._w).mul_by_inv_sqrt2()
        new_w = (self._z - self._w).mul_by_inv_sqrt2()
        return DOmegaUnitary(new_z, new_w, self._n + 4)

    def mul_by_H_and_T_power_from_left(self, m: int) -> DOmegaUnitary:
        return self.mul_by_T_power_from_left(m).mul_by_H_from_left()

    def mul_by_X_from_left(self) -> DOmegaUnitary:
        return DOmegaUnitary(self._w, self._z, self._n + 4)

    def mul_by_W_from_left(self) -> DOmegaUnitary:
        return DOmegaUnitary(
            self._z.mul_by_omega(), self._w.mul_by_omega(), self._n + 2
        )

    def mul_by_W_power_from_left(self, m: int) -> DOmegaUnitary:
        if m >= 8 or m < 0:
            m &= 0b111
        return DOmegaUnitary(
            self._z.mul_by_omega_power(m),
            self._w.mul_by_omega_power(m),
            self._n + (m << 1),
        )

    def renew_denomexp(self, new_k: int) -> DOmegaUnitary:
        return DOmegaUnitary(self._z, self._w, self._n, new_k)

    def reduce_denomexp(self) -> DOmegaUnitary:
        new_z = self._z.reduce_denomexp()
        new_w = self._w.reduce_denomexp()
        return DOmegaUnitary(new_z, new_w, self._n)

    @classmethod
    def identity(cls) -> DOmegaUnitary:
        return cls(DOmega.from_int(1), DOmega.from_int(0), 0)

    @classmethod
    def from_circuit(cls, circuit: QuantumCircuit) -> DOmegaUnitary:
        unitary = cls.identity()
        for g in reversed(circuit):
            if isinstance(g, HGate):
                unitary = unitary.renew_denomexp(unitary.k + 1).mul_by_H_from_left()
            elif isinstance(g, TGate):
                unitary = unitary.mul_by_T_from_left()
            elif isinstance(g, SGate):
                unitary = unitary.mul_by_S_from_left()
            elif isinstance(g, SXGate):
                unitary = unitary.mul_by_X_from_left()
            elif isinstance(g, WGate):
                unitary = unitary.mul_by_W_from_left()
            else:
                raise ValueError
        return unitary.reduce_denomexp()

    @classmethod
    def from_gates(cls, gates: str) -> DOmegaUnitary:
        unitary = cls.identity()
        for g in reversed(gates):
            if g == "H":
                unitary = unitary.renew_denomexp(unitary.k + 1).mul_by_H_from_left()
            elif g == "T":
                unitary = unitary.mul_by_T_from_left()
            elif g == "S":
                unitary = unitary.mul_by_S_from_left()
            elif g == "X":
                unitary = unitary.mul_by_X_from_left()
            elif g == "W":
                unitary = unitary.mul_by_W_from_left()
            else:
                raise ValueError
        return unitary.reduce_denomexp()
