from __future__ import annotations

import string
from functools import cached_property

import mpmath

from .mymath import RealNum, einsum, from_matrix_to_tensor, from_tensor_to_matrix
from .quantum_circuit import QuantumCircuit
from .quantum_gate import CxGate, HGate, SGate, SingleQubitGate, SXGate, TGate, WGate
from .ring import OMEGA, DOmega


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


class DOmegaMatrix:
    def __init__(
        self, mat: list[list[DOmega]], wires: list[int], k: int = 0, phase: RealNum = 0
    ) -> None:
        n = len(wires)
        if (len(mat) != 2**n) or any(len(row) != 2**n for row in mat):
            raise ValueError(
                f"Matrix must be a {2**n}x{2**n} square matrix to match wires"
                f"(got {len(mat)}x{len(mat[0]) if len(mat) > 0 else 0})"
            )

        self._mat: list[list[DOmega]] = mat
        self._wires: list[int] = wires
        self._k: int = k
        self._phase: mpmath.mpf = mpmath.mpf(phase) % (2 * mpmath.mp.pi)

    @property
    def mat(self) -> list[list[DOmega]]:
        return self._mat

    @property
    def wires(self) -> list[int]:
        return self._wires

    @property
    def k(self) -> int:
        return self._k

    @property
    def phase(self) -> mpmath.mpf:
        return self._phase

    @phase.setter
    def phase(self, phase: RealNum) -> None:
        self._phase = mpmath.mpf(phase) % (2 * mpmath.mp.pi)

    @classmethod
    def from_domega_unitary(
        cls, unitary: DOmegaUnitary, wires: list[int], phase: RealNum = 0
    ) -> DOmegaMatrix:
        return DOmegaMatrix(unitary.to_matrix, phase=phase, wires=wires)

    @classmethod
    def from_single_qubit_gate(cls, g: SingleQubitGate) -> DOmegaMatrix:
        return DOmegaMatrix.from_domega_unitary(
            DOmegaUnitary.from_circuit(QuantumCircuit.from_list([g])),
            wires=[g.target_qubit],
        )

    @classmethod
    def from_single_qubit_circuit(
        cls, circuit: QuantumCircuit, wires: list[int]
    ) -> DOmegaMatrix:
        return DOmegaMatrix.from_domega_unitary(
            DOmegaUnitary.from_circuit(circuit), phase=circuit.phase, wires=wires
        )

    @classmethod
    def from_w_gate(cls, g: WGate) -> DOmegaMatrix:
        return DOmegaMatrix([[DOmega(OMEGA, 0)]], wires=[])

    @classmethod
    def from_cx_gate(cls, g: CxGate) -> DOmegaMatrix:
        wires = [g.control_qubit, g.target_qubit]
        mat = [[DOmega.from_int(0) for j in range(4)] for i in range(4)]
        for i, j in [(0, 0), (1, 1), (2, 3), (3, 2)]:
            mat[i][j] = DOmega.from_int(1)
        return DOmegaMatrix(mat, wires)

    @cached_property
    def to_matrix(self) -> list[list[DOmega]]:
        return self._mat

    @cached_property
    def to_complex_matrix(self) -> mpmath.matrix:
        return mpmath.matrix(
            [[x.to_complex for x in row] for row in self._mat]
        ) * mpmath.exp(1.0j * self._phase)

    def __repr__(self) -> str:
        return (
            f"DOmegaMatrix({repr(self._mat)}, {self._wires}, {self._k}, {self._phase})"
        )

    def __eq__(self, other: DOmegaMatrix | object) -> bool:
        if isinstance(other, DOmegaMatrix):
            return self._mat == other.mat and self._k == other.k
        else:
            return False

    def __matmul__(self, other: DOmegaMatrix) -> DOmegaMatrix:
        if isinstance(other, DOmegaMatrix):
            all_wires = list(set(self._wires + other.wires))
            n_total = len(all_wires)

            labels = list(string.ascii_lowercase)
            if (len(self._wires) + len(other.wires)) * 2 > len(labels):
                raise ValueError("Too many qubits for automatic einsum labeling")

            out_row_labels = [labels[i] for i in range(n_total)]
            out_col_labels = [labels[n_total + i] for i in range(n_total)]

            self_row_labels = [out_row_labels[all_wires.index(w)] for w in self._wires]
            self_col_labels = [out_col_labels[all_wires.index(w)] for w in self._wires]

            other_row_labels = [out_row_labels[all_wires.index(w)] for w in other.wires]
            other_col_labels = [out_col_labels[all_wires.index(w)] for w in other.wires]

            shared_wires = set(self._wires) & set(other.wires)
            for idx, w in enumerate(shared_wires):
                i = self._wires.index(w)
                j = other.wires.index(w)
                other_row_labels[j] = labels[n_total * 2 + idx]
                self_col_labels[i] = labels[n_total * 2 + idx]

            einsum_str = (
                "".join(self_row_labels + self_col_labels)
                + ","
                + "".join(other_row_labels + other_col_labels)
                + "->"
                + "".join(out_row_labels + out_col_labels)
            )

            A_tensor = from_matrix_to_tensor(self._mat, len(self._wires))
            B_tensor = from_matrix_to_tensor(other.mat, len(other.wires))
            C_tensor = from_tensor_to_matrix(
                einsum(einsum_str, A_tensor, B_tensor), n_total
            )

            return DOmegaMatrix(
                C_tensor,
                k=self._k + other.k,
                phase=self._phase + other.phase,
                wires=all_wires,
            )
        else:
            return NotImplemented

    @classmethod
    def identity(cls, wires: list[int]) -> DOmegaMatrix:
        n = 2 ** len(wires)
        mat = [
            [DOmega.from_int(1) if i == j else DOmega.from_int(0) for j in range(n)]
            for i in range(n)
        ]
        return cls(mat, wires)
