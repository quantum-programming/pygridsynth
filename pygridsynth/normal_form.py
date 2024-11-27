from enum import Enum


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
CONJ3_TABLE = [(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 0), (0, 0, 3, 0),
               (0, 1, 0, 0), (0, 1, 1, 0), (0, 1, 2, 0), (0, 1, 3, 0),
               (1, 0, 0, 0), (2, 0, 3, 6), (1, 1, 2, 2), (2, 1, 3, 6),
               (1, 0, 2, 0), (2, 1, 1, 0), (1, 1, 0, 6), (2, 0, 1, 4),
               (2, 0, 0, 0), (1, 1, 3, 4), (2, 1, 0, 0), (1, 0, 1, 2),
               (2, 1, 2, 2), (1, 1, 1, 0), (2, 0, 2, 6), (1, 0, 3, 2)]
CINV_TABLE = [(0, 0, 0, 0), (0, 0, 3, 0), (0, 0, 2, 0), (0, 0, 1, 0),
              (0, 1, 0, 0), (0, 1, 1, 6), (0, 1, 2, 4), (0, 1, 3, 2),
              (2, 0, 0, 0), (1, 0, 1, 2), (2, 1, 0, 0), (1, 1, 3, 4),
              (2, 1, 1, 2), (1, 1, 1, 6), (2, 0, 2, 2), (1, 0, 3, 4),
              (1, 0, 0, 0), (2, 1, 3, 6), (1, 1, 2, 2), (2, 0, 3, 6),
              (1, 0, 2, 0), (2, 1, 1, 6), (1, 1, 0, 2), (2, 0, 1, 6)]
TCONJ_TABLE = [(Axis.I, 0, 0), (Axis.I, 1, 7),
               (Axis.H, 3, 3), (Axis.H, 2, 0),
               (Axis.SH, 0, 5), (Axis.SH, 1, 4)]


class Clifford():
    def __init__(self, a, b, c, d):
        if a >= 3 or a < 0:
            a %= 3
        if b >= 2 or b < 0:
            b &= 1
        if c >= 4 or c < 0:
            c &= 0b11
        if d >= 8 or d < 0:
            d &= 0b111
        self._a = a
        self._b = b
        self._c = c
        self._d = d

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def c(self):
        return self._c

    @property
    def d(self):
        return self._d

    def __repr__(self):
        return f"Clifford({self._a}, {self._b}, {self._c}, {self._d})"

    def __str__(self):
        return f"E^{self._a} X^{self._b} S^{self._c} ω^{self._d}"

    @classmethod
    def from_str(cls, g):
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

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self._a == other.a and self._b == other.b
                    and self._c == other.c and self._d == other.d)
        else:
            return False

    @classmethod
    def _conj2(cls, c, b):
        return CONJ2_TABLE[c << 1 | b]

    @classmethod
    def _conj3(cls, b, c, a):
        return CONJ3_TABLE[a << 3 | b << 2 | c]

    @classmethod
    def _cinv(cls, a, b, c):
        return CINV_TABLE[a << 3 | b << 2 | c]

    @classmethod
    def _tconj(cls, a, b):
        return TCONJ_TABLE[a << 1 | b]

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            a1, b1, c1, d1 = self.__class__._conj3(self._b, self._c, other.a)
            c2, d2 = self.__class__._conj2(c1, other.b)
            new_a = self._a + a1
            new_b = b1 + other.b
            new_c = c2 + other.c
            new_d = d2 + d1 + self._d + other.d
            return self.__class__(new_a, new_b, new_c, new_d)
        else:
            return NotImplemented

    def inv(self):
        a1, b1, c1, d1 = self.__class__._cinv(self._a, self._b, self._c)
        return self.__class__(a1, b1, c1, d1 - self._d)

    def decompose_coset(self):
        if self._a == 0:
            return Axis.I, self
        elif self._a == 1:
            return Axis.H, CLIFFORD_H.inv() * self
        elif self._a == 2:
            return Axis.SH, CLIFFORD_SH.inv() * self

    def decompose_tconj(self):
        axis, c1, d1 = self.__class__._tconj(self._a, self._b)
        return axis, self.__class__(0, self._b, c1 + self._c, d1 + self._d)

    def to_gates(self):
        axis, c = self.decompose_coset()
        return ("" if axis == Axis.I else axis.name) + "X" * c.b + "S" * c.c + "W" * c.d


class NormalForm():
    def __init__(self, syllables, c):
        self._syllables = syllables
        self._c = c

    @property
    def syllables(self):
        return self._syllables

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, c):
        self._c = c

    def __repr__(self):
        return f"NormalForm({repr(self._syllables)}, {repr(self._c)})"

    def _append_gate(self, g):
        if g in ["H", "S", "X", "W"]:
            self.c *= Clifford.from_str(g)
        elif g == "T":
            axis, new_c = self.c.decompose_tconj()
            if axis == Axis.I:
                if len(self._syllables) == 0:
                    self._syllables.append(Syllable.T)
                elif self._syllables[-1] == Syllable.T:
                    self._syllables[-1].pop()
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

    @classmethod
    def from_gates(cls, gates):
        normal_form = NormalForm([], CLIFFORD_I)
        for g in gates:
            normal_form._append_gate(g)
        return normal_form

    def to_gates(self):
        gates = ""
        for syllable in self._syllables:
            if syllable != Syllable.I:
                gates += syllable.name
        gates += self._c.to_gates()
        return "I" if gates == "" else gates


CLIFFORD_I = Clifford(0, 0, 0, 0)
CLIFFORD_X = Clifford(0, 1, 0, 0)
CLIFFORD_H = Clifford(1, 0, 1, 5)
CLIFFORD_S = Clifford(0, 0, 1, 0)
CLIFFORD_W = Clifford(0, 0, 0, 1)
CLIFFORD_SH = CLIFFORD_S * CLIFFORD_H
CLIFFORD_HS = CLIFFORD_H * CLIFFORD_S
CLIFFORD_SHS = CLIFFORD_S * CLIFFORD_H * CLIFFORD_S
