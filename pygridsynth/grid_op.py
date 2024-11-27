from functools import cached_property
import mpmath

from .ring import ZOmega, DOmega


class EllipsePair():
    def __init__(self, A, B):
        self.A = A
        self.B = B

    @property
    def skew(self):
        return self.A.skew + self.B.skew

    @property
    def bias(self):
        return self.B.bias / self.A.bias

    def __rmul__(self, other):
        if isinstance(other, GridOp):
            return self.__class__(other * self.A, other.conj_sq2 * self.B)
        else:
            return NotImplemented


class GridOp():
    def __init__(self, u0, u1):
        self._u0 = u0
        self._u1 = u1
        # d0 + b0 + d1 + b1 : even
        # a0 + c0 + a1 + c1 : even

    @property
    def u0(self):
        return self._u0

    @property
    def u1(self):
        return self._u1

    @property
    def a0(self):
        return self._u0.a

    @property
    def b0(self):
        return self._u0.b

    @property
    def c0(self):
        return self._u0.c

    @property
    def d0(self):
        return self._u0.d

    @property
    def a1(self):
        return self._u1.a

    @property
    def b1(self):
        return self._u1.b

    @property
    def c1(self):
        return self._u1.c

    @property
    def d1(self):
        return self._u1.d

    def __str__(self):
        return (f"[[{self.d0}{self.c0 - self.a0:+}/√2, {self.d1}{self.c1 - self.a1:+}/√2],\n"
                f" [{self.b0}{self.c0 + self.a0:+}/√2, {self.b1}{self.c1 + self.a1:+}/√2]]")

    @cached_property
    def _det_vec(self):
        return self._u0.conj * self._u1

    @cached_property
    def is_special(self):
        v = self._det_vec
        return v.a + v.c == 0 and (v.b == 1 or v.b == -1)

    @cached_property
    def toMat(self):
        return mpmath.matrix([[self._u0.real, self._u1.real], [self._u0.imag, self._u1.imag]])

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            return GridOp(self * other.u0, self * other.u1)
        elif isinstance(other, ZOmega):
            new_d = (self.d0 * other.d + self.d1 * other.b
                     + (self.c1 - self.a1 + self.c0 - self.a0) // 2 * other.c
                     + (self.c1 - self.a1 - self.c0 + self.a0) // 2 * other.a)
            new_c = (self.c0 * other.d + self.c1 * other.b
                     + (self.b1 + self.d1 + self.b0 + self.d0) // 2 * other.c
                     + (self.b1 + self.d1 - self.b0 - self.d0) // 2 * other.a)
            new_b = (self.b0 * other.d + self.b1 * other.b
                     + (self.c1 + self.a1 + self.c0 + self.a0) // 2 * other.c
                     + (self.c1 + self.a1 - self.c0 - self.a0) // 2 * other.a)
            new_a = (self.a0 * other.d + self.a1 * other.b
                     + (self.b1 - self.d1 + self.b0 - self.d0) // 2 * other.c
                     + (self.b1 - self.d1 - self.b0 + self.d0) // 2 * other.a)
            return ZOmega(new_a, new_b, new_c, new_d)
        elif isinstance(other, DOmega):
            return DOmega(self * other.u, other.k)
        else:
            return NotImplemented

    @cached_property
    def inv(self):
        if not self.is_special:
            return None

        new_c0 = (self.c1 + self.a1 - self.c0 - self.a0) // 2
        new_a0 = (- self.c1 - self.a1 - self.c0 - self.a0) // 2
        new_u0 = ZOmega(new_a0, -self.b0, new_c0, self.b1)
        new_c1 = (- self.c1 + self.a1 + self.c0 - self.a0) // 2
        new_a1 = (self.c1 - self.a1 + self.c0 - self.a0) // 2
        new_u1 = ZOmega(new_a1, self.d0, new_c1, -self.d1)
        if self._det_vec.b == -1:
            new_u0 = -new_u0
            new_u1 = -new_u1
        return GridOp(new_u0, new_u1)

    def __pow__(self, other):
        if isinstance(other, int):
            if other < 0:
                return self.inv ** (-other)

            new = self.__class__(ZOmega(0, 0, 0, 1), ZOmega(0, 1, 0, 0))
            tmp = self
            while other > 0:
                if other & 1:
                    new *= tmp
                tmp *= tmp
                other >>= 1
            return new
        else:
            return NotImplemented

    @cached_property
    def adj(self):
        new_c0 = (self.c1 - self.a1 + self.c0 - self.a0) // 2
        new_a0 = (self.c1 - self.a1 - self.c0 + self.a0) // 2
        new_u0 = ZOmega(new_a0, self.d1, new_c0, self.d0)
        new_c1 = (self.c1 + self.a1 + self.c0 + self.a0) // 2
        new_a1 = (self.c1 + self.a1 - self.c0 - self.a0) // 2
        new_u1 = ZOmega(new_a1, self.b1, new_c1, self.b0)
        return self.__class__(new_u0, new_u1)

    @cached_property
    def conj_sq2(self):
        return self.__class__(self._u0.conj_sq2, self._u1.conj_sq2)
