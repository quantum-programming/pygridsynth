from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property

import mpmath

from .grid_op import GridOp
from .mymath import RealNum, sqrt


class Interval:
    def __init__(self, l: RealNum, r: RealNum) -> None:
        self.l: mpmath.mpf = mpmath.mpf(l)
        self.r: mpmath.mpf = mpmath.mpf(r)

    def __str__(self) -> str:
        return f"[{self.l}, {self.r}]"

    def __add__(self, other: RealNum | Interval) -> Interval:
        if isinstance(other, RealNum):
            return Interval(self.l + other, self.r + other)
        elif isinstance(other, Interval):
            return Interval(self.l + other.l, self.r + other.r)
        else:
            return NotImplemented

    def __radd__(self, other: RealNum | Interval) -> Interval:
        return self + other

    def __sub__(self, other: RealNum | Interval) -> Interval:
        return self + (-other)

    def __rsub__(self, other: RealNum | Interval) -> Interval:
        return (-self) + other

    def __neg__(self) -> Interval:
        return Interval(-self.r, -self.l)

    def __mul__(self, other: RealNum) -> Interval:
        if isinstance(other, RealNum):
            if other >= 0:
                return Interval(self.l * other, self.r * other)
            else:
                return Interval(self.r * other, self.l * other)
        else:
            return NotImplemented

    def __rmul__(self, other: RealNum) -> Interval:
        return self * other

    def __truediv__(self, other: RealNum) -> Interval:
        if isinstance(other, RealNum):
            if other > 0:
                return Interval(self.l / other, self.r / other)
            else:
                return Interval(self.r / other, self.l / other)
        else:
            return NotImplemented

    @cached_property
    def width(self) -> mpmath.mpf:
        return self.r - self.l

    def fatten(self, eps: RealNum) -> Interval:
        return Interval(self.l - eps, self.r + eps)

    def within(self, x: RealNum) -> bool:
        return self.l <= x <= self.r


class Rectangle:
    def __init__(self, x_l: RealNum, x_r: RealNum, y_l: RealNum, y_r: RealNum):
        self.I_x: Interval = Interval(x_l, x_r)
        self.I_y: Interval = Interval(y_l, y_r)

    def __str__(self) -> str:
        return f"{self.I_x}×{self.I_y}"

    def __mul__(self, other: RealNum) -> Rectangle:
        if isinstance(other, RealNum):
            if other >= 0:
                new_I_x = self.I_x * other
                new_I_y = self.I_y * other
                return Rectangle(new_I_x.l, new_I_x.r, new_I_y.l, new_I_y.r)
            else:
                new_I_x = self.I_x * other
                new_I_y = self.I_y * other
                return Rectangle(new_I_x.r, new_I_x.l, new_I_y.r, new_I_y.l)
        else:
            return NotImplemented

    def __rmul__(self, other: RealNum) -> Rectangle:
        return self * other

    @cached_property
    def area(self) -> mpmath.mpf:
        return self.I_x.width * self.I_y.width

    def plot(self, ax, color: str = "black") -> None:
        x = [self.I_x.l, self.I_x.l, self.I_x.r, self.I_x.r, self.I_x.l]
        y = [self.I_y.l, self.I_y.r, self.I_y.r, self.I_y.l, self.I_y.l]
        ax.plot(x, y, c=color)


class Ellipse:
    def __init__(self, D: mpmath.matrix, p: mpmath.matrix):
        self.D: mpmath.matrix = D.copy()
        self.p: mpmath.matrix = p.copy()

    @property
    def px(self) -> mpmath.mpf:
        return self.p[0]

    @px.setter
    def px(self, px: RealNum) -> None:
        self.p[0] = mpmath.mpf(px)

    @property
    def py(self) -> mpmath.mpf:
        return self.p[1]

    @py.setter
    def py(self, py: RealNum) -> None:
        self.p[1] = mpmath.mpf(py)

    @property
    def a(self) -> mpmath.mpf:
        return self.D[0, 0]

    @a.setter
    def a(self, a: mpmath.mpf) -> None:
        self.D[0, 0] = mpmath.mpf(a)

    @property
    def b(self) -> mpmath.mpf:
        return self.D[0, 1]

    @b.setter
    def b(self, b: RealNum) -> None:
        self.D[0, 1] = mpmath.mpf(b)
        self.D[1, 0] = mpmath.mpf(b)

    @property
    def d(self) -> mpmath.mpf:
        return self.D[1, 1]

    @d.setter
    def d(self, d: RealNum) -> None:
        self.D[1, 1] = mpmath.mpf(d)

    def inside(self, v: mpmath.matrix) -> bool:
        x = v[0] - self.px
        y = v[1] - self.py
        tmp = self.a * x * x + 2 * self.b * x * y + self.d * y * y
        return tmp <= 1

    def bbox(self) -> Rectangle:
        sqrt_det = self.sqrt_det
        w = sqrt(self.d) / sqrt_det
        h = sqrt(self.a) / sqrt_det
        return Rectangle(self.px - w, self.px + w, self.py - h, self.py + h)

    def __mul__(self, other: RealNum) -> Ellipse:
        if isinstance(other, RealNum):
            return Ellipse(self.D * (1 / other) ** 2, self.p * other)
        else:
            return NotImplemented

    def __rmul__(self, other: RealNum | GridOp) -> Ellipse:
        if isinstance(other, RealNum):
            return self * other
        elif isinstance(other, GridOp):
            if other.inv is None:
                raise ValueError(
                    f"Cannot compute inverse: grid operator {other} has no inverse."
                )
            M00 = other.inv.to_matrix[0, 0]
            M01 = other.inv.to_matrix[0, 1]
            M10 = other.inv.to_matrix[1, 0]
            M11 = other.inv.to_matrix[1, 1]
            a = self.a * M00 * M00 + 2 * self.b * M00 * M10 + self.d * M10 * M10
            b = (
                self.a * M00 * M01
                + self.b * (M00 * M11 + M01 * M10)
                + self.d * M10 * M11
            )
            d = self.a * M01 * M01 + 2 * self.b * M11 * M01 + self.d * M11 * M11
            new_D = mpmath.matrix([[a, b], [b, d]])

            M00 = other.to_matrix[0, 0]
            M01 = other.to_matrix[0, 1]
            M10 = other.to_matrix[1, 0]
            M11 = other.to_matrix[1, 1]
            px = M00 * self.px + M01 * self.py
            py = M10 * self.px + M11 * self.py
            new_p = mpmath.matrix([px, py])

            return Ellipse(new_D, new_p)
        else:
            return NotImplemented

    def __add__(self, other: RealNum) -> Ellipse:
        return self * other

    def __truediv__(self, other: RealNum) -> Ellipse:
        if isinstance(other, RealNum):
            return Ellipse(self.D * other**2, self.p / other)
        else:
            return NotImplemented

    @property
    def area(self) -> mpmath.mpf:
        return mpmath.mp.pi / self.sqrt_det

    @property
    def sqrt_det(self) -> mpmath.mpf:
        det = self.d * self.a - self.b**2
        return sqrt(det)

    def normalize(self) -> Ellipse:
        return Ellipse(self.D / self.sqrt_det, self.p * sqrt(self.sqrt_det))

    @property
    def skew(self) -> mpmath.mpf:
        return self.b**2

    @property
    def bias(self) -> mpmath.mpf:
        return self.d / self.a

    def plot(self, ax, n: int = 5000) -> None:
        eig_val, eig_vec = mpmath.eigsy(self.D)
        vx = [self.px] * n
        vy = [self.py] * n
        for i in range(n):
            t = mpmath.mp.pi * 2 * i / n
            vx[i] += eig_vec[0, 0] * mpmath.cos(t) / sqrt(eig_val[0])
            vx[i] += eig_vec[0, 1] * mpmath.sin(t) / sqrt(eig_val[1])
            vy[i] += eig_vec[1, 0] * mpmath.cos(t) / sqrt(eig_val[0])
            vy[i] += eig_vec[1, 1] * mpmath.sin(t) / sqrt(eig_val[1])
        ax.plot(vx, vy, c="orangered")


class EllipsePair:
    def __init__(self, A: Ellipse, B: Ellipse) -> None:
        self.A: Ellipse = A
        self.B: Ellipse = B

    @property
    def skew(self) -> mpmath.mpf:
        return self.A.skew + self.B.skew

    @property
    def bias(self) -> mpmath.mpf:
        return self.B.bias / self.A.bias

    def __rmul__(self, other: GridOp) -> EllipsePair:
        if isinstance(other, GridOp):
            return EllipsePair(other * self.A, other.conj_sq2 * self.B)
        else:
            return NotImplemented


class ConvexSet(ABC):
    def __init__(self, ellipse: Ellipse) -> None:
        self._ellipse: Ellipse = ellipse

    @abstractmethod
    def inside(self, u: mpmath.matrix) -> bool:
        pass

    @property
    def ellipse(self) -> Ellipse:
        return self._ellipse

    @abstractmethod
    def intersect(
        self, u0: mpmath.matrix, v: mpmath.matrix
    ) -> tuple[mpmath.mpf, mpmath.mpf] | None:
        pass
