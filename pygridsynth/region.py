from abc import ABC, abstractmethod
from functools import cached_property
import numbers
import mpmath

from .mymath import sqrt
from .grid_op import GridOp


class Interval():
    def __init__(self, l, r):
        self.l = l
        self.r = r

    def __str__(self):
        return f"[{self.l}, {self.r}]"

    def __add__(self, other):
        if isinstance(other, numbers.Real):
            return self.__class__(self.l + other, self.r + other)
        elif isinstance(other, self.__class__):
            return self.__class__(self.l + other.l, self.r + other.r)
        else:
            return NotImplemented

    def __radd__(self, other):
        if isinstance(other, numbers.Real):
            return self + other
        elif isinstance(other, self.__class__):
            return self + other
        else:
            return NotImplemented

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __neg__(self):
        return self.__class__(- self.r, - self.l)

    def __mul__(self, other):
        if isinstance(other, numbers.Real):
            if other >= 0:
                return self.__class__(self.l * other, self.r * other)
            else:
                return self.__class__(self.r * other, self.l * other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, numbers.Real):
            if other > 0:
                return self.__class__(self.l / other, self.r / other)
            else:
                return self.__class__(self.r / other, self.l / other)
        else:
            return NotImplemented

    @cached_property
    def width(self):
        return self.r - self.l

    def fatten(self, eps):
        return Interval(self.l - eps, self.r + eps)

    def within(self, x):
        return self.l <= x <= self.r


class Rectangle():
    def __init__(self, x_l, x_r, y_l, y_r):
        self.I_x = Interval(x_l, x_r)
        self.I_y = Interval(y_l, y_r)

    def __str__(self):
        return f"{self.I_x}×{self.I_y}"

    def __mul__(self, other):
        if isinstance(other, numbers.Real):
            if other >= 0:
                new_I_x = self.I_x * other
                new_I_y = self.I_y * other
                return self.__class__(new_I_x.l, new_I_x.r, new_I_y.l, new_I_y.r)
            else:
                new_I_x = self.I_x * other
                new_I_y = self.I_y * other
                return self.__class__(new_I_x.r, new_I_x.l, new_I_y.r, new_I_y.l)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, numbers.Real):
            return self * other
        else:
            return NotImplemented

    @cached_property
    def area(self):
        return self.I_x.width * self.I_y.width

    def plot(self, ax, color='black'):
        x = [self.I_x.l, self.I_x.l, self.I_x.r, self.I_x.r, self.I_x.l]
        y = [self.I_y.l, self.I_y.r, self.I_y.r, self.I_y.l, self.I_y.l]
        ax.plot(x, y, c=color)


class Ellipse():
    def __init__(self, D, p):
        self.D = D
        self.p = p

    @property
    def px(self):
        return self.p[0]

    @px.setter
    def px(self, px):
        self.p[0] = px

    @property
    def py(self):
        return self.p[1]

    @py.setter
    def py(self, py):
        self.p[1] = py

    @property
    def a(self):
        return self.D[0, 0]

    @a.setter
    def a(self, a):
        self.D[0, 0] = a

    @property
    def b(self):
        return self.D[0, 1]

    @b.setter
    def b(self, b):
        self.D[0, 1] = b
        self.D[1, 0] = b

    @property
    def d(self):
        return self.D[1, 1]

    @d.setter
    def d(self, d):
        self.D[1, 1] = d

    def inside(self, v):
        x = v[0] - self.px
        y = v[1] - self.py
        tmp = self.a * x * x + 2 * self.b * x * y + self.d * y * y
        return tmp <= 1

    def bbox(self):
        sqrt_det = self.sqrt_det
        w = sqrt(self.d) / sqrt_det
        h = sqrt(self.a) / sqrt_det
        return Rectangle(self.px - w, self.px + w, self.py - h, self.py + h)

    def __mul__(self, other):
        if isinstance(other, numbers.Real):
            return self.__class__(self.D * (1 / other) ** 2, self.p * other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, GridOp):
            M00 = other.inv.toMat[0, 0]
            M01 = other.inv.toMat[0, 1]
            M10 = other.inv.toMat[1, 0]
            M11 = other.inv.toMat[1, 1]
            a = self.a * M00 * M00 + 2 * self.b * M00 * M10 + self.d * M10 * M10
            b = self.a * M00 * M01 + self.b * (M00 * M11 + M01 * M10) + self.d * M10 * M11
            d = self.a * M01 * M01 + 2 * self.b * M11 * M01 + self.d * M11 * M11
            new_D = mpmath.matrix([[a, b], [b, d]])

            M00 = other.toMat[0, 0]
            M01 = other.toMat[0, 1]
            M10 = other.toMat[1, 0]
            M11 = other.toMat[1, 1]
            px = M00 * self.px + M01 * self.py
            py = M10 * self.px + M11 * self.py
            new_p = mpmath.matrix([px, py])

            return self.__class__(new_D, new_p)
        elif isinstance(other, numbers.Real):
            return self * other
        else:
            return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, numbers.Real):
            return self.__class__(self.D * other ** 2, self.p / other)
        else:
            return NotImplemented

    @property
    def area(self):
        return mpmath.mp.pi / self.sqrt_det

    @property
    def sqrt_det(self):
        det = self.d * self.a - self.b ** 2
        return sqrt(det)

    def normalize(self):
        return self.__class__(self.D / self.sqrt_det, self.p * sqrt(self.sqrt_det))

    @property
    def skew(self):
        return self.b ** 2

    @property
    def bias(self):
        return self.d / self.a

    def plot(self, ax, n=5000):
        eig_val, eig_vec = mpmath.eigsy(self.D)
        vx = [self.px] * n
        vy = [self.py] * n
        for i in range(n):
            t = mpmath.mp.pi * 2 * i / n
            vx[i] += eig_vec[0, 0] * mpmath.cos(t) / sqrt(eig_val[0])
            vx[i] += eig_vec[0, 1] * mpmath.sin(t) / sqrt(eig_val[1])
            vy[i] += eig_vec[1, 0] * mpmath.cos(t) / sqrt(eig_val[0])
            vy[i] += eig_vec[1, 1] * mpmath.sin(t) / sqrt(eig_val[1])
        ax.plot(vx, vy, c='orangered')


class ConvexSet(ABC):
    def __init__(self, ellipse):
        self._ellipse = ellipse

    @abstractmethod
    def inside(self, u):
        pass

    @property
    def ellipse(self):
        return self._ellipse

    @abstractmethod
    def intersect(self, u, v):
        pass
