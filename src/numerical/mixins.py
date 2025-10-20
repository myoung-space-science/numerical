import typing

import numpy
import numpy.typing

from . import _operators
from ._operations import (
    unary,
    binary,
    mytype,
)


T = typing.TypeVar('T')


class Orderable:
    """Operator support for orderable numerical quantities."""

    def __lt__(self, other):
        return binary(_operators.lt, self, other)

    def __le__(self, other):
        return binary(_operators.le, self, other)

    def __gt__(self, other):
        return binary(_operators.gt, self, other)

    def __ge__(self, other):
        return binary(_operators.ge, self, other)


class Comparable(Orderable):
    """Operator support for comparable numerical quantities."""

    def __eq__(self, other):
        return binary(_operators.eq, self, other)

    def __ne__(self, other):
        return binary(_operators.ne, self, other)


class Additive:
    """Operator support for additive numerical objects."""

    @mytype
    def __add__(self, other) -> typing.Self:
        return binary(_operators.add, self, other)

    @mytype
    def __radd__(self, other) -> typing.Self:
        return binary(_operators.add, other, self)

    @mytype
    def __sub__(self, other) -> typing.Self:
        return binary(_operators.sub, self, other)

    @mytype
    def __rsub__(self, other) -> typing.Self:
        return binary(_operators.sub, other, self)


class Multiplicative:
    """Operator support for multiplicative numerical objects."""

    @mytype
    def __mul__(self, other) -> typing.Self:
        return binary(_operators.mul, self, other)

    @mytype
    def __rmul__(self, other) -> typing.Self:
        return binary(_operators.mul, other, self)

    @mytype
    def __truediv__(self, other) -> typing.Self:
        return binary(_operators.truediv, self, other)

    @mytype
    def __rtruediv__(self, other) -> typing.Self:
        return binary(_operators.truediv, other, self)


class Algebraic(Additive, Multiplicative):
    """Operator support for algebraic numerical objects."""

    @mytype
    def __pow__(self, other, mod: int | None = None) -> typing.Self:
        return binary(_operators.pow, self, other, mod=mod)


class Complex(Algebraic):
    """Operator support for complex-valued numerical objects."""

    @mytype
    def __abs__(self) -> typing.Self:
        return unary(_operators.abs, self)

    @mytype
    def __pos__(self) -> typing.Self:
        return unary(_operators.pos, self)

    @mytype
    def __neg__(self) -> typing.Self:
        return unary(_operators.neg, self)


class Real(Comparable, Complex):
    """Operator support for real-valued numerical objects."""

    @mytype
    def __rpow__(self, other, mod: int | None = None) -> typing.Self:
        return binary(_operators.pow, other, self, mod=mod)

    @mytype
    def __floordiv__(self, other) -> typing.Self:
        return binary(_operators.floordiv, self, other)

    @mytype
    def __rfloordiv__(self, other) -> typing.Self:
        return binary(_operators.floordiv, other, self)

    @mytype
    def __mod__(self, other) -> typing.Self:
        return binary(_operators.mod, self, other)

    @mytype
    def __rmod__(self, other) -> typing.Self:
        return binary(_operators.mod, other, self)


class Value(Comparable, Complex):
    """Operator support for singular numerical objects."""

    def __complex__(self) -> complex:
        return unary(complex, self)

    def __float__(self) -> float:
        return unary(float, self)

    def __int__(self) -> int:
        return unary(int, self)

    @mytype
    def __round__(self) -> typing.Self:
        return unary(_operators.round, self)


class Sequence(Comparable, Complex, typing.Generic[T]):
    """Operator support for numerical sequences."""

    def __contains__(self, x, /) -> bool:
        return binary(_operators.contains, self, x)

    def __len__(self) -> int:
        return unary(_operators.len, self)

    def __iter__(self):
        return unary(_operators.iter, self)

    def __getitem__(self, i: typing.SupportsIndex, /) -> T:
        return binary(_operators.getitem, self, i)

    def __array__(self, *args, **kwargs) -> numpy.typing.NDArray:
        return unary(numpy.array, self, *args, **kwargs)



