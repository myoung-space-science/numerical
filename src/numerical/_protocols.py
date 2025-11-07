import abc
import numbers

import numpy
import numpy.typing

from . import typeface


T = typeface.TypeVar('T')


@typeface.runtime_checkable
class Orderable(typeface.Protocol):
    """Protocol for objects that support relative ordering.

    Classes that implement this protocol must define the following methods

    - `__lt__`
    - `__le__`
    - `__gt__`
    - `__ge__`

    The most appropriate return type will often be `bool` for each method, but
    exceptions exist (cf. `numpy.ndarray`).
    """

    __module__ = __package__

    __slots__ = ()

    @abc.abstractmethod
    def __lt__(self, other): ...

    @abc.abstractmethod
    def __le__(self, other): ...

    @abc.abstractmethod
    def __gt__(self, other): ...

    @abc.abstractmethod
    def __ge__(self, other): ...


@typeface.runtime_checkable
class Comparable(Orderable, typeface.Protocol):
    """Protocol for orderable objects that also support equality operations.

    Classes that implement this protocol must implement the `~Orderable`
    protocol, and must define the following methods

    - `__eq__`
    - `__ne__`

    The most appropriate return type will often be `bool` for each method, but
    exceptions exist (cf. `numpy.ndarray`).
    """

    __module__ = __package__

    __slots__ = ()

    @abc.abstractmethod
    def __eq__(self, other): ...

    @abc.abstractmethod
    def __ne__(self, other): ...


@typeface.runtime_checkable
class Additive(typeface.Protocol):
    """Protocol for additive objects.

    Classes that implement this protocol must define the following methods

    - `__add__`
    - `__radd__`
    - `__sub__`
    - `__rsub__`

    Each method may return whatever type is appropriate for the class.
    """

    __module__ = __package__

    __slots__ = ()

    @abc.abstractmethod
    def __add__(self, other) -> typeface.Self: ...

    @abc.abstractmethod
    def __radd__(self, other) -> typeface.Self: ...

    @abc.abstractmethod
    def __sub__(self, other) -> typeface.Self: ...

    @abc.abstractmethod
    def __rsub__(self, other) -> typeface.Self: ...


@typeface.runtime_checkable
class Multiplicative(typeface.Protocol):
    """Protocol for multiplicative objects.

    Classes that implement this protocol must define the following methods

    - `__mul__`
    - `__rmul__`
    - `__truediv__`
    - `__rtruediv__`

    Each method may return whatever type is appropriate for the class.

    Notes
    -----
    - The floor- and modular-division operators do not appear in this protocol,
      because their distinction from "true" division really only makes sense in
      the context of real-valued numerical objects, whereas a more general set
      of objects may implement this protocol (e.g., symbolic expressions).
    """

    __module__ = __package__

    __slots__ = ()

    @abc.abstractmethod
    def __mul__(self, other) -> typeface.Self: ...

    @abc.abstractmethod
    def __rmul__(self, other) -> typeface.Self: ...

    @abc.abstractmethod
    def __truediv__(self, other) -> typeface.Self: ...

    @abc.abstractmethod
    def __rtruediv__(self, other) -> typeface.Self: ...


@typeface.runtime_checkable
class Algebraic(Additive, Multiplicative, typeface.Protocol):
    """Protocol for algebraic objects.

    Classes that implement this protocol must implement the `~Additive` and
    `~Multiplicative` protocols, and must define the `__pow__` method, which may
    return whatever type is appropriate for the class.

    Notes
    -----
    - The formal definition of an algebraic quantity requires that
      exponentiation involve only constant, rational exponents. However, this
      protocol does not place any restrictions on the type of exponent.
    - This protocol differs from the `~Additive` and `~Multiplicative`
      protocols, which require forward and reverse versions of their operators,
      by not requiring `__rpow__`. Doing so would imply that any algebraic
      object should be allowed as an exponent, which is not generally true. Of
      course, this doesn't prevent a concrete object from defining `__rpow__`.
    """

    __module__ = __package__

    __slots__ = ()

    @abc.abstractmethod
    def __pow__(self, other) -> typeface.Self: ...


@typeface.runtime_checkable
class Complex(Algebraic, typeface.Protocol):
    """Protocol for complex-valued objects.

    Classes that implement this protocol must implement the `~Algebraic`
    protocol, and must define the following methods

    - `__abs__`
    - `__pos__`
    - `__neg__`

    Each method may return whatever type is appropriate for the class.
    """

    __module__ = __package__

    __slots__ = ()

    @abc.abstractmethod
    def __abs__(self) -> typeface.Self: ...

    @abc.abstractmethod
    def __pos__(self) -> typeface.Self: ...

    @abc.abstractmethod
    def __neg__(self) -> typeface.Self: ...


@typeface.runtime_checkable
class Real(Comparable, Complex, typeface.Protocol):
    """Protocol for real-valued numerical objects.

    Classes that implement this protocol must implement the `~Comparable` and
    `~Complex` protocols, and must define the following methods

    - `__rpow__`
    - `__floordiv__`
    - `__rfloordiv__`
    - `__mod__`
    - `__rmod__`

    Each method may return whatever type is appropriate for the class.
    """

    __module__ = __package__

    __slots__ = ()

    @abc.abstractmethod
    def __rpow__(self, other) -> typeface.Self: ...

    @abc.abstractmethod
    def __floordiv__(self, other) -> typeface.Self: ...

    @abc.abstractmethod
    def __rfloordiv__(self, other) -> typeface.Self: ...

    @abc.abstractmethod
    def __mod__(self, other) -> typeface.Self: ...

    @abc.abstractmethod
    def __rmod__(self, other) -> typeface.Self: ...


@typeface.runtime_checkable
class Value(Comparable, Complex, typeface.Protocol):
    """Protocol for singular numerical objects.

    Classes that implement this protocol must implement the `~Comparable` and
    `~Complex` protocols, as well as the following methods

    - `__complex__`, which must return an instance of `complex`
    - `__float__`, which must return an instance of `float`
    - `__int__`, which must return an instance of `int`
    - `__round__`, which may return whatever type is appropriate to the class
    """

    __module__ = __package__

    __slots__ = ()

    @abc.abstractmethod
    def __complex__(self) -> complex: ...

    @abc.abstractmethod
    def __float__(self) -> float: ...

    @abc.abstractmethod
    def __int__(self) -> int: ...

    @abc.abstractmethod
    def __round__(self) -> typeface.Self: ...


DT = typeface.TypeVar('DT', bound=numbers.Real)

AT = typeface.TypeVar('AT', numpy.integer, numpy.floating)

@typeface.runtime_checkable
class Sequence(Comparable, Complex, typeface.Protocol[DT]):
    """Protocol for numerical sequences.

    Classes that implement this protocol must implement the `~Comparable` and
    `~Complex` protocols, as well as the following methods

    - `__contains__`, which should return a `bool`
    - `__len__`, which should return an `int`
    - `__iter__`, which should return an iterator over the value type
    - `__getitem__`, which may return an appropriate value type
    - `__array__`, which should return a `numpy.ndarray`

    Notes
    -----
    This protocol is more restrictive than `collections.abc.Sequence`, which
    only requires `__getitem__` and `__len__` in order to define additional
    mixin methods that include `__contains__` and `__iter__`.
    """

    __module__ = __package__

    __slots__ = ()

    @abc.abstractmethod
    def __contains__(self, v, /) -> bool: ...

    @abc.abstractmethod
    def __len__(self) -> int: ...

    @abc.abstractmethod
    def __iter__(self): ...

    @abc.abstractmethod
    def __getitem__(self, i, /) -> DT | typeface.Sequence[DT]: ...

    @abc.abstractmethod
    def __array__(self, *args, **kwargs) -> numpy.typing.NDArray[AT]: ...

