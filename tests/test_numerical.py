import numbers
import typing

import numpy

import numerical


ObjectType = typing.TypeVar('ObjectType', bound=numerical.Object)

class EqualityMixin(numerical.Object):
    """Mixin class that implements `__eq__` for testing."""

    def __eq__(self, other):
        """Called for self == other."""
        if isinstance(other, numerical.Object):
            return self._data == other._data
        return self._data == other


class Orderable(
    EqualityMixin,
    numerical.Object,
    numerical.mixins.Orderable,
    numerical.Orderable): ...


class Comparable(
    EqualityMixin,
    numerical.Object,
    numerical.mixins.Comparable,
    numerical.Comparable): ...


class Additive(
    EqualityMixin,
    numerical.Object,
    numerical.mixins.Additive,
    numerical.Additive): ...


class Multiplicative(
    EqualityMixin,
    numerical.Object,
    numerical.mixins.Multiplicative,
    numerical.Multiplicative): ...


class Algebraic(
    EqualityMixin,
    numerical.Object,
    numerical.mixins.Algebraic,
    numerical.Algebraic): ...


class Complex(
    EqualityMixin,
    numerical.Object,
    numerical.mixins.Complex,
    numerical.Complex): ...


class Real(
    EqualityMixin,
    numerical.Object,
    numerical.mixins.Real,
    numerical.Real): ...


class Value(
    EqualityMixin,
    numerical.Object,
    numerical.mixins.Value,
    numerical.Value): ...


class Sequence(
    EqualityMixin,
    numerical.Object,
    numerical.mixins.Sequence,
    numerical.Sequence): ...


class Scalar(Value, numerical.mixins.NumpyMixin): ...


class Array(Sequence, numerical.mixins.NumpyMixin): ...


def test_types():
    """Make sure instances pass instance checks."""
    x = 2
    assert isinstance(Orderable(x), numerical.Orderable)
    assert isinstance(Comparable(x), numerical.Comparable)
    assert isinstance(Additive(x), numerical.Additive)
    assert isinstance(Multiplicative(x), numerical.Multiplicative)
    assert isinstance(Algebraic(x), numerical.Algebraic)
    assert isinstance(Complex(x), numerical.Complex)
    assert isinstance(Real(x), numerical.Real)
    assert isinstance(Sequence(x), numerical.Sequence)
    assert isinstance(Value(x), numerical.Value)


def test_orderable():
    """Test operations on orderable objects."""
    x = 2
    y = 3
    a = Orderable(x)
    b = Orderable(y)
    check_orderable(a, b, x, y)


def test_comparable():
    """Test operations on comparable objects."""
    x = 2
    y = 3
    a = Comparable(x)
    b = Comparable(y)
    check_comparable(a, b, x, y)


def test_complex():
    """Test operations on complex objects."""
    x = 2
    y = 3
    a = Complex(x)
    b = Complex(y)
    check_complex(a, b, x, y)


PROTOCOLS = {
    'orderable': (
        numerical.Orderable,
    ),
    'comparable': (
        numerical.Orderable,
        numerical.Comparable,
    ),
    'additive': (
        numerical.Additive,
    ),
    'multiplicative': (
        numerical.Multiplicative,
    ),
    'algebraic': (
        numerical.Additive,
        numerical.Multiplicative,
        numerical.Algebraic,
    ),
    'complex': (
        numerical.Algebraic,
        numerical.Complex,
    ),
    'real': (
        numerical.Comparable,
        numerical.Complex,
        numerical.Real,
    ),
    'value': (
        numerical.Comparable,
        numerical.Complex,
        numerical.Value,
    ),
    'sequence': (
        numerical.Comparable,
        numerical.Complex,
        numerical.Sequence,
    ),
}


OPERATORS = {
    'additive': (
        numerical.operators.add,
        numerical.operators.sub,
    ),
    'multiplicative': (
        numerical.operators.mul,
        numerical.operators.truediv,
    ),
    'unary': (
        numerical.operators.abs,
        numerical.operators.pos,
        numerical.operators.neg,
    )
}


def test_sequence() -> None:
    """Test base types and operations on numerical sequences."""
    x = numpy.array([2.1, 3.4])
    a = Sequence(x)
    y = numpy.array([21, 34])
    b = Sequence(y)
    check_comparable(a, b, x, y)
    # NOTE: We only check operations on (`Sequence`, `Sequence`) and
    # (`Sequence`, `numpy.ndarray`) because our test class does not implement
    # `__array_ufunc__`, so the operation on (`numpy.ndarray`, `Sequence`) falls
    # through to the `numpy.ndarray` implementation.
    pairs = ((a, b), (a, y))
    for pair in pairs:
        for f in OPERATORS['additive']:
            forward = f(*pair)
            assert isinstance(forward, type(a))
            assert numpy.all(forward == Additive(f(x, y)))
        for f in OPERATORS['multiplicative']:
            forward = f(*pair)
            assert isinstance(forward, type(a))
            assert numpy.all(forward == Multiplicative(f(x, y)))
    f = numerical.operators.pow
    for pair in pairs:
        result = f(*pair)
        assert isinstance(result, type(a))
        assert numpy.all(result == Complex(f(x, y)))
    for f in OPERATORS['unary']:
        assert numpy.all(f(a) == Complex(f(x)))


def test_value() -> None:
    """Test base types and operations on numerical values."""
    x = 2.1
    a = Value(x)
    y = 3
    b = Value(y)
    check_comparable(a, b, x, y)
    check_complex(a, b, x, y)
    types = (
        complex,
        float,
        int,
    )
    for t in types:
        result = t(a)
        assert isinstance(result, t)
        assert result == t(x)
    f = numerical.operators.round
    result = f(a)
    assert isinstance(result, Value)
    assert result == Value(f(x))


def test_real() -> None:
    """Test base types and operations on real objects."""
    x = 2
    a = Real(x)
    keys = set(PROTOCOLS) - {'value', 'sequence'}
    for key in keys:
        assert isinstance(a, PROTOCOLS[key])
    y = 3
    b = Real(y)
    check_real(a, b, x, y)


def test_numpy() -> None:
    """Test support for numpy functions."""
    x = 4.0
    s = Scalar(x)
    assert numpy.all(numpy.sqrt(s) == Scalar(numpy.sqrt(x)))
    y = numpy.array([4.0, 9.0])
    a = Array(y)
    assert numpy.all(numpy.sqrt(a) == Array(numpy.sqrt(y)))


def check_real(
    a: ObjectType,
    b: ObjectType,
    x: numbers.Number,
    y: numbers.Number,
) -> None:
    """Check operations on real objects."""
    check_comparable(a, b, x, y)
    check_complex(a, b, x, y)
    pairs = ((a, b), (a, y), (x, b))
    operators = (
        numerical.operators.floordiv,
        numerical.operators.mod,
        # NOTE: The function `check_complex` already tests forward `pow`, but
        # it's simpler to just check forward and reverse here, despite the
        # redundancy.
        numerical.operators.pow,
    )
    for pair in pairs:
        for f in operators:
            forward = f(*pair)
            reverse = f(*pair[::-1])
            assert isinstance(forward, type(a))
            assert isinstance(reverse, type(a))
            assert forward == Real(f(x, y))
            assert reverse == Real(f(y, x))


def check_complex(
    a: ObjectType,
    b: ObjectType,
    x: numbers.Number,
    y: numbers.Number,
) -> None:
    """Check operations on complex objects."""
    check_multiplicative(a, b, x, y)
    check_additive(a, b, x, y)
    f = numerical.operators.pow
    pairs = ((a, b), (a, y))
    for pair in pairs:
        result = f(*pair)
        assert isinstance(result, type(a))
        assert result == Complex(f(x, y))
    for f in OPERATORS['unary']:
        assert f(a) == Complex(f(x))


def check_multiplicative(
    a: ObjectType,
    b: ObjectType,
    x: numbers.Number,
    y: numbers.Number,
) -> None:
    """Check operations on multiplicative objects."""
    pairs = ((a, b), (a, y), (x, b))
    for pair in pairs:
        for f in OPERATORS['multiplicative']:
            forward = f(*pair)
            reverse = f(*pair[::-1])
            assert isinstance(forward, type(a))
            assert isinstance(reverse, type(a))
            assert numpy.all(forward == Multiplicative(f(x, y)))
            assert numpy.all(reverse == Multiplicative(f(y, x)))


def check_additive(
    a: Additive,
    b: Additive,
    x: numbers.Number,
    y: numbers.Number,
) -> None:
    """Check operations on additive objects."""
    pairs = ((a, b), (a, y), (x, b))
    for pair in pairs:
        for f in OPERATORS['additive']:
            forward = f(*pair)
            reverse = f(*pair[::-1])
            assert isinstance(forward, type(a))
            assert isinstance(reverse, type(a))
            assert numpy.all(forward == Additive(f(x, y)))
            assert numpy.all(reverse == Additive(f(y, x)))


def check_comparable(
    a: numerical.Comparable,
    b: numerical.Comparable,
    x: numbers.Number,
    y: numbers.Number,
) -> None:
    """Check operations on comparable objects."""
    c = Comparable(x)
    assert a is not c
    assert numpy.all(a == c)
    assert numpy.all(b != c)
    assert numpy.all(a == x)
    assert numpy.all(b == y)
    assert numpy.all(a != y)
    assert numpy.all(b != x)
    check_orderable(a, b, x, y)


def check_orderable(
    a: numerical.Orderable,
    b: numerical.Orderable,
    x: numbers.Number,
    y: numbers.Number,
) -> None:
    """Check operations on orderable objects."""
    assert numpy.all(a < y)
    assert numpy.all(a <= y)
    assert numpy.all(b > x)
    assert numpy.all(b >= x)
    assert numpy.all(a < b)
    assert numpy.all(a <= b)
    assert numpy.all(b > a)
    assert numpy.all(b >= a)

