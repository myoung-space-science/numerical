"""
Functions that check or operate on numerical data.
"""

import typing
import numbers
import numpy
import numpy.typing

from ._exceptions import DataTypeError
from ._types import Quantity


T = typing.TypeVar('T')

def isintegral(a):
    """True if `a` is an object of integral type of has integral data.

    This function exists to provide a single instance check against all integral
    types relevant to this package.
    """
    x = a._data if isinstance(a, Quantity) else a
    return isinstance(x, (numbers.Integral, numpy.integer))


@typing.overload
def hasdtype(
    a: Quantity | numpy.typing.ArrayLike,
    t: numpy.typing.DTypeLike,
    /,
): ...

@typing.overload
def hasdtype(
    a: Quantity | numpy.typing.ArrayLike,
    t: tuple[numpy.typing.DTypeLike],
    /,
): ...

def hasdtype(a, t, /):
    """True if `a` has one of the given data types.
    
    This function wraps `numpy.issubdtype` to allow the caller to pass more than
    one data type at a time, similar to the behavior of the built-in functions
    `isinstance` and `issubclass`.

    Parameters
    ----------
    a : physical object or array-like
        The object to check.
    t : dtype-like or tuple of dtype-like
        One or more objects that can be interpreted as `numpy` data types.

    Notes
    -----
    - If `a` is a physical object, this function will operate on `a._data`.
    - If the array-like operand (either `a` or `a._data`, as appropriate) is not
      a `numpy.ndarray`, this function will first convert it to one.
    """
    x = a._data if isinstance(a, Quantity) else a
    y = x if isinstance(x, numpy.ndarray) else numpy.array(x)
    dtype = y.dtype
    if isinstance(t, tuple):
        return any(numpy.issubdtype(dtype, i) for i in t)
    return numpy.issubdtype(dtype, t)


def ismonotonic(
    a: numpy.typing.ArrayLike | Quantity[numpy.typing.ArrayLike],
    order: typing.Literal['increasing', 'decreasing'] | None = None,
    strict: bool=False,
) -> bool:
    """True if `a` is strictly increasing or strictly decreasing.
    
    Parameters
    ----------
    a : array_like
        An array-like object or a numerical object with array-like data.
    order : {None, 'increasing', 'decreasing'} (optional)
        Whether the numerical data must be monotonicly increasing or decreasing.
        The default behavior is to check for either.
    strict : boolean (optional)
        If true, require that the numerical data be strictly increasing or
        decreasing. The default behavior is only to require that the numerical
        data be non-decreasing or non-increasing, respectively.
    """
    x = a._data if isinstance(a, Quantity) else a
    try:
        d = numpy.diff(x)
    except Exception as err:
        raise DataTypeError(
            f"Cannot compute the monotonicity of data in {a}"
        ) from err
    if strict is False:
        decreasing = numpy.all(d <= 0)
        increasing = numpy.all(d >= 0)
        if order is None:
            return decreasing or increasing
        if order == 'decreasing':
            return decreasing
        if order == 'increasing':
            return increasing
    decreasing = numpy.all(d < 0)
    increasing = numpy.all(d > 0)
    if order is None:
        return decreasing or increasing
    if order == 'decreasing':
        return decreasing
    if order == 'increasing':
        return increasing


def isequal(a, b):
    """True if `a` and `b` have equal numeric data.

    This is a convenience function that allows the caller to test whether two
    objects are numerically equivalent, even if they aren't strictly equal.
    """
    x = a._data if isinstance(a, Quantity) else a
    y = b._data if isinstance(b, Quantity) else b
    return numpy.array_equal(x, y)


def isclose(a: Quantity, b: numbers.Real) -> bool:
    """True if `b` is close to a value in `a`'s numeric data.

    This function is similar to (and, if fact, uses) `numpy.isclose`. The
    primary distinctions are that this function compares a single value to the
    real-valued data of a variable object and returns a single boolean value.
    
    Parameters
    ----------
    a : numerical quantity
        The object that may contain `b`.

    b : real number
        The value for which to search.

    Notes
    -----
    - This function exists to handle cases in which floating-point arithmetic
      has caused a numeric operation to return an imprecise result, especially
      for small numbers (e.g., certain unit conversions). It will first test for
      standard containment via `in` before attempting to determine if `b` is
      close enough, within a very strict tolerance, to any member of `a`.
    """
    data = a._data
    try:
        iter(data)
    except TypeError:
        return a == b or numpy.isclose(b, data, atol=0.0)
    if b in data:
        return True
    if b < numpy.min(data) or b > numpy.max(data):
        return False
    return numpy.any([numpy.isclose(b, data, atol=0.0)])


_NT = typing.TypeVar('_NT', bound=numbers.Complex)


class Nearest(typing.NamedTuple):
    """The result of searching an array for a target value."""

    index: int
    value: _NT


def nearest(
    values: typing.Iterable[_NT],
    target: _NT,
    bound: str=None,
) -> Nearest:
    """Find the value in a collection nearest the target value.
    
    Parameters
    ----------
    values : iterable of numbers
        An iterable collection of numbers to compare to the target value. Must
        support conversion to a `numpy.ndarray`.

    target : number
        A single numerical value for which to search in `values`. Must be
        coercible to the type of `values`.

    bound : {None, 'lower', 'upper'}
        The constraint to apply when finding the nearest value:

        - None: no constraint
        - 'lower': ensure that the nearest value is equal to or greater than the
          target value (in other words, the target value is a lower bound for
          the nearest value)
        - 'upper': ensure that the nearest value is equal to or less than the
          target value (in other words, the target value is an upper bound for
          the nearest value)

    Returns
    -------
    Nearest
        A named tuple with `value` and `index` fields, respectively containing
        the value in `values` closest to `target` (given the constraint set by
        `bound`, if any) and the index of `value` in `values`. If the array
        corresponding to `values` is one-dimensional, `index` will be an
        integer; otherwise, it will be a tuple with one entry for each
        dimension.

    Notes
    -----
    This function is based on the top answer to this StackOverflow question:
    https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    However, a lower-voted answer (and the comments) has some suggestions for a
    bisection-based method.
    """

    array = numpy.asarray(values)
    index = numpy.argmin(numpy.abs(array - target))
    if bound == 'lower':
        try:
            while array[index] < target:
                index += 1
        except IndexError:
            index = -1
    elif bound == 'upper':
        try:
            while array[index] > target:
                index -= 1
        except IndexError:
            index = 0
    if array.ndim > 1:
        index = numpy.unravel_index(index, array.shape)
    return Nearest(index=index, value=array[index])


