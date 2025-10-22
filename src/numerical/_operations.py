import functools

from ._types import Quantity
from . import typeface


def unary(f: typeface.Callable, a, *args, **kwargs):
    """Implement the unary operation f(a)."""
    x = a._data if isinstance(a, Quantity) else a
    return f(x, *args, **kwargs)


def binary(f: typeface.Callable, a, b, *args, **kwargs):
    """Implement the binary operation f(a, b)."""
    x = a._data if isinstance(a, Quantity) else a
    y = b._data if isinstance(b, Quantity) else b
    return f(x, y, *args, **kwargs)


def mytype(method: typeface.Callable):
    """Convert the result of `method` to an instance of its class."""
    @functools.wraps(method)
    def wrapper(self: Quantity, *args, **kwargs):
        return type(self)(method(self, *args, **kwargs))
    return wrapper

