import typing

from ._types import Quantity


def unary(f: typing.Callable, a, *args, **kwargs):
    """Implement the unary operation f(a)."""
    x = a._data if isinstance(a, Quantity) else a
    return f(x, *args, **kwargs)


def binary(f: typing.Callable, a, b, *args, **kwargs):
    """Implement the binary operation f(a, b)."""
    x = a._data if isinstance(a, Quantity) else a
    y = b._data if isinstance(b, Quantity) else b
    return f(x, y, *args, **kwargs)

