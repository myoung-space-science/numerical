"""
A namespace for standard numerical operators.
"""

import builtins
import operator
import typing


T = typing.TypeVar('T')

class Operator:
    """Base class for enhanced operators."""

    _DEFINED: dict[str, typing.Self] = {}

    def __new__(cls, *args):
        """Create or return a singleton instance."""
        if args in cls._DEFINED:
            return cls._DEFINED[args]
        instance = super().__new__(cls)
        cls._DEFINED[args] = instance
        return instance

    def __init__(self, __f: typing.Callable[..., T], operation: str):
        """Initialize a new instance."""
        self._f = __f
        self._operation = operation

    def __repr__(self):
        """Called for repr(self)."""
        return self._operation

    def __call__(self, *args, **kwds):
        """Called for self(*args, **kwds)."""
        return self._f(*args, **kwds)


abs = Operator(builtins.abs, r'abs(a)')
pos = Operator(operator.pos, r'+a')
neg = Operator(operator.neg, r'-a')
round = Operator(builtins.round, r'round(a)')
eq = Operator(operator.eq, r'a == b')
ne = Operator(operator.ne, r'a != b')
lt = Operator(operator.lt, r'a < b')
le = Operator(operator.le, r'a <= b')
gt = Operator(operator.gt, r'a > b')
ge = Operator(operator.ge, r'a >= b')
add = Operator(operator.add, r'a + b')
sub = Operator(operator.sub, r'a - b')
mul = Operator(operator.mul, r'a * b')
truediv = Operator(operator.truediv, r'a / b')
floordiv = Operator(operator.floordiv, r'a // b')
mod = Operator(operator.mod, r'a % b')
pow = Operator(builtins.pow, r'a ** b')
contains = Operator(operator.contains, r'x in a')
len = Operator(builtins.len, r'len(a)')
iter = Operator(builtins.iter, r'iter(a)')
getitem = Operator(operator.getitem, r'a[i]')

