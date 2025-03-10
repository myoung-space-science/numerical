import typing


T = typing.TypeVar('T')


class Quantity(typing.Generic[T]):
    """The base for all numerical types."""

    def __init__(self, data: T):
        self._data = data

    def __repr__(self) -> str:
        """Called for repr(self)."""
        return f"{self.__class__.__qualname__}({self})"

    def __str__(self):
        """Called for str(self)."""
        return str(self._data)

