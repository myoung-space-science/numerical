from . import typeface


T = typeface.TypeVar('T')


@typeface.runtime_checkable
class Quantity(typeface.Protocol[T]):
    """Protocol for numerical types.

    A numerical quantity is the simplest numerical type. It defines a single
    `_data` attribute. You can use this class in instance checks to determine
    whether or not to expect an object to behave consistently with formal
    numeric objects (i.e., instances of `~Object`).
    """

    _data: T


class Object(typeface.Generic[T]):
    """The base for all concrete numerical types."""

    def __init__(self, data: T):
        self._data = data

    def __repr__(self) -> str:
        """Called for repr(self)."""
        return f"{self.__class__.__qualname__}({self})"

    def __str__(self):
        """Called for str(self)."""
        return str(self._data)

