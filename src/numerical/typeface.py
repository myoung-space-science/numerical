"""
Support for type annotations.

This module provides a single interface to type annotations. For example,
suppose `BestType` is available in the `typing` module starting with Python
version 3.X, and is available in the `typing_extensions` module for earlier
versions. This module will check the version of the active Python interpreter
and will import from either `typing` or `typing_extensions` as appropriate, so
that other modules may simply access `typehelp.BestType`.
"""

from typing import *
from typing_extensions import *

__all__ = ()

def __getattr__(name: str) -> type:
    """Get a built-in type annotation."""
    try:
        attr = globals()[name]
    except KeyError as err:
        raise AttributeError(
            f"Could not find a type annotation for {name!r}"
        ) from err
    return attr

