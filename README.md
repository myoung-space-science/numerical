# A numerical tower for non-scalar objects

This package is my attempt to create a type hierarchy similar to that of [numbers.py](https://peps.python.org/pep-3141/), without the assumption that the underlying numerical data is scalar.

The following protocol classes are at the core of this package:
* `Orderable`
  * requires `__lt__`, `__le__`, `__gt__`, `__ge__`
* `Comparable(Orderable)`
  * requires `__eq__` and `__ne__`
* `Additive`
  * requires `__add__`, `__radd__`, `__sub__`, and `__rsub__`
* `Multiplicative`
  * requires `__mul__`, `__rmul__`, `__truediv__`, and `__rtruediv__`
* `Algebraic(Additive, Multiplicative)`
  * requires `__pow__`
* `Complex(Algebraic)`
  * requires `__abs__`, `__pos__`, `__neg__`
* `Real(Comparable, Complex)`
  * requires `__rpow__`, `__floordiv__`, `__rfloordiv__`, `__mod__`, and `__rmod__`
* `Value(Comparable, Complex)`
  * requires `__complex__`, `__float__`, `__int__`, and `__round__`
* `Sequence(Comparable, Complex)`
  * requires `__contains__`, `__iter__`, `__len__`, `__getitem__`, and `__array__`

In plain words, an **orderable** object is one that can be put in order compared to something else. A **comparable** object is on that is orderable and can be directly compared to something else for equality. An **additive** object is one that can be added and subtracted from another. A **multiplicative** object is one that can be multiplied or divided by another. An **algebraic** object is one that is additive and multiplicative, and can be the base in an exponential operation. A **complex** object is an algebraic object that has absolute, positive, and negative representations. A **real** object is a comparable and complex object that can be the exponent in an exponential operation, and can participate in some division operations beyond those of a multiplicative object. A **value** is a comparable and complex object with singular numeric data &mdash; this aligns most closely with a complex number. Finally, a **sequence** is a comparable and complex object that behaves like a `collections.abc.Sequence` and can be converted to a `numpy.ndarray`.

This package also provides a mixin class for each protocol in `numerical.mixins` that users can combine with the abstract base class `numerical.Quantity` to build concrete implementations of the above protocols.

