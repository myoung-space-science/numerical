import numpy
import pytest

import numerical


class Object(numerical.Object):
    """A concrete test object."""

    def __init__(self, data, color=None) -> None:
        super().__init__(data)
        self.color = color

    def __eq__(self, other) -> bool:
        if isinstance(other, Object):
            return (
                numpy.array_equal(self._data, other._data)
                and
                self.color == other.color
            )
        return False

    def __ne__(self, other) -> bool:
        return not (self == other)


def test_isintegral():
    """Test `numerical.data.isintegral`."""
    true = [
        1,
        int(numpy.array(1)),
        numpy.array([1])[0],
        Object(1),
    ]
    for arg in true:
        assert numerical.data.isintegral(arg)
    false = [
        1.0,
        1+0j,
        '1',
        numpy.array(1),
        numpy.array([1]),
    ]
    for arg in false:
        assert not numerical.data.isintegral(arg)


def test_hasdtype():
    """Test `numerical.data.hasdtype`."""
    this = Object([1.5, 3])
    assert numerical.data.hasdtype(this, numpy.floating)


def test_ismonotonic():
    """Test the function that checks for monotonicity."""
    increasing = [1, 2, 3, 4]
    decreasing = [4, 3, 2, 1]
    non_decreasing = [1, 2, 2, 3]
    non_increasing = [3, 2, 2, 1]
    non_monotonic = [1, 3, 2, 4]
    assert numerical.data.ismonotonic(increasing)
    assert numerical.data.ismonotonic(decreasing)
    assert numerical.data.ismonotonic(non_decreasing)
    assert numerical.data.ismonotonic(non_increasing)
    assert not numerical.data.ismonotonic(non_monotonic)
    assert numerical.data.ismonotonic(increasing, order='increasing')
    assert numerical.data.ismonotonic(decreasing, order='decreasing')
    assert numerical.data.ismonotonic(non_decreasing, order='increasing')
    assert numerical.data.ismonotonic(non_increasing, order='decreasing')
    assert not numerical.data.ismonotonic(increasing, order='decreasing')
    assert not numerical.data.ismonotonic(decreasing, order='increasing')
    assert not numerical.data.ismonotonic(non_decreasing, order='decreasing')
    assert not numerical.data.ismonotonic(non_increasing, order='increasing')
    assert numerical.data.ismonotonic(
        increasing,
        order='increasing',
        strict=True,
    )
    assert numerical.data.ismonotonic(
        decreasing,
        order='decreasing',
        strict=True,
    )
    assert not numerical.data.ismonotonic(
        non_decreasing,
        order='increasing',
        strict=True,
    )
    assert not numerical.data.ismonotonic(
        non_increasing,
        order='decreasing',
        strict=True,
    )
    with pytest.raises(numerical.data.DataTypeError):
        numerical.data.ismonotonic('sequence')
    with pytest.raises(numerical.data.DataTypeError):
        numerical.data.ismonotonic(['1234'])
    with pytest.raises(numerical.data.DataTypeError):
        numerical.data.ismonotonic(['1', '2', '3', '4'])


def test_isequal():
    """Test `numerical.data.isequal`."""
    values = [
        1.5,
        numpy.array([1.5]),
        numpy.array([1.5, -1.5]),
    ]
    for v in values:
        x = Object(v, color='red')
        y = Object(v, color='blue')
        assert x != y
        assert numerical.data.isequal(x, v)
        assert numerical.data.isequal(y, v)
        assert numerical.data.isequal(v, x)
        assert numerical.data.isequal(v, y)
        assert numerical.data.isequal(v, v)
        assert numerical.data.isequal(x, y)
        assert numerical.data.isequal(x, y)


def test_isclose():
    """Test `numerical.data.isclose`."""
    this = Object(numpy.array([1.1e-30, 2e-30]))
    tests = [
        {'value': 1.1e-30,     'in': True,  'contains': True},
        {'value': 2.0e-30,     'in': True,  'contains': True},
        {'value': 1.99999e-30, 'in': False, 'contains': True},
    ]
    for test in tests:
        assert (test['value'] in this._data) == test['in']
        assert numerical.data.isclose(this, test['value']) == test['contains']
    value = 1.99999e-30
    scalar = Object(2.0e-30)
    assert value != scalar
    assert numerical.data.isclose(scalar, value)


def test_nearest():
    values = [0.1, 0.2, 0.3]
    basic = {
        0.11: (0, 0.1),
        0.15: (0, 0.1),
        0.20: (1, 0.2),
    }
    for target, (index, value) in basic.items():
        found = numerical.data.nearest(values, target)
        assert found.index == index
        assert found.value == value
    for target in [0.21, 0.25, 0.29]:
        found = numerical.data.nearest(values, target, bound='lower')
        assert found.index == 2
        assert found.value == 0.3
        found = numerical.data.nearest(values, target, bound='upper')
        assert found.index == 1
        assert found.value == 0.2
    values = numpy.arange(3.0 * 4.0 * 5.0).reshape(3, 4, 5)
    found = numerical.data.nearest(values, 32.9)
    assert found.index == (1, 2, 3)
    assert found.value == 33.0


