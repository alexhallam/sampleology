import pytest
from hypothesis import given, strategies as st
from add import add, add_positive


@given(st.integers(), st.integers())
def test_add_commutative(a, b):
    """Test that addition is commutative: a + b = b + a"""
    assert add(a, b) == add(b, a)


@given(st.integers(), st.integers(), st.integers())
def test_add_associative(a, b, c):
    """Test that addition is associative: (a + b) + c = a + (b + c)"""
    assert add(add(a, b), c) == add(a, add(b, c))


@given(st.integers())
def test_add_zero(a):
    """Test that adding zero returns the original number: a + 0 = a"""
    assert add(a, 0) == a
    assert add(0, a) == a


@given(st.integers(min_value=1), st.integers(min_value=1))
def test_add_positive_valid_inputs(a, b):
    """Test that add_positive works with valid positive inputs"""
    result = add_positive(a, b)
    assert result == a + b
    assert result > 0


@given(st.integers(max_value=0), st.integers(min_value=1))
def test_add_positive_first_negative(a, b):
    """Test that add_positive raises ValueError when first argument is not positive"""
    with pytest.raises(ValueError, match="Both numbers must be positive"):
        add_positive(a, b)
