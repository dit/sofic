def test_is_equal_process_accepts_sympy_probabilities():
    """Regression: object-dtype symbol matrices broke ``matrix_rank``."""
    import pytest

    sp = pytest.importorskip("sympy")

    from sofic.examples import golden_mean

    assert golden_mean(sp.Rational(1, 2)).is_equal_process(golden_mean(0.5))
    assert not golden_mean(sp.Rational(1, 3)).is_equal_process(golden_mean(0.5))
