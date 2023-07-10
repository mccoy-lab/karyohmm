"""Testing module for utility functions."""


import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from karyohmm_utils import emission_baf, mat_dosage, pat_dosage
from scipy.integrate import trapezoid


@pytest.mark.parametrize(
    "hap,state,expected",
    [
        ([0, 1], (0, -1, 1, -1), 0),
        ([0, 1], (1, -1, 1, -1), 1),
        ([1, 1], (0, 1, 1, -1), 2),
        ([1, 1], (-1, -1, -1, -1), -1),
    ],
)
def test_mat_dosage(hap, state, expected):
    """Test the maternal dosage function."""
    assert mat_dosage(hap, state) == expected


@pytest.mark.parametrize(
    "hap,state,expected",
    [
        ([0, 1], (0, -1, 1, -1), 1),
        ([0, 1], (1, -1, 0, -1), 0),
        ([0, 1], (-1, -1, 0, -1), 0),
        ([0, 1], (1, -1, 0, 1), 1.0),
        ([1, 1], (1, -1, 0, 1), 2.0),
        ([1, 0], (1, -1, 0, 0), 2.0),
        ([1, 1], (-1, -1, -1, -1), -1),
    ],
)
def test_pat_dosage(hap, state, expected):
    """Test the paternal dosage function."""
    assert pat_dosage(hap, state) == expected


@given(
    pi0=st.floats(
        min_value=0, max_value=1, exclude_min=True, exclude_max=True, allow_nan=False
    ),
    sigma=st.floats(
        min_value=1e-2,
        max_value=0.5,
        exclude_min=True,
        exclude_max=True,
        allow_nan=False,
    ),
    m=st.integers(min_value=0, max_value=1),
    p=st.integers(min_value=0, max_value=1),
    k=st.integers(min_value=2, max_value=2),
)
def test_emission_disomy(pi0, sigma, m, p, k):
    """Test that the emission under disomy integrates to 1."""
    bs = np.linspace(0, 1, 1000)
    e = [np.exp(emission_baf(b, m=m, p=p, k=k, pi0=pi0, std_dev=sigma)) for b in bs]
    integrand = trapezoid(e, bs)
    assert np.isclose(integrand, 1, atol=1e-03)
