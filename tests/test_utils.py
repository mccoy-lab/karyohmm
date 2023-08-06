"""Testing module for utility functions."""


import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from karyohmm_utils import emission_baf, mat_dosage, pat_dosage
from scipy.integrate import trapezoid
from utils import sim_joint_het


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
    nsibs=st.integers(min_value=1, max_value=10),
    switch=st.booleans(),
)
def test_sim_joint_het(switch, pi0, sigma, nsibs):
    """Test the simulation of the joint heterozygotes."""
    true_haps1, true_haps2, haps1, haps2, bafs = sim_joint_het(
        switch=switch, mix_prop=pi0, std_dev=sigma, nsibs=nsibs
    )
    assert true_haps1.ndim == 2
    assert true_haps2.ndim == 2
    assert len(bafs) == nsibs
    if not switch:
        assert np.all(true_haps1 == haps1)
    else:
        assert ~np.all(true_haps1 == haps1)
