"""Testing module for utility functions."""


import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from karyohmm_utils import emission_baf, mat_dosage, pat_dosage
from scipy.integrate import trapezoid
from scipy.stats import truncnorm
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
    m=st.integers(min_value=0, max_value=1),
)
def test_emission_monosomy(pi0, sigma, m):
    """Test that the emission under disomy integrates to 1."""
    bs = np.linspace(0, 1, 1000)
    e = [np.exp(emission_baf(b, m=m, p=0, k=1, pi0=pi0, std_dev=sigma)) for b in bs]
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
    m=st.integers(min_value=0, max_value=2),
    p=st.integers(min_value=0, max_value=1),
)
def test_emission_trisomy(pi0, sigma, m, p):
    """Test that the emission under disomy integrates to 1."""
    bs = np.linspace(0, 1, 1000)
    e = [np.exp(emission_baf(b, m=m, p=p, k=3, pi0=pi0, std_dev=sigma)) for b in bs]
    integrand = trapezoid(e, bs)
    assert np.isclose(integrand, 1, atol=1e-03)


@given(
    baf=st.floats(min_value=0, max_value=0, allow_nan=False),
    m=st.integers(min_value=0, max_value=1),
    p=st.integers(min_value=0, max_value=1),
    sigma=st.floats(
        min_value=1e-2,
        max_value=0.5,
        exclude_min=True,
        exclude_max=True,
        allow_nan=False,
    ),
    k=st.integers(min_value=2, max_value=3),
)
def test_emission_overall(baf, m, p, k, sigma):
    """Test the overall admissions distribution."""
    mu = (m + p) / k
    a, b = 0, 1
    true_logpdf = truncnorm.logpdf(
        baf, (a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma
    )
    baf_log_pdf = emission_baf(baf, m=m, p=p, k=k, pi0=1e-12, std_dev=sigma)
    assert np.isclose(baf_log_pdf, true_logpdf)


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
    true_haps1, true_haps2, haps1, haps2, bafs, genos = sim_joint_het(
        switch=switch, mix_prop=pi0, std_dev=sigma, nsibs=nsibs
    )
    assert true_haps1.ndim == 2
    assert true_haps2.ndim == 2
    assert len(bafs) == nsibs
    # Make sure that the first ones are hets and second are homozyg
    assert np.all(np.sum(true_haps1, axis=0) == 1)
    assert np.all(np.sum(haps1, axis=0) == 1)
    assert np.all(np.sum(true_haps2, axis=0) != 1)
    assert np.all(np.sum(haps2, axis=0) != 1)
    if not switch:
        assert np.all(true_haps1 == haps1)
    else:
        assert ~np.all(true_haps1 == haps1)
