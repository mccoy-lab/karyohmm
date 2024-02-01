"""Test suite for simulation of PGT-A data."""
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from karyohmm import PGTSim, PGTSimMosaic

pgt_sim = PGTSim()
pgt_sim_mosaic = PGTSimMosaic()


@given(
    length=st.floats(min_value=1e2, max_value=1e8),
    m=st.integers(min_value=2, max_value=4000),
)
@settings(max_examples=10, deadline=1000)
def test_pgt_sim(length, m):
    """Test for PGT simulations."""
    data = pgt_sim.full_ploidy_sim(m=m, length=length)
    assert data["m"] == m
    assert data["length"] == length
    assert np.max(data["pos"]) <= length


@given(
    length=st.floats(min_value=1e2, max_value=1e8),
    m=st.integers(min_value=2, max_value=4000),
    nsib=st.integers(min_value=2, max_value=5),
)
@settings(max_examples=10, deadline=5000)
def test_pgt_siblings(length, m, nsib):
    """Test for PGT simulations."""
    data = pgt_sim.sibling_euploid_sim(nsibs=nsib, m=m, length=length)
    assert data["m"] == m
    assert data["length"] == length
    assert data["nsibs"] == nsib
    assert np.max(data["pos"]) <= length
    for i in range(nsib):
        assert f"baf_embryo{i}" in data.keys()
