"""Test suite for simulation of PGT-A data."""
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from karyohmm import PGTSim, PGTSimMosaic, PGTSimSegmental

pgt_sim = PGTSim()
pgt_sim_mosaic = PGTSimMosaic()
pgt_sim_segmental = PGTSimSegmental()


@given(
    length=st.floats(min_value=1e2, max_value=1e8),
    m=st.integers(min_value=1000, max_value=5000),
    ploidy=st.integers(min_value=0, max_value=3),
)
@settings(max_examples=100, deadline=1000)
def test_pgt_sim(length, m, ploidy):
    """Test for PGT simulations."""
    data = pgt_sim.full_ploidy_sim(m=m, ploidy=ploidy, length=length)
    assert data["m"] == m
    assert data["length"] == length
    assert np.max(data["pos"]) <= length
    assert "baf_embryo" in data.keys()
    assert np.any(data["baf_embryo"] == 0.0) | np.any(data["baf_embryo"] == 1.0)


@given(
    length=st.floats(min_value=1e2, max_value=1e8),
    m=st.integers(min_value=2, max_value=1000),
    nsib=st.integers(min_value=2, max_value=5),
)
@settings(max_examples=20, deadline=5000)
def test_pgt_siblings(length, m, nsib):
    """Test for PGT simulations."""
    data = pgt_sim.sibling_euploid_sim(nsibs=nsib, m=m, length=length)
    assert data["m"] == m
    assert data["length"] == length
    assert data["nsibs"] == nsib
    assert np.max(data["pos"]) <= length
    for i in range(nsib):
        assert f"baf_embryo{i}" in data.keys()
        assert f"geno_embryo{i}" in data.keys()


@given(
    m=st.integers(min_value=1000, max_value=5000),
    ploidy=st.integers(min_value=0, max_value=3),
    frac_chrom=st.floats(min_value=0.1, max_value=0.5),
)
@settings(max_examples=10, deadline=5000)
def test_pgt_segmental(m, ploidy, frac_chrom):
    """Test for PGT segmental aneuploidy simulation."""
    mean_size = np.round(m * frac_chrom)
    data = pgt_sim_segmental.full_segmental_sim(m=m, ploidy=ploidy, mean_size=mean_size)
    assert data["m"] == m
    assert "baf_embryo" in data.keys()
    assert np.any(data["baf_embryo"] == 0.0) | np.any(data["baf_embryo"] == 1.0)
    if ploidy != 2:
        assert not np.all(data["ploidies"] == 2)
    else:
        assert np.all(data["ploidies"] == 2)
