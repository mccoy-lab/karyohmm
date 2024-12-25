"""Test suite for simulation of PGT-A data."""
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from karyohmm import PGTSim, PGTSimMosaic, PGTSimSegmental, PGTSimVCF

pgt_sim = PGTSim()
pgt_sim_mosaic = PGTSimMosaic()
pgt_sim_segmental = PGTSimSegmental()


@given(
    length=st.floats(min_value=1e2, max_value=1e8),
    m=st.integers(min_value=1000, max_value=5000),
    ploidy=st.integers(min_value=0, max_value=3),
)
@settings(max_examples=100, deadline=5000)
def test_pgt_sim(length, m, ploidy):
    """Test for PGT simulations."""
    data = pgt_sim.full_ploidy_sim(m=m, ploidy=ploidy, length=length)
    assert data["m"] == m
    assert data["length"] == length
    assert np.max(data["pos"]) <= length
    assert "baf" in data.keys()
    if ploidy > 0:
        assert np.any(data["baf"] == 0.0) | np.any(data["baf"] == 1.0)


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
    assert "baf" in data.keys()
    assert np.any(data["baf"] == 0.0) | np.any(data["baf"] == 1.0)
    if ploidy != 2:
        assert not np.all(data["ploidies"] == 2)
    else:
        assert np.all(data["ploidies"] == 2)


@pytest.fixture
def valid_vcf_file():
    """Small VCF file shipped with the project."""
    return "data/chr1.subset.1kg_phase3.phased.n50.vcf.gz"


def test_pgt_vcf(valid_vcf_file):
    """Test reading in a VCF and outputting key parameters."""
    pgt_vcf = PGTSimVCF()
    mat_haps, pat_haps, pos, afs = pgt_vcf.gen_parental_haplotypes(
        vcf_fp=valid_vcf_file,
        maternal_id="HG00096",
        paternal_id="HG00107",
        gts012=True,
        threads=4,
    )
    assert mat_haps.ndim == 2
    assert pat_haps.ndim == 2
    assert mat_haps.size == pat_haps.size
    assert mat_haps.shape[0] == 2
    assert pat_haps.shape[0] == 2
    assert mat_haps.shape[1] == pos.size
    assert pat_haps.shape[1] == pos.size
    assert afs.size == pos.size
    assert np.all(np.isin(mat_haps, [0, 1]))
    assert np.all(np.isin(pat_haps, [0, 1]))


def test_pgt_sim_from_vcf(valid_vcf_file):
    """Test actually simulating from pre-existing haplotypes."""
    pgt_vcf = PGTSimVCF()
    pgt_sim = PGTSim()
    mat_haps, pat_haps, pos, afs = pgt_vcf.gen_parental_haplotypes(
        vcf_fp=valid_vcf_file,
        maternal_id="HG00096",
        paternal_id="HG00107",
        gts012=True,
        threads=4,
    )
    results = pgt_sim.sim_from_haps(mat_haps=mat_haps, pat_haps=pat_haps, pos=pos)
    assert "baf" in results


@given(
    length=st.floats(min_value=1e2, max_value=1e8),
    m=st.integers(min_value=100, max_value=3000),
    k=st.integers(min_value=2, max_value=20),
)
@settings(max_examples=10, deadline=5000)
def test_pgt_sim_ref_panel(length, m, k):
    """Test for PGT simulations."""
    data = pgt_sim.full_ploidy_sim(m=m, ploidy=2, length=length)
    assert data["m"] == m
    assert data["length"] == length
    assert np.max(data["pos"]) <= length
    assert "baf" in data.keys()
    ref_panel = pgt_sim.sim_haplotype_ref_panel(
        haps=data["mat_haps"], pos=data["pos"], panel_size=k
    )
    assert ref_panel.ndim == 2
    assert ref_panel.shape[0] == k
    assert ref_panel.shape[1] == m
