"""Test suite for karyoHMM."""
import numpy as np
import pytest
from utils import full_ploidy_sim

from karyohmm import MetaHMM

# --- Generating test data for applications --- #
data_disomy = full_ploidy_sim(m=2000, seed=42)
data_trisomy = full_ploidy_sim(m=2000, ploidy=3, mat_skew=0, seed=42)
data_monosomy = full_ploidy_sim(m=2000, ploidy=1, mat_skew=0, seed=42)
data_nullisomy = full_ploidy_sim(m=2000, ploidy=0, mat_skew=0, seed=42)


def test_data_integrity(data=data_disomy):
    """Test for some basic data sanity checks ..."""
    for x in ["baf_embryo", "lrr_embryo", "mat_haps", "pat_haps"]:
        assert x in data
    baf = data["baf_embryo"]
    mat_haps = data["mat_haps"]
    pat_haps = data["pat_haps"]
    assert np.all((baf <= 1.0) & (baf >= 0.0))
    assert baf.size == mat_haps.shape[1]
    assert mat_haps.shape == pat_haps.shape
    assert np.all((mat_haps == 0) | (mat_haps == 1))
    assert np.all((pat_haps == 0) | (pat_haps == 1))


# --- Testing the metadata implementations --- #
@pytest.mark.parametrize("data,logr", [(data_disomy, False), (data_trisomy, False)])
def test_forward_algorithm(data, logr):
    """Test the implementation of the forward algorithm."""
    hmm = MetaHMM(logr=logr)
    _, _, _, karyotypes, loglik = hmm.forward_algorithm(
        bafs=data["baf_embryo"],
        lrrs=data["lrr_embryo"],
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
        logr=logr,
    )
    assert loglik < 0
    if logr:
        assert karyotypes.size == 23
    else:
        assert karyotypes.size == 21


@pytest.mark.parametrize("data,logr", [(data_disomy, False), (data_trisomy, False)])
def test_backward_algorithm(data, logr):
    """Test the implementation of the backward algorithm."""
    hmm = MetaHMM(logr=logr)
    _, _, _, karyotypes, loglik = hmm.backward_algorithm(
        bafs=data["baf_embryo"],
        lrrs=data["lrr_embryo"],
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
        logr=logr,
    )
    assert loglik < 0
    if logr:
        assert karyotypes.size == 23
    else:
        assert karyotypes.size == 21


@pytest.mark.parametrize("data,logr", [(data_disomy, False), (data_trisomy, False)])
def test_likelihood_similarity(data, logr):
    """Test that the forward & backward algorithms give similar loglik."""
    hmm = MetaHMM(logr=logr)
    _, _, _, _, a_loglik = hmm.forward_algorithm(
        bafs=data["baf_embryo"],
        lrrs=data["lrr_embryo"],
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
        logr=logr,
    )
    _, _, _, _, b_loglik = hmm.backward_algorithm(
        bafs=data["baf_embryo"],
        lrrs=data["lrr_embryo"],
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
        logr=logr,
    )
    assert np.isclose(a_loglik, b_loglik, atol=1e-1)


def test_fwd_bwd_algorithm():
    """Test the properties of the output from the fwd-bwd algorithm."""
    pass
