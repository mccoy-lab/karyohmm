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
def test_fwd_bwd_algorithm(data, logr):
    """Test the properties of the output from the fwd-bwd algorithm."""
    hmm = MetaHMM(logr=logr)
    gammas, _, karyotypes = hmm.forward_backward(
        bafs=data["baf_embryo"],
        lrrs=data["lrr_embryo"],
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
        logr=logr,
    )
    post_dict = hmm.posterior_karyotypes(gammas, karyotypes)
    for x in ["0", "1m", "1p", "2", "3m", "3p"]:
        assert x in post_dict
    assert np.isclose(sum([post_dict[k] for k in post_dict]), 1.0)


@pytest.mark.parametrize("data,logr", [(data_disomy, False), (data_trisomy, False)])
def test_est_pi0_sigma(data, logr):
    """Test the optimization routine on the forward-algorithm likelihood."""
    hmm = MetaHMM(logr=logr)
    pi0_est, sigma_est = hmm.est_sigma_pi0(
        bafs=data["baf_embryo"],
        lrrs=data["lrr_embryo"],
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
        logr=logr,
    )
    assert (pi0_est > 0) and (pi0_est < 1.0)
    assert (sigma_est > 0) and (sigma_est < 1.0)


def test_string_rep(logr=False):
    """Test that the string representation of states makes sense."""
    hmm = MetaHMM(logr=logr)
    for s in hmm.states:
        x = hmm.get_state_str(s)
        m = sum([j >= 0 for j in s])
        if m == 0:
            assert x == "0"
        elif m == 1:
            if x[0] == "m":
                assert s in hmm.m_monosomy_states
            else:
                assert s in hmm.p_monosomy_states
        else:
            assert len(x) == 2 * m


@pytest.mark.parametrize(
    "r,a,logr", [(1e-3, 1e-7, False), (1e-3, 1e-9, False), (1e-3, 1e-10, True)]
)
def test_transition_matrices(r, a, logr):
    """Test that transition matrices obey the rules."""
    hmm = MetaHMM(logr=logr)
    A = hmm.create_transition_matrix(hmm.karyotypes, r=r, a=a)
    for i in range(A.shape[0]):
        assert np.isclose(np.sum(np.exp(A[i, :])), 1.0)
