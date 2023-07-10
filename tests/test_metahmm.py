"""Test suite for karyoHMM MetaHMM."""
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
    for x in ["baf_embryo", "mat_haps", "pat_haps"]:
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
@pytest.mark.parametrize("data", [data_disomy, data_trisomy, data_monosomy])
def test_forward_algorithm(data):
    """Test the implementation of the forward algorithm."""
    hmm = MetaHMM()
    _, _, _, karyotypes, loglik = hmm.forward_algorithm(
        bafs=data["baf_embryo"],
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
    )


@pytest.mark.parametrize("data", [data_disomy, data_trisomy, data_monosomy])
def test_backward_algorithm(data):
    """Test the implementation of the backward algorithm."""
    hmm = MetaHMM()
    _, _, _, karyotypes, loglik = hmm.backward_algorithm(
        bafs=data["baf_embryo"],
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
    )


@pytest.mark.parametrize("data", [data_disomy, data_trisomy, data_monosomy])
def test_fwd_bwd_algorithm(data):
    """Test the properties of the output from the fwd-bwd algorithm."""
    hmm = MetaHMM()
    gammas, _, karyotypes = hmm.forward_backward(
        bafs=data["baf_embryo"],
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
    )
    # all of the columns must have a sum to 1
    assert np.all(np.isclose(np.sum(np.exp(gammas), axis=0), 1.0))
    post_dict = hmm.posterior_karyotypes(gammas, karyotypes)
    for x in ["0", "1m", "1p", "2", "3m", "3p"]:
        assert x in post_dict
    assert np.isclose(sum([post_dict[k] for k in post_dict]), 1.0)


@pytest.mark.parametrize("data", [data_disomy, data_trisomy, data_monosomy])
def test_est_pi0_sigma(data):
    """Test the optimization routine on the forward-algorithm likelihood."""
    hmm = MetaHMM()
    pi0_est, sigma_est = hmm.est_sigma_pi0(
        bafs=data["baf_embryo"],
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
    )
    assert (pi0_est > 0) and (pi0_est < 1.0)
    assert (sigma_est > 0) and (sigma_est < 1.0)


def test_string_rep():
    """Test that the string representation of states makes sense."""
    hmm = MetaHMM()
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


@pytest.mark.parametrize("r,a", [(1e-3, 1e-7), (1e-3, 1e-9), (1e-3, 1e-10)])
def test_transition_matrices(r, a):
    """Test that transition matrices obey the rules."""
    hmm = MetaHMM()
    A = hmm.create_transition_matrix(hmm.karyotypes, r=r, a=a)
    for i in range(A.shape[0]):
        assert np.isclose(np.sum(np.exp(A[i, :])), 1.0)


@pytest.mark.parametrize(
    "data", [data_disomy, data_trisomy, data_monosomy, data_nullisomy]
)
def test_ploidy_correctness(data):
    """This actually tests that the posterior inference of whole-chromosome aneuploidies is correct here."""
    hmm = MetaHMM()
    gammas, _, karyotypes = hmm.forward_backward(
        bafs=data["baf_embryo"],
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
        pi0=0.7,
        std_dev=0.15,
    )
    # all of the columns must have a sum to 1
    assert np.all(np.isclose(np.sum(np.exp(gammas), axis=0), 1.0))
    post_dict = hmm.posterior_karyotypes(gammas, karyotypes)
    max_post = np.max([post_dict[p] for p in post_dict])
    for x in ["0", "1m", "1p", "2", "3m", "3p"]:
        assert x in post_dict
    assert post_dict[data["aploid"]] == max_post
