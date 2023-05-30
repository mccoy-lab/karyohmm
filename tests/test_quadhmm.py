"""Testing module for QuadHMM."""

import numpy as np
import pytest
from utils import sibling_euploid_sim

from karyohmm import QuadHMM

# --- Generating test data for applications --- #
data_disomy_sibs = sibling_euploid_sim(
    m=4000, nsibs=3, std_dev=0.15, switch_err_rate=1e-2, seed=42
)


@pytest.mark.parametrize("data", [data_disomy_sibs])
def test_forward_algorithm(data):
    """Test the forward algorithm implementation of the QuadHMM."""
    hmm = QuadHMM()
    _, _, _, karyotypes, loglik = hmm.forward_algorithm(
        bafs=[data["baf_embryo0"], data["baf_embryo1"]],
        mat_haps=data["mat_haps_true"],
        pat_haps=data["pat_haps_true"],
    )
    assert loglik < 0


@pytest.mark.parametrize("data", [data_disomy_sibs])
def test_viterbi_algorithm(data):
    """Test the viterbi algorithm in the QuadHMM."""
    hmm = QuadHMM()
    path, states, deltas, psi = hmm.viterbi_algorithm(
        bafs=[data["baf_embryo0"], data["baf_embryo1"]],
        mat_haps=data["mat_haps_true"],
        pat_haps=data["pat_haps_true"],
    )
    assert np.all((path >= 0) & (path < len(states)))
    res_path = hmm.restrict_path(path)
    assert np.all((res_path >= 0) & (res_path <= 3))


@pytest.mark.parametrize("data", [data_disomy_sibs])
def test_recomb_isolation(data):
    """Test the viterbi algorithm in the QuadHMM."""
    hmm = QuadHMM()
    path01, _, _, _ = hmm.viterbi_algorithm(
        bafs=[data["baf_embryo0"], data["baf_embryo1"]],
        mat_haps=data["mat_haps_true"],
        pat_haps=data["pat_haps_true"],
    )
    res_path01 = hmm.restrict_path(path01)
    path02, _, _, _ = hmm.viterbi_algorithm(
        bafs=[data["baf_embryo0"], data["baf_embryo2"]],
        mat_haps=data["mat_haps_true"],
        pat_haps=data["pat_haps_true"],
    )
    res_path02 = hmm.restrict_path(path02)
    mat_rec, pat_rec = hmm.isolate_recomb(res_path01, res_path02)
    # True recombination events ...
    zs_maternal0 = data["zs_maternal0"]
    zs_paternal0 = data["zs_paternal0"]
    n_mat_rec = np.sum(zs_maternal0[:-1] != zs_maternal0[1:])
    n_pat_rec = np.sum(zs_paternal0[:-1] != zs_paternal0[1:])
    assert len(mat_rec) == n_mat_rec
    assert len(pat_rec) == n_pat_rec
