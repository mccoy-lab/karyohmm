"""Testing module for QuadHMM."""

import numpy as np
import pytest

from karyohmm import PGTSim, QuadHMM

# --- Generating test data for applications --- #
pgt_sim = PGTSim()
data_disomy_sibs = pgt_sim.sibling_euploid_sim(
    m=4000, nsibs=3, std_dev=0.15, switch_err_rate=1e-2, seed=42
)
data_disomy_sibs_v2 = pgt_sim.sibling_euploid_sim(
    m=4000, nsibs=3, std_dev=0.2, switch_err_rate=1e-2, seed=24
)

data_disomy_sibs_v3 = pgt_sim.sibling_euploid_sim(
    m=4000, nsibs=3, std_dev=0.25, switch_err_rate=2e-2, seed=18
)


@pytest.mark.parametrize(
    "data", [data_disomy_sibs, data_disomy_sibs_v2, data_disomy_sibs_v3]
)
def test_forward_algorithm(data):
    """Test the forward algorithm implementation of the QuadHMM."""
    hmm = QuadHMM()
    _, _, _, _, loglik = hmm.forward_algorithm(
        bafs=[data["baf_embryo0"], data["baf_embryo1"]],
        pos=data["pos"],
        mat_haps=data["mat_haps_true"],
        pat_haps=data["pat_haps_true"],
    )


@pytest.mark.parametrize("data", [data_disomy_sibs])
def test_backward_algorithm(data):
    """Test the backward algorithm implementation of the QuadHMM."""
    hmm = QuadHMM()
    _, _, _, _, loglik = hmm.forward_algorithm(
        bafs=[data["baf_embryo0"], data["baf_embryo1"]],
        pos=data["pos"],
        mat_haps=data["mat_haps_true"],
        pat_haps=data["pat_haps_true"],
    )


@pytest.mark.parametrize("data", [data_disomy_sibs])
def test_forward_vs_backward_loglik(data):
    """Test that the log-likelihood from forward algorithm is equal to the backward."""
    hmm = QuadHMM()
    _, _, _, _, fwd_loglik = hmm.forward_algorithm(
        bafs=[data["baf_embryo0"], data["baf_embryo1"]],
        pos=data["pos"],
        mat_haps=data["mat_haps_true"],
        pat_haps=data["pat_haps_true"],
    )
    _, _, _, _, bwd_loglik = hmm.backward_algorithm(
        bafs=[data["baf_embryo0"], data["baf_embryo1"]],
        pos=data["pos"],
        mat_haps=data["mat_haps_true"],
        pat_haps=data["pat_haps_true"],
    )
    assert np.isclose(fwd_loglik, bwd_loglik, atol=1e-6)


@pytest.mark.parametrize(
    "data", [data_disomy_sibs, data_disomy_sibs_v2, data_disomy_sibs_v3]
)
def test_viterbi_algorithm(data):
    """Test the viterbi algorithm in the QuadHMM."""
    hmm = QuadHMM()
    path, states, deltas, psi = hmm.viterbi_algorithm(
        bafs=[data["baf_embryo0"], data["baf_embryo1"]],
        pos=data["pos"],
        mat_haps=data["mat_haps_true"],
        pat_haps=data["pat_haps_true"],
    )
    assert np.all((path >= 0) & (path < len(states)))
    res_path = hmm.restrict_path(path)
    assert np.all((res_path >= 0) & (res_path <= 3))


@pytest.mark.parametrize(
    "data", [data_disomy_sibs, data_disomy_sibs_v2, data_disomy_sibs_v3]
)
def test_recomb_isolation_map(data):
    """Test the viterbi algorithm in the QuadHMM."""
    hmm = QuadHMM()
    res_path01 = hmm.map_path(
        bafs=[data["baf_embryo0"], data["baf_embryo1"]],
        pos=data["pos"],
        mat_haps=data["mat_haps_true"],
        pat_haps=data["pat_haps_true"],
    )
    res_path02 = hmm.map_path(
        bafs=[data["baf_embryo0"], data["baf_embryo2"]],
        pos=data["pos"],
        mat_haps=data["mat_haps_true"],
        pat_haps=data["pat_haps_true"],
    )
    mat_rec, pat_rec, _, _ = hmm.isolate_recomb(res_path01, [res_path02])
    # True recombination events ...
    zs_maternal0 = data["zs_maternal0"]
    zs_paternal0 = data["zs_paternal0"]
    n_mat_rec = np.sum(zs_maternal0[:-1] != zs_maternal0[1:])
    n_pat_rec = np.sum(zs_paternal0[:-1] != zs_paternal0[1:])
    assert len(mat_rec) == n_mat_rec
    assert len(pat_rec) == n_pat_rec


@pytest.mark.parametrize(
    "data", [data_disomy_sibs, data_disomy_sibs_v2, data_disomy_sibs_v3]
)
def test_recomb_isolation_viterbi(data):
    """Test the viterbi algorithm in the QuadHMM."""
    hmm = QuadHMM()
    res_path01 = hmm.viterbi_path(
        bafs=[data["baf_embryo0"], data["baf_embryo1"]],
        pos=data["pos"],
        mat_haps=data["mat_haps_true"],
        pat_haps=data["pat_haps_true"],
    )
    res_path02 = hmm.viterbi_path(
        bafs=[data["baf_embryo0"], data["baf_embryo2"]],
        pos=data["pos"],
        mat_haps=data["mat_haps_true"],
        pat_haps=data["pat_haps_true"],
    )
    mat_rec, pat_rec, _, _ = hmm.isolate_recomb(res_path01, [res_path02])
    # True recombination events ...
    zs_maternal0 = data["zs_maternal0"]
    zs_paternal0 = data["zs_paternal0"]
    n_mat_rec = np.sum(zs_maternal0[:-1] != zs_maternal0[1:])
    n_pat_rec = np.sum(zs_paternal0[:-1] != zs_paternal0[1:])
    assert len(mat_rec) == n_mat_rec
    assert len(pat_rec) == n_pat_rec


@pytest.mark.parametrize(
    "data", [data_disomy_sibs, data_disomy_sibs_v2, data_disomy_sibs_v3]
)
def test_estimate_sharing(data):
    """Test the estimate of haplotype sharing across sibling embryos."""
    hmm = QuadHMM()
    mat_haplo_len, pat_haplo_len, both_haplo_len, tot_len = hmm.est_sharing(
        bafs=[data["baf_embryo0"], data["baf_embryo1"]],
        pos=data["pos"],
        mat_haps=data["mat_haps_true"],
        pat_haps=data["pat_haps_true"],
    )
    assert tot_len <= data["length"]
    assert tot_len == (data["pos"][-1] - data["pos"][0])
