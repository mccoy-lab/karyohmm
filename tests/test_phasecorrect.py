"""Testing module for phase correction using embryo BAF."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from karyohmm import PGTSim, PhaseCorrect, QuadHMM

pgt_sim = PGTSim()
data_disomy_sibs_null = pgt_sim.sibling_euploid_sim(
    m=4000, nsibs=3, std_dev=0.1, mix_prop=0.6, switch_err_rate=0.0, seed=42
)

data_disomy_sibs_test_1percent = pgt_sim.sibling_euploid_sim(
    m=4000, nsibs=3, std_dev=0.1, mix_prop=0.6, switch_err_rate=1e-2, seed=42
)

data_disomy_sibs_test_3percent = pgt_sim.sibling_euploid_sim(
    m=4000, nsibs=3, std_dev=0.1, mix_prop=0.6, switch_err_rate=3e-2, seed=42
)


@pytest.mark.parametrize(
    "data",
    [
        data_disomy_sibs_null,
        data_disomy_sibs_test_1percent,
        data_disomy_sibs_test_3percent,
    ],
)
def test_switch_err_est(data):
    """Test the forward algorithm implementation of the QuadHMM."""
    phase_correct = PhaseCorrect(
        mat_haps=data["mat_haps_real"], pat_haps=data["pat_haps_real"], pos=data["pos"]
    )
    phase_correct.add_true_haps(
        true_mat_haps=data["mat_haps_true"],
        true_pat_haps=data["pat_haps_true"],
    )
    n_switch, _, switch_err_rate, _, _, _ = phase_correct.estimate_switch_err_true()
    if data["mat_switch"].size > 0:
        assert switch_err_rate > 0
    else:
        assert switch_err_rate == 0
    n_switch, _, switch_err_rate, _, _, _ = phase_correct.estimate_switch_err_true(
        maternal=False
    )
    if data["pat_switch"].size > 0:
        assert switch_err_rate > 0
    else:
        assert switch_err_rate == 0


@pytest.mark.parametrize(
    "data",
    [
        data_disomy_sibs_test_1percent,
        data_disomy_sibs_test_3percent,
    ],
)
def test_phase_correct_true(data):
    """Test the phase-correction routine."""
    phase_correct = PhaseCorrect(
        mat_haps=data["mat_haps_real"], pat_haps=data["pat_haps_real"], pos=data["pos"]
    )
    phase_correct.add_true_haps(
        true_mat_haps=data["mat_haps_true"],
        true_pat_haps=data["pat_haps_true"],
    )
    # 1. Apply phase correction for the maternal haplotypes
    phase_correct.add_baf(
        embryo_bafs=[data[f"baf_embryo{i}"] for i in range(data["nsibs"])]
    )
    phase_correct.lod_phase_correct(pi0=0.6, std_dev=0.1)
    _, _, switch_err_rate_raw, _, _, _ = phase_correct.estimate_switch_err_true()
    _, _, switch_err_rate_fixed, _, _, _ = phase_correct.estimate_switch_err_true(
        fixed=True
    )
    assert switch_err_rate_fixed < switch_err_rate_raw


@pytest.mark.parametrize(
    "data",
    [
        data_disomy_sibs_test_1percent,
        data_disomy_sibs_test_3percent,
    ],
)
def test_phase_correct_empirical(data):
    """Test the phase-correction routine."""
    phase_correct = PhaseCorrect(
        mat_haps=data["mat_haps_real"], pat_haps=data["pat_haps_real"], pos=data["pos"]
    )
    phase_correct.add_true_haps(
        true_mat_haps=data["mat_haps_true"], true_pat_haps=data["pat_haps_true"]
    )
    # 1. Apply phase correction for the maternal haplotypes
    phase_correct.add_baf(
        embryo_bafs=[data[f"baf_embryo{i}"] for i in range(data["nsibs"])]
    )
    phase_correct.lod_phase_correct(pi0=0.6, std_dev=0.1)
    # 2. Estimate empirical switch error rates
    _, _, switch_err_rate_raw, _, _, _ = phase_correct.estimate_switch_err_empirical()
    _, _, switch_err_rate_fixed, _, _, _ = phase_correct.estimate_switch_err_empirical(
        fixed=True
    )
    assert switch_err_rate_fixed < switch_err_rate_raw


@pytest.mark.parametrize(
    "data",
    [
        data_disomy_sibs_test_1percent,
        data_disomy_sibs_test_3percent,
    ],
)
def test_phase_correct_viterbi(data):
    """Test a phase correction using the viterbi-copying path under disomy."""
    phase_correct = PhaseCorrect(
        mat_haps=data["mat_haps_real"], pat_haps=data["pat_haps_real"], pos=data["pos"]
    )
    phase_correct.add_true_haps(
        true_mat_haps=data["mat_haps_true"], true_pat_haps=data["pat_haps_true"]
    )
    # 1. Add in the BAF per sibling
    phase_correct.add_baf(
        embryo_bafs=[data[f"baf_embryo{i}"] for i in range(data["nsibs"])]
    )
    # 2. Estimate the noise parameters
    phase_correct.est_sigma_pi0s()
    # 3. Estimate the fixed haplotypes
    (
        mat_haps,
        pat_haps,
        n_mis_mat_tot,
        n_mis_pat_tot,
    ) = phase_correct.viterbi_phase_correct(niter=5)
    assert n_mis_mat_tot.size == 5
    assert n_mis_pat_tot.size == 5
    for i in range(1, 5):
        # Assert that we are always improving the phasing errors here
        assert n_mis_mat_tot[i] <= n_mis_mat_tot[i - 1]
        assert n_mis_pat_tot[i] <= n_mis_pat_tot[i - 1]
    _, _, switch_err_rate_raw, _, _, _ = phase_correct.estimate_switch_err_true()
    _, _, switch_err_rate_fixed, _, _, _ = phase_correct.estimate_switch_err_true(
        fixed=True
    )
    assert switch_err_rate_fixed < switch_err_rate_raw


@pytest.mark.parametrize(
    "data",
    [
        data_disomy_sibs_test_1percent,
        data_disomy_sibs_test_3percent,
    ],
)
def test_recomb_isolation(data):
    """Test recombination isolation."""
    phase_correct = PhaseCorrect(
        mat_haps=data["mat_haps_real"], pat_haps=data["pat_haps_real"], pos=data["pos"]
    )
    phase_correct.add_true_haps(
        true_mat_haps=data["mat_haps_true"], true_pat_haps=data["pat_haps_true"]
    )
    # 1. Add in the BAF per sibling
    phase_correct.add_baf(
        embryo_bafs=[data[f"baf_embryo{i}"] for i in range(data["nsibs"])]
    )
    # 2. Estimate the noise parameters
    phase_correct.est_sigma_pi0s()
    # 3. Estimate the fixed haplotypes
    (
        mat_haps,
        pat_haps,
        n_mis_mat_tot,
        n_mis_pat_tot,
    ) = phase_correct.viterbi_phase_correct(niter=2)
    hmm = QuadHMM()
    res_path01 = hmm.map_path(
        bafs=[data["baf_embryo0"], data["baf_embryo1"]],
        pos=phase_correct.pos,
        mat_haps=phase_correct.mat_haps_fixed,
        pat_haps=phase_correct.pat_haps_fixed,
    )
    res_path02 = hmm.map_path(
        bafs=[data["baf_embryo0"], data["baf_embryo2"]],
        pos=phase_correct.pos,
        mat_haps=phase_correct.mat_haps_fixed,
        pat_haps=phase_correct.pat_haps_fixed,
    )
    mat_rec, pat_rec, _, _ = hmm.isolate_recomb(res_path01, [res_path02])
    # True recombination events ...
    zs_maternal0 = data["zs_maternal0"]
    zs_paternal0 = data["zs_paternal0"]
    n_mat_rec = np.sum(zs_maternal0[:-1] != zs_maternal0[1:])
    n_pat_rec = np.sum(zs_paternal0[:-1] != zs_paternal0[1:])
    assert len(mat_rec) == n_mat_rec
    assert len(pat_rec) == n_pat_rec
