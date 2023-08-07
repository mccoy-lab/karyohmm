"""Testing module for phase correction using embryo BAF."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from utils import sibling_euploid_sim, sim_joint_het

from karyohmm import PhaseCorrect

data_disomy_sibs_null = sibling_euploid_sim(
    m=4000, nsibs=3, std_dev=0.1, switch_err_rate=0.0, seed=42
)

data_disomy_sibs_test_1percent = sibling_euploid_sim(
    m=4000, nsibs=3, std_dev=0.1, switch_err_rate=1e-2, seed=42
)

data_disomy_sibs_test_3percent = sibling_euploid_sim(
    m=4000, nsibs=3, std_dev=0.1, switch_err_rate=3e-2, seed=42
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
        mat_haps=data["mat_haps_real"], pat_haps=data["pat_haps_real"]
    )
    phase_correct.add_true_haps(
        true_mat_haps=data["mat_haps_true"], true_pat_haps=data["pat_haps_true"]
    )
    n_switch, _, switch_err_rate, _ = phase_correct.estimate_switch_err_true()
    if data["mat_switch"].size > 0:
        assert switch_err_rate > 0
    else:
        assert switch_err_rate == 0
    n_switch, _, switch_err_rate, _ = phase_correct.estimate_switch_err_true(
        maternal=False
    )
    if data["pat_switch"].size > 0:
        assert switch_err_rate > 0
    else:
        assert switch_err_rate == 0


@pytest.mark.parametrize(
    "data",
    [
        data_disomy_sibs_null,
        data_disomy_sibs_test_1percent,
        data_disomy_sibs_test_3percent,
    ],
)
def test_phase_correct_true(data):
    """Test the phase-correction routine."""
    phase_correct = PhaseCorrect(
        mat_haps=data["mat_haps_real"], pat_haps=data["pat_haps_real"]
    )
    phase_correct.add_true_haps(
        true_mat_haps=data["mat_haps_true"], true_pat_haps=data["pat_haps_true"]
    )
    # 1. Apply phase correction for the maternal haplotypes
    phase_correct.add_baf(
        embryo_bafs=[data[f"baf_embryo{i}"] for i in range(data["nsibs"])]
    )
    phase_correct.phase_correct(pi0=0.6, std_dev=0.1)
    _, _, switch_err_rate_raw, _ = phase_correct.estimate_switch_err_true()
    _, _, switch_err_rate_fixed, _ = phase_correct.estimate_switch_err_true(fixed=True)
    assert switch_err_rate_fixed <= switch_err_rate_raw


@pytest.mark.parametrize(
    "data",
    [
        data_disomy_sibs_null,
        data_disomy_sibs_test_1percent,
        data_disomy_sibs_test_3percent,
    ],
)
def test_phase_correct_empirical(data):
    """Test the phase-correction routine."""
    phase_correct = PhaseCorrect(
        mat_haps=data["mat_haps_real"], pat_haps=data["pat_haps_real"]
    )
    phase_correct.add_true_haps(
        true_mat_haps=data["mat_haps_true"], true_pat_haps=data["pat_haps_true"]
    )
    # 1. Apply phase correction for the maternal haplotypes
    phase_correct.add_baf(
        embryo_bafs=[data[f"baf_embryo{i}"] for i in range(data["nsibs"])]
    )
    phase_correct.phase_correct(pi0=0.6, std_dev=0.1)
    # 2. Estimate empirical switch error rates
    _, _, switch_err_rate_raw, _ = phase_correct.estimate_switch_err_empirical()
    _, _, switch_err_rate_fixed, _ = phase_correct.estimate_switch_err_empirical(
        fixed=True
    )
    assert switch_err_rate_fixed <= switch_err_rate_raw


@given(
    pi0=st.floats(
        min_value=0.5, max_value=1, exclude_min=True, exclude_max=True, allow_nan=False
    ),
    sigma=st.floats(
        min_value=1e-2,
        max_value=0.1,
        exclude_min=True,
        exclude_max=False,
        allow_nan=False,
    ),
    nsibs=st.integers(min_value=3, max_value=10),
    switch=st.booleans(),
    seed=st.integers(min_value=1, max_value=1000),
)
def test_phase_correct_simple(switch, pi0, sigma, nsibs, seed):
    """Implement a more simple assessment of phase correction.

    NOTE: We have truncated this more to avoid excess noise that incurs some false-inferences.
    """
    # 1. Simulate a switched setup ...
    true_haps1, true_haps2, haps1, haps2, bafs, _ = sim_joint_het(
        switch=switch,
        mix_prop=pi0,
        meta_seed=seed,
        std_dev=sigma,
        nsibs=nsibs,
    )
    # 2. Implement the phase correction ...
    phase_correct = PhaseCorrect(mat_haps=haps1, pat_haps=haps2)
    phase_correct.add_true_haps(true_mat_haps=true_haps1, true_pat_haps=true_haps2)
    phase_correct.add_baf(embryo_bafs=bafs)
    phase_correct.phase_correct(pi0=pi0, std_dev=sigma)
    n_switches_real, _, _, _ = phase_correct.estimate_switch_err_true(fixed=False)
    n_switches_fixed, _, _, _ = phase_correct.estimate_switch_err_true(fixed=True)
    if switch:
        # If there was a switch, we should fix it now...
        assert n_switches_real > 0
        assert n_switches_fixed == 0
    else:
        # If there was not a switch, we should not need to fix it...
        assert n_switches_real == 0
        assert n_switches_fixed == 0
