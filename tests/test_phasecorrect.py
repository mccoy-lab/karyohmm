"""Testing module for phase correction using embryo BAF."""

import numpy as np
import pytest
from utils import sibling_euploid_sim

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
    n_switch, _, switch_err_rate, _ = phase_correct.estimate_switch_err()
    if data["mat_switch"].size > 0:
        assert switch_err_rate > 0
    else:
        assert switch_err_rate == 0
    n_switch, _, switch_err_rate, _ = phase_correct.estimate_switch_err(maternal=False)
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
def test_phase_correct(data):
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
    _, _, switch_err_rate_raw, _ = phase_correct.estimate_switch_err()
    _, _, switch_err_rate_fixed, _ = phase_correct.estimate_switch_err(fixed=True)
    assert switch_err_rate_fixed <= switch_err_rate_raw
