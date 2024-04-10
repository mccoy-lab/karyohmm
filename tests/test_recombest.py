"""Testing module for phase correction using embryo BAF."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from karyohmm import PGTSim, RecombEst, PhaseCorrect

pgt_sim = PGTSim()
data_disomy_sibs_null = pgt_sim.sibling_euploid_sim(
    m=4000, nsibs=3, std_dev=0.1, mix_prop=0.6, switch_err_rate=0.0, seed=42
)

data_disomy_sibs_test_1percent = pgt_sim.sibling_euploid_sim(
    m=4000, nsibs=3, std_dev=0.1, mix_prop=0.6, switch_err_rate=1e-2, seed=42
)

data_disomy_sibs_test_2percent = pgt_sim.sibling_euploid_sim(
    m=4000, nsibs=3, std_dev=0.1, mix_prop=0.6, switch_err_rate=2e-2, seed=42
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
