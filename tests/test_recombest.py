"""Testing module for phase correction using embryo BAF."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from karyohmm import PGTSim, RecombEst

pgt_sim = PGTSim()
data_disomy_sibs_null = pgt_sim.sibling_euploid_sim(
    m=4000, nsibs=3, std_dev=0.1, mix_prop=0.6, switch_err_rate=0.0, seed=42
)

# data_disomy_sibs_test_1percent = pgt_sim.sibling_euploid_sim(
#     m=4000, nsibs=3, std_dev=0.1, mix_prop=0.6, switch_err_rate=1e-2, seed=42
# )

# data_disomy_sibs_test_2percent = pgt_sim.sibling_euploid_sim(
#     m=4000, nsibs=3, std_dev=0.1, mix_prop=0.6, switch_err_rate=2e-2, seed=42
# )

# data_disomy_sibs_test_3percent = pgt_sim.sibling_euploid_sim(
#     m=4000, nsibs=3, std_dev=0.1, mix_prop=0.6, switch_err_rate=3e-2, seed=42
# )


@pytest.mark.parametrize(
    "data",
    [
        data_disomy_sibs_null,
    ],
)
def test_recomb_est_init(data):
    """Test the forward algorithm implementation of the QuadHMM."""
    recomb_est = RecombEst(
        mat_haps=data["mat_haps_real"], pat_haps=data["pat_haps_real"], pos=data["pos"]
    )
    recomb_est.add_baf(
        embryo_bafs=[data[f"baf_embryo{i}"] for i in range(data["nsibs"])]
    )
