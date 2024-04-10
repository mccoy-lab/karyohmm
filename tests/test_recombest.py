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


# @given(
#     pi0=st.floats(
#         min_value=0.1, max_value=0.9, exclude_min=True, exclude_max=True, allow_nan=False
#     ),
#     sigma=st.floats(
#         min_value=5e-2,
#         max_value=0.5,
#         exclude_min=True,
#         exclude_max=True,
#         allow_nan=False,
#     ),
#     nsibs=st.integers(min_value=2, max_value=8)
# )
# @settings(max_examples=10, deadline=5000)


@pytest.mark.parametrize(
    "sigma,pi0,nsibs",
    [
        (0.05, 0.9, 3),
    ],
)
def test_recomb_isolate_paternal(sigma, pi0, nsibs):
    """Test that recombinations can be isolated in realistic conditions."""
    data = pgt_sim.sibling_euploid_sim(
        m=4000,
        nsibs=nsibs,
        std_dev=sigma,
        mix_prop=pi0,
        rec_prob=1e-3,
        switch_err_rate=0.0,
        seed=42 + nsibs,
    )
    recomb_est = RecombEst(
        mat_haps=data["mat_haps_real"], pat_haps=data["pat_haps_real"], pos=data["pos"]
    )
    # Add in the BAF values
    recomb_est.add_baf(
        embryo_bafs=[data[f"baf_embryo{i}"] for i in range(data["nsibs"])]
    )
    # Set the parameters here ...
    recomb_est.embryo_pi0s = np.array([pi0 for _ in range(nsibs)])
    recomb_est.embryo_sigmas = np.array([sigma for _ in range(nsibs)])
    # Obtain the true paternal recombination events for the template embryo ...
    zs_paternal0 = data["zs_paternal0"]
    n_pat_rec = np.sum(zs_paternal0[:-1] != zs_paternal0[1:])
    _, pat_recomb_events = recomb_est.isolate_recomb_events()
    filt_pat_recomb_events = recomb_est.refine_recomb_events(pat_recomb_events)
    # Make sure that the number matches up!
    assert n_pat_rec == len(filt_pat_recomb_events)
