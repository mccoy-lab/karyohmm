"""Test suite for mosaic cell-fraction estimation."""
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from karyohmm_utils import logsumexp

from karyohmm import MosaicEst, PGTSim

# --- Generating test data for applications --- #
pgt_sim = PGTSim()
data_disomy = pgt_sim.full_ploidy_sim(m=2000, length=10e6, std_dev=0.15, seed=42)


def test_mosaic_est_init(data=data_disomy):
    """Test the intialization of the mosaic estimation."""
    m_est = MosaicEst(
        mat_haps=data["mat_haps"], pat_haps=data["pat_haps"], bafs=data["baf_embryo"]
    )
    assert m_est.het_bafs is None


def test_baf_hets(data=data_disomy):
    """Test isolation of BAFs from heterozygotes."""
    m_est = MosaicEst(
        mat_haps=data["mat_haps"], pat_haps=data["pat_haps"], bafs=data["baf_embryo"]
    )
    m_est.baf_hets()
    assert m_est.het_bafs is not None


@given(
    sw_err=st.floats(min_value=1e-8, max_value=0.05),
    t_rate=st.floats(min_value=1e-8, max_value=0.2),
)
def test_transition_matrices(sw_err, t_rate):
    """Test that creation of the transition matrices is well-reasoned."""
    m_est = MosaicEst(
        mat_haps=data_disomy["mat_haps"],
        pat_haps=data_disomy["pat_haps"],
        bafs=data_disomy["baf_embryo"],
    )
    m_est.create_transition_matrix(switch_err=sw_err, t_rate=t_rate)
    # Assert that all of the rows sum up to 1
    assert np.isclose(logsumexp(m_est.A[0, :]), 0.0)
    assert np.isclose(logsumexp(m_est.A[1, :]), 0.0)
    assert np.isclose(logsumexp(m_est.A[2, :]), 0.0)


@given(theta=st.floats(min_value=0.0, max_value=0.5))
def test_est_cf(theta):
    """Test numerical accuracy of cell-fraction estimation."""
    m_est = MosaicEst(
        mat_haps=data_disomy["mat_haps"],
        pat_haps=data_disomy["pat_haps"],
        bafs=data_disomy["baf_embryo"],
    )
    cf_gain = m_est.est_cf(theta=theta, gain=True)
    cf_loss = m_est.est_cf(theta=theta, gain=False)
    if np.isnan(theta):
        assert np.isnan(cf_gain) and np.isnan(cf_loss)
    assert (cf_gain >= 0) and (cf_gain <= 1.0)
    if theta >= (2 / 3 - 1 / 2):
        # This is the maximal value for a chromosomal gain ...
        assert cf_gain == 1.0
    if theta == 0.0:
        # No possible gain or loss
        assert cf_gain == 0.0
        assert cf_loss == 0.0
    if theta == 0.5:
        assert (cf_gain == 1.0) and (cf_loss == 1.0)
