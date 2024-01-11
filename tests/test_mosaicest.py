"""Test suite for mosaic cell-fraction estimation."""
import numpy as np
import pytest

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


def test_mosaic_est_w_sigma(data=data_disomy):
    """Test estimates of cell-fraction for mosaicism."""
    m_est = MosaicEst(
        mat_haps=data["mat_haps"], pat_haps=data["pat_haps"], bafs=data["baf_embryo"]
    )
    m_est.baf_hets()
    ci_mle_theta, ci_cf = m_est.est_mle_mosaic(sigma=0.15)
    assert ci_mle_theta[0] <= ci_mle_theta[1]
    assert ci_mle_theta[1] <= ci_mle_theta[2]
    assert ci_cf[0] <= ci_cf[1]
    assert ci_cf[1] <= ci_cf[2]
    # It should be a very low mosaic cell-fraction ...
    assert ci_cf[1] <= 1e-2

def test_mosaic_est_wo_sigma(data=data_disomy):
    """Test estimates of cell-fraction for mosaicism."""
    m_est = MosaicEst(
        mat_haps=data["mat_haps"], pat_haps=data["pat_haps"], bafs=data["baf_embryo"]
    )
    m_est.baf_hets()
    ci_mle_theta, ci_cf = m_est.est_mle_mosaic()
    assert ci_mle_theta[0] <= ci_mle_theta[1]
    assert ci_mle_theta[1] <= ci_mle_theta[2]
    assert ci_cf[0] <= ci_cf[1]
    assert ci_cf[1] <= ci_cf[2]
    # It should be a very low mosaic cell-fraction still ...
    assert ci_cf[1] <= 1e-2