"""Test suite for karyoHMM DuoHMM."""

from karyohmm import MccEst, PGTSim
import numpy as np

# --- Generating test data for applications in the DuoHMM setting --- #
pgt_sim = PGTSim()
data_disomy = pgt_sim.full_ploidy_sim(m=1000, mix_prop=0.01, std_dev=0.1, seed=42)


def test_init():
    """Test the initialization of the estimator."""
    x = MccEst()


def test_loglik_trio(
    bafs=np.array([0.6, 0.5]),
    mat_haps=np.array([[0, 0], [0, 0]]),
    pat_haps=np.array([[1, 1], [1, 1]]),
):
    mcc = MccEst()
    ll = mcc.loglik_mcc_trio(
        bafs=bafs, mat_haps=mat_haps, pat_haps=pat_haps, c=0.1, std_dev=0.1
    )
    assert ll < 0


def test_loglik_poc(
    bafs=np.array([0.6, 0.5]),
    mat_haps=np.array([[0, 0], [0, 0]]),
    freqs=np.array([0.3, 0.3]),
):
    mcc = MccEst()
    ll = mcc.loglik_mcc_poc(
        bafs=bafs, mat_haps=mat_haps, freqs=freqs, c=0.1, std_dev=0.1
    )
    assert ll < 0
