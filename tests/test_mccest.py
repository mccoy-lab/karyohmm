"""Test suite for karyoHMM DuoHMM."""

from karyohmm import MccEst, PGTSim
import numpy as np

# --- Generating test data for applications in the DuoHMM setting --- #
pgt_sim = PGTSim()
data_disomy = pgt_sim.full_ploidy_sim(m=1000, mix_prop=0.01, std_dev=0.1, seed=42)


def test_init():
    """Test the initialization of the estimator."""
    _ = MccEst()


def test_loglik_trio(
    bafs=np.array([0.6, 0.5]),
    mat_haps=np.array([[0, 0], [0, 0]]),
    pat_haps=np.array([[1, 1], [1, 1]]),
):
    mcc = MccEst()
    ll = mcc.loglik_mcc_trio(
        bafs=bafs, mat_haps=mat_haps, pat_haps=pat_haps, c=0.1, std_dev=0.1
    )
    assert ~np.isnan(ll)


def test_loglik_poc(
    bafs=np.array([0.6, 0.5]),
    mat_haps=np.array([[0, 0], [0, 0]]),
    freqs=np.array([0.3, 0.3]),
):
    mcc = MccEst()
    ll = mcc.loglik_mcc_poc(
        bafs=bafs, mat_haps=mat_haps, freqs=freqs, c=0.1, std_dev=0.1
    )
    assert ~np.isnan(ll)


def test_mle_trio(
    bafs=np.array([0.6, 0.5]),
    mat_haps=np.array([[0, 0], [0, 0]]),
    pat_haps=np.array([[1, 1], [1, 1]]),
):
    mcc = MccEst()
    (c_est, std_dev) = mcc.est_mcc_trio(bafs=bafs, mat_haps=mat_haps, pat_haps=pat_haps)
    assert (c_est >= 0) and (c_est <= 0.5)


def test_mle_poc(
    bafs=np.array([0.6, 0.5]),
    mat_haps=np.array([[0, 0], [0, 0]]),
    freqs=np.array([0.3, 0.3]),
):
    mcc = MccEst()
    (c_est, std_dev) = mcc.est_mcc_poc(bafs=bafs, mat_haps=mat_haps, freqs=freqs)
    assert (c_est >= 0) and (c_est <= 0.5)


def test_ci_trio(
    bafs=np.array([0.6, 0.5]),
    mat_haps=np.array([[0, 0], [0, 0]]),
    pat_haps=np.array([[1, 1], [1, 1]]),
):
    mcc = MccEst()
    (c_est, std_dev) = mcc.est_mcc_trio(bafs=bafs, mat_haps=mat_haps, pat_haps=pat_haps)
    (lower95_c, x, upper95_c) = mcc.mcc_ci_trio(
        bafs=bafs, mat_haps=mat_haps, pat_haps=pat_haps, c_hat=c_est, std_dev=std_dev
    )
    assert x == c_est
    assert lower95_c <= upper95_c


def test_ci_poc(
    bafs=np.array([0.6, 0.5]),
    mat_haps=np.array([[0, 0], [0, 0]]),
    freqs=np.array([0.3, 0.3]),
):
    mcc = MccEst()
    (c_est, std_dev) = mcc.est_mcc_poc(bafs=bafs, mat_haps=mat_haps, freqs=freqs)
    (lower95_c, x, upper95_c) = mcc.mcc_ci_poc(
        bafs=bafs, mat_haps=mat_haps, freqs=freqs, c_hat=c_est, std_dev=std_dev
    )
    assert x == c_est
    assert lower95_c <= upper95_c


def test_realistic_ci_trio(m=10000, n=5, c=0.1):
    """Simulate n realistic chromosomes with a high-degree of contamination."""
    pgt_sim = PGTSim()
    data = [
        pgt_sim.full_ploidy_sim(m=m, mix_prop=0.00, std_dev=0.1, seed=i + 1)
        for i in range(n)
    ]
    cc_bafs = np.hstack(
        [
            pgt_sim.sim_cell_contamination(
                baf=data[i]["baf"], haps=data[i]["mat_haps"], fraction=c, seed=i + 1
            )
            for i in range(n)
        ]
    )
    mat_haps = np.hstack([data[i]["mat_haps"] for i in range(n)])
    pat_haps = np.hstack([data[i]["pat_haps"] for i in range(n)])
    mcc = MccEst()
    (c_est, std_dev) = mcc.est_mcc_trio(
        bafs=cc_bafs, mat_haps=mat_haps, pat_haps=pat_haps
    )
    (lower95_c, x, upper95_c) = mcc.mcc_ci_trio(
        bafs=cc_bafs, mat_haps=mat_haps, pat_haps=pat_haps, c_hat=c_est, std_dev=std_dev
    )
    assert x == c_est
    assert lower95_c <= upper95_c
