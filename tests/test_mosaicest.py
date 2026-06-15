"""Test suite for mosaic cell-fraction estimation."""

import numpy as np
import pytest
from unittest.mock import patch
from hypothesis import given
from hypothesis import strategies as st
from karyohmm_utils import logsumexp

from karyohmm import MosaicEst, PGTSim
from karyohmm.simulator import PGTSimMosaic

# ---------------------------------------------------------------------------
# Shared simulation fixtures
# ---------------------------------------------------------------------------

pgt_sim = PGTSim()
data_disomy = pgt_sim.full_ploidy_sim(m=5000, length=1e7, std_dev=0.1, seed=42)
data_trisomy = pgt_sim.full_ploidy_sim(m=5000, ploidy=3, length=1e7, std_dev=0.1, seed=42)
data_monosomy = pgt_sim.full_ploidy_sim(m=5000, ploidy=1, length=1e7, std_dev=0.1, seed=42)

# seed=42 → aploid 3m (maternal gain) and 1m (paternal loss, maternal retained)
# seed=44 → aploid 3p (paternal gain) and 1p (maternal loss, paternal retained)
# Using two seeds gives all four parental origins.
data_pat_gain = pgt_sim.full_ploidy_sim(m=5000, ploidy=3, length=1e7, std_dev=0.1, seed=44)
data_mat_loss = pgt_sim.full_ploidy_sim(m=5000, ploidy=1, length=1e7, std_dev=0.1, seed=44)
data_disomy_44 = pgt_sim.full_ploidy_sim(m=5000, length=1e7, std_dev=0.1, seed=44)

# Coherent mosaic fixtures: weighted blend of full-ploidy sims sharing the
# same parental haplotypes (same seed ⟹ same mat_haps / pat_haps / pos).
_CF_GAIN = 0.4
_CF_LOSS = 0.4


def _blend(base, aneu, cf):
    d = dict(base)
    d["baf"] = (1 - cf) * base["baf"] + cf * aneu["baf"]
    d["lrr"] = (1 - cf) * base["lrr"] + cf * aneu["lrr"]
    d["sigmas"] = (1 - cf) * base["sigmas"] + cf * aneu["sigmas"]
    return d


# maternal gain (3m) + paternal loss (1m) at seed=42
_data_mosaic_mat_gain = _blend(data_disomy, data_trisomy, _CF_GAIN)
_data_mosaic_pat_loss = _blend(data_disomy, data_monosomy, _CF_LOSS)

# paternal gain (3p) + maternal loss (1p) at seed=44
_data_mosaic_pat_gain = _blend(data_disomy_44, data_pat_gain, _CF_GAIN)
_data_mosaic_mat_loss = _blend(data_disomy_44, data_mat_loss, _CF_LOSS)

# Aliases used by the existing tests below (seed=42 pair)
_data_mosaic_gain = _data_mosaic_mat_gain
_data_mosaic_loss = _data_mosaic_pat_loss


def _make(data):
    """Construct MosaicEst from a simulation result dict."""
    return MosaicEst(
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
        bafs=data["baf"],
        pos=data["pos"],
        lrrs=data["lrr"],
        sigmas=data["sigmas"],
    )


# ---------------------------------------------------------------------------
# Construction and preprocessing
# ---------------------------------------------------------------------------


def test_init_runs_preprocessing():
    """After construction, het sites are phased and transition matrix is ready."""
    m = _make(data_disomy)
    assert m.n_het > 0
    assert m.phased_baf.size == m.n_het
    assert m.A is not None


def test_no_lrr_warns():
    """Omitting LRR emits a UserWarning rather than raising."""
    with pytest.warns(UserWarning, match="lrrs not provided"):
        MosaicEst(
            mat_haps=data_disomy["mat_haps"],
            pat_haps=data_disomy["pat_haps"],
            bafs=data_disomy["baf"],
            pos=data_disomy["pos"],
        )


def test_phased_baf_disomy_centred():
    """Phased BAF at disomy het sites should have mean near zero."""
    m = _make(data_disomy)
    assert abs(np.mean(m.phased_baf)) < 0.05


def test_n_het_minimum():
    """Raises if fewer than 10 expected-het sites exist."""
    rng = np.random.default_rng(0)
    # All-homozygous parental haplotypes → no expected het sites
    mh = np.zeros((2, 200), dtype=np.int8)
    ph = np.zeros((2, 200), dtype=np.int8)
    with pytest.raises(ValueError, match="Fewer than 10"):
        MosaicEst(
            mat_haps=mh, pat_haps=ph,
            bafs=rng.uniform(size=200),
            pos=np.sort(rng.uniform(high=1e7, size=200)),
            lrrs=np.zeros(200), sigmas=np.ones(200),
        )


@given(
    sw_err=st.floats(min_value=1e-8, max_value=0.05),
    t_rate=st.floats(min_value=1e-8, max_value=0.2),
)
def test_transition_matrix_rows_sum_to_one(sw_err, t_rate):
    """All 5 rows of the log-transition matrix must sum to 0 (probability 1).

    Rows corresponding to aneuploid states contain -inf entries for the
    cross-type transitions (gain↔loss); logsumexp handles these correctly.
    """
    m = _make(data_disomy)
    m.create_transition_matrix(switch_err=sw_err, t_rate=t_rate)
    assert m.A.shape == (5, 5)
    for row in range(5):
        assert np.isclose(logsumexp(m.A[row, :]), 0.0)


# ---------------------------------------------------------------------------
# forward_algo_full — likelihood sanity checks
# ---------------------------------------------------------------------------


def test_forward_loglik_is_finite():
    """forward_algo_full returns a finite log-likelihood for valid cf."""
    m = _make(data_disomy)
    _, _, ll = m.forward_algo_full(cf=0.0)
    assert np.isfinite(ll)


def test_forward_loglik_decreases_at_cf_zero_for_disomy():
    """For a disomy sample cf=0 should have the highest or near-highest likelihood."""
    m = _make(data_disomy)
    ll0 = m.forward_algo_full(cf=0.0)[2]
    ll_high = m.forward_algo_full(cf=0.4)[2]
    assert ll0 > ll_high


def test_forward_loglik_increases_with_cf_for_trisomy():
    """For a full trisomy the likelihood should increase moving away from cf=0."""
    m = _make(data_trisomy)
    ll0 = m.forward_algo_full(cf=0.0)[2]
    ll_true = m.forward_algo_full(cf=0.9)[2]
    assert ll_true > ll0


# ---------------------------------------------------------------------------
# est_mle_cf — point estimation (disomy null; full cases covered in origin section)
# ---------------------------------------------------------------------------


def test_mle_cf_disomy_near_zero():
    """Disomy: MLE cell fraction should be effectively zero."""
    m = _make(data_disomy)
    m.est_mle_cf()
    assert m.mle_cf is not None
    assert not np.isnan(m.mle_cf)
    assert m.mle_cf < 0.05


# ---------------------------------------------------------------------------
# ci_mle_cf — confidence intervals
# ---------------------------------------------------------------------------


def _check_ci(ci):
    assert ci[0] <= ci[1] <= ci[2]
    assert 0.0 <= ci[0]
    assert ci[2] <= 1.0


def test_ci_ordered_disomy():
    m = _make(data_disomy)
    m.est_mle_cf()
    _check_ci(m.ci_mle_cf())


def test_ci_ordered_trisomy():
    m = _make(data_trisomy)
    m.est_mle_cf()
    _check_ci(m.ci_mle_cf())


def test_ci_ordered_monosomy():
    m = _make(data_monosomy)
    m.est_mle_cf()
    _check_ci(m.ci_mle_cf())


def test_ci_mosaic_gain_reasonable():
    """95% CI for 40% mosaic gain: ordered, non-trivial width, and mle_cf > 0."""
    m = _make(_data_mosaic_gain)
    m.est_mle_cf()
    ci = m.ci_mle_cf()
    _check_ci(ci)
    # CI should be non-degenerate and centre well above zero
    assert (ci[2] - ci[0]) > 0.01
    assert ci[1] > 0.1


# ---------------------------------------------------------------------------
# lrt_cf — disomy null; aneuploid LRT is verified inside _assert_case above
# ---------------------------------------------------------------------------


def test_lrt_cf_disomy_not_significant():
    """LRT statistic should be below the chi2(1) 0.05 threshold for a true disomy."""
    m = _make(data_disomy)
    m.est_mle_cf()
    assert m.lrt_cf() < _CHI2_THRESH


# ---------------------------------------------------------------------------
# Robustness to parental haplotype errors
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# infer_origin — complete coverage of all four parental origins
#
# Convention in the simulator:
#   aploid='3m' → extra maternal copy      → maternal-gain   (seed=42)
#   aploid='1m' → maternal copy retained   → paternal-loss   (seed=42)
#   aploid='3p' → extra paternal copy      → paternal-gain   (seed=44)
#   aploid='1p' → paternal copy retained   → maternal-loss   (seed=44)
# ---------------------------------------------------------------------------

_CHI2_THRESH = 3.84  # chi2(1) at p=0.05


def _assert_case(data, expected_origin, min_cf, label=""):
    """Shared helper: fit, check cf, check origin, check LRT."""
    m = _make(data)
    m.est_mle_cf()
    assert not np.isnan(m.mle_cf), f"{label}: mle_cf is nan"
    assert m.mle_cf > min_cf, f"{label}: mle_cf={m.mle_cf:.4f} not > {min_cf}"
    assert m.infer_origin() == expected_origin, (
        f"{label}: got {m.infer_origin()!r}, want {expected_origin!r}"
    )
    assert m.lrt_cf() > _CHI2_THRESH, f"{label}: LRT not significant"


def test_infer_origin_state_names_complete():
    """STATE_NAMES covers all five possible return values of infer_origin."""
    expected = {
        "neutral", "maternal-gain", "paternal-gain", "maternal-loss", "paternal-loss"
    }
    assert set(MosaicEst.STATE_NAMES) == expected


def test_disomy_is_neutral_and_lrt_small():
    """Disomy: mle_cf ≈ 0, labelled neutral, LRT not significant."""
    m = _make(data_disomy)
    m.est_mle_cf()
    assert m.mle_cf < 0.05
    assert m.infer_origin() == "neutral"
    assert m.lrt_cf() < _CHI2_THRESH


# --- Full aneuploidies ---

def test_full_maternal_gain():
    """Full maternal trisomy (3m): mle_cf ≈ 1, origin = maternal-gain, LRT large."""
    _assert_case(data_trisomy, "maternal-gain", min_cf=0.8, label="full-maternal-gain")


def test_full_paternal_gain():
    """Full paternal trisomy (3p): mle_cf ≈ 1, origin = paternal-gain, LRT large."""
    _assert_case(data_pat_gain, "paternal-gain", min_cf=0.8, label="full-paternal-gain")


def test_full_paternal_loss():
    """Full paternal monosomy (1m / maternal retained): origin = paternal-loss."""
    _assert_case(data_monosomy, "paternal-loss", min_cf=0.8, label="full-paternal-loss")


def test_full_maternal_loss():
    """Full maternal monosomy (1p / paternal retained): origin = maternal-loss."""
    _assert_case(data_mat_loss, "maternal-loss", min_cf=0.8, label="full-maternal-loss")


# --- Mosaic aneuploidies (40% cell fraction) ---

def test_mosaic_maternal_gain():
    """40% maternal-gain mosaic: mle_cf > 0.2, correct origin, LRT significant."""
    _assert_case(
        _data_mosaic_mat_gain, "maternal-gain", min_cf=0.2,
        label="mosaic-maternal-gain",
    )


def test_mosaic_paternal_gain():
    """40% paternal-gain mosaic: mle_cf > 0.2, correct origin, LRT significant."""
    _assert_case(
        _data_mosaic_pat_gain, "paternal-gain", min_cf=0.2,
        label="mosaic-paternal-gain",
    )


def test_mosaic_paternal_loss():
    """40% paternal-loss mosaic: mle_cf > 0.2, correct origin, LRT significant."""
    _assert_case(
        _data_mosaic_pat_loss, "paternal-loss", min_cf=0.2,
        label="mosaic-paternal-loss",
    )


def test_mosaic_maternal_loss():
    """40% maternal-loss mosaic: mle_cf > 0.2, correct origin, LRT significant."""
    _assert_case(
        _data_mosaic_mat_loss, "maternal-loss", min_cf=0.2,
        label="mosaic-maternal-loss",
    )


# ---------------------------------------------------------------------------
# Coverage of error / boundary paths
# ---------------------------------------------------------------------------


def test_forward_algo_full_first_site_is_het():
    """forward_algo_full correctly adds BAF emission when site 0 is a het site."""
    # Build a tiny synthetic case where the first SNP is expected-het so the
    # `if is_het[0]` branch inside forward_algo_full is exercised.
    rng = np.random.default_rng(7)
    m_sites = 200
    pos = np.sort(rng.uniform(high=1e7, size=m_sites))
    # Force site 0 to be a het site: mat hom-ref (0,0), pat hom-alt (1,1)
    mat_haps = rng.integers(0, 2, size=(2, m_sites))
    pat_haps = rng.integers(0, 2, size=(2, m_sites))
    mat_haps[:, 0] = 0
    pat_haps[:, 0] = 1
    # Ensure enough other het sites exist so _baf_hets doesn't raise
    for i in range(1, 30):
        mat_haps[:, i] = 0
        pat_haps[:, i] = 1
    bafs = rng.uniform(size=m_sites)
    lrrs = rng.normal(size=m_sites)
    sigmas = np.abs(rng.normal(loc=0.2, size=m_sites)) + 0.05
    m = MosaicEst(mat_haps=mat_haps, pat_haps=pat_haps, bafs=bafs, pos=pos,
                  lrrs=lrrs, sigmas=sigmas)
    assert m.het_idx[0] == 0  # confirm first site is het
    _, _, ll = m.forward_algo_full(cf=0.1)
    assert np.isfinite(ll)


def test_est_mle_cf_failure_sets_nan():
    """If forward_algo_full raises, est_mle_cf stores nan rather than crashing."""
    m = _make(data_disomy)
    with patch.object(m, "forward_algo_full", side_effect=ValueError("synthetic failure")):
        m.est_mle_cf()
    assert np.isnan(m.mle_cf)


def test_ci_mle_cf_boundary_low():
    """ci_mle_cf uses one-sided finite difference when mle_cf < h."""
    m = _make(data_disomy)
    m.est_mle_cf()
    # Use h larger than the near-zero mle_cf to force the `cf < h` branch
    ci = m.ci_mle_cf(h=max(m.mle_cf + 1e-4, 1e-3))
    _check_ci(ci)


def test_ci_mle_cf_boundary_high():
    """ci_mle_cf uses one-sided finite difference when mle_cf is near 1."""
    m = _make(data_trisomy)
    m.est_mle_cf()
    # Use h larger than (0.999 - mle_cf) to force the high-boundary branch
    gap = 0.999 - m.mle_cf
    if gap > 0:
        ci = m.ci_mle_cf(h=gap + 1e-4)
        _check_ci(ci)


def test_ci_mle_cf_exception_handler():
    """ci_mle_cf returns [nan, nan, nan] when the Hessian computation fails."""
    m = _make(data_disomy)
    m.est_mle_cf()
    with patch.object(m, "forward_algo_full", side_effect=ZeroDivisionError):
        ci = m.ci_mle_cf()
    assert all(np.isnan(v) for v in ci)


def test_lrt_cf_auto_calls_est_mle_cf():
    """lrt_cf runs est_mle_cf internally when mle_cf has not been set yet."""
    m = _make(data_trisomy)
    assert m.mle_cf is None
    lrt = m.lrt_cf()
    assert m.mle_cf is not None  # was set as a side-effect
    assert np.isfinite(lrt) and lrt > 0


def test_lrt_cf_nan_when_mle_failed():
    """lrt_cf returns nan when the MLE failed (mle_cf is nan)."""
    m = _make(data_disomy)
    m.mle_cf = np.nan  # simulate a prior optimisation failure
    assert np.isnan(m.lrt_cf())


def test_robust_to_switch_errors():
    """Switch errors in parental haps do not affect mle_cf (het sites immune)."""
    from karyohmm.simulator import PGTSimBase

    sim = PGTSimBase()
    mh, ph, _, _ = sim.create_switch_errors(
        data_disomy["mat_haps"], data_disomy["pat_haps"], err_rate=0.1, seed=1
    )
    m_clean = _make(_data_mosaic_gain)
    m_sw = MosaicEst(
        mat_haps=mh, pat_haps=ph,
        bafs=_data_mosaic_gain["baf"], pos=_data_mosaic_gain["pos"],
        lrrs=_data_mosaic_gain["lrr"], sigmas=_data_mosaic_gain["sigmas"],
    )
    m_clean.est_mle_cf()
    m_sw.est_mle_cf()
    assert abs(m_clean.mle_cf - m_sw.mle_cf) < 0.01


def test_robust_to_genotyping_errors():
    """Up to 5% genotyping error in parental haps causes < 0.02 drift in mle_cf."""
    from karyohmm.simulator import PGTSimBase

    sim = PGTSimBase()
    _, mh_e = sim.create_genotyping_errors(data_disomy["mat_haps"], err_rate=0.05, seed=1)
    _, ph_e = sim.create_genotyping_errors(data_disomy["pat_haps"], err_rate=0.05, seed=2)
    m_clean = _make(_data_mosaic_gain)
    m_ge = MosaicEst(
        mat_haps=mh_e, pat_haps=ph_e,
        bafs=_data_mosaic_gain["baf"], pos=_data_mosaic_gain["pos"],
        lrrs=_data_mosaic_gain["lrr"], sigmas=_data_mosaic_gain["sigmas"],
    )
    m_clean.est_mle_cf()
    m_ge.est_mle_cf()
    assert abs(m_clean.mle_cf - m_ge.mle_cf) < 0.02
