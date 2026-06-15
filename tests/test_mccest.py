"""Test suite for MCC estimation: unphased, phase-aware, and genome-wide methods.

Design notes
------------
The phased HMM tracks which maternal haplotype was transmitted to the POC via a
2-state recombination HMM.  The CI-width advantage over the unphased model is real
but modest (~3-5 %) and only manifests when:
  (a) maternal sites are heterozygous (so phase matters),
  (b) consecutive sites share an LD block (so the HMM can resolve the phase state),
  (c) contamination c is comparable to or smaller than BAF noise sigma.
PGTSim haplotypes have essentially no LD (mean run-length ~2), so those are used
only for basic sanity checks rather than CI-comparison assertions.

``sim_cell_contamination(fraction=f)`` produces a BAF shift of roughly f/2 at
informative sites, which the MLE recovers as a model-c of ≈ f/2.  Integration tests
therefore use the MLE-based LRT rather than asserting ll(fraction) > ll(0), and
accuracy tests use BAF drawn directly from the MCC emission model.

Fixture groups
--------------
_sc_*    Single-chromosome, unphased unit tests (m=500, seed=7)
_ph_*    Single-chromosome, phased unit tests (m=2000, seed=42, contaminated)
_int_*   Genome-wide integration tests via PGTSim (3 chroms, m=2000, seeds 700+)
_model_* Genome-wide accuracy tests via direct emission sampling (3 chroms, m=2000)
"""

import numpy as np
from karyohmm import MccEst, PGTSim

# ===========================================================================
# Fixture group 1 — single-chromosome, unphased  (m=500, seed=7)
# ===========================================================================

_sc_sim = PGTSim()
_sc_data = _sc_sim.full_ploidy_sim(m=500, mix_prop=0.0, std_dev=0.1, seed=7)
_sc_bafs = _sc_data["baf"]
_sc_mat  = _sc_data["mat_haps"]
_sc_pat  = _sc_data["pat_haps"]
_sc_freq = _sc_data["af"]

# ===========================================================================
# Fixture group 2 — single-chromosome, phased  (m=2000, seed=42, contaminated)
# ===========================================================================

_ph_sim = PGTSim()
_ph_data = _ph_sim.full_ploidy_sim(m=2000, mix_prop=0.0, std_dev=0.1, seed=42)
_ph_cc_bafs = _ph_sim.sim_cell_contamination(
    baf=_ph_data["baf"], haps=_ph_data["mat_haps"], fraction=0.10, seed=42
)
_ph_mat_haps = _ph_data["mat_haps"]
_ph_pat_haps = _ph_data["pat_haps"]
_ph_freqs    = _ph_data["af"]
_ph_pos      = np.sort(
    np.random.default_rng(0).integers(1, 50_000_000, _ph_data["baf"].size).astype(float)
)

# ===========================================================================
# Fixture group 3 — genome-wide, PGTSim  (3 chroms × m=2000, seeds 700+/800+)
# Contamination added via sim_cell_contamination; MLE recovers model-c ≈ 0.05.
# ===========================================================================

_N_CHROMS = 3
_int_sim = PGTSim()
_int_chrom_data = [
    _int_sim.full_ploidy_sim(m=2000, mix_prop=0.0, std_dev=0.10, seed=700 + i)
    for i in range(_N_CHROMS)
]
_int_baf_list   = [d["baf"]      for d in _int_chrom_data]
_int_mat_list   = [d["mat_haps"] for d in _int_chrom_data]
_int_pat_list   = [d["pat_haps"] for d in _int_chrom_data]
_int_freqs_list = [d["af"]       for d in _int_chrom_data]
_int_pos_list   = [d["pos"]      for d in _int_chrom_data]
_int_cc_baf_list = [
    _int_sim.sim_cell_contamination(
        baf=_int_chrom_data[i]["baf"],
        haps=_int_chrom_data[i]["mat_haps"],
        fraction=0.10,
        seed=800 + i,
    )
    for i in range(_N_CHROMS)
]

# ===========================================================================
# Fixture group 4 — genome-wide, model-consistent  (3 chroms × m=2000)
# BAF drawn directly from the MCC emission so accuracy tests can assert
# abs(c_est - c_true) < 0.04.
# ===========================================================================

def _make_ld_block_data(n=4000, c_true=0.08, std_dev=0.12, block_len=400, seed=11):
    """Synthetic trio data drawn from the phase-aware MCC emission model.

    Alternating LD blocks let the HMM resolve the transmitted haplotype.
    Paternal is all-AA (pg=0) for maximum per-site information.
    """
    rng = np.random.default_rng(seed)
    mat_haps = np.zeros((2, n), dtype=np.int32)
    phase = 0
    for start in range(0, n, block_len):
        end = min(start + block_len, n)
        mat_haps[0, start:end] = phase
        mat_haps[1, start:end] = 1 - phase
        phase = 1 - phase
    pat_haps = np.zeros((2, n), dtype=np.int32)
    pos = np.linspace(1, 150_000_000, n)
    freqs = np.full(n, 0.5)
    m_h = mat_haps[0, :]
    mus = np.where(m_h == 1, 0.5 + c_true / 2, 0.0)
    bafs = np.clip(rng.normal(mus, std_dev), 0.0, 1.0)
    return bafs, mat_haps, pat_haps, freqs, pos, c_true


def _make_ld_block_poc_data(n=4000, c_true=0.08, std_dev=0.12, block_len=400,
                             freq=0.05, seed=11):
    """Synthetic POC data drawn from the phase-aware MCC emission under HWE.

    Low paternal allele frequency (~90 % AA paternal) keeps the contamination
    signal detectable after marginalising over the unobserved paternal genotype.
    """
    rng = np.random.default_rng(seed)
    mat_haps = np.zeros((2, n), dtype=np.int32)
    phase = 0
    for start in range(0, n, block_len):
        end = min(start + block_len, n)
        mat_haps[0, start:end] = phase
        mat_haps[1, start:end] = 1 - phase
        phase = 1 - phase
    pos = np.linspace(1, 150_000_000, n)
    freqs = np.full(n, freq)
    m_h = mat_haps[0, :]
    p0, p1, p2 = (1 - freq) ** 2, 2 * freq * (1 - freq), freq ** 2
    pg = rng.choice([0, 1, 2], size=n, p=[p0, p1, p2])
    mus = np.zeros(n)
    for i in range(n):
        mhi, pgi = int(m_h[i]), int(pg[i])
        if pgi == 0:
            mus[i] = 0.5 + c_true / 2 if mhi == 1 else 0.0
        elif pgi == 1:
            mus[i] = 0.5 if mhi == 1 else rng.choice([c_true / 2, 0.5 - c_true / 2])
        else:
            mus[i] = 1.0 - c_true / 2 if mhi == 1 else 0.5 + c_true / 2
    bafs = np.clip(rng.normal(mus, std_dev), 0.0, 1.0)
    return bafs, mat_haps, freqs, pos, c_true


def _make_genome_ld_trio(n_chroms=3, m=2000, c_true=0.10, std_dev=0.12,
                          block_len=400, seed=11):
    """Multi-chromosome trio data from the phase-aware MCC emission (LD-block)."""
    baf_list, mat_list, pat_list, freqs_list, pos_list = [], [], [], [], []
    for i in range(n_chroms):
        rng = np.random.default_rng(seed + i * 100)
        mat_haps = np.zeros((2, m), dtype=np.int32)
        phase = 0
        for start in range(0, m, block_len):
            end = min(start + block_len, m)
            mat_haps[0, start:end] = phase
            mat_haps[1, start:end] = 1 - phase
            phase = 1 - phase
        pat_haps = np.zeros((2, m), dtype=np.int32)
        pos = np.linspace(1 + i * 200_000_000, (i + 1) * 200_000_000, m)
        freqs = np.full(m, 0.5)
        m_h = mat_haps[0, :]
        mus = np.where(m_h == 1, 0.5 + c_true / 2, 0.0)
        bafs = np.clip(rng.normal(mus, std_dev), 0.0, 1.0)
        baf_list.append(bafs)
        mat_list.append(mat_haps)
        pat_list.append(pat_haps)
        freqs_list.append(freqs)
        pos_list.append(pos)
    return baf_list, mat_list, pat_list, freqs_list, pos_list, c_true


def _make_genome_ld_poc(n_chroms=3, m=2000, c_true=0.10, std_dev=0.12,
                         block_len=400, freq=0.05, seed=11):
    """Multi-chromosome POC data from the phase-aware MCC emission (LD-block)."""
    baf_list, mat_list, freqs_list, pos_list = [], [], [], []
    for i in range(n_chroms):
        rng = np.random.default_rng(seed + i * 100)
        mat_haps = np.zeros((2, m), dtype=np.int32)
        phase = 0
        for start in range(0, m, block_len):
            end = min(start + block_len, m)
            mat_haps[0, start:end] = phase
            mat_haps[1, start:end] = 1 - phase
            phase = 1 - phase
        pos = np.linspace(1 + i * 200_000_000, (i + 1) * 200_000_000, m)
        freqs = np.full(m, freq)
        m_h = mat_haps[0, :]
        p0, p1, p2 = (1 - freq) ** 2, 2 * freq * (1 - freq), freq ** 2
        pg = rng.choice([0, 1, 2], size=m, p=[p0, p1, p2])
        mus = np.zeros(m)
        for j in range(m):
            mhi, pgi = int(m_h[j]), int(pg[j])
            if pgi == 0:
                mus[j] = 0.5 + c_true / 2 if mhi == 1 else 0.0
            elif pgi == 1:
                mus[j] = 0.5 if mhi == 1 else rng.choice([c_true / 2, 0.5 - c_true / 2])
            else:
                mus[j] = 1.0 - c_true / 2 if mhi == 1 else 0.5 + c_true / 2
        bafs = np.clip(rng.normal(mus, std_dev), 0.0, 1.0)
        baf_list.append(bafs)
        mat_list.append(mat_haps)
        freqs_list.append(freqs)
        pos_list.append(pos)
    return baf_list, mat_list, freqs_list, pos_list, c_true


(
    _model_baf_trio, _model_mat_trio, _model_pat_trio,
    _model_freqs_trio, _model_pos_trio, _model_c_trio,
) = _make_genome_ld_trio(seed=11)

(
    _model_baf_poc, _model_mat_poc,
    _model_freqs_poc, _model_pos_poc, _model_c_poc,
) = _make_genome_ld_poc(seed=11)


# ===========================================================================
# Initialisation
# ===========================================================================


def test_init():
    """MccEst initialises without error."""
    _ = MccEst()


# ===========================================================================
# Single-chromosome unphased — loglik, MLE, CI
# ===========================================================================


def test_loglik_trio(
    bafs=np.array([0.6, 0.5]),
    mat_haps=np.array([[0, 0], [0, 0]]),
    pat_haps=np.array([[1, 1], [1, 1]]),
):
    mcc = MccEst()
    ll = mcc.loglik_mcc_trio(bafs=bafs, mat_haps=mat_haps, pat_haps=pat_haps, c=0.1, std_dev=0.1)
    assert ~np.isnan(ll)


def test_loglik_poc(
    bafs=np.array([0.6, 0.5]),
    mat_haps=np.array([[0, 0], [0, 0]]),
    freqs=np.array([0.3, 0.3]),
):
    mcc = MccEst()
    ll = mcc.loglik_mcc_poc(bafs=bafs, mat_haps=mat_haps, freqs=freqs, c=0.1, std_dev=0.1)
    assert ~np.isnan(ll)


def test_mle_trio(
    bafs=np.array([0.6, 0.5]),
    mat_haps=np.array([[0, 0], [0, 0]]),
    pat_haps=np.array([[1, 1], [1, 1]]),
):
    mcc = MccEst()
    c_est, _ = mcc.est_mcc_trio(bafs=bafs, mat_haps=mat_haps, pat_haps=pat_haps)
    assert 0.0 <= c_est <= 0.5


def test_mle_poc(
    bafs=np.array([0.6, 0.5]),
    mat_haps=np.array([[0, 0], [0, 0]]),
    freqs=np.array([0.3, 0.3]),
):
    mcc = MccEst()
    c_est, _ = mcc.est_mcc_poc(bafs=bafs, mat_haps=mat_haps, freqs=freqs)
    assert 0.0 <= c_est <= 0.5


def test_ci_trio(
    bafs=np.array([0.6, 0.5]),
    mat_haps=np.array([[0, 0], [0, 0]]),
    pat_haps=np.array([[1, 1], [1, 1]]),
):
    mcc = MccEst()
    c_est, std_dev = mcc.est_mcc_trio(bafs=bafs, mat_haps=mat_haps, pat_haps=pat_haps)
    lower, x, upper = mcc.mcc_ci_trio(bafs=bafs, mat_haps=mat_haps, pat_haps=pat_haps,
                                       c_hat=c_est, std_dev=std_dev)
    assert x == c_est
    assert lower <= upper


def test_ci_poc(
    bafs=np.array([0.6, 0.5]),
    mat_haps=np.array([[0, 0], [0, 0]]),
    freqs=np.array([0.3, 0.3]),
):
    mcc = MccEst()
    c_est, std_dev = mcc.est_mcc_poc(bafs=bafs, mat_haps=mat_haps, freqs=freqs)
    lower, x, upper = mcc.mcc_ci_poc(bafs=bafs, mat_haps=mat_haps, freqs=freqs,
                                      c_hat=c_est, std_dev=std_dev)
    assert x == c_est
    assert lower <= upper


# ===========================================================================
# Single-chromosome unphased — CI boundary / fallback
# ===========================================================================


def test_ci_poc_lower_fallback():
    """mcc_ci_poc with c_hat=0 falls back to lower_CI=0 without raising."""
    mcc = MccEst()
    lower, x, upper = mcc.mcc_ci_poc(_sc_bafs, _sc_mat, _sc_freq, c_hat=0.0, std_dev=0.1)
    assert lower == 0.0
    assert x == 0.0
    assert upper >= 0.0


def test_ci_poc_upper_fallback():
    """mcc_ci_poc with c_hat=0.5 falls back to upper_CI=0.5 without raising."""
    mcc = MccEst()
    _, x, upper = mcc.mcc_ci_poc(_sc_bafs, _sc_mat, _sc_freq, c_hat=0.5, std_dev=0.1)
    assert upper == 0.5
    assert x == 0.5


def test_ci_trio_lower_fallback():
    """mcc_ci_trio with c_hat=0 falls back to lower_CI=0 without raising."""
    mcc = MccEst()
    lower, _, _ = mcc.mcc_ci_trio(_sc_bafs, _sc_mat, _sc_pat, c_hat=0.0, std_dev=0.1)
    assert lower == 0.0


def test_ci_trio_upper_fallback():
    """mcc_ci_trio with c_hat=0.5 falls back to upper_CI=0.5 without raising."""
    mcc = MccEst()
    _, _, upper = mcc.mcc_ci_trio(_sc_bafs, _sc_mat, _sc_pat, c_hat=0.5, std_dev=0.1)
    assert upper == 0.5


# ===========================================================================
# Single-chromosome unphased — loglik edge cases and input validation
# ===========================================================================


def test_loglik_trio_c_at_boundaries():
    """loglik_mcc_trio is finite at c=0 and c=0.5."""
    mcc = MccEst()
    for c in [0.0, 0.5]:
        assert np.isfinite(mcc.loglik_mcc_trio(_sc_bafs, _sc_mat, _sc_pat, c=c, std_dev=0.1))


def test_loglik_poc_c_at_boundaries():
    """loglik_mcc_poc is finite at c=0 and c=0.5."""
    mcc = MccEst()
    for c in [0.0, 0.5]:
        assert np.isfinite(mcc.loglik_mcc_poc(_sc_bafs, _sc_mat, _sc_freq, c=c, std_dev=0.1))


def test_loglik_trio_all_hom_maternal():
    """loglik_mcc_trio is finite when all maternal sites are homozygous (mg=0 or mg=2)."""
    mcc = MccEst()
    n = 50
    mat_hom = np.ones((2, n), dtype=np.int32)
    pat_het = np.vstack([np.zeros(n, dtype=np.int32), np.ones(n, dtype=np.int32)])
    ll = mcc.loglik_mcc_trio(np.full(n, 0.75), mat_hom, pat_het, c=0.1, std_dev=0.1)
    assert np.isfinite(ll)


def test_loglik_poc_all_hom_maternal():
    """loglik_mcc_poc is finite when all maternal sites are homozygous."""
    mcc = MccEst()
    n = 50
    ll = mcc.loglik_mcc_poc(np.full(n, 0.1), np.zeros((2, n), dtype=np.int32),
                              np.full(n, 0.4), c=0.1, std_dev=0.1)
    assert np.isfinite(ll)


def test_ci_poc_nondefault_alpha():
    """mcc_ci_poc with alpha=0.90 returns a narrower interval than alpha=0.95."""
    mcc = MccEst()
    c_est, std_dev = mcc.est_mcc_poc(_sc_bafs, _sc_mat, _sc_freq)
    lo90, _, hi90 = mcc.mcc_ci_poc(_sc_bafs, _sc_mat, _sc_freq, c_hat=c_est, std_dev=std_dev, alpha=0.90)
    lo95, _, hi95 = mcc.mcc_ci_poc(_sc_bafs, _sc_mat, _sc_freq, c_hat=c_est, std_dev=std_dev, alpha=0.95)
    assert (hi90 - lo90) <= (hi95 - lo95)


def test_ci_trio_nondefault_alpha():
    """mcc_ci_trio with alpha=0.90 returns a narrower interval than alpha=0.95."""
    mcc = MccEst()
    c_est, std_dev = mcc.est_mcc_trio(_sc_bafs, _sc_mat, _sc_pat)
    lo90, _, hi90 = mcc.mcc_ci_trio(_sc_bafs, _sc_mat, _sc_pat, c_hat=c_est, std_dev=std_dev, alpha=0.90)
    lo95, _, hi95 = mcc.mcc_ci_trio(_sc_bafs, _sc_mat, _sc_pat, c_hat=c_est, std_dev=std_dev, alpha=0.95)
    assert (hi90 - lo90) <= (hi95 - lo95)


def test_est_mcc_poc_nondefault_algo():
    """est_mcc_poc with algo='L-BFGS-B' returns a valid estimate."""
    mcc = MccEst()
    c_est, s_est = mcc.est_mcc_poc(_sc_bafs, _sc_mat, _sc_freq, algo="L-BFGS-B")
    assert 0.0 <= c_est <= 0.5
    assert s_est > 0.0


def test_est_mcc_trio_nondefault_algo():
    """est_mcc_trio with algo='L-BFGS-B' returns a valid estimate."""
    mcc = MccEst()
    c_est, s_est = mcc.est_mcc_trio(_sc_bafs, _sc_mat, _sc_pat, algo="L-BFGS-B")
    assert 0.0 <= c_est <= 0.5
    assert s_est > 0.0


def test_loglik_trio_invalid_c_too_large():
    """loglik_mcc_trio raises on c > 0.5."""
    mcc = MccEst()
    try:
        mcc.loglik_mcc_trio(_sc_bafs, _sc_mat, _sc_pat, c=0.6, std_dev=0.1)
        assert False, "Expected AssertionError"
    except AssertionError:
        pass


def test_loglik_trio_invalid_std_dev():
    """loglik_mcc_trio raises on std_dev=0."""
    mcc = MccEst()
    try:
        mcc.loglik_mcc_trio(_sc_bafs, _sc_mat, _sc_pat, c=0.1, std_dev=0.0)
        assert False, "Expected AssertionError"
    except AssertionError:
        pass


def test_loglik_poc_invalid_freq():
    """loglik_mcc_poc raises on freq > 1."""
    mcc = MccEst()
    try:
        mcc.loglik_mcc_poc(_sc_bafs, _sc_mat, np.full(_sc_bafs.size, 1.5), c=0.1, std_dev=0.1)
        assert False, "Expected AssertionError"
    except AssertionError:
        pass


# ===========================================================================
# Single-chromosome phased — loglik
# ===========================================================================


def test_loglik_phased_trio_finite():
    """loglik_mcc_phased_trio returns a finite scalar."""
    mcc = MccEst()
    ll = mcc.loglik_mcc_phased_trio(
        np.array([0.6, 0.4]),
        np.array([[0, 1], [1, 0]], dtype=np.int32),
        np.array([[1, 0], [0, 1]], dtype=np.int32),
        np.array([1000.0, 2000.0]),
        c=0.1, std_dev=0.1,
    )
    assert np.isfinite(ll)


def test_loglik_phased_poc_finite():
    """loglik_mcc_phased_poc returns a finite scalar."""
    mcc = MccEst()
    ll = mcc.loglik_mcc_phased_poc(
        np.array([0.6, 0.4]),
        np.array([[0, 1], [1, 0]], dtype=np.int32),
        np.array([0.3, 0.4]),
        np.array([1000.0, 2000.0]),
        c=0.1, std_dev=0.1,
    )
    assert np.isfinite(ll)


def test_loglik_phased_trio_all_hom_matches_unphased():
    """With all homozygous maternal sites phased and unphased likelihoods agree."""
    mcc = MccEst()
    mat_haps = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.int32)
    pat_haps = np.array([[1, 0, 1, 0], [1, 0, 1, 0]], dtype=np.int32)
    bafs = np.array([0.5, 0.1, 0.9, 0.5])
    pos  = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    ll_p = mcc.loglik_mcc_phased_trio(bafs, mat_haps, pat_haps, pos, c=0.1, std_dev=0.1)
    ll_u = mcc.loglik_mcc_trio(bafs, mat_haps, pat_haps, c=0.1, std_dev=0.1)
    assert np.isclose(ll_p, ll_u, atol=1e-5)


def test_loglik_phased_trio_increases_with_contamination():
    """Phased trio loglik is higher at c_true than c=0 on model-consistent data."""
    mcc = MccEst()
    bafs, mat_haps, pat_haps, _, pos, c_true = _make_ld_block_data()
    assert mcc.loglik_mcc_phased_trio(bafs, mat_haps, pat_haps, pos, c=c_true) > \
           mcc.loglik_mcc_phased_trio(bafs, mat_haps, pat_haps, pos, c=0.0)


def test_loglik_phased_poc_increases_with_contamination():
    """Phased POC loglik is higher at c_true than c=0 on model-consistent data."""
    mcc = MccEst()
    bafs, mat_haps, freqs, pos, c_true = _make_ld_block_poc_data()
    assert mcc.loglik_mcc_phased_poc(bafs, mat_haps, freqs, pos, c=c_true) > \
           mcc.loglik_mcc_phased_poc(bafs, mat_haps, freqs, pos, c=0.0)


def test_loglik_phased_trio_c_at_boundaries():
    """loglik_mcc_phased_trio is finite at c=0 and c=0.5."""
    mcc = MccEst()
    for c in [0.0, 0.5]:
        assert np.isfinite(mcc.loglik_mcc_phased_trio(_ph_cc_bafs, _ph_mat_haps, _ph_pat_haps, _ph_pos, c=c))


def test_loglik_phased_poc_c_at_boundaries():
    """loglik_mcc_phased_poc is finite at c=0 and c=0.5."""
    mcc = MccEst()
    for c in [0.0, 0.5]:
        assert np.isfinite(mcc.loglik_mcc_phased_poc(_ph_cc_bafs, _ph_mat_haps, _ph_freqs, _ph_pos, c=c))


def test_loglik_phased_trio_nondefault_r():
    """loglik_mcc_phased_trio changes with the recombination rate r."""
    mcc = MccEst()
    bafs, mat_haps, pat_haps, _, pos, _ = _make_ld_block_data()
    ll_lo = mcc.loglik_mcc_phased_trio(bafs, mat_haps, pat_haps, pos, c=0.05, r=1e-9)
    ll_hi = mcc.loglik_mcc_phased_trio(bafs, mat_haps, pat_haps, pos, c=0.05, r=1e-6)
    assert ll_lo != ll_hi


def test_loglik_phased_poc_nondefault_r():
    """loglik_mcc_phased_poc changes with the recombination rate r."""
    mcc = MccEst()
    bafs, mat_haps, freqs, pos, _ = _make_ld_block_poc_data()
    ll_lo = mcc.loglik_mcc_phased_poc(bafs, mat_haps, freqs, pos, c=0.05, r=1e-9)
    ll_hi = mcc.loglik_mcc_phased_poc(bafs, mat_haps, freqs, pos, c=0.05, r=1e-6)
    assert ll_lo != ll_hi


# ===========================================================================
# Single-chromosome phased — MLE
# ===========================================================================


def test_mle_phased_trio_bounds():
    """Phased trio MLE c is in [0, 0.5] and sigma > 0."""
    mcc = MccEst()
    c_est, s_est = mcc.est_mcc_phased_trio(_ph_cc_bafs, _ph_mat_haps, _ph_pat_haps, _ph_pos)
    assert 0.0 <= c_est <= 0.5
    assert s_est > 0.0


def test_mle_phased_poc_bounds():
    """Phased POC MLE c is in [0, 0.5] and sigma > 0."""
    mcc = MccEst()
    c_est, s_est = mcc.est_mcc_phased_poc(_ph_cc_bafs, _ph_mat_haps, _ph_freqs, _ph_pos)
    assert 0.0 <= c_est <= 0.5
    assert s_est > 0.0


def test_mle_phased_trio_accuracy():
    """Phased trio MLE recovers c within 0.04 on LD-block model-consistent data."""
    mcc = MccEst()
    bafs, mat_haps, pat_haps, _, pos, c_true = _make_ld_block_data(c_true=0.10)
    c_est, _ = mcc.est_mcc_phased_trio(bafs, mat_haps, pat_haps, pos)
    assert abs(c_est - c_true) < 0.04


def test_mle_phased_poc_accuracy():
    """Phased POC MLE recovers c within 0.04 on LD-block model-consistent data."""
    mcc = MccEst()
    bafs, mat_haps, freqs, pos, c_true = _make_ld_block_poc_data(c_true=0.10)
    c_est, _ = mcc.est_mcc_phased_poc(bafs, mat_haps, freqs, pos)
    assert abs(c_est - c_true) < 0.04


# ===========================================================================
# Single-chromosome phased — CI
# ===========================================================================


def test_ci_phased_trio_wellformed():
    """Phased trio CI satisfies lower <= estimate <= upper."""
    mcc = MccEst()
    c_est, s_est = mcc.est_mcc_phased_trio(_ph_cc_bafs, _ph_mat_haps, _ph_pat_haps, _ph_pos)
    lower, x, upper = mcc.mcc_ci_phased_trio(_ph_cc_bafs, _ph_mat_haps, _ph_pat_haps, _ph_pos,
                                               c_hat=c_est, std_dev=s_est)
    assert lower <= x <= upper


def test_ci_phased_poc_wellformed():
    """Phased POC CI satisfies lower <= estimate <= upper."""
    mcc = MccEst()
    c_est, s_est = mcc.est_mcc_phased_poc(_ph_cc_bafs, _ph_mat_haps, _ph_freqs, _ph_pos)
    lower, x, upper = mcc.mcc_ci_phased_poc(_ph_cc_bafs, _ph_mat_haps, _ph_freqs, _ph_pos,
                                              c_hat=c_est, std_dev=s_est)
    assert lower <= x <= upper


def test_ci_phased_trio_covers_truth():
    """Phased trio 95% CI covers c_true on LD-block model-consistent data."""
    mcc = MccEst()
    bafs, mat_haps, pat_haps, _, pos, c_true = _make_ld_block_data(c_true=0.10)
    c_est, s_est = mcc.est_mcc_phased_trio(bafs, mat_haps, pat_haps, pos)
    lower, _, upper = mcc.mcc_ci_phased_trio(bafs, mat_haps, pat_haps, pos,
                                              c_hat=c_est, std_dev=s_est)
    assert lower <= c_true <= upper


def test_ci_phased_poc_covers_truth():
    """Phased POC 95% CI covers c_true on LD-block model-consistent data."""
    mcc = MccEst()
    bafs, mat_haps, freqs, pos, c_true = _make_ld_block_poc_data(c_true=0.10)
    c_est, s_est = mcc.est_mcc_phased_poc(bafs, mat_haps, freqs, pos)
    lower, _, upper = mcc.mcc_ci_phased_poc(bafs, mat_haps, freqs, pos,
                                             c_hat=c_est, std_dev=s_est)
    assert lower <= c_true <= upper


def test_ci_phased_trio_lower_fallback():
    """mcc_ci_phased_trio with c_hat=0 falls back to lower_CI=0 without raising."""
    mcc = MccEst()
    lower, x, upper = mcc.mcc_ci_phased_trio(_ph_cc_bafs, _ph_mat_haps, _ph_pat_haps, _ph_pos,
                                               c_hat=0.0, std_dev=0.1)
    assert lower == 0.0
    assert x == 0.0
    assert upper >= 0.0


def test_ci_phased_trio_upper_fallback():
    """mcc_ci_phased_trio with c_hat=0.5 falls back to upper_CI=0.5 without raising."""
    mcc = MccEst()
    _, x, upper = mcc.mcc_ci_phased_trio(_ph_cc_bafs, _ph_mat_haps, _ph_pat_haps, _ph_pos,
                                           c_hat=0.5, std_dev=0.1)
    assert upper == 0.5
    assert x == 0.5


def test_ci_phased_poc_lower_fallback():
    """mcc_ci_phased_poc with c_hat=0 falls back to lower_CI=0 without raising."""
    mcc = MccEst()
    lower, x, _ = mcc.mcc_ci_phased_poc(_ph_cc_bafs, _ph_mat_haps, _ph_freqs, _ph_pos,
                                          c_hat=0.0, std_dev=0.1)
    assert lower == 0.0
    assert x == 0.0


def test_ci_phased_poc_upper_fallback():
    """mcc_ci_phased_poc with c_hat=0.5 falls back to upper_CI=0.5 without raising."""
    mcc = MccEst()
    _, x, upper = mcc.mcc_ci_phased_poc(_ph_cc_bafs, _ph_mat_haps, _ph_freqs, _ph_pos,
                                          c_hat=0.5, std_dev=0.1)
    assert upper == 0.5
    assert x == 0.5


def test_ci_phased_poc_nondefault_alpha():
    """mcc_ci_phased_poc with alpha=0.90 is narrower than alpha=0.95."""
    mcc = MccEst()
    c_est, s_est = mcc.est_mcc_phased_poc(_ph_cc_bafs, _ph_mat_haps, _ph_freqs, _ph_pos)
    lo90, _, hi90 = mcc.mcc_ci_phased_poc(_ph_cc_bafs, _ph_mat_haps, _ph_freqs, _ph_pos,
                                           c_hat=c_est, std_dev=s_est, alpha=0.90)
    lo95, _, hi95 = mcc.mcc_ci_phased_poc(_ph_cc_bafs, _ph_mat_haps, _ph_freqs, _ph_pos,
                                           c_hat=c_est, std_dev=s_est, alpha=0.95)
    assert (hi90 - lo90) <= (hi95 - lo95)


def test_ci_phased_trio_nondefault_alpha():
    """mcc_ci_phased_trio with alpha=0.90 is narrower than alpha=0.95."""
    mcc = MccEst()
    c_est, s_est = mcc.est_mcc_phased_trio(_ph_cc_bafs, _ph_mat_haps, _ph_pat_haps, _ph_pos)
    lo90, _, hi90 = mcc.mcc_ci_phased_trio(_ph_cc_bafs, _ph_mat_haps, _ph_pat_haps, _ph_pos,
                                            c_hat=c_est, std_dev=s_est, alpha=0.90)
    lo95, _, hi95 = mcc.mcc_ci_phased_trio(_ph_cc_bafs, _ph_mat_haps, _ph_pat_haps, _ph_pos,
                                            c_hat=c_est, std_dev=s_est, alpha=0.95)
    assert (hi90 - lo90) <= (hi95 - lo95)


def test_phased_ci_narrower_than_unphased_trio():
    """On LD-structured data the phased trio CI is narrower than the unphased CI."""
    mcc = MccEst()
    bafs, mat_haps, pat_haps, _, pos, _ = _make_ld_block_data(c_true=0.08, std_dev=0.12, seed=11)
    c_p, s_p = mcc.est_mcc_phased_trio(bafs, mat_haps, pat_haps, pos)
    lo_p, _, hi_p = mcc.mcc_ci_phased_trio(bafs, mat_haps, pat_haps, pos, c_hat=c_p, std_dev=s_p)
    c_u, s_u = mcc.est_mcc_trio(bafs, mat_haps, pat_haps)
    lo_u, _, hi_u = mcc.mcc_ci_trio(bafs, mat_haps, pat_haps, c_hat=c_u, std_dev=s_u)
    assert (hi_p - lo_p) < (hi_u - lo_u), (
        f"phased ({hi_p - lo_p:.5f}) >= unphased ({hi_u - lo_u:.5f})"
    )


def test_phased_ci_narrower_than_unphased_poc():
    """On LD-structured data the phased POC CI is narrower than the unphased CI."""
    mcc = MccEst()
    bafs, mat_haps, freqs, pos, _ = _make_ld_block_poc_data(
        c_true=0.08, std_dev=0.12, freq=0.05, seed=11
    )
    c_pp, s_pp = mcc.est_mcc_phased_poc(bafs, mat_haps, freqs, pos)
    lo_pp, _, hi_pp = mcc.mcc_ci_phased_poc(bafs, mat_haps, freqs, pos, c_hat=c_pp, std_dev=s_pp)
    c_up, s_up = mcc.est_mcc_poc(bafs, mat_haps, freqs)
    lo_up, _, hi_up = mcc.mcc_ci_poc(bafs, mat_haps, freqs, c_hat=c_up, std_dev=s_up)
    assert (hi_pp - lo_pp) < (hi_up - lo_up), (
        f"phased ({hi_pp - lo_pp:.5f}) >= unphased ({hi_up - lo_up:.5f})"
    )


# ===========================================================================
# Single-chromosome phased — input validation
# ===========================================================================


def test_loglik_phased_trio_invalid_c():
    """loglik_mcc_phased_trio raises on c > 0.5."""
    mcc = MccEst()
    try:
        mcc.loglik_mcc_phased_trio(_ph_cc_bafs, _ph_mat_haps, _ph_pat_haps, _ph_pos, c=0.6)
        assert False, "Expected AssertionError"
    except AssertionError:
        pass


def test_loglik_phased_poc_invalid_r():
    """loglik_mcc_phased_poc raises on r=0."""
    mcc = MccEst()
    try:
        mcc.loglik_mcc_phased_poc(_ph_cc_bafs, _ph_mat_haps, _ph_freqs, _ph_pos, r=0.0)
        assert False, "Expected AssertionError"
    except AssertionError:
        pass


def test_loglik_phased_trio_unsorted_pos():
    """loglik_mcc_phased_trio raises when pos is not strictly increasing."""
    mcc = MccEst()
    bad_pos = _ph_pos.copy()
    bad_pos[10] = bad_pos[9]
    try:
        mcc.loglik_mcc_phased_trio(_ph_cc_bafs, _ph_mat_haps, _ph_pat_haps, bad_pos)
        assert False, "Expected AssertionError"
    except AssertionError:
        pass


# ===========================================================================
# Genome-wide loglik — finite and additivity (unphased + phased)
# ===========================================================================


def test_loglik_genome_poc_finite():
    """loglik_mcc_genome_poc returns a finite scalar."""
    mcc = MccEst()
    assert np.isfinite(
        mcc.loglik_mcc_genome_poc(_int_baf_list, _int_mat_list, _int_freqs_list, c=0.1, std_dev=0.1)
    )


def test_loglik_genome_trio_finite():
    """loglik_mcc_genome_trio returns a finite scalar."""
    mcc = MccEst()
    assert np.isfinite(
        mcc.loglik_mcc_genome_trio(_int_baf_list, _int_mat_list, _int_pat_list, c=0.1, std_dev=0.1)
    )


def test_loglik_genome_phased_poc_finite():
    """loglik_mcc_genome_phased_poc returns a finite scalar."""
    mcc = MccEst()
    assert np.isfinite(
        mcc.loglik_mcc_genome_phased_poc(
            _int_baf_list, _int_mat_list, _int_freqs_list, _int_pos_list, c=0.1, std_dev=0.1
        )
    )


def test_loglik_genome_phased_trio_finite():
    """loglik_mcc_genome_phased_trio returns a finite scalar."""
    mcc = MccEst()
    assert np.isfinite(
        mcc.loglik_mcc_genome_phased_trio(
            _int_baf_list, _int_mat_list, _int_pat_list, _int_pos_list, c=0.1, std_dev=0.1
        )
    )


def test_loglik_genome_poc_equals_sum():
    """loglik_mcc_genome_poc equals the sum of per-chromosome POC logliks."""
    mcc = MccEst()
    c, std_dev = 0.08, 0.12
    ll_g = mcc.loglik_mcc_genome_poc(_int_baf_list, _int_mat_list, _int_freqs_list, c=c, std_dev=std_dev)
    ll_s = sum(mcc.loglik_mcc_poc(b, m, f, c=c, std_dev=std_dev)
               for b, m, f in zip(_int_baf_list, _int_mat_list, _int_freqs_list))
    assert np.isclose(ll_g, ll_s)


def test_loglik_genome_trio_equals_sum():
    """loglik_mcc_genome_trio equals the sum of per-chromosome trio logliks."""
    mcc = MccEst()
    c, std_dev = 0.08, 0.12
    ll_g = mcc.loglik_mcc_genome_trio(_int_baf_list, _int_mat_list, _int_pat_list, c=c, std_dev=std_dev)
    ll_s = sum(mcc.loglik_mcc_trio(b, m, p, c=c, std_dev=std_dev)
               for b, m, p in zip(_int_baf_list, _int_mat_list, _int_pat_list))
    assert np.isclose(ll_g, ll_s)


def test_loglik_genome_phased_poc_equals_sum():
    """loglik_mcc_genome_phased_poc equals the sum of per-chromosome phased POC logliks."""
    mcc = MccEst()
    c, std_dev = 0.08, 0.12
    ll_g = mcc.loglik_mcc_genome_phased_poc(
        _int_baf_list, _int_mat_list, _int_freqs_list, _int_pos_list, c=c, std_dev=std_dev
    )
    ll_s = sum(
        mcc.loglik_mcc_phased_poc(b, m, f, pos, c=c, std_dev=std_dev)
        for b, m, f, pos in zip(_int_baf_list, _int_mat_list, _int_freqs_list, _int_pos_list)
    )
    assert np.isclose(ll_g, ll_s)


def test_loglik_genome_phased_trio_equals_sum():
    """loglik_mcc_genome_phased_trio equals the sum of per-chromosome phased trio logliks."""
    mcc = MccEst()
    c, std_dev = 0.08, 0.12
    ll_g = mcc.loglik_mcc_genome_phased_trio(
        _int_baf_list, _int_mat_list, _int_pat_list, _int_pos_list, c=c, std_dev=std_dev
    )
    ll_s = sum(
        mcc.loglik_mcc_phased_trio(b, m, p, pos, c=c, std_dev=std_dev)
        for b, m, p, pos in zip(_int_baf_list, _int_mat_list, _int_pat_list, _int_pos_list)
    )
    assert np.isclose(ll_g, ll_s)


# ===========================================================================
# Genome-wide MLE — bounds and per-chrom length/bounds
# ===========================================================================


def test_est_mcc_genome_poc_bounds():
    """est_mcc_genome_poc returns c in [0, 0.5] and sigma > 0."""
    mcc = MccEst()
    c_est, s_est = mcc.est_mcc_genome_poc(_int_cc_baf_list, _int_mat_list, _int_freqs_list)
    assert 0.0 <= c_est <= 0.5 and s_est > 0.0


def test_est_mcc_genome_trio_bounds():
    """est_mcc_genome_trio returns c in [0, 0.5] and sigma > 0."""
    mcc = MccEst()
    c_est, s_est = mcc.est_mcc_genome_trio(_int_cc_baf_list, _int_mat_list, _int_pat_list)
    assert 0.0 <= c_est <= 0.5 and s_est > 0.0


def test_est_mcc_genome_phased_poc_bounds():
    """est_mcc_genome_phased_poc returns c in [0, 0.5] and sigma > 0."""
    mcc = MccEst()
    c_est, s_est = mcc.est_mcc_genome_phased_poc(
        _int_cc_baf_list, _int_mat_list, _int_freqs_list, _int_pos_list
    )
    assert 0.0 <= c_est <= 0.5 and s_est > 0.0


def test_est_mcc_genome_phased_trio_bounds():
    """est_mcc_genome_phased_trio returns c in [0, 0.5] and sigma > 0."""
    mcc = MccEst()
    c_est, s_est = mcc.est_mcc_genome_phased_trio(
        _int_cc_baf_list, _int_mat_list, _int_pat_list, _int_pos_list
    )
    assert 0.0 <= c_est <= 0.5 and s_est > 0.0


def test_est_mcc_per_chrom_poc_length_and_bounds():
    """est_mcc_per_chrom_poc returns one valid (c, sigma) per chromosome."""
    mcc = MccEst()
    results = mcc.est_mcc_per_chrom_poc(_int_cc_baf_list, _int_mat_list, _int_freqs_list)
    assert len(results) == _N_CHROMS
    for c_est, s_est in results:
        assert 0.0 <= c_est <= 0.5 and s_est > 0.0


def test_est_mcc_per_chrom_trio_length_and_bounds():
    """est_mcc_per_chrom_trio returns one valid (c, sigma) per chromosome."""
    mcc = MccEst()
    results = mcc.est_mcc_per_chrom_trio(_int_cc_baf_list, _int_mat_list, _int_pat_list)
    assert len(results) == _N_CHROMS
    for c_est, s_est in results:
        assert 0.0 <= c_est <= 0.5 and s_est > 0.0


def test_est_mcc_per_chrom_phased_poc_length_and_bounds():
    """est_mcc_per_chrom_phased_poc returns one valid (c, sigma) per chromosome."""
    mcc = MccEst()
    results = mcc.est_mcc_per_chrom_phased_poc(
        _int_cc_baf_list, _int_mat_list, _int_freqs_list, _int_pos_list
    )
    assert len(results) == _N_CHROMS
    for c_est, s_est in results:
        assert 0.0 <= c_est <= 0.5 and s_est > 0.0


def test_est_mcc_per_chrom_phased_trio_length_and_bounds():
    """est_mcc_per_chrom_phased_trio returns one valid (c, sigma) per chromosome."""
    mcc = MccEst()
    results = mcc.est_mcc_per_chrom_phased_trio(
        _int_cc_baf_list, _int_mat_list, _int_pat_list, _int_pos_list
    )
    assert len(results) == _N_CHROMS
    for c_est, s_est in results:
        assert 0.0 <= c_est <= 0.5 and s_est > 0.0


# ===========================================================================
# Integration — contamination detection and phased vs unphased LRT comparison
# Uses PGTSim data; tests MLE-based LRT (MLE > 0 and LRT at MLE > 0).
# ===========================================================================


def test_genome_trio_detects_contamination():
    """Unphased trio genome MLE is > 0.01 on contaminated PGTSim data."""
    mcc = MccEst()
    c_hat, _ = mcc.est_mcc_genome_trio(_int_cc_baf_list, _int_mat_list, _int_pat_list)
    assert c_hat > 0.01, f"c_hat = {c_hat:.4f}"


def test_genome_poc_detects_contamination():
    """Unphased POC genome MLE is > 0.01 on contaminated PGTSim data."""
    mcc = MccEst()
    c_hat, _ = mcc.est_mcc_genome_poc(_int_cc_baf_list, _int_mat_list, _int_freqs_list)
    assert c_hat > 0.01, f"c_hat = {c_hat:.4f}"


def test_genome_phased_trio_detects_contamination():
    """Phase-aware trio genome MLE is > 0.01 on contaminated PGTSim data."""
    mcc = MccEst()
    c_hat, _ = mcc.est_mcc_genome_phased_trio(
        _int_cc_baf_list, _int_mat_list, _int_pat_list, _int_pos_list
    )
    assert c_hat > 0.01, f"c_hat = {c_hat:.4f}"


def test_genome_phased_poc_detects_contamination():
    """Phase-aware POC genome MLE is > 0.01 on contaminated PGTSim data."""
    mcc = MccEst()
    c_hat, _ = mcc.est_mcc_genome_phased_poc(
        _int_cc_baf_list, _int_mat_list, _int_freqs_list, _int_pos_list
    )
    assert c_hat > 0.01, f"c_hat = {c_hat:.4f}"


def test_phased_unphased_lrt_both_positive_trio():
    """Phased and unphased trio LRTs are both positive on contaminated PGTSim data."""
    mcc = MccEst()
    c_u, s_u = mcc.est_mcc_genome_trio(_int_cc_baf_list, _int_mat_list, _int_pat_list)
    lrt_u = 2 * (
        mcc.loglik_mcc_genome_trio(_int_cc_baf_list, _int_mat_list, _int_pat_list, c=c_u, std_dev=s_u)
        - mcc.loglik_mcc_genome_trio(_int_cc_baf_list, _int_mat_list, _int_pat_list, c=0.0, std_dev=s_u)
    )
    c_p, s_p = mcc.est_mcc_genome_phased_trio(
        _int_cc_baf_list, _int_mat_list, _int_pat_list, _int_pos_list
    )
    lrt_p = 2 * (
        mcc.loglik_mcc_genome_phased_trio(
            _int_cc_baf_list, _int_mat_list, _int_pat_list, _int_pos_list, c=c_p, std_dev=s_p
        )
        - mcc.loglik_mcc_genome_phased_trio(
            _int_cc_baf_list, _int_mat_list, _int_pat_list, _int_pos_list, c=0.0, std_dev=s_p
        )
    )
    assert lrt_u > 0, f"Unphased trio LRT = {lrt_u:.2f}"
    assert lrt_p > 0, f"Phased trio LRT = {lrt_p:.2f}"
    assert abs(c_u - c_p) < 0.05, f"phased={c_p:.3f} unphased={c_u:.3f} differ > 0.05"


def test_phased_unphased_lrt_both_positive_poc():
    """Phased and unphased POC LRTs are both positive on contaminated PGTSim data."""
    mcc = MccEst()
    c_u, s_u = mcc.est_mcc_genome_poc(_int_cc_baf_list, _int_mat_list, _int_freqs_list)
    lrt_u = 2 * (
        mcc.loglik_mcc_genome_poc(_int_cc_baf_list, _int_mat_list, _int_freqs_list, c=c_u, std_dev=s_u)
        - mcc.loglik_mcc_genome_poc(_int_cc_baf_list, _int_mat_list, _int_freqs_list, c=0.0, std_dev=s_u)
    )
    c_p, s_p = mcc.est_mcc_genome_phased_poc(
        _int_cc_baf_list, _int_mat_list, _int_freqs_list, _int_pos_list
    )
    lrt_p = 2 * (
        mcc.loglik_mcc_genome_phased_poc(
            _int_cc_baf_list, _int_mat_list, _int_freqs_list, _int_pos_list, c=c_p, std_dev=s_p
        )
        - mcc.loglik_mcc_genome_phased_poc(
            _int_cc_baf_list, _int_mat_list, _int_freqs_list, _int_pos_list, c=0.0, std_dev=s_p
        )
    )
    assert lrt_u > 0, f"Unphased POC LRT = {lrt_u:.2f}"
    assert lrt_p > 0, f"Phased POC LRT = {lrt_p:.2f}"
    assert abs(c_u - c_p) < 0.05, f"phased={c_p:.3f} unphased={c_u:.3f} differ > 0.05"


# ===========================================================================
# Integration — MLE accuracy on model-consistent data (all four variants)
# ===========================================================================


def test_genome_mle_trio_accuracy():
    """Genome-wide unphased trio MLE recovers c within 0.04 on model-consistent data."""
    mcc = MccEst()
    c_est, _ = mcc.est_mcc_genome_trio(_model_baf_trio, _model_mat_trio, _model_pat_trio)
    assert abs(c_est - _model_c_trio) < 0.04, f"|{c_est:.3f} - {_model_c_trio}| >= 0.04"


def test_genome_mle_poc_accuracy():
    """Genome-wide unphased POC MLE recovers c within 0.04 on model-consistent data."""
    mcc = MccEst()
    c_est, _ = mcc.est_mcc_genome_poc(_model_baf_poc, _model_mat_poc, _model_freqs_poc)
    assert abs(c_est - _model_c_poc) < 0.04, f"|{c_est:.3f} - {_model_c_poc}| >= 0.04"


def test_genome_phased_mle_trio_accuracy():
    """Genome-wide phased trio MLE recovers c within 0.04 on model-consistent data."""
    mcc = MccEst()
    c_est, _ = mcc.est_mcc_genome_phased_trio(
        _model_baf_trio, _model_mat_trio, _model_pat_trio, _model_pos_trio
    )
    assert abs(c_est - _model_c_trio) < 0.04, f"|{c_est:.3f} - {_model_c_trio}| >= 0.04"


def test_genome_phased_mle_poc_accuracy():
    """Genome-wide phased POC MLE recovers c within 0.04 on model-consistent data."""
    mcc = MccEst()
    c_est, _ = mcc.est_mcc_genome_phased_poc(
        _model_baf_poc, _model_mat_poc, _model_freqs_poc, _model_pos_poc
    )
    assert abs(c_est - _model_c_poc) < 0.04, f"|{c_est:.3f} - {_model_c_poc}| >= 0.04"


# ===========================================================================
# Integration — per-chromosome consistency with genome-wide MLE
# ===========================================================================


def test_per_chrom_trio_consistent_with_genome():
    """Per-chrom unphased trio estimates are within 0.05 of the genome-wide MLE."""
    mcc = MccEst()
    c_g, _ = mcc.est_mcc_genome_trio(_int_cc_baf_list, _int_mat_list, _int_pat_list)
    for i, (c_chr, _) in enumerate(
        mcc.est_mcc_per_chrom_trio(_int_cc_baf_list, _int_mat_list, _int_pat_list)
    ):
        assert abs(c_chr - c_g) < 0.05, f"chrom {i}: |{c_chr:.3f} - {c_g:.3f}| >= 0.05"


def test_per_chrom_poc_consistent_with_genome():
    """Per-chrom unphased POC estimates are within 0.05 of the genome-wide MLE."""
    mcc = MccEst()
    c_g, _ = mcc.est_mcc_genome_poc(_int_cc_baf_list, _int_mat_list, _int_freqs_list)
    for i, (c_chr, _) in enumerate(
        mcc.est_mcc_per_chrom_poc(_int_cc_baf_list, _int_mat_list, _int_freqs_list)
    ):
        assert abs(c_chr - c_g) < 0.05, f"chrom {i}: |{c_chr:.3f} - {c_g:.3f}| >= 0.05"


def test_per_chrom_phased_trio_consistent_with_genome():
    """Per-chrom phased trio estimates are within 0.05 of the genome-wide MLE."""
    mcc = MccEst()
    c_g, _ = mcc.est_mcc_genome_phased_trio(
        _int_cc_baf_list, _int_mat_list, _int_pat_list, _int_pos_list
    )
    for i, (c_chr, _) in enumerate(
        mcc.est_mcc_per_chrom_phased_trio(
            _int_cc_baf_list, _int_mat_list, _int_pat_list, _int_pos_list
        )
    ):
        assert abs(c_chr - c_g) < 0.05, f"chrom {i}: |{c_chr:.3f} - {c_g:.3f}| >= 0.05"


# ===========================================================================
# Concatenation equivalence — guards the unphased genome-wide loglik optimisation
# ===========================================================================


def test_genome_poc_concat_equals_sum():
    """loglik_mcc_genome_poc (concat impl) equals the naive per-chrom sum."""
    mcc = MccEst()
    for c, std_dev in [(0.0, 0.1), (0.05, 0.1), (0.10, 0.15)]:
        ll_g = mcc.loglik_mcc_genome_poc(_int_baf_list, _int_mat_list, _int_freqs_list,
                                          c=c, std_dev=std_dev)
        ll_s = sum(mcc.loglik_mcc_poc(b, m, f, c=c, std_dev=std_dev)
                   for b, m, f in zip(_int_baf_list, _int_mat_list, _int_freqs_list))
        assert np.isclose(ll_g, ll_s, rtol=1e-10), f"c={c}: {ll_g:.6f} != {ll_s:.6f}"


def test_genome_trio_concat_equals_sum():
    """loglik_mcc_genome_trio (concat impl) equals the naive per-chrom sum."""
    mcc = MccEst()
    for c, std_dev in [(0.0, 0.1), (0.05, 0.1), (0.10, 0.15)]:
        ll_g = mcc.loglik_mcc_genome_trio(_int_baf_list, _int_mat_list, _int_pat_list,
                                           c=c, std_dev=std_dev)
        ll_s = sum(mcc.loglik_mcc_trio(b, m, p, c=c, std_dev=std_dev)
                   for b, m, p in zip(_int_baf_list, _int_mat_list, _int_pat_list))
        assert np.isclose(ll_g, ll_s, rtol=1e-10), f"c={c}: {ll_g:.6f} != {ll_s:.6f}"


# ===========================================================================
# Realistic end-to-end CI test (concatenated multi-chromosome data)
# ===========================================================================


def test_realistic_ci_trio(m=10000, n=5, c=0.1):
    """Simulate n realistic chromosomes, estimate contamination, and verify CI shape."""
    pgt_sim = PGTSim()
    data = [pgt_sim.full_ploidy_sim(m=m, mix_prop=0.0, std_dev=0.1, seed=i + 1)
            for i in range(n)]
    cc_bafs = np.hstack([
        pgt_sim.sim_cell_contamination(baf=data[i]["baf"], haps=data[i]["mat_haps"],
                                        fraction=c, seed=i + 1)
        for i in range(n)
    ])
    mat_haps = np.hstack([data[i]["mat_haps"] for i in range(n)])
    pat_haps = np.hstack([data[i]["pat_haps"] for i in range(n)])
    mcc = MccEst()
    c_est, std_dev = mcc.est_mcc_trio(bafs=cc_bafs, mat_haps=mat_haps, pat_haps=pat_haps)
    lower, x, upper = mcc.mcc_ci_trio(bafs=cc_bafs, mat_haps=mat_haps, pat_haps=pat_haps,
                                       c_hat=c_est, std_dev=std_dev)
    assert x == c_est
    assert lower <= upper
