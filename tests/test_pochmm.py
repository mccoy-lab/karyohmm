"""Test suite for karyoHMM PocHMM."""

from karyohmm import PGTSim, PocHMM
import numpy as np
import pytest


# --- Generating test data for applications in the PocHMM (duo) setting --- #
pgt_sim = PGTSim()
data_disomy = pgt_sim.full_ploidy_sim(m=1000, mix_prop=0.7, std_dev=0.1, seed=42)
data_nullisomy = pgt_sim.full_ploidy_sim(
    m=1000, ploidy=0, mat_skew=1.0, mix_prop=0.7, std_dev=0.1, seed=42
)
# Trisomy: maternal-origin (3m) and paternal-origin (3p)
data_mat_trisomy = pgt_sim.full_ploidy_sim(
    m=1000, ploidy=3, mat_skew=1.0, mix_prop=0.7, std_dev=0.1, seed=42
)
data_pat_trisomy = pgt_sim.full_ploidy_sim(
    m=1000, ploidy=3, mat_skew=0.0, mix_prop=0.7, std_dev=0.1, seed=42
)
# Monosomy: maternal-origin (1p: maternal chrom lost, paternal survives) and
#           paternal-origin (1m: paternal chrom lost, maternal survives)
data_mat_origin_mono = pgt_sim.full_ploidy_sim(
    m=1000, ploidy=1, mat_skew=1.0, mix_prop=0.7, std_dev=0.1, seed=42
)
data_pat_origin_mono = pgt_sim.full_ploidy_sim(
    m=1000, ploidy=1, mat_skew=0.0, mix_prop=0.7, std_dev=0.1, seed=42
)

# Convenience alias kept for parametrize tests that cover all karyotypes
_all_data = [
    data_disomy,
    data_nullisomy,
    data_mat_trisomy,
    data_pat_trisomy,
    data_mat_origin_mono,
    data_pat_origin_mono,
]

# --- Reference-panel test data ---
# Simulate a reference panel of 200 haplotypes (100 diploid samples), then draw
# maternal and paternal haplotypes directly from that panel to simulate an embryo.
# This mirrors realistic usage of sim_haplotype_ref_panel + infer_missing_af.
np.random.seed(7)
_base_haps, _, _ = pgt_sim.draw_parental_genotypes(m=1000, seed=7)
_base_pos = np.sort(np.random.uniform(high=1e7, size=1000))
_ref_panel_200 = pgt_sim.sim_haplotype_ref_panel(
    _base_haps, _base_pos, panel_size=200, seed=7
)
# Rows 0-1 → maternal haplotypes; rows 2-3 → paternal haplotypes
_panel_mat = _ref_panel_200[:2, :].astype(np.uint16)
_panel_pat = _ref_panel_200[2:4, :].astype(np.uint16)
_, _, _panel_mat_hap1, _panel_pat_hap1, _panel_aploid = pgt_sim.sim_haplotype_paths(
    _panel_mat, _panel_pat, _base_pos, ploidy=2, seed=7
)
_, _panel_baf = pgt_sim.sim_b_allele_freq(
    _panel_mat_hap1, _panel_pat_hap1, ploidy=2, std_dev=0.1, mix_prop=0.7, seed=7
)
_panel_lrr, _panel_sigmas = pgt_sim.sim_logR_ratio(
    _panel_mat_hap1, _panel_pat_hap1, ploidy=2, seed=7
)
data_from_ref_panel = {
    "mat_haps": _panel_mat,
    "pat_haps": _panel_pat,
    "baf": _panel_baf,
    "lrr": _panel_lrr,
    "sigmas": _panel_sigmas,
    "pos": _base_pos,
    "aploid": _panel_aploid,
    "ref_panel": _ref_panel_200,
}

# Map karyotype labels to integer ploidy levels (for tests that check ploidy
# without requiring exact parental-origin identification).
_PLOIDY_LEVEL = {"0": 0, "1m": 1, "1p": 1, "2": 2, "3m": 3, "3p": 3}


def bph(states):
    """Identify states that are BPH - both parental homologs."""
    idx = []
    for i, s in enumerate(states):
        assert len(s) == 4
        k = 0
        for j in range(4):
            k += s[j] >= 0
        if k == 3:
            if s[1] != -1:
                if s[0] != s[1]:
                    idx.append(i)
            if s[3] != -1:
                if s[2] != s[3]:
                    idx.append(i)
    return idx


def sph(states):
    """Identify states that are SPH - single parental homolog."""
    idx = []
    for i, s in enumerate(states):
        assert len(s) == 4
        k = 0
        for j in range(4):
            k += s[j] >= 0
        if k == 3:
            if s[1] != -1:
                if s[0] == s[1]:
                    idx.append(i)
            if s[3] != -1:
                if s[2] == s[3]:
                    idx.append(i)
    return idx


def test_bph_sph():
    """Test that BPH vs. SPH give you the correct states."""
    hmm = PocHMM()
    sph_idx = sph(hmm.states)
    bph_idx = bph(hmm.states)
    assert len(sph_idx) == 8
    assert len(bph_idx) == 4
    for i in sph_idx:
        x = hmm.states[i]
        assert (x[0] == x[1]) or (x[2] == x[3])
    for i in bph_idx:
        x = hmm.states[i]
        assert (x[0] != x[1]) or (x[2] != x[3])


@pytest.mark.parametrize("data", _all_data)
def test_data_integrity(data):
    """Test basic data sanity checks."""
    for x in ["baf", "lrr", "sigmas", "mat_haps", "pat_haps"]:
        assert x in data
    baf = data["baf"]
    pos = data["pos"]
    mat_haps = data["mat_haps"]
    pat_haps = data["pat_haps"]
    assert np.all((baf <= 1.0) & (baf >= 0.0))
    assert baf.size == mat_haps.shape[1]
    assert baf.size == pos.size
    assert mat_haps.shape == pat_haps.shape
    assert np.all((mat_haps == 0) | (mat_haps == 1))
    assert np.all((pat_haps == 0) | (pat_haps == 1))


@pytest.mark.parametrize("data", _all_data)
def test_pochmm_forward(data):
    """Forward loglik changes when LRR is included vs. blanked."""
    baf = data["baf"]
    lrr = data["lrr"]
    sigmas = data["sigmas"]
    pos = data["pos"]
    mat_haps = data["mat_haps"]
    pochmm = PocHMM()
    _, _, _, _, loglik = pochmm.forward_algorithm(
        bafs=baf, lrrs=lrr, sigmas=sigmas, pos=pos, haps=mat_haps, freqs=None
    )
    _, _, _, _, loglik_no_lrr = pochmm.forward_algorithm(
        bafs=baf,
        lrrs=np.repeat(-9, baf.size),
        sigmas=np.ones(baf.size),
        pos=pos,
        haps=mat_haps,
        freqs=None,
    )
    assert loglik != loglik_no_lrr


@pytest.mark.parametrize("data", _all_data)
def test_pochmm_backward(data):
    """Backward loglik changes when LRR is included vs. blanked."""
    baf = data["baf"]
    lrr = data["lrr"]
    sigmas = data["sigmas"]
    pos = data["pos"]
    mat_haps = data["mat_haps"]
    pochmm = PocHMM()
    _, _, _, _, loglik = pochmm.backward_algorithm(
        bafs=baf, lrrs=lrr, sigmas=sigmas, pos=pos, haps=mat_haps, freqs=None
    )
    _, _, _, _, loglik_no_lrr = pochmm.backward_algorithm(
        bafs=baf,
        lrrs=np.repeat(-9, baf.size),
        sigmas=np.ones(baf.size),
        pos=pos,
        haps=mat_haps,
        freqs=None,
    )
    assert loglik != loglik_no_lrr


@pytest.mark.parametrize("data", _all_data)
def test_pochmm_fwd_bwd(data):
    """Forward and backward log-likelihoods agree."""
    baf = data["baf"]
    lrr = data["lrr"]
    sigmas = data["sigmas"]
    pos = data["pos"]
    mat_haps = data["mat_haps"]
    pochmm = PocHMM()
    _, _, _, _, loglik = pochmm.forward_algorithm(
        bafs=baf, lrrs=lrr, sigmas=sigmas, pos=pos, haps=mat_haps, freqs=None
    )
    _, _, _, _, loglik2 = pochmm.backward_algorithm(
        bafs=baf, lrrs=lrr, sigmas=sigmas, pos=pos, haps=mat_haps, freqs=None
    )
    assert np.isclose(loglik, loglik2)


@pytest.mark.parametrize("data", _all_data)
def test_pochmm_fwdbwd(data):
    """Forward-backward posteriors are valid probability distributions."""
    baf = data["baf"]
    lrr = data["lrr"]
    sigmas = data["sigmas"]
    pos = data["pos"]
    mat_haps = data["mat_haps"]
    pochmm = PocHMM()
    gammas, _, _ = pochmm.forward_backward(
        bafs=baf, lrrs=lrr, sigmas=sigmas, pos=pos, haps=mat_haps, freqs=None
    )
    assert np.all(np.isclose(np.sum(np.exp(gammas), axis=0), 1.0))
    pp = np.exp(gammas)
    assert np.all((pp >= 0.0) & (pp <= 1.0))


@pytest.mark.parametrize("use_lrr", [True, False], ids=["with_lrr", "no_lrr"])
@pytest.mark.parametrize(
    "data",
    [
        data_disomy,
        data_mat_trisomy,
        data_pat_trisomy,
        data_mat_origin_mono,
        data_pat_origin_mono,
    ],
    ids=["disomy", "mat_trisomy", "pat_trisomy", "mat_origin_mono", "pat_origin_mono"],
)
def test_pochmm_ploidy_correctness(data, use_lrr):
    """Argmax karyotype matches the true simulated karyotype, with and without LRR."""
    baf = data["baf"]
    lrr = data["lrr"] if use_lrr else np.repeat(-9.0, baf.size)
    sigmas = data["sigmas"] if use_lrr else np.ones(baf.size)
    pos = data["pos"]
    mat_haps = data["mat_haps"]
    # Use paternal allele frequencies as a proxy for an external reference panel
    freqs = data["pat_haps"].sum(axis=0) / 2.0
    hmm = PocHMM()
    gammas, _, karyotypes = hmm.forward_backward(
        bafs=baf,
        lrrs=lrr,
        sigmas=sigmas,
        pos=pos,
        haps=mat_haps,
        pi0=0.7,
        std_dev=0.1,
        freqs=freqs,
    )
    assert np.all(np.isclose(np.sum(np.exp(gammas), axis=0), 1.0))
    post_dict = hmm.posterior_karyotypes(gammas, karyotypes)
    for x in ["0", "1m", "1p", "2", "3m", "3p"]:
        assert x in post_dict
    max_post = max(post_dict.values())
    assert post_dict[data["aploid"]] == max_post


@pytest.mark.parametrize("use_lrr", [True, False], ids=["with_lrr", "no_lrr"])
@pytest.mark.parametrize(
    "data",
    [
        data_disomy,
        data_mat_trisomy,
        data_pat_trisomy,
        data_mat_origin_mono,
        data_pat_origin_mono,
    ],
    ids=["disomy", "mat_trisomy", "pat_trisomy", "mat_origin_mono", "pat_origin_mono"],
)
def test_pochmm_ploidy_correctness_uniform_freqs(data, use_lrr):
    """Ploidy level is correctly identified using a uniform (even) prior on allele frequencies.

    With freqs=0.5 at all sites the model lacks information to distinguish parental
    origin, so only the integer ploidy level (0/1/2/3) is asserted.
    """
    baf = data["baf"]
    lrr = data["lrr"] if use_lrr else np.repeat(-9.0, baf.size)
    sigmas = data["sigmas"] if use_lrr else np.ones(baf.size)
    freqs = np.full(baf.size, 0.5)
    hmm = PocHMM()
    gammas, _, karyotypes = hmm.forward_backward(
        bafs=baf,
        lrrs=lrr,
        sigmas=sigmas,
        pos=data["pos"],
        haps=data["mat_haps"],
        pi0=0.7,
        std_dev=0.1,
        freqs=freqs,
    )
    assert np.all(np.isclose(np.sum(np.exp(gammas), axis=0), 1.0))
    post_dict = hmm.posterior_karyotypes(gammas, karyotypes)
    for x in ["0", "1m", "1p", "2", "3m", "3p"]:
        assert x in post_dict
    best = max(post_dict, key=post_dict.get)
    assert _PLOIDY_LEVEL[str(best)] == _PLOIDY_LEVEL[str(data["aploid"])]


def test_pochmm_ploidy_correctness_from_ref_panel():
    """Exact karyotype is recovered when parents are drawn from a haplotype reference panel
    and paternal allele frequencies are inferred via infer_missing_af instead of using
    the true paternal haplotypes.

    Setup: 200 haplotypes (100 diploid samples) are simulated with
    sim_haplotype_ref_panel; rows 0-1 become the maternal haplotypes and rows 2-3
    the paternal haplotypes for the embryo simulation.
    """
    data = data_from_ref_panel
    hmm = PocHMM()
    mat_geno = data["mat_haps"].sum(axis=0).astype(np.int32)
    freqs = hmm.infer_missing_af(data["baf"], mat_geno, data["ref_panel"], data["pos"])
    assert freqs.shape == data["baf"].shape
    assert np.all((freqs >= 0.0) & (freqs <= 1.0))
    gammas, _, karyotypes = hmm.forward_backward(
        bafs=data["baf"],
        lrrs=data["lrr"],
        sigmas=data["sigmas"],
        pos=data["pos"],
        haps=data["mat_haps"],
        pi0=0.7,
        std_dev=0.1,
        freqs=freqs,
    )
    assert np.all(np.isclose(np.sum(np.exp(gammas), axis=0), 1.0))
    post_dict = hmm.posterior_karyotypes(gammas, karyotypes)
    for x in ["0", "1m", "1p", "2", "3m", "3p"]:
        assert x in post_dict
    assert post_dict[data["aploid"]] == max(post_dict.values())


def test_infer_missing_af_properties():
    """infer_missing_af returns a valid allele-frequency array of the correct shape."""
    baf = data_disomy["baf"]
    mat_geno = data_disomy["mat_haps"].sum(axis=0).astype(np.int32)
    panel = pgt_sim.sim_haplotype_ref_panel(
        data_disomy["mat_haps"], data_disomy["pos"], panel_size=50, seed=1
    )
    hmm = PocHMM()
    freqs = hmm.infer_missing_af(baf, mat_geno, panel, data_disomy["pos"])
    assert freqs.shape == baf.shape
    assert np.all((freqs >= 0.0) & (freqs <= 1.0))
    assert not np.any(np.isnan(freqs))


def test_infer_missing_af_no_anchors_fallback():
    """infer_missing_af falls back to the panel mean when no anchor sites exist."""
    baf = data_disomy["baf"]
    panel = pgt_sim.sim_haplotype_ref_panel(
        data_disomy["mat_haps"], data_disomy["pos"], panel_size=50, seed=1
    )
    hmm = PocHMM()
    # Force no anchor sites by making all genotypes heterozygous (geno==1 never triggers)
    het_geno = np.ones(baf.size, dtype=np.int32)
    freqs = hmm.infer_missing_af(baf, het_geno, panel, data_disomy["pos"])
    expected = panel.mean(axis=0)
    assert freqs.shape == baf.shape
    assert np.allclose(freqs, expected)


def test_genotype_parent():
    """genotype_parent returns a (3, m) log-posterior matrix that sums to 1 per site."""
    baf = data_disomy["baf"]
    lrr = data_disomy["lrr"]
    sigmas = data_disomy["sigmas"]
    hmm = PocHMM()
    gammas, _, _ = hmm.forward_backward(
        bafs=baf,
        lrrs=lrr,
        sigmas=sigmas,
        pos=data_disomy["pos"],
        haps=data_disomy["mat_haps"],
        pi0=0.7,
        std_dev=0.1,
    )
    geno_dosage = hmm.genotype_parent(
        bafs=baf,
        lrrs=lrr,
        sigmas=sigmas,
        haps=data_disomy["mat_haps"],
        gammas=gammas,
        pi0=0.7,
        std_dev=0.1,
    )
    assert geno_dosage.shape == (3, baf.size)
    col_sums = np.exp(geno_dosage).sum(axis=0)
    assert np.allclose(col_sums, 1.0, atol=1e-4)
    assert np.all((np.exp(geno_dosage) >= 0.0) & (np.exp(geno_dosage) <= 1.0 + 1e-10))
