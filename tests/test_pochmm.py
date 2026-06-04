"""Test suite for karyoHMM PocHMM."""

from karyohmm import PGTSim, PocHMM
import numpy as np
import pytest


# --- Generating test data for applications in the DuoHMM setting --- #
pgt_sim = PGTSim()
data_disomy = pgt_sim.full_ploidy_sim(m=1000, mix_prop=0.7, std_dev=0.1, seed=42)
data_trisomy = pgt_sim.full_ploidy_sim(
    m=1000, ploidy=3, mat_skew=1.0, mix_prop=0.7, std_dev=0.1, seed=42
)
data_monosomy = pgt_sim.full_ploidy_sim(
    m=1000, ploidy=1, mat_skew=1.0, mix_prop=0.7, std_dev=0.1, seed=42
)
data_nullisomy = pgt_sim.full_ploidy_sim(
    m=1000, ploidy=0, mat_skew=1.0, mix_prop=0.7, std_dev=0.1, seed=42
)


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
                    # Both maternal homologs present
                    idx.append(i)
            if s[3] != -1:
                if s[2] != s[3]:
                    # Both paternal homologs present
                    idx.append(i)
    # Returns indices of both maternal & paternal BPH
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
                    # Both maternal homologs present
                    idx.append(i)
            if s[3] != -1:
                if s[2] == s[3]:
                    # Both paternal homologs present
                    idx.append(i)
    # Returns indices of both maternal & paternal SPH
    return idx


def test_bph_sph():
    """Test that BPH vs. SPH give you the correct states."""
    hmm = PocHMM()
    sph_idx = sph(hmm.states)
    bph_idx = bph(hmm.states)
    # The total number of states shouldb be 12
    assert len(sph_idx) == 8
    assert len(bph_idx) == 4
    for i in sph_idx:
        x = hmm.states[i]
        assert (x[0] == x[1]) or (x[2] == x[3])
    for i in bph_idx:
        x = hmm.states[i]
        assert (x[0] != x[1]) or (x[2] != x[3])


@pytest.mark.parametrize(
    "data",
    [data_disomy, data_trisomy, data_nullisomy, data_monosomy],
)
def test_data_integrity(data):
    """Test for some basic data sanity checks ..."""
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


@pytest.mark.parametrize(
    "data",
    [data_disomy, data_trisomy, data_nullisomy, data_monosomy],
)
def test_pochmm_forward(data):
    """Test the forward algorithm both with and without including the lrr."""
    baf = data["baf"]
    lrr = data["lrr"]
    sigmas = data["sigmas"]
    pos = data["pos"]
    mat_haps = data["mat_haps"]
    lrrs_blank = np.repeat(-9, baf.size)
    pochmm = PocHMM()
    _, _, _, _, loglik = pochmm.forward_algorithm(
        bafs=baf, lrrs=lrr, sigmas=sigmas, pos=pos, haps=mat_haps, freqs=None
    )
    _, _, _, _, loglik2 = pochmm.forward_algorithm(
        bafs=baf,
        lrrs=lrrs_blank,
        sigmas=np.ones(baf.size),
        pos=pos,
        haps=mat_haps,
        freqs=None,
    )
    assert loglik != loglik2


@pytest.mark.parametrize(
    "data",
    [data_disomy, data_trisomy, data_nullisomy, data_monosomy],
)
def test_pochmm_backward(data):
    """Test the forward algorithm both with and without including the lrr."""
    baf = data["baf"]
    lrr = data["lrr"]
    sigmas = data["sigmas"]
    pos = data["pos"]
    mat_haps = data["mat_haps"]
    lrrs_blank = np.repeat(-9, baf.size)
    pochmm = PocHMM()
    _, _, _, _, loglik = pochmm.backward_algorithm(
        bafs=baf, lrrs=lrr, sigmas=sigmas, pos=pos, haps=mat_haps, freqs=None
    )
    _, _, _, _, loglik2 = pochmm.backward_algorithm(
        bafs=baf,
        lrrs=lrrs_blank,
        sigmas=np.ones(baf.size),
        pos=pos,
        haps=mat_haps,
        freqs=None,
    )
    assert loglik != loglik2


@pytest.mark.parametrize(
    "data",
    [data_disomy, data_trisomy, data_nullisomy, data_monosomy],
)
def test_pochmm_fwd_bwd(data):
    """Test the forward algorithm both with and without including the lrr."""
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
        bafs=baf,
        lrrs=lrr,
        sigmas=sigmas,
        pos=pos,
        haps=mat_haps,
        freqs=None,
    )
    assert np.isclose(loglik, loglik2)


@pytest.mark.parametrize(
    "data",
    [data_disomy, data_trisomy, data_nullisomy, data_monosomy],
)
def test_pochmm_fwdbwd(data):
    """Test the forward algorithm both with and without including the lrr."""
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
    pp_fwdbwd = np.exp(gammas)
    assert np.all((pp_fwdbwd >= 0.0) & (pp_fwdbwd <= 1.0))


@pytest.mark.parametrize(
    "data",
    [data_disomy, data_trisomy, data_monosomy],
)
def test_pochmm_ploidy_correctness(data):
    """Test the forward algorithm both with and without including the lrr."""
    baf = data["baf"]
    lrr = data["lrr"]
    sigmas = data["sigmas"]
    pos = data["pos"]
    mat_haps = data["mat_haps"]
    pat_haps = data["pat_haps"]
    hmm = PocHMM()
    gammas, states, karyotypes = hmm.forward_backward(
        bafs=baf,
        lrrs=lrr,
        sigmas=sigmas,
        pos=pos,
        haps=mat_haps,
        pi0=0.7,
        std_dev=0.1,
        freqs=pat_haps.sum(axis=0) / 2.0,
    )
    assert np.all(np.isclose(np.sum(np.exp(gammas), axis=0), 1.0))
    post_dict = hmm.posterior_karyotypes(gammas, karyotypes)
    max_post = np.max([post_dict[p] for p in post_dict])
    for x in ["0", "1m", "1p", "2", "3m", "3p"]:
        assert x in post_dict
    assert post_dict[data["aploid"]] == max_post
    assert max_post >= 0.95
