"""Test suite for karyoHMM DuoHMM."""
import numpy as np
import pytest
from karyohmm_utils import create_index_arrays, transition_kernel
from scipy.special import logsumexp as logsumexp_sp

from karyohmm import DuoHMM, PGTSim

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
    hmm = DuoHMM()
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


def test_data_integrity(data=data_disomy):
    """Test for some basic data sanity checks ..."""
    for x in ["baf_embryo", "mat_haps", "pat_haps"]:
        assert x in data
    baf = data["baf_embryo"]
    pos = data["pos"]
    mat_haps = data["mat_haps"]
    pat_haps = data["pat_haps"]
    assert np.all((baf <= 1.0) & (baf >= 0.0))
    assert baf.size == mat_haps.shape[1]
    assert baf.size == pos.size
    assert mat_haps.shape == pat_haps.shape
    assert np.all((mat_haps == 0) | (mat_haps == 1))
    assert np.all((pat_haps == 0) | (pat_haps == 1))


# --- Testing the metadata implementations --- #
@pytest.mark.parametrize("data", [data_disomy, data_trisomy, data_monosomy])
def test_forward_algorithm(data):
    """Test the implementation of the forward algorithm."""
    hmm = DuoHMM()
    _, _, _, karyotypes, loglik = hmm.forward_algorithm(
        bafs=data["baf_embryo"],
        pos=data["pos"],
        haps=data["mat_haps"],
    )


@pytest.mark.parametrize("data", [data_disomy, data_trisomy, data_monosomy])
def test_backward_algorithm(data):
    """Test the implementation of the backward algorithm."""
    hmm = DuoHMM()
    _, _, _, karyotypes, loglik = hmm.backward_algorithm(
        bafs=data["baf_embryo"],
        pos=data["pos"],
        haps=data["mat_haps"],
    )


@pytest.mark.parametrize("data", [data_disomy])
def test_forward_vs_backward_loglik(data):
    """Test that the log-likelihood from forward algorithm is equal to the backward."""
    hmm = DuoHMM()

    _, _, _, _, fwd_loglik = hmm.forward_algorithm(
        bafs=data["baf_embryo"],
        pos=data["pos"],
        haps=data["mat_haps"],
    )
    _, _, _, _, bwd_loglik = hmm.backward_algorithm(
        bafs=data["baf_embryo"],
        pos=data["pos"],
        haps=data["mat_haps"],
    )
    assert np.isclose(fwd_loglik, bwd_loglik, atol=1e-6)


@pytest.mark.parametrize("data", [data_disomy])
def test_disomy_model(data):
    """Test implementation under a pure-disomy model."""
    hmm = DuoHMM(disomy=True)
    gammas, _, karyotypes = hmm.forward_backward(
        bafs=data["baf_embryo"], pos=data["pos"], haps=data["mat_haps"]
    )
    assert np.all(np.isclose(np.sum(np.exp(gammas), axis=0), 1.0))


@pytest.mark.parametrize("data", [data_disomy, data_trisomy, data_monosomy])
def test_fwd_bwd_algorithm(data):
    """Test the properties of the output from the fwd-bwd algorithm."""
    hmm = DuoHMM()
    gammas, _, karyotypes = hmm.forward_backward(
        bafs=data["baf_embryo"],
        pos=data["pos"],
        haps=data["mat_haps"],
    )
    # all of the columns must have a sum to 1
    assert np.all(np.isclose(np.sum(np.exp(gammas), axis=0), 1.0))
    post_dict = hmm.posterior_karyotypes(gammas, karyotypes)
    for x in ["0", "1m", "1p", "2", "3m", "3p"]:
        assert x in post_dict
    assert np.isclose(sum([post_dict[k] for k in post_dict]), 1.0)


@pytest.mark.parametrize("data", [data_disomy])
def test_est_pi0_sigma(data):
    """Test the optimization routine on the forward-algorithm likelihood."""
    hmm = DuoHMM()
    pi0_est, sigma_est = hmm.est_sigma_pi0(
        bafs=data["baf_embryo"], pos=data["pos"], haps=data["mat_haps"]
    )
    assert (pi0_est > 0) and (pi0_est < 1.0)
    assert (sigma_est > 0) and (sigma_est < 1.0)
    assert np.isclose(pi0_est, 0.7, atol=1e-1)
    assert np.isclose(sigma_est, 0.1, atol=5e-2)


@pytest.mark.parametrize(
    "data,pi0_bounds",
    [
        (data_disomy, (0, 1)),
        (data_disomy, 0.1),
        (data_disomy, (0.1, 0.5, 0.9)),
        (data_disomy, (0.99, 0.01)),
    ],
)
def test_est_pi0_sigma_bad_pi0_bounds(data, pi0_bounds):
    """Test the pi0 bounds as input to the MLE estimation."""
    with pytest.raises(Exception):
        hmm = DuoHMM()
        pi0_est, sigma_est = hmm.est_sigma_pi0(
            bafs=data["baf_embryo"],
            pos=data["pos"],
            haps=data["mat_haps"],
            pi0_bounds=pi0_bounds,
        )


@pytest.mark.parametrize(
    "data,sigma_bounds",
    [(data_disomy, (0, 1)), (data_disomy, 0.1), (data_disomy, (0.99, 0.01))],
)
def test_est_pi0_sigma_bad_sigma_bounds(data, sigma_bounds):
    """Test the sigma bounds as input to the MLE estimation."""
    with pytest.raises(Exception):
        hmm = DuoHMM()
        pi0_est, sigma_est = hmm.est_sigma_pi0(
            bafs=data["baf_embryo"],
            pos=data["pos"],
            haps=data["mat_haps"],
            sigma_bounds=sigma_bounds,
        )


def test_string_rep():
    """Test that the string representation of states makes sense."""
    hmm = DuoHMM()
    for s in hmm.states:
        x = hmm.get_state_str(s)
        m = sum([j >= 0 for j in s])
        if m == 0:
            assert x == "0"
        elif m == 1:
            if x[0] == "m":
                assert s in hmm.m_monosomy_states
            else:
                assert s in hmm.p_monosomy_states
        elif m == 2:
            assert len(x) == 2 * m
            assert x in ["m0p0", "m0p1", "m1p0", "m1p1"]
        else:
            assert len(x) == 2 * m
            assert (
                ("m0m1" in x)
                | ("p0p1" in x)
                | ("m0m0" in x)
                | ("m1m1" in x)
                | ("p0p0" in x)
                | ("p1p1" in x)
            )


@pytest.mark.parametrize("r,a", [(1e-8, 1e-2), (1e-8, 1e-4), (1e-8, 1e-6)])
def test_transition_matrices(r, a):
    """Test that transition matrices obey the rules."""
    hmm = DuoHMM()
    K0, K1 = create_index_arrays(hmm.karyotypes)
    A = transition_kernel(K0, K1, r=r, a=a)
    for i in range(A.shape[0]):
        assert np.isclose(np.exp(logsumexp_sp(A[i, :])), 1.0)


@pytest.mark.parametrize(
    "r,a,d",
    [(1e-8, 1e-2, 1e5), (1e-8, 1e-3, 1e4), (1e-8, 1e-2, 1e3), (1e-8, 1e-1, 1e8)],
)
def test_transition_matrices_dist(r, a, d):
    """Test how transition matrices scale with distance."""
    assert r < a
    hmm = DuoHMM()
    K0, K1 = create_index_arrays(hmm.karyotypes)
    A = transition_kernel(K0, K1, d=d, r=r, a=a)
    for i in range(A.shape[0]):
        assert np.isclose(np.exp(logsumexp_sp(A[i, :])), 1.0)


@pytest.mark.parametrize(
    "data", [data_disomy, data_trisomy, data_monosomy, data_nullisomy]
)
def test_ploidy_correctness(data):
    """This actually tests that the posterior inference of whole-chromosome aneuploidies is correct here."""
    hmm = DuoHMM()
    gammas, _, karyotypes = hmm.forward_backward(
        bafs=data["baf_embryo"],
        pos=data["pos"],
        haps=data["mat_haps"],
        pi0=0.7,
        std_dev=0.1,
    )
    # all of the columns must have a sum to 1
    assert np.all(np.isclose(np.sum(np.exp(gammas), axis=0), 1.0))
    post_dict = hmm.posterior_karyotypes(gammas, karyotypes)
    max_post = np.max([post_dict[p] for p in post_dict])
    for x in ["0", "1m", "1p", "2", "3m", "3p"]:
        assert x in post_dict
    assert post_dict[data["aploid"]] == max_post
    assert post_dict[data["aploid"]] > 0.95


@pytest.mark.parametrize(
    "data", [data_disomy, data_trisomy, data_monosomy, data_nullisomy]
)
def test_ploidy_correctness_mle(data):
    """This actually tests that the posterior inference of whole-chromosome aneuploidies is correct under mle."""
    hmm = DuoHMM()
    pi0_est, sigma_est = hmm.est_sigma_pi0(
        bafs=data["baf_embryo"],
        pos=data["pos"],
        haps=data["mat_haps"],
    )
    gammas, _, karyotypes = hmm.forward_backward(
        bafs=data["baf_embryo"],
        pos=data["pos"],
        haps=data["mat_haps"],
        pi0=pi0_est,
        std_dev=sigma_est,
    )
    # all of the columns must have a sum to 1
    assert np.all(np.isclose(np.sum(np.exp(gammas), axis=0), 1.0))
    post_dict = hmm.posterior_karyotypes(gammas, karyotypes)
    max_post = np.max([post_dict[p] for p in post_dict])
    for x in ["0", "1m", "1p", "2", "3m", "3p"]:
        assert x in post_dict
    assert post_dict[data["aploid"]] == max_post
    assert post_dict[data["aploid"]] > 0.95


@pytest.mark.parametrize("data", [data_disomy])
def test_genotype_parent(data):
    """Test being able to genotype the unobserved parent."""
    hmm = DuoHMM()
    pi0_est, sigma_est = hmm.est_sigma_pi0(
        bafs=data["baf_embryo"],
        pos=data["pos"],
        haps=data["mat_haps_prime"],
        global_opt=False,
    )
    gammas, _, karyotypes = hmm.forward_backward(
        bafs=data["baf_embryo"],
        pos=data["pos"],
        haps=data["mat_haps"],
        pi0=pi0_est,
        std_dev=sigma_est,
    )
    geno_dosage = hmm.genotype_parent(
        bafs=data["baf_embryo"],
        haps=data["mat_haps"],
        gammas=gammas,
        maternal=True,
        pi0=pi0_est,
        std_dev=sigma_est,
    )
    # NOTE: this assumes the parent is diploid at these sites
    assert geno_dosage.ndim == 2
    assert geno_dosage.shape[0] == 3
