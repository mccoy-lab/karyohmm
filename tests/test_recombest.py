"""Testing module for phase correction using embryo BAF."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from karyohmm import MetaHMM, PGTSim, PhaseCorrect, RecombEst

pgt_sim = PGTSim()
meta_hmm = MetaHMM(disomy=True)

data_disomy_sibs_null = pgt_sim.sibling_euploid_sim(
    m=4000, nsibs=3, std_dev=0.1, mix_prop=0.6, switch_err_rate=0.0, seed=42
)


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


@pytest.mark.parametrize(
    "sigma,pi0,nsibs,seed",
    [
        (0.05, 0.2, 3, 1),
        (0.05, 0.4, 3, 2),
        (0.05, 0.6, 3, 3),
        (0.05, 0.8, 3, 4),
        (0.1, 0.2, 3, 5),
        (0.1, 0.4, 3, 6),
        (0.1, 0.6, 3, 7),
        (0.1, 0.8, 3, 8),
        (0.175, 0.2, 3, 1),
        (0.175, 0.4, 3, 2),
        (0.175, 0.6, 3, 3),
        (0.175, 0.8, 3, 4),
        (0.2, 0.2, 3, 5),
        (0.2, 0.4, 3, 6),
        (0.2, 0.6, 3, 7),
        (0.2, 0.8, 3, 8),
    ],
)
def test_rec_paternal_expected_baf_perfect_phase(sigma, pi0, nsibs, seed):
    """Test that recombinations can be isolated in expected conditions."""
    data = pgt_sim.sibling_euploid_sim(
        m=4000,
        nsibs=nsibs,
        std_dev=sigma,
        mix_prop=pi0,
        rec_prob=1e-4,
        switch_err_rate=0.0,
        seed=seed,
    )
    recomb_est = RecombEst(
        mat_haps=data["mat_haps_true"], pat_haps=data["pat_haps_true"], pos=data["pos"]
    )
    # Add in the expected BAF ...
    recomb_est.add_baf(
        embryo_bafs=[data[f"geno_embryo{i}"] / 2.0 for i in range(data["nsibs"])]
    )
    # Set the parameters here ...
    recomb_est.embryo_pi0s = np.array([pi0 for _ in range(nsibs)])
    recomb_est.embryo_sigmas = np.array([sigma for _ in range(nsibs)])
    # Obtain the true paternal recombination events for the template embryo ...
    zs_paternal0 = data["zs_paternal0"]
    n_pat_rec = np.sum(zs_paternal0[:-1] != zs_paternal0[1:])
    _, llr_z, filt_pat_recomb_events = recomb_est.isolate_recomb_events(maternal=False)
    # Make sure that the numbers match up for number of recombination events ...
    assert n_pat_rec == len(filt_pat_recomb_events)


@pytest.mark.parametrize(
    "sigma,pi0,nsibs,seed",
    [
        (0.05, 0.2, 3, 1),
        (0.05, 0.4, 3, 2),
        (0.05, 0.6, 3, 3),
        (0.05, 0.8, 3, 4),
        (0.1, 0.2, 3, 5),
        (0.1, 0.4, 3, 6),
        (0.1, 0.6, 3, 7),
        (0.1, 0.8, 3, 8),
        (0.175, 0.2, 3, 1),
        (0.175, 0.4, 3, 2),
        (0.175, 0.6, 3, 3),
        (0.175, 0.8, 3, 4),
        (0.2, 0.2, 3, 5),
        (0.2, 0.4, 3, 6),
        (0.2, 0.6, 3, 7),
        (0.2, 0.8, 3, 8),
    ],
)
def test_rec_paternal_inferred_baf_perfect_phase(sigma, pi0, nsibs, seed):
    """Test that recombinations can be isolated using inferred estimates for allelic intensity."""
    data = pgt_sim.sibling_euploid_sim(
        m=4000,
        nsibs=nsibs,
        std_dev=sigma,
        mix_prop=pi0,
        rec_prob=1e-4,
        switch_err_rate=0.0,
        seed=42 + seed,
    )
    recomb_est = RecombEst(
        mat_haps=data["mat_haps_true"], pat_haps=data["pat_haps_true"], pos=data["pos"]
    )
    # Set the parameters here ...
    recomb_est.embryo_pi0s = np.array([pi0 for _ in range(nsibs)])
    recomb_est.embryo_sigmas = np.array([sigma for _ in range(nsibs)])
    expected_baf = []
    for i in range(nsibs):
        dosages = meta_hmm.genotype_embryo(
            bafs=data[f"baf_embryo{i}"],
            pos=data["pos"],
            mat_haps=data["mat_haps_true"],
            pat_haps=data["pat_haps_true"],
            std_dev=recomb_est.embryo_sigmas[i],
            pi0=recomb_est.embryo_pi0s[i],
        )
        # Calculate the expected BAF for that embryo ...
        e_baf_i = (
            dosages[0, :] * 0.0 + dosages[1, :] * 1.0 + dosages[2, :] * 2.0
        ) / 2.0
        expected_baf.append(e_baf_i)
    recomb_est.add_baf(embryo_bafs=expected_baf)

    # Obtain the true paternal recombination events for the template embryo ...
    zs_paternal0 = data["zs_paternal0"]
    n_pat_rec = np.sum(zs_paternal0[:-1] != zs_paternal0[1:])
    _, llr_z, filt_pat_recomb_events = recomb_est.isolate_recomb_events(maternal=False)
    # Make sure that the numbers match up for number of recombination events ...
    assert n_pat_rec == len(filt_pat_recomb_events)


@pytest.mark.parametrize(
    "sigma,pi0,nsibs,seed",
    [
        (0.05, 0.2, 3, 1),
        (0.05, 0.8, 3, 4),
        (0.1, 0.2, 3, 5),
        (0.1, 0.8, 3, 8),
        (0.175, 0.2, 3, 1),
        (0.175, 0.8, 3, 4),
        (0.2, 0.2, 3, 5),
        (0.2, 0.8, 3, 8),
    ],
)
def test_rec_paternal_inferred_baf_fixphase(sigma, pi0, nsibs, seed):
    """Test that recombinations can be isolated using inferred estimates for allelic intensity."""
    data = pgt_sim.sibling_euploid_sim(
        m=4000,
        nsibs=nsibs,
        std_dev=sigma,
        mix_prop=pi0,
        rec_prob=1e-4,
        switch_err_rate=1e-2,
        seed=42 + seed,
    )
    # Actually run the phase-correction routine ...
    phase_correct = PhaseCorrect(
        mat_haps=data["mat_haps_real"], pat_haps=data["pat_haps_real"], pos=data["pos"]
    )
    phase_correct.embryo_pi0s = np.array([pi0 for _ in range(nsibs)])
    phase_correct.embryo_sigmas = np.array([sigma for _ in range(nsibs)])
    phase_correct.add_baf(
        embryo_bafs=[data[f"baf_embryo{i}"] for i in range(data["nsibs"])]
    )
    (
        mat_haps,
        pat_haps,
        n_mis_mat_tot,
        n_mis_pat_tot,
    ) = phase_correct.viterbi_phase_correct(niter=1)
    # Use the phase-corrected haplotypes as input for the recombination estimation + expected genotype estimation ...
    recomb_est = RecombEst(mat_haps=mat_haps, pat_haps=pat_haps, pos=data["pos"])
    # Set the parameters + calculate the expected BAF
    recomb_est.embryo_pi0s = np.array([pi0 for _ in range(nsibs)])
    recomb_est.embryo_sigmas = np.array([sigma for _ in range(nsibs)])
    expected_baf = []
    for i in range(nsibs):
        dosages = meta_hmm.genotype_embryo(
            bafs=data[f"baf_embryo{i}"],
            pos=data["pos"],
            mat_haps=mat_haps,
            pat_haps=pat_haps,
            std_dev=recomb_est.embryo_sigmas[i],
            pi0=recomb_est.embryo_pi0s[i],
        )
        # Calculate the expected BAF for that embryo ...
        e_baf_i = (
            dosages[0, :] * 0.0 + dosages[1, :] * 1.0 + dosages[2, :] * 2.0
        ) / 2.0
        expected_baf.append(e_baf_i)
    recomb_est.add_baf(embryo_bafs=expected_baf)
    # Obtain the true paternal recombination events for the template embryo under consideration ...
    zs_paternal0 = data["zs_paternal0"]
    n_pat_rec = np.sum(zs_paternal0[:-1] != zs_paternal0[1:])
    filt_pat_recomb_events = recomb_est.estimate_crossovers(maternal=False)
    # Make sure that the numbers match up for number of recombination events detected
    assert n_pat_rec == len(filt_pat_recomb_events)
    # Check that the positions are correct?
    if n_pat_rec > 0:
        for x in np.where(zs_paternal0[:-1] != zs_paternal0[1:])[0]:
            found = False
            for s,e in filt_pat_recomb_events:
                if  (recomb_est.pos[x] >= s) and (recomb_est.pos[x] <= e):
                    found = True
                    break
            assert found  



@pytest.mark.parametrize(
    "sigma,pi0,nsibs,seed,m,seqlen",
    [
        (0.15, 0.6, 3, 2, 2000, 1e7),
        (0.15, 0.6, 3, 3, 3000, 1e7),
        (0.15, 0.6, 3, 4, 4000, 1e7),
        (0.15, 0.6, 3, 5, 5000, 1e7),
        (0.15, 0.6, 3, 42, 3983, 48e6),
        (0.15, 0.6, 3, 43, 3983, 48e6),
        (0.15, 0.6, 3, 44, 3983, 48e6),
        (0.15, 0.6, 3, 45, 3983, 48e6),
    ],
)
def test_rec_paternal_inferred_baf_perfect_phase_diff_density(
    sigma, pi0, nsibs, seed, m, seqlen
):
    """Test that recombinations can be isolated using inferred estimates for allelic intensity.

    This incorporates realistic chromosome sizes from chr21 on the cytoSNP array.
    """
    assert m > 0
    assert seqlen > 0
    e_rec = (seqlen / 1e6 * 1e-2) / m
    data = pgt_sim.sibling_euploid_sim(
        m=m,
        nsibs=nsibs,
        std_dev=sigma,
        mix_prop=pi0,
        length=seqlen,
        rec_prob=e_rec,
        switch_err_rate=0.0,
        seed=42 + seed,
    )
    recomb_est = RecombEst(
        mat_haps=data["mat_haps_true"], pat_haps=data["pat_haps_true"], pos=data["pos"]
    )
    # Set the parameters here ...
    recomb_est.embryo_pi0s = np.array([pi0 for _ in range(nsibs)])
    recomb_est.embryo_sigmas = np.array([sigma for _ in range(nsibs)])
    expected_baf = []
    for i in range(nsibs):
        dosages = meta_hmm.genotype_embryo(
            bafs=data[f"baf_embryo{i}"],
            pos=data["pos"],
            mat_haps=data["mat_haps_true"],
            pat_haps=data["pat_haps_true"],
            std_dev=recomb_est.embryo_sigmas[i],
            pi0=recomb_est.embryo_pi0s[i],
        )
        # Calculate the expected BAF for that embryo ...
        e_baf_i = (
            dosages[0, :] * 0.0 + dosages[1, :] * 1.0 + dosages[2, :] * 2.0
        ) / 2.0
        expected_baf.append(e_baf_i)
    recomb_est.add_baf(embryo_bafs=expected_baf)

    # Obtain the true paternal recombination events for the template embryo ...
    zs_paternal0 = data["zs_paternal0"]
    n_pat_rec = np.sum(zs_paternal0[:-1] != zs_paternal0[1:])
    _, llr_z, filt_pat_recomb_events = recomb_est.isolate_recomb_events(maternal=False)
    # Make sure that the numbers match up for number of recombination events ...
    assert n_pat_rec == len(filt_pat_recomb_events)


@pytest.mark.parametrize(
    "sigma,pi0,nsibs,seed,m,seqlen",
    [
        (0.15, 0.6, 3, 42, 3983, 48e6),
        (0.15, 0.6, 3, 43, 3983, 48e6),
        (0.15, 0.6, 3, 44, 3983, 48e6),
        (0.15, 0.6, 3, 45, 3983, 48e6),
    ],
)
def test_rec_paternal_inferred_baf_diff_density(sigma, pi0, nsibs, seed, m, seqlen):
    """Test that recombinations can be isolated using inferred estimates for allelic intensity.

    This incorporates realistic chromosome sizes from chr21 on the cytoSNP array.
    """
    assert m > 0
    assert seqlen > 0
    e_rec = (seqlen / 1e6 * 1e-2) / m
    data = pgt_sim.sibling_euploid_sim(
        m=m,
        length=seqlen,
        nsibs=nsibs,
        std_dev=sigma,
        mix_prop=pi0,
        rec_prob=e_rec,
        switch_err_rate=1.5e-2,
        seed=42 + seed,
    )
    # Actually run the phase-correction routine ...
    phase_correct = PhaseCorrect(
        mat_haps=data["mat_haps_real"], pat_haps=data["pat_haps_real"], pos=data["pos"]
    )
    phase_correct.embryo_pi0s = np.array([pi0 for _ in range(nsibs)])
    phase_correct.embryo_sigmas = np.array([sigma for _ in range(nsibs)])
    phase_correct.add_baf(
        embryo_bafs=[data[f"baf_embryo{i}"] for i in range(data["nsibs"])]
    )
    (
        mat_haps,
        pat_haps,
        n_mis_mat_tot,
        n_mis_pat_tot,
    ) = phase_correct.viterbi_phase_correct(niter=1)
    # Use the phase-corrected haplotypes as input for the recombination estimation + expected genotype estimation ...
    recomb_est = RecombEst(mat_haps=mat_haps, pat_haps=pat_haps, pos=data["pos"])
    # Set the parameters + calculate the expected BAF
    recomb_est.embryo_pi0s = np.array([pi0 for _ in range(nsibs)])
    recomb_est.embryo_sigmas = np.array([sigma for _ in range(nsibs)])
    expected_baf = []
    for i in range(nsibs):
        dosages = meta_hmm.genotype_embryo(
            bafs=data[f"baf_embryo{i}"],
            pos=data["pos"],
            mat_haps=mat_haps,
            pat_haps=pat_haps,
            std_dev=recomb_est.embryo_sigmas[i],
            pi0=recomb_est.embryo_pi0s[i],
        )
        # Calculate the expected BAF for that embryo ...
        e_baf_i = (
            dosages[0, :] * 0.0 + dosages[1, :] * 1.0 + dosages[2, :] * 2.0
        ) / 2.0
        expected_baf.append(e_baf_i)
    recomb_est.add_baf(embryo_bafs=expected_baf)
    # Obtain the true paternal recombination events for the template embryo under consideration ...
    zs_paternal0 = data["zs_paternal0"]
    n_pat_rec = np.sum(zs_paternal0[:-1] != zs_paternal0[1:])
    _, llr_z, filt_pat_recomb_events = recomb_est.isolate_recomb_events(maternal=False)
    # Make sure that the numbers match up for number of recombination events detected
    assert n_pat_rec == len(filt_pat_recomb_events)
