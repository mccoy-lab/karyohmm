"""Testing module for phase correction using embryo BAF."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from karyohmm import MetaHMM, PGTSim, RecombEst

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
    "sigma,pi0,nsibs",
    [
        (0.175, 0.8, 3),
        (0.125, 0.5, 3),
        (0.105, 0.2, 3),
    ],
)
def test_recomb_isolate_paternal_rec_expected_baf(sigma, pi0, nsibs):
    """Test that recombinations can be isolated in expected conditions."""
    data = pgt_sim.sibling_euploid_sim(
        m=4000,
        nsibs=nsibs,
        std_dev=sigma,
        mix_prop=pi0,
        rec_prob=1e-4,
        switch_err_rate=0.0,
        seed=42,
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
    _, llr_z, pat_recomb_events = recomb_est.isolate_recomb_events()
    filt_pat_recomb_events = recomb_est.refine_recomb_events(pat_recomb_events, npad=5)
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
def test_recomb_isolate_paternal_rec_inferred_baf(sigma, pi0, nsibs, seed):
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
        e_baf_i = (
            dosages[0, :] * 0.0 + dosages[1, :] * 1.0 + dosages[2, :] * 2.0
        ) / 2.0
        expected_baf.append(e_baf_i)
    # Add in the expected BAF ...
    recomb_est.add_baf(embryo_bafs=expected_baf)

    # Obtain the true paternal recombination events for the template embryo ...
    zs_paternal0 = data["zs_paternal0"]
    n_pat_rec = np.sum(zs_paternal0[:-1] != zs_paternal0[1:])
    _, llr_z, pat_recomb_events = recomb_est.isolate_recomb_events()
    filt_pat_recomb_events = recomb_est.refine_recomb_events(pat_recomb_events, npad=5)
    # Make sure that the numbers match up for number of recombination events ...
    assert n_pat_rec == len(filt_pat_recomb_events)
