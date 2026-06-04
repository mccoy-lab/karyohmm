"""
Karyohmm is an HMM-based model for aneuploidy detection.

Karyohmm implements methods for haplotype based analyses
of copy number changes from array intensity data when
parental genotypes are available.

Modules available are:

- MetaHMM: module for whole chromosome aneuploidy determination via HMMs.
- QuadHMM: module leveraging multi-sibling design for evaluating crossover recombination estimation.
- PocHMM: module for inference of aneuploidy with a single parent in products-of-conception
- MosaicEst: module for estimating mosaic cell fraction from heterozygote baf imbalance.
- PhaseCorrect: module which implements Mendelian phase correction for parental haplotypes.
- RecombEst: module implementing simpler detection of crossover recombination based on Coop et al 2007.

"""

import warnings

import numpy as np
from karyohmm_utils import (
    backward_algo,
    backward_algo_duo,
    backward_algo_sibs,
    emission_baf,
    emission_lrr,
    forward_algo,
    forward_algo_duo,
    forward_algo_sibs,
    logaddexp,
    logsumexp,
    loglik_mcc,
    mat_dosage,
    norm_logl,
    pat_dosage,
    viterbi_algo,
    viterbi_algo_sibs,
)
from scipy.optimize import brentq, minimize
from scipy.special import logsumexp as logsumexp_sp
from scipy.stats import chi2


class AneuploidyHMM:
    """Base class for defining all the aneuploidy HMM."""

    def __init__(self):
        """Initialize the base aneuploidy HMM class."""
        self.ploidy = 2
        self.aploid = None

    def get_state_str(self, state):
        """Obtain a string representation of the HMM state.

        Arguments:
            - state: A tuple (typically length 4)

        Returns:
            - state_str (`str`): string that represents the state

        """
        t = []
        for i, s in enumerate(state):
            if s != -1:
                if i < 2:
                    t.append(f"m{s}")
                else:
                    t.append(f"p{s}")
        if not t:
            t.append("0")
        return "".join(t)

    def est_sigma_pi0(
        self,
        bafs,
        pos,
        mat_haps,
        pat_haps,
        algo="L-BFGS-B",
        pi0_bounds=(0.01, 0.99),
        sigma_bounds=(1e-2, 1.0),
        **kwargs,
    ):
        """Estimate sigma and pi0 under the B-Allele Frequency model using optimization of forward algorithm likelihood.

        Arguments:
            - bafs (`np.array`): B-allele frequencies across the all m sites
            - pos (`np.array`): basepair positions of the SNPs
            - mat_haps (`np.array`): a 2 x m array of 0/1 maternal haplotypes
            - pat_haps (`np.array`): a 2 x m array of 0/1 paternal haplotypes
            - algo (`str`): one of Nelder-Mead, L-BFGS-B, or Powell algorithms for optimization

        Returns:
            - pi0_est (`float`): estimate of sparsity parameter (pi0) for B-allele emission model
            - sigma_est (`float`): estimate of noise parameter (sigma) for B-allele emission model

        """
        assert algo in ["Nelder-Mead", "L-BFGS-B", "Powell"]
        assert (len(pi0_bounds) == 2) and (len(sigma_bounds) == 2)
        assert (pi0_bounds[0] > 0) and (pi0_bounds[1] > 0)
        assert (pi0_bounds[0] < 1) and (pi0_bounds[1] < 1)
        assert pi0_bounds[0] < pi0_bounds[1]
        assert (sigma_bounds[0] > 0) and (sigma_bounds[1] > 0)
        assert sigma_bounds[0] < sigma_bounds[1]
        mid_pi0 = np.mean(pi0_bounds)
        mid_sigma = np.mean(sigma_bounds)
        opt_res = minimize(
            lambda x: (
                -self.forward_algorithm(
                    bafs=bafs,
                    pos=pos,
                    mat_haps=mat_haps,
                    pat_haps=pat_haps,
                    pi0=x[0],
                    std_dev=x[1],
                    **kwargs,
                )[4]
            ),
            x0=[mid_pi0, mid_sigma],
            method=algo,
            bounds=[pi0_bounds, sigma_bounds],
            tol=1e-4,
            options={"disp": True, "ftol": 1e-4, "xtol": 1e-4},
        )
        pi0_est = opt_res.x[0]
        sigma_est = opt_res.x[1]
        return pi0_est, sigma_est

    def full_param_inference(
        self,
        bafs,
        pos,
        mat_haps,
        pat_haps,
        algo="L-BFGS-B",
        pi0_bounds=(0.01, 0.99),
        sigma_bounds=(1e-2, 1.0),
    ):
        """

        Full parameter inference under the B-allele frequency model using naive optimization of the forward algorithm.

        Arguments:
            - bafs (`np.array`): B-allele frequencies across the all m sites
            - pos (`np.array`): basepair positions of the SNPs
            - mat_haps (`np.array`): a 2 x m array of 0/1 maternal haplotypes
            - pat_haps (`np.array`): a 2 x m array of 0/1 paternal haplotypes
            - algo (`str`): one of Nelder-Mead, L-BFGS-B, or Powell algorithms for optimization

        """
        raise NotImplementedError(
            "We have not implemented full joint parameter inference yet!"
        )


class MetaHMM(AneuploidyHMM):
    """A meta-HMM that evaluates all possible ploidy states for allele intensity data."""

    def __init__(self, disomy=False, upd=False):
        """Initialize the MetaHMM class for determining chromosomal aneuploidy.

        Arguments:
            - disomy (`bool`): assume disomy to limit the model statespace

        Returns: MetaHMM object

        """
        super().__init__()
        self.ploidy = 0
        self.aploid = "meta"
        self.nullisomy_state = [(-1, -1, -1, -1)]
        self.p_monosomy_states = [(-1, -1, 1, -1), (-1, -1, 0, -1)]
        self.m_monosomy_states = [(0, -1, -1, -1), (1, -1, -1, -1)]
        self.disomy_states = [
            (0, -1, 0, -1),
            (0, -1, 1, -1),
            (1, -1, 0, -1),
            (1, -1, 1, -1),
        ]
        # First two are the female states ...
        self.m_trisomy_states = [
            (0, 0, 0, -1),
            (0, 1, 0, -1),
            (1, 1, 0, -1),
            (0, 0, 1, -1),
            (0, 1, 1, -1),
            (1, 1, 1, -1),
        ]
        self.p_trisomy_states = [
            (0, -1, 0, 0),
            (1, -1, 0, 0),
            (0, -1, 0, 1),
            (1, -1, 0, 1),
            (0, -1, 1, 1),
            (1, -1, 1, 1),
        ]
        self.m_upd_states = [
            (0, 0, -1, -1),
            (1, 1, -1, -1),
            (0, 1, -1, -1),
            (1, 0, -1, -1),
        ]
        self.p_upd_states = [
            (-1, -1, 0, 0),
            (-1, -1, 1, 1),
            (-1, -1, 0, 1),
            (-1, -1, 1, 0),
        ]
        if disomy:
            self.aploid = "disomy"
            self.states = self.disomy_states
            self.karyotypes = np.array(["2", "2", "2", "2"], dtype=str)
        else:
            if upd:
                self.aploid = "meta+upd"
                self.states = (
                    self.nullisomy_state
                    + self.m_monosomy_states
                    + self.p_monosomy_states
                    + self.disomy_states
                    + self.m_trisomy_states
                    + self.p_trisomy_states
                    + self.p_upd_states
                    + self.m_upd_states
                )
                self.karyotypes = np.array(
                    [
                        "0",
                        "1m",
                        "1m",
                        "1p",
                        "1p",
                        "2",
                        "2",
                        "2",
                        "2",
                        "3m",
                        "3m",
                        "3m",
                        "3m",
                        "3m",
                        "3m",
                        "3p",
                        "3p",
                        "3p",
                        "3p",
                        "3p",
                        "3p",
                        "2p0",
                        "2p0",
                        "2p1",
                        "2p1",
                        "2m0",
                        "2m0",
                        "2m1",
                        "2m1",
                    ],
                    dtype=str,
                )
            else:
                self.states = (
                    self.nullisomy_state
                    + self.m_monosomy_states
                    + self.p_monosomy_states
                    + self.disomy_states
                    + self.m_trisomy_states
                    + self.p_trisomy_states
                )
                self.karyotypes = np.array(
                    [
                        "0",
                        "1m",
                        "1m",
                        "1p",
                        "1p",
                        "2",
                        "2",
                        "2",
                        "2",
                        "3m",
                        "3m",
                        "3m",
                        "3m",
                        "3m",
                        "3m",
                        "3p",
                        "3p",
                        "3p",
                        "3p",
                        "3p",
                        "3p",
                    ],
                    dtype=str,
                )

    def _warn_missing_lrr(self, lrrs):
        """Warn when UPD states are active but LRR data is absent."""
        if self.aploid == "meta+upd" and np.all(lrrs == -9.0):
            warnings.warn(
                "All LRR values are missing (sentinel -9.0) but UPD states are "
                "included in the model. UPD states share copy number 2 with disomy "
                "and rely primarily on BAF patterns when LRR is unavailable, which "
                "may reduce power to distinguish UPD from normal disomy.",
                UserWarning,
                stacklevel=3,
            )

    def forward_algorithm(
        self,
        bafs,
        lrrs,
        sigmas,
        pos,
        mat_haps,
        pat_haps,
        pi0=0.5,
        std_dev=0.25,
        r=1e-8,
        a=1e-2,
        unphased=False,
        mat_err=0.0,
        pat_err=0.0,
    ):
        """Forward HMM algorithm under a multi-ploidy model.

        Arguments:
            - bafs (`np.array`): B-allele frequencies across the all m sites
            - pos (`np.array`): m-length vector of basepair positions for sites
            - mat_haps (`np.array`): a 2 x m array of 0/1 maternal haplotypes
            - pat_haps (`np.array`): a 2 x m array of 0/1 paternal haplotypes
            - pi0 (`float`): sparsity parameter for B-allele emission model
            - std_dev (`float`): standard deviation for B-allele emission model
            - r (`float`): intra-karyotype transition rate (recombination)
            - a (`float`): inter-karyotype transition rate
            - unphased (`bool`): run the model in unphased mode
            - mat_err (`float`): per-site maternal genotyping error rate
            - pat_err (`float`): per-site paternal genotyping error rate

        Returns:
            - alphas (`np.array`): forward variable from hmm across k states
            - scaler (`np.array`): m-length array of scale parameters
            - states (`list`): tuple representation of states
            - karyotypes (`np.array`):  array of karyotypes in the MetaHMM model
            - loglik (`float`): total log-likelihood of B-allele frequency

        """
        assert bafs.ndim == 1
        assert lrrs.ndim == 1
        assert sigmas.ndim == 1
        assert bafs.size == lrrs.size
        assert sigmas.size == lrrs.size
        assert pos.ndim == 1
        assert (mat_haps.ndim == 2) & (pat_haps.ndim == 2)
        assert (pi0 > 0) & (pi0 < 1.0)
        assert std_dev > 0
        assert bafs.size == mat_haps.shape[1]
        assert bafs.size == pos.size
        assert mat_haps.shape == pat_haps.shape
        assert np.all(pos[1:] > pos[:-1])
        assert r < 0.5 and r > 0
        assert a < 0.5 and a > 0
        assert 0.0 <= mat_err < 0.5
        assert 0.0 <= pat_err < 0.5
        alphas, scaler, _, _, loglik = forward_algo(
            bafs,
            lrrs,
            sigmas,
            pos,
            mat_haps,
            pat_haps,
            self.states,
            self.karyotypes,
            r=r,
            a=a,
            pi0=pi0,
            std_dev=std_dev,
            unphased=int(unphased),
            mat_err=mat_err,
            pat_err=pat_err,
        )
        return alphas, scaler, self.states, self.karyotypes, loglik

    def backward_algorithm(
        self,
        bafs,
        lrrs,
        sigmas,
        pos,
        mat_haps,
        pat_haps,
        pi0=0.5,
        std_dev=0.25,
        r=1e-8,
        a=1e-2,
        unphased=False,
        mat_err=0.0,
        pat_err=0.0,
    ):
        """Backward HMM algorithm under a given statespace model.

        Arguments:
            - bafs (`np.array`): B-allele frequencies across the all m sites
            - pos (`np.array`): m-length vector of basepair positions for sites
            - mat_haps (`np.array`): a 2 x m array of 0/1 maternal haplotypes
            - pat_haps (`np.array`): a 2 x m array of 0/1 paternal haplotypes
            - pi0 (`float`): sparsity parameter for B-allele emission model
            - std_dev (`float`): standard deviation for B-allele emission model
            - r (`float`): intra-karyotype transition rate (recombination)
            - a (`float`): inter-karyotype transition rate
            - unphased (`bool`): run the model in unphased mode
            - mat_err (`float`): per-site maternal genotyping error rate
            - pat_err (`float`): per-site paternal genotyping error rate

        Returns:
            - betas (`np.array`): backward variables from hmm across the k states
            - scaler (`np.array`): m-length array of scale parameters
            - states (`list`): tuple representation of states
            - karyotypes (`np.array`):  array of karyotypes in the MetaHMM model
            - loglik (`float`): total log-likelihood of B-allele frequency

        """
        assert bafs.ndim == 1
        assert lrrs.ndim == 1
        assert sigmas.ndim == 1
        assert bafs.size == lrrs.size
        assert sigmas.size == bafs.size
        assert pos.ndim == 1
        assert (mat_haps.ndim == 2) & (pat_haps.ndim == 2)
        assert (pi0 > 0) & (pi0 < 1.0)
        assert std_dev > 0
        assert bafs.size == mat_haps.shape[1]
        assert bafs.size == pos.size
        assert mat_haps.shape == pat_haps.shape
        assert np.all(pos[1:] > pos[:-1])
        assert r < 0.5 and r > 0
        assert a < 0.5 and a > 0
        assert 0.0 <= mat_err < 0.5
        assert 0.0 <= pat_err < 0.5
        betas, scaler, _, _, loglik = backward_algo(
            bafs,
            lrrs,
            sigmas,
            pos,
            mat_haps,
            pat_haps,
            self.states,
            self.karyotypes,
            r=r,
            a=a,
            pi0=pi0,
            std_dev=std_dev,
            unphased=int(unphased),
            mat_err=mat_err,
            pat_err=pat_err,
        )
        return betas, scaler, self.states, self.karyotypes, loglik

    def forward_backward(
        self,
        bafs,
        lrrs,
        sigmas,
        pos,
        mat_haps,
        pat_haps,
        pi0=0.2,
        std_dev=0.25,
        r=1e-8,
        a=1e-2,
        unphased=False,
        mat_err=0.0,
        pat_err=0.0,
    ):
        """Run the forward-backward algorithm across all states.

        Arguments:
            - bafs (`np.array`): B-allele frequencies across the all m sites
            - pos (`np.array`): m-length vector of basepair positions for sites
            - mat_haps (`np.array`): a 2 x m array of 0/1 maternal haplotypes
            - pat_haps (`np.array`): a 2 x m array of 0/1 paternal haplotypes
            - pi0 (`float`): sparsity parameter for B-allele emission model
            - std_dev (`float`): standard deviation for B-allele emission model
            - r (`float`): intra-karyotype transition rate (recombination)
            - a (`float`): inter-karyotype transition rate
            - unphased (`bool`): run the model in unphased mode
            - mat_err (`float`): per-site maternal genotyping error rate
            - pat_err (`float`): per-site paternal genotyping error rate

        Returns:
            - gammas (`np.array`): log posterior density of being in each of k hidden states
            - states (`list`): tuple representation of states
            - karyotypes (`np.array`):  array of karyotypes in the model

        """
        self._warn_missing_lrr(lrrs)
        alphas, _, states, karyotypes, _ = self.forward_algorithm(
            bafs,
            lrrs,
            sigmas,
            pos,
            mat_haps,
            pat_haps,
            pi0=pi0,
            std_dev=std_dev,
            r=r,
            a=a,
            unphased=unphased,
            mat_err=mat_err,
            pat_err=pat_err,
        )
        betas, _, _, _, _ = self.backward_algorithm(
            bafs,
            lrrs,
            sigmas,
            pos,
            mat_haps,
            pat_haps,
            pi0=pi0,
            std_dev=std_dev,
            r=r,
            a=a,
            unphased=unphased,
            mat_err=mat_err,
            pat_err=pat_err,
        )
        gammas = (alphas + betas) - logsumexp_sp(alphas + betas, axis=0)
        return gammas, states, karyotypes

    def viterbi_algorithm(
        self,
        bafs,
        lrrs,
        sigmas,
        pos,
        mat_haps,
        pat_haps,
        pi0=0.2,
        std_dev=0.25,
        r=1e-8,
        a=1e-2,
        unphased=False,
        mat_err=0.0,
        pat_err=0.0,
    ):
        """Implement the viterbi traceback through karyotypic states.

        Arguments:
            - bafs (`np.array`): B-allele frequencies across the all m sites
            - pos (`np.array`): m-length vector of basepair positions for sites
            - mat_haps (`np.array`): a 2 x m array of 0/1 maternal haplotypes
            - pat_haps (`np.array`): a 2 x m array of 0/1 paternal haplotypes
            - pi0 (`float`): sparsity parameter for B-allele emission model
            - std_dev (`float`): standard deviation for B-allele emission model
            - r (`float`): intra-karyotype transition rate
            - a (`float`): inter-karyotype transition rate
            - unphased (`bool`): run the model in unphased mode
            - mat_err (`float`): per-site maternal genotyping error rate
            - pat_err (`float`): per-site paternal genotyping error rate

        Returns:
            - path (`np.array`): most likely copying path through k states in the model
            - states (`list`): tuple representation of states
            - deltas (`np.array`): delta variable (maximum path probability at step m)
            - psi (`np.array`): storage vector for psi variable

        """
        self._warn_missing_lrr(lrrs)
        assert bafs.ndim == 1
        assert lrrs.ndim == 1
        assert sigmas.ndim == 1
        assert bafs.size == lrrs.size
        assert lrrs.size == sigmas.size
        assert pos.ndim == 1
        assert (mat_haps.ndim == 2) & (pat_haps.ndim == 2)
        assert (pi0 > 0) & (pi0 < 1.0)
        assert std_dev > 0
        assert bafs.size == pos.size
        assert bafs.size == mat_haps.shape[1]
        assert mat_haps.shape == pat_haps.shape
        assert np.all(pos[1:] > pos[:-1])
        assert 0.0 <= mat_err < 0.5
        assert 0.0 <= pat_err < 0.5
        path, states, deltas, psi = viterbi_algo(
            bafs,
            lrrs,
            sigmas,
            pos,
            mat_haps,
            pat_haps,
            self.states,
            self.karyotypes,
            r=r,
            a=a,
            pi0=pi0,
            std_dev=std_dev,
            unphased=int(unphased),
            mat_err=mat_err,
            pat_err=pat_err,
        )
        return path, states, deltas, psi

    def marginal_posterior_karyotypes(self, gammas, karyotypes):
        """Obtain the marginal posterior (not logged) probability over karyotypic states.

        Arguments:
            - gammas (`np.array`): a k x m array of log-posterior density across sites
            - karyotypes (`np.array`): karyotype labels for all k states

        Returns:
            - gamma_karyo: (`np.array`): collapsed posteriors

        """
        assert gammas.ndim == 2
        assert gammas.shape[0] == karyotypes.size
        k, m = gammas.shape
        nk = np.unique(karyotypes).size
        gamma_karyo = np.zeros(shape=(nk, m))
        _, idx = np.unique(karyotypes, return_index=True)
        uniq_karyo = karyotypes[np.sort(idx)]
        for i, k in enumerate(uniq_karyo):
            # This is just the summed version of the posteriors ...
            gamma_karyo[i, :] = np.sum(np.exp(gammas[(karyotypes == k), :]), axis=0)
        return gamma_karyo, uniq_karyo

    def posterior_karyotypes(self, gammas, karyotypes):
        """Obtain full posterior on karyotypes chromosome-wide.

        NOTE: this is the weighted proportion of time spent in each karyotypic state-space.

        Arguments:
            - gammas (`np.array`): a k x m array of log-posterior density across sites
            - karyotypes (`np.array`): karyotype labels for all k states

        Returns:
            - kar_prob (`dict`): dictionary of posterior probability per-snp

        """
        assert gammas.ndim == 2
        assert gammas.shape[0] == karyotypes.size
        k, m = gammas.shape
        kar_prob = {}
        for k in np.unique(karyotypes):
            kar_prob[k] = np.sum(np.exp(gammas[(karyotypes == k), :])) / m
        return kar_prob

    def genotype_embryo(
        self,
        bafs,
        lrrs,
        sigmas,
        pos,
        mat_haps,
        pat_haps,
        pi0=0.2,
        std_dev=0.25,
        r=1e-8,
        a=1e-2,
        unphased=False,
        viterbi=False,
    ):
        """Obtain genotype dosages for a putative disomic embryo.

        Arguments:
            - bafs (`np.array`): B-allele frequencies across the all m sites
            - pos (`np.array`): m-length vector of basepair positions for sites
            - mat_haps (`np.array`): a 2 x m array of 0/1 maternal haplotypes
            - pat_haps (`np.array`): a 2 x m array of 0/1 paternal haplotypes
            - pi0 (`float`): sparsity parameter for B-allele emission model
            - std_dev (`float`): standard deviation for B-allele emission model
            - r (`float`): intra-karyotype transition rate
            - a (`float`): inter-karyotype transition rate
            - unphased (`bool`): run the model in unphased mode
            - viterbi (`bool`): estimate the embryo genotypes using the viterbi algorithm

        Returns:
            - dosages (`np.array`): a 3 x M array of genotype probabilities (RR, RA, AA)

        """
        if self.aploid != "disomy":
            raise ValueError(
                "Obtaining non-disomic embryo genotypes is not currently supported!"
            )
        # Run the forward-backward algorithm on this ...
        if viterbi:
            path, states, _, _ = self.viterbi_algorithm(
                bafs=bafs,
                lrrs=lrrs,
                sigmas=sigmas,
                pos=pos,
                mat_haps=mat_haps,
                pat_haps=pat_haps,
                pi0=pi0,
                std_dev=std_dev,
                r=r,
                a=a,
                unphased=False,
            )
            dosages = np.zeros(shape=(3, pos.size), dtype=np.float32)
            for i, x in enumerate(path):
                cur_geno = int(
                    mat_dosage(mat_haps[:, i], states[x])
                    + pat_dosage(pat_haps[:, i], states[x])
                )
                assert (cur_geno >= 0) and (cur_geno <= 2)
                dosages[cur_geno, i] = 1.0
        else:
            gammas, states, _ = self.forward_backward(
                bafs=bafs,
                pos=pos,
                lrrs=lrrs,
                sigmas=sigmas,
                mat_haps=mat_haps,
                pat_haps=pat_haps,
                pi0=pi0,
                std_dev=std_dev,
                r=r,
                a=a,
                unphased=False,
            )
            # Calculate the genotype dosage of the alternative allele
            dosages = np.zeros(shape=(3, pos.size), dtype=np.float32)
            for i in range(pos.size):
                for j, x in enumerate(states):
                    cur_geno = int(
                        mat_dosage(mat_haps[:, i], x) + pat_dosage(pat_haps[:, i], x)
                    )
                    assert (cur_geno >= 0) and (cur_geno <= 2)
                    # This is analogous to the PL field (but on raw scale)
                    dosages[cur_geno, i] += np.exp(gammas[j, i])
            # Dosages are oriented towards the alt allele (bottom row is the alt/alt homozygote)
        return dosages

    def flag_parental_genotype_errors(
        self,
        gammas,
        states,
        bafs,
        mat_haps,
        pat_haps,
        pi0=0.2,
        std_dev=0.25,
    ):
        """Identify per-site parental genotype calls that are inconsistent with the posterior.

        For each site, computes a posterior-weighted log Bayes factor comparing the
        best alternative parental genotype (fixing the other parent) to the called one.
        A positive score means the data, given the inferred copying state, prefers a
        different parental genotype at that site.

        Arguments:
            - gammas (`np.array`): k x m log-posterior array from forward_backward
            - states (`list`): list of state tuples (output of forward_backward)
            - bafs (`np.array`): m-length array of B-allele frequencies
            - mat_haps (`np.array`): 2 x m array of 0/1 maternal haplotypes
            - pat_haps (`np.array`): 2 x m array of 0/1 paternal haplotypes
            - pi0 (`float`): sparsity parameter for BAF emission
            - std_dev (`float`): noise parameter for BAF emission

        Returns:
            - mat_err_score (`np.array`): m-length array; positive values flag likely maternal genotype errors
            - pat_err_score (`np.array`): m-length array; positive values flag likely paternal genotype errors

        """
        assert gammas.ndim == 2
        assert gammas.shape == (len(states), bafs.size)
        assert mat_haps.shape == (2, bafs.size)
        assert pat_haps.shape == (2, bafs.size)

        m = bafs.size
        mat_err_score = np.zeros(m)
        pat_err_score = np.zeros(m)
        all_genos = [[0, 0], [0, 1], [1, 0], [1, 1]]

        for i in range(m):
            mat_h = [int(mat_haps[0, i]), int(mat_haps[1, i])]
            pat_h = [int(pat_haps[0, i]), int(pat_haps[1, i])]
            for j, s in enumerate(states):
                w = np.exp(gammas[j, i])
                if w < 1e-12:
                    continue
                k = sum(v >= 0 for v in s)
                if k == 0:
                    continue
                m_d = mat_dosage(mat_h, s)
                p_d = pat_dosage(pat_h, s)
                called_loglik = emission_baf(
                    bafs[i], m_d, p_d, pi0=pi0, std_dev=std_dev, k=k
                )
                best_alt_mat = called_loglik
                for alt_mat in all_genos:
                    if alt_mat != mat_h:
                        alt_loglik = emission_baf(
                            bafs[i],
                            mat_dosage(alt_mat, s),
                            p_d,
                            pi0=pi0,
                            std_dev=std_dev,
                            k=k,
                        )
                        if alt_loglik > best_alt_mat:
                            best_alt_mat = alt_loglik
                mat_err_score[i] += w * (best_alt_mat - called_loglik)
                best_alt_pat = called_loglik
                for alt_pat in all_genos:
                    if alt_pat != pat_h:
                        alt_loglik = emission_baf(
                            bafs[i],
                            m_d,
                            pat_dosage(alt_pat, s),
                            pi0=pi0,
                            std_dev=std_dev,
                            k=k,
                        )
                        if alt_loglik > best_alt_pat:
                            best_alt_pat = alt_loglik
                pat_err_score[i] += w * (best_alt_pat - called_loglik)

        return mat_err_score, pat_err_score


class QuadHMM(AneuploidyHMM):
    """HMM for sibling euploid embryos based on Roach et al 2010 but for BAF data."""

    def __init__(self):
        """Initialize the QuadHMM model."""
        self.ploidy = 2
        self.aploid = "2"
        self.single_states = [
            (0, -1, 0, -1),
            (0, -1, 1, -1),
            (1, -1, 0, -1),
            (1, -1, 1, -1),
        ]
        self.states = []
        for i in self.single_states:
            for j in self.single_states:
                self.states.append((i, j))
        self.karyotypes = np.repeat("2", len(self.states)).astype(str)

    def forward_algorithm(
        self,
        bafs,
        pos,
        mat_haps,
        pat_haps,
        pi0=(0.7, 0.7),
        std_dev=(0.15, 0.15),
        r=1e-8,
        mat_err=0.0,
        pat_err=0.0,
    ):
        """Implement the forward algorithm for QuadHMM model.

        Arguments:
            - bafs (`list`): list of two arrays of B-allele frequencies across m sites for two siblings
            - pos (`np.array`): m-length vector of basepair positions for sites
            - mat_haps (`np.array`): a 2 x m array of 0/1 maternal haplotypes
            - pat_haps (`np.array`): a 2 x m array of 0/1 paternal haplotypes
            - pi0 (`tuple - float`): sparsity parameter for B-allele emission model
            - std_dev (`tuple - float`): standard deviation for B-allele emission model
            - r (`float`): inter-state transition rate
            - mat_err (`float`): per-site maternal genotyping error rate
            - pat_err (`float`): per-site paternal genotyping error rate

        Returns:
            - alphas (`np.array`): forward variable from hmm across 4 sibling states
            - scaler (`np.array`): m-length array of scale parameters
            - states (`list`): tuple representation of the 4 states
            - karyotypes (`np.array`):  array of karyotypes (default: None)
            - loglik (`float`): total log-likelihood of joint sibling B-allele frequencies

        """
        assert len(bafs) == 2
        assert bafs[0].size == pos.size
        assert bafs[0].size == bafs[1].size
        assert bafs[0].size == mat_haps.shape[1]
        assert (mat_haps.ndim == 2) and (pat_haps.ndim == 2)
        assert mat_haps.size == pat_haps.size
        assert np.all(pos[1:] > pos[:-1])
        alphas, scaler, states, karyotypes, loglik = forward_algo_sibs(
            bafs,
            pos,
            mat_haps,
            pat_haps,
            states=self.states,
            karyotypes=self.karyotypes,
            r=r,
            pi0=pi0,
            std_dev=std_dev,
            mat_err=mat_err,
            pat_err=pat_err,
        )
        return alphas, scaler, states, karyotypes, loglik

    def backward_algorithm(
        self,
        bafs,
        pos,
        mat_haps,
        pat_haps,
        pi0=(0.7, 0.7),
        std_dev=(0.15, 0.15),
        r=1e-8,
        mat_err=0.0,
        pat_err=0.0,
    ):
        """Implement the backward algorithm for QuadHMM model.

        Arguments:
            - bafs (`list`): list of two arrays of B-allele frequencies across m sites for two siblings
            - pos (`np.array`): m-length vector of basepair positions for sites
            - mat_haps (`np.array`): a 2 x m array of 0/1 maternal haplotypes
            - pat_haps (`np.array`): a 2 x m array of 0/1 paternal haplotypes
            - pi0 (`tuple - float`): sparsity parameter for B-allele emission model
            - std_dev (`tuple - float`): standard deviation for B-allele emission model
            - r (`float`): inter-state transition rate
            - mat_err (`float`): per-site maternal genotyping error rate
            - pat_err (`float`): per-site paternal genotyping error rate

        Returns:
            - betas (`np.array`): backward variable from hmm across 4 sibling states
            - scaler (`np.array`): m-length array of scale parameters
            - states (`list`): tuple representation of the 4 states
            - karyotypes (`np.array`):  array of karyotypes (default: None)
            - loglik (`float`): total log-likelihood of joint sibling B-allele frequencies

        """
        assert len(bafs) == 2
        assert bafs[0].size == pos.size
        assert bafs[0].size == bafs[1].size
        assert bafs[0].size == mat_haps.shape[1]
        assert (mat_haps.ndim == 2) and (pat_haps.ndim == 2)
        assert mat_haps.size == pat_haps.size
        assert np.all(pos[1:] > pos[:-1])
        alphas, scaler, states, karyotypes, loglik = backward_algo_sibs(
            bafs,
            pos,
            mat_haps,
            pat_haps,
            states=self.states,
            karyotypes=self.karyotypes,
            r=r,
            pi0=pi0,
            std_dev=std_dev,
            mat_err=mat_err,
            pat_err=pat_err,
        )
        return alphas, scaler, states, karyotypes, loglik

    def forward_backward(
        self,
        bafs,
        pos,
        mat_haps,
        pat_haps,
        pi0=(0.7, 0.7),
        std_dev=(0.15, 0.15),
        r=1e-8,
        a=1e-2,
        mat_err=0.0,
        pat_err=0.0,
    ):
        """Implement the forward-backward algorithm for the QuadHMM model.

        Arguments:
            - bafs (`list`): list of two arrays of B-allele frequencies across m sites for two siblings
            - pos (`np.array`): m-length vector of basepair positions for sites
            - mat_haps (`np.array`): a 2 x m array of 0/1 maternal haplotypes
            - pat_haps (`np.array`): a 2 x m array of 0/1 paternal haplotypes
            - pi0 (`tuple - float`): sparsity parameter for B-allele emission model
            - std_dev (`tuple - float`): standard deviation for B-allele emission model
            - r (`float`): inter-state transition rate
            - mat_err (`float`): per-site maternal genotyping error rate
            - pat_err (`float`): per-site paternal genotyping error rate

        Returns:
            - gammas (`np.array`): log posterior density of being in each of 4 hidden states
            - states (`list`): tuple representation of the states
            - karyotypes (`np.array`): None

        """
        alphas, _, states, _, _ = forward_algo_sibs(
            bafs,
            pos,
            mat_haps,
            pat_haps,
            states=self.states,
            karyotypes=self.karyotypes,
            pi0=pi0,
            std_dev=std_dev,
            r=r,
            mat_err=mat_err,
            pat_err=pat_err,
        )
        betas, _, _, _, _ = backward_algo_sibs(
            bafs,
            pos,
            mat_haps,
            pat_haps,
            states=self.states,
            karyotypes=self.karyotypes,
            pi0=pi0,
            std_dev=std_dev,
            r=r,
            mat_err=mat_err,
            pat_err=pat_err,
        )
        gammas = (alphas + betas) - logsumexp_sp(alphas + betas, axis=0)
        return gammas, states, None

    def viterbi_algorithm(
        self,
        bafs,
        pos,
        mat_haps,
        pat_haps,
        pi0=(0.7, 0.7),
        std_dev=(0.15, 0.15),
        r=1e-8,
        mat_err=0.0,
        pat_err=0.0,
    ):
        """Viterbi algorithm definition in a QuadHMM-context.

        Arguments:
            - bafs (`list`): list of two arrays of B-allele frequencies across m sites for two siblings
            - pos (`np.array`): m-length vector of basepair positions for sites
            - mat_haps (`np.array`): a 2 x m array of 0/1 maternal haplotypes
            - pat_haps (`np.array`): a 2 x m array of 0/1 paternal haplotypes
            - pi0 (`tuple - float`): sparsity parameter for B-allele emission model
            - std_dev (`tuple - float`): standard deviation for B-allele emission model
            - r (`float`): inter-state transition rate
            - mat_err (`float`): per-site maternal genotyping error rate
            - pat_err (`float`): per-site paternal genotyping error rate

        Returns:
            - path (`np.array`): most likely copying path through k states in the model
            - states (`list`): tuple representation of states
            - deltas (`np.array`): delta variable (maximum path probability at step m)
            - psi (`np.array`): storage vector for psi variable

        """
        path, states, deltas, psi = viterbi_algo_sibs(
            bafs,
            pos,
            mat_haps,
            pat_haps,
            states=self.states,
            karyotypes=self.karyotypes,
            pi0=pi0,
            std_dev=std_dev,
            r=r,
            mat_err=mat_err,
            pat_err=pat_err,
        )
        return path, states, deltas, psi

    def restrict_path(self, path):
        """Break down states into the same categories as Roach et al for determining recombinations.

        The state definitions used for tracing sibling haplotypes are:
            - (0) maternal-haploidentical
            - (1) paternal-haploidentical
            - (2) identical
            - (3) non-identical

        Arguments:
            - path (`np.array`): a sequence of states output from the viterbi algorithm (16 possible states)

        Returns:
            - refined_path (`np.array`): a sequence of m states in 0,1,2,3 indicating states from Roach et al.

        """
        maternal_haploidentical = []
        paternal_haploidentical = []
        identical = []
        non_identical = []
        for i, (x, y) in enumerate(self.states):
            if x == y:
                identical.append(i)
            elif (x[0] == y[0]) and (x[2] != y[2]):
                maternal_haploidentical.append(i)
            elif (x[2] == y[2]) and (x[0] != y[0]):
                paternal_haploidentical.append(i)
            else:
                non_identical.append(i)
        # Refining the path estimation to only the Roach et al 2010 states
        refined_path = np.zeros(path.size)
        for i in range(path.size):
            if path[i] in maternal_haploidentical:
                refined_path[i] = 0
            elif path[i] in paternal_haploidentical:
                refined_path[i] = 1
            elif path[i] in identical:
                refined_path[i] = 2
            elif path[i] in non_identical:
                refined_path[i] = 3
            else:
                raise ValueError("Incorrect path estimate!")
        return refined_path

    def restrict_states(self):
        """Break down states into the same categories as Roach et al for determining recombinations.

        Returns:
            - maternal_haploidentical (`list`): indexes of maternally haploidentical states
            - paternal_haploidentical (`list`): indexes of paternally haploidentical states
            - identical (`list`): indexes of identical states (siblings share same haplotypes)
            - non-identical (`list`): indexes of non-identical states

        """
        maternal_haploidentical = []
        paternal_haploidentical = []
        identical = []
        non_identical = []
        for i, (x, y) in enumerate(self.states):
            if x == y:
                identical.append(i)
            elif (x[0] == y[0]) and (x[2] != y[2]):
                maternal_haploidentical.append(i)
            elif (x[2] == y[2]) and (x[0] != y[0]):
                paternal_haploidentical.append(i)
            else:
                non_identical.append(i)
        return (
            maternal_haploidentical,
            paternal_haploidentical,
            identical,
            non_identical,
        )

    def viterbi_path(
        self,
        bafs,
        pos,
        mat_haps,
        pat_haps,
        pi0=(0.7, 0.7),
        std_dev=(0.15, 0.15),
        r=1e-8,
    ):
        """Obtain the restricted Viterbi path for traceback.

        Arguments:
            - bafs (`list`): list of two arrays of B-allele frequencies across m sites for two siblings
            - pos (`np.arary`): m-length vector of basepair positions for sites
            - mat_haps (`np.array`): a 2 x m array of 0/1 maternal haplotypes
            - pat_haps (`np.array`): a 2 x m array of 0/1 paternal haplotypes
            - pi0 (`float`): sparsity parameter for B-allele emission model
            - std_dev (`float`): standard deviation for B-allele emission model
            - r (`float`): inter-state transition rate

        Returns:
            - res_path (`np.array`): most likely copying path through 4 states in from Roach et al.

        """
        path, _, _, _ = self.viterbi_algorithm(
            bafs, pos, mat_haps, pat_haps, pi0=pi0, std_dev=std_dev, r=r
        )
        res_path = self.restrict_path(path)
        return res_path

    def map_path(
        self,
        bafs,
        pos,
        mat_haps,
        pat_haps,
        pi0=(0.7, 0.7),
        std_dev=(0.15, 0.15),
        r=1e-8,
    ):
        """Obtain the Maximum A-Posteriori Path across restricted states.

        Arguments:
            - bafs (`list`): list of two arrays of B-allele frequencies across m sites for two siblings
            - pos (`np.array`): m-length vector of basepair positions for sites
            - mat_haps (`np.array`): a 2 x m array of 0/1 maternal haplotypes
            - pat_haps (`np.array`): a 2 x m array of 0/1 paternal haplotypes
            - pi0 (`tuple float`): sparsity parameter for B-allele emission model
            - std_dev (`tuple float`): standard deviation for B-allele emission model
            - r (`float`): inter-state transition rate

        Returns:
            - map_path (`np.array`): max-a-posteriori copying path through 4 sibling copying states

        """
        gammas, _, _ = self.forward_backward(
            bafs, pos, mat_haps, pat_haps, pi0=pi0, std_dev=std_dev, r=r
        )
        (
            maternal_haploidentical,
            paternal_haploidentical,
            identical,
            non_identical,
        ) = self.restrict_states()
        red_gammas = np.zeros(shape=(4, gammas.shape[1]))
        red_gammas[0, :] = np.exp(gammas)[maternal_haploidentical, :].sum(axis=0)
        red_gammas[1, :] = np.exp(gammas)[paternal_haploidentical, :].sum(axis=0)
        red_gammas[2, :] = np.exp(gammas)[identical, :].sum(axis=0)
        red_gammas[3, :] = np.exp(gammas)[non_identical, :].sum(axis=0)
        red_gammas = np.log(red_gammas)
        map_path = np.argmax(red_gammas, axis=0)
        return map_path

    def flag_parental_genotype_errors(
        self,
        gammas,
        states,
        bafs,
        mat_haps,
        pat_haps,
        pi0=(0.7, 0.7),
        std_dev=(0.15, 0.15),
    ):
        """Identify per-site parental genotype calls that are inconsistent with the posterior.

        For each site, computes a posterior-weighted log Bayes factor comparing the
        best alternative parental genotype (fixing the other parent) to the called one.
        Both siblings observe the same parental genotype, so the joint sibling BAF
        likelihood is used when scoring alternatives.  A positive score means the
        data, given the inferred copying state, prefers a different parental genotype
        at that site.

        Arguments:
            - gammas (`np.array`): k x m log-posterior array from forward_backward
            - states (`list`): list of paired state tuples (output of forward_backward)
            - bafs (`list`): list of two m-length BAF arrays for the two siblings
            - mat_haps (`np.array`): 2 x m array of 0/1 maternal haplotypes
            - pat_haps (`np.array`): 2 x m array of 0/1 paternal haplotypes
            - pi0 (`tuple`): (pi0_sib0, pi0_sib1) sparsity parameters for BAF emission
            - std_dev (`tuple`): (std_sib0, std_sib1) noise parameters for BAF emission

        Returns:
            - mat_err_score (`np.array`): m-length array; positive values flag likely maternal genotype errors
            - pat_err_score (`np.array`): m-length array; positive values flag likely paternal genotype errors

        """
        assert gammas.ndim == 2
        assert len(bafs) == 2
        assert bafs[0].size == bafs[1].size
        assert gammas.shape == (len(states), bafs[0].size)
        assert mat_haps.shape == (2, bafs[0].size)
        assert pat_haps.shape == (2, bafs[0].size)

        m = bafs[0].size
        mat_err_score = np.zeros(m)
        pat_err_score = np.zeros(m)
        all_genos = [[0, 0], [0, 1], [1, 0], [1, 1]]

        for i in range(m):
            mat_h = [int(mat_haps[0, i]), int(mat_haps[1, i])]
            pat_h = [int(pat_haps[0, i]), int(pat_haps[1, i])]
            for j, (s0, s1) in enumerate(states):
                w = np.exp(gammas[j, i])
                if w < 1e-12:
                    continue
                m_d0 = mat_dosage(mat_h, s0)
                p_d0 = pat_dosage(pat_h, s0)
                m_d1 = mat_dosage(mat_h, s1)
                p_d1 = pat_dosage(pat_h, s1)
                called_loglik = emission_baf(
                    bafs[0][i], m_d0, p_d0, pi0=pi0[0], std_dev=std_dev[0], k=2
                ) + emission_baf(
                    bafs[1][i], m_d1, p_d1, pi0=pi0[1], std_dev=std_dev[1], k=2
                )
                best_alt_mat = called_loglik
                for alt_mat in all_genos:
                    if alt_mat != mat_h:
                        alt_loglik = emission_baf(
                            bafs[0][i],
                            mat_dosage(alt_mat, s0),
                            p_d0,
                            pi0=pi0[0],
                            std_dev=std_dev[0],
                            k=2,
                        ) + emission_baf(
                            bafs[1][i],
                            mat_dosage(alt_mat, s1),
                            p_d1,
                            pi0=pi0[1],
                            std_dev=std_dev[1],
                            k=2,
                        )
                        if alt_loglik > best_alt_mat:
                            best_alt_mat = alt_loglik
                mat_err_score[i] += w * (best_alt_mat - called_loglik)
                best_alt_pat = called_loglik
                for alt_pat in all_genos:
                    if alt_pat != pat_h:
                        alt_loglik = emission_baf(
                            bafs[0][i],
                            m_d0,
                            pat_dosage(alt_pat, s0),
                            pi0=pi0[0],
                            std_dev=std_dev[0],
                            k=2,
                        ) + emission_baf(
                            bafs[1][i],
                            m_d1,
                            pat_dosage(alt_pat, s1),
                            pi0=pi0[1],
                            std_dev=std_dev[1],
                            k=2,
                        )
                        if alt_loglik > best_alt_pat:
                            best_alt_pat = alt_loglik
                pat_err_score[i] += w * (best_alt_pat - called_loglik)

        return mat_err_score, pat_err_score

    def det_recomb_sex(self, i, j):
        """Determine the parental origin of the recombination event.

        Arguments:
            - i (`int`): state index for previous state
            - j (`int`): state index for current state

        Returns:
            - m (`int`): sex of haplotype on which transition occurred (0: maternal, 1: paternal)

        """
        assert i != j
        m = -1
        if i == 0 and j == 3:
            # maternal haploidentity -> non-identity
            m = 0
        if i == 0 and j == 2:
            # maternal haploidentity -> identity
            m = 1
        if i == 1 and j == 3:
            # paternal haploidentity -> non-identity
            m = 1
        if i == 1 and j == 2:
            # paternal haploidentity -> identity
            m = 0
        if i == 2 and j == 0:
            # identical -> maternal haploidentity
            m = 1
        if i == 2 and j == 1:
            # identical -> paternal haploidentity
            m = 0
        if i == 3 and j == 0:
            # non-identical -> maternal haploidentity
            m = 0
        if i == 3 and j == 1:
            # non-identical -> paternal haploidentity
            m = 1
        return m

    def isolate_recomb(self, path_xy, path_xzs, window=20):
        """Isolate key recombination events from a pair of refined paths.

        Arguments:
            - path_xy (`np.array`): numpy array of path for focal pair of siblings
            - path_xzs (`list`): list of sibling paths
            - window: number of SNPs that the closest transition must be in (e.g. minimum resolution)

        Returns:
            - mat_recomb_lst (`list`): list of maternal recombination snp indexes
            - pat_recomb_lst (`list`): list of paternal recombination snp indexes
            - mat_recomb (`dict`): dictionary of snp-index x sibling counts for maternal recombinations
            - pat_recomb (`dict`): dictionary of snp-index x sibling counts for paternal recombinations

        """
        mat_recomb = {}
        pat_recomb = {}
        for path_xz in path_xzs:
            assert path_xy.size == path_xz.size
            transitions_01 = np.where(path_xy[:-1] != path_xy[1:])[0]
            transitions_02 = np.where(path_xz[:-1] != path_xz[1:])[0]
            for r in transitions_01:
                # This is the targetted transition that we want to assign to maternal or paternal position.
                i0, j0 = path_xy[r], path_xy[r + 1]
                dists = np.sqrt((transitions_02 - r) ** 2)
                if np.any(dists < window):
                    # This gets the closest matching one
                    r2 = transitions_02[np.argmin(dists)]
                    i1, j1 = path_xz[r2], path_xz[r2 + 1]
                    m = self.det_recomb_sex(i0, j0)
                    m2 = self.det_recomb_sex(i1, j1)
                    if m == 0 and m2 == 0:
                        if r not in mat_recomb:
                            mat_recomb[r] = 1
                        else:
                            mat_recomb[r] = mat_recomb[r] + 1
                    if m == 1 and m2 == 1:
                        if r not in pat_recomb:
                            pat_recomb[r] = 1
                        else:
                            pat_recomb[r] = pat_recomb[r] + 1

        # NOTE: here we just get positions, and if they are supported by the majority rule ...
        mat_recomb_lst = [k for k in mat_recomb if mat_recomb[k] > len(path_xzs) / 2]
        pat_recomb_lst = [k for k in pat_recomb if pat_recomb[k] > len(path_xzs) / 2]
        # This returns the list of tuples on the recombination positions and minimum distances across the traces.
        return mat_recomb_lst, pat_recomb_lst, mat_recomb, pat_recomb

    def est_sharing(
        self,
        bafs,
        pos,
        mat_haps,
        pat_haps,
        pi0=(0.7, 0.7),
        std_dev=(0.15, 0.15),
        r=1e-8,
    ):
        """Estimate the proportion of the chromosome that is shared identically across siblings.

        Arguments:
            - bafs (`list`): list of two arrays of B-allele frequencies across m sites for two siblings
            - pos (`np.array`): m-length vector of basepair positions for sites
            - mat_haps (`np.array`): a 2 x m array of 0/1 maternal haplotypes
            - pat_haps (`np.array`): a 2 x m array of 0/1 paternal haplotypes
            - pi0 (`tuple float`): sparsity parameter for B-allele emission model
            - std_dev (`tuple float`): standard deviation for B-allele emission model
            - r (`float`): inter-state transition rate

        Returns:
            -  mat_haplo_len (`float`): the total length maternal haplo-identity
            -  pat_haplo_len (`float`): the total length paternal haplo-identity
            -  both_haplo_len (`float`): the total length identical between siblings
            -  tot_len (`float`): the total length of the variants typed on the chromosome

        """
        # 1. Estimate the simplified viterbi path through these states
        cur_viterbi_path = self.viterbi_path(
            bafs=bafs,
            pos=pos,
            mat_haps=mat_haps,
            pat_haps=pat_haps,
            pi0=pi0,
            std_dev=std_dev,
            r=r,
        )
        # 2. Estimate the amount of the chromosome in haplo-identical states
        maternal_haplo_idx = np.where(cur_viterbi_path == 0)[0]
        paternal_haplo_idx = np.where(cur_viterbi_path == 1)[0]
        both_haplo_idx = np.where(cur_viterbi_path == 2)[0]
        # 3. Now estimate the fraction of the genome explicitly ...
        aggregate_length = lambda p, idx: np.sum(
            [
                p1 - p0
                for (p0, p1, x0, x1) in zip(
                    p[idx][:-1],
                    p[idx][1:],
                    idx[:-1],
                    idx[1:],
                )
                if (x1 - 1 == x0)
            ]
        )
        tot_len = pos[-1] - pos[0]
        mat_haplo_len = aggregate_length(pos, maternal_haplo_idx)
        pat_haplo_len = aggregate_length(pos, paternal_haplo_idx)
        both_haplo_len = aggregate_length(pos, both_haplo_idx)
        return mat_haplo_len, pat_haplo_len, both_haplo_len, tot_len


class PocHMM(MetaHMM):
    """Class for estimating ploidy variation in duo-based PoC data."""

    def __init__(self, disomy=False):
        """Initialize the HMM."""
        super().__init__()

    def est_sigma_pi0(
        self,
        bafs,
        lrrs,
        sigmas,
        pos,
        haps,
        freqs=None,
        maternal=True,
        algo="L-BFGS-B",
        pi0_bounds=(0.01, 0.99),
        sigma_bounds=(1e-2, 0.5),
        **kwargs,
    ):
        """Estimate sigma and pi0 under the B-Allele Frequency model using optimization of forward algorithm likelihood.

        Arguments:
            - bafs (`np.array`): B-allele frequencies across the all m sites
            - pos (`np.array`): basepair positions of the SNPs
            - haps (`np.array`): a 2 x m array of 0/1 maternal haplotypes
            - freqs (`np.array`): an m array of allele frequencies as priors
            - algo (`str`): one of Nelder-Mead, L-BFGS-B, or Powell algorithms for optimization
            - pi0_bounds (`tuple`): bounds for acceptable values of pi0 parameter
            - sigma_bounds (`tuple`): bounds for acceptable values for sigma

        Returns:
            - pi0_est (`float`): estimate of sparsity parameter (pi0) for B-allele emission model
            - sigma_est (`float`): estimate of noise parameter (sigma) for B-allele emission model

        """
        assert algo in ["Nelder-Mead", "L-BFGS-B", "Powell"]
        assert (len(pi0_bounds) == 2) and (len(sigma_bounds) == 2)
        assert (pi0_bounds[0] > 0) and (pi0_bounds[1] > 0)
        assert (pi0_bounds[0] < 1) and (pi0_bounds[1] < 1)
        assert pi0_bounds[0] < pi0_bounds[1]
        assert (sigma_bounds[0] > 0) and (sigma_bounds[1] > 0)
        assert sigma_bounds[0] < sigma_bounds[1]
        mid_pi0 = np.mean(pi0_bounds)
        mid_sigma = np.mean(sigma_bounds)
        opt_res = minimize(
            lambda x: (
                -self.forward_algorithm(
                    bafs=bafs,
                    lrrs=lrrs,
                    sigmas=sigmas,
                    pos=pos,
                    haps=haps,
                    freqs=freqs,
                    maternal=maternal,
                    pi0=x[0],
                    std_dev=x[1],
                    **kwargs,
                )[4]
            ),
            x0=[mid_pi0, mid_sigma],
            method=algo,
            bounds=[pi0_bounds, sigma_bounds],
            tol=1e-4,
            options={"disp": True, "ftol": 1e-4, "xtol": 1e-4},
        )
        pi0_est = opt_res.x[0]
        sigma_est = opt_res.x[1]
        return pi0_est, sigma_est

    def forward_algorithm(
        self,
        bafs,
        lrrs,
        sigmas,
        pos,
        haps,
        freqs=None,
        maternal=True,
        pi0=0.2,
        std_dev=0.25,
        r=1e-8,
        a=1e-2,
        unphased=False,
        obs_err=0.0,
    ):
        """Forward algorithm for duos.

        Arguments:
            - bafs (`np.array`): B-allele frequencies across the all m sites
            - pos (`np.array`): m-length vector of basepair positions for sites
            - haps (`np.array`): a 2 x m array of 0/1 parental haplotypes
            - freqs (`np.array`): an m-length array of freqs
            - pi0 (`float`): sparsity parameter for B-allele emission model
            - std_dev (`float`): standard deviation for B-allele emission model
            - r (`float`): intra-karyotype transition rate (recombination)
            - a (`float`): inter-karyotype transition rate
            - unphased (`bool`): run the model in unphased mode
            - obs_err (`float`): per-site genotyping error rate for the observed parent

        Returns:
            - alphas (`np.array`): forward variable from hmm across k states
            - scaler (`np.array`): m-length array of scale parameters
            - states (`list`): tuple representation of states
            - karyotypes (`np.array`):  array of karyotypes in the MetaHMM model
            - loglik (`float`): total log-likelihood of B-allele frequency

        """
        assert bafs.ndim == 1
        assert lrrs.ndim == 1
        assert sigmas.ndim == 1
        assert bafs.size == lrrs.size
        assert lrrs.size == sigmas.size
        assert pos.ndim == 1
        assert haps.ndim == 2
        assert (pi0 > 0) & (pi0 < 1.0)
        assert std_dev > 0
        assert bafs.size == haps.shape[1]
        assert bafs.size == pos.size
        assert np.all(pos[1:] > pos[:-1])
        assert r < 0.5 and r > 0
        assert a < 0.5 and a > 0
        assert 0.0 <= obs_err < 0.5
        if freqs is not None:
            assert freqs.size == bafs.size
        else:
            freqs = np.repeat(-1, bafs.size)
        alphas, scaler, _, _, loglik = forward_algo_duo(
            bafs,
            lrrs,
            sigmas,
            pos,
            haps,
            freqs,
            self.states,
            self.karyotypes,
            maternal=maternal,
            r=r,
            a=a,
            pi0=pi0,
            std_dev=std_dev,
            obs_err=obs_err,
        )
        return alphas, scaler, self.states, self.karyotypes, loglik

    def backward_algorithm(
        self,
        bafs,
        lrrs,
        sigmas,
        pos,
        haps,
        freqs=None,
        maternal=True,
        pi0=0.2,
        std_dev=0.25,
        r=1e-8,
        a=1e-2,
        unphased=False,
        obs_err=0.0,
    ):
        """Backward algorithm for duos.

        Arguments:
            - bafs (`np.array`): B-allele frequencies across the all m sites
            - pos (`np.array`): m-length vector of basepair positions for sites
            - haps (`np.array`): a 2 x m array of 0/1 parental haplotypes
            - freqs (`np.array`): an m-length array of freqs
            - pi0 (`float`): sparsity parameter for B-allele emission model
            - std_dev (`float`): standard deviation for B-allele emission model
            - r (`float`): intra-karyotype transition rate (recombination)
            - a (`float`): inter-karyotype transition rate
            - unphased (`bool`): run the model in unphased mode
            - obs_err (`float`): per-site genotyping error rate for the observed parent

        Returns:
            - betas (`np.array`): backward variable from hmm across k states
            - scaler (`np.array`): m-length array of scale parameters
            - states (`list`): tuple representation of states
            - karyotypes (`np.array`):  array of karyotypes in the MetaHMM model
            - loglik (`float`): total log-likelihood of B-allele frequency

        """
        assert bafs.ndim == 1
        assert lrrs.ndim == 1
        assert sigmas.ndim == 1
        assert bafs.size == lrrs.size
        assert lrrs.size == sigmas.size
        assert pos.ndim == 1
        assert haps.ndim == 2
        assert (pi0 > 0) & (pi0 < 1.0)
        assert std_dev > 0
        assert bafs.size == haps.shape[1]
        assert bafs.size == pos.size
        assert np.all(pos[1:] > pos[:-1])
        assert r < 0.5 and r > 0
        assert a < 0.5 and a > 0
        assert 0.0 <= obs_err < 0.5
        if freqs is not None:
            assert freqs.size == bafs.size
        else:
            freqs = np.repeat(-1, bafs.size)
        betas, scaler, _, _, loglik = backward_algo_duo(
            bafs,
            lrrs,
            sigmas,
            pos,
            haps,
            freqs,
            self.states,
            self.karyotypes,
            maternal=maternal,
            r=r,
            a=a,
            pi0=pi0,
            std_dev=std_dev,
            obs_err=obs_err,
        )
        return betas, scaler, self.states, self.karyotypes, loglik

    def forward_backward(
        self,
        bafs,
        lrrs,
        sigmas,
        pos,
        haps,
        freqs=None,
        maternal=True,
        pi0=0.2,
        std_dev=0.25,
        r=1e-8,
        a=1e-2,
        unphased=False,
        obs_err=0.0,
    ):
        """Forward-backward algorithm for duos.

        Arguments:
            - bafs (`np.array`): B-allele frequencies across the all m sites
            - pos (`np.array`): m-length vector of basepair positions for sites
            - haps (`np.array`): a 2 x m array of 0/1 parental haplotypes
            - freqs (`np.array`): an m-length array of freqs
            - pi0 (`float`): sparsity parameter for B-allele emission model
            - std_dev (`float`): standard deviation for B-allele emission model
            - r (`float`): intra-karyotype transition rate (recombination)
            - a (`float`): inter-karyotype transition rate
            - unphased (`bool`): run the model in unphased mode
            - obs_err (`float`): per-site genotyping error rate for the observed parent

        Returns:
            - gammas (`np.array`): log posterior density of being in each of k hidden states
            - states (`list`): tuple representation of states
            - karyotypes (`np.array`):  array of karyotypes in the MetaHMM model

        """
        alphas, _, states, karyotypes, _ = self.forward_algorithm(
            bafs,
            lrrs,
            sigmas,
            pos,
            haps,
            freqs,
            maternal=maternal,
            pi0=pi0,
            std_dev=std_dev,
            r=r,
            a=a,
            unphased=unphased,
            obs_err=obs_err,
        )
        betas, _, _, _, _ = self.backward_algorithm(
            bafs,
            lrrs,
            sigmas,
            pos,
            haps,
            freqs,
            maternal=maternal,
            pi0=pi0,
            std_dev=std_dev,
            r=r,
            a=a,
            unphased=unphased,
            obs_err=obs_err,
        )
        gammas = (alphas + betas) - logsumexp_sp(alphas + betas, axis=0)
        return gammas, states, karyotypes

    def genotype_parent(
        self,
        bafs,
        lrrs,
        sigmas,
        haps,
        gammas,
        freqs=None,
        maternal=True,
        pi0=0.2,
        std_dev=0.25,
    ):
        """Obtain a matrix of genotype dosages/posteriors for the unobserved parent."""
        assert bafs.ndim == 1
        assert lrrs.ndim == 1
        assert sigmas.ndim == 1
        assert bafs.size == lrrs.size
        assert sigmas.size == sigmas.size
        assert haps.ndim == 2
        assert (pi0 > 0) & (pi0 < 1.0)
        assert std_dev > 0
        assert bafs.size == haps.shape[1]
        assert gammas.ndim == 2
        assert gammas.shape[0] == self.karyotypes.size
        if freqs is not None:
            assert freqs.size == bafs.size
        else:
            freqs = np.repeat(0.5, bafs.size)
        ks = [sum([s >= 0 for s in state]) for state in self.states]
        n = bafs.size
        m = len(self.states)
        geno_dosage = np.zeros(shape=(4, n), dtype=np.float32)
        geno = [[0, 0], [0, 1], [1, 0], [1, 1]]
        for i in range(n):
            f = freqs[i]

            for idx, (x, p) in enumerate(
                zip(geno, ((1 - f) ** 2, f * (1 - f), f * (1 - f), f**2))
            ):
                cur_emissions = np.zeros(m)
                for j in range(m):
                    if maternal:
                        m_ij = mat_dosage(haps[:, i], self.states[j])
                        p_ij = pat_dosage(x, self.states[j])
                    else:
                        m_ij = mat_dosage(x, self.states[j])
                        p_ij = pat_dosage(haps[:, i], self.states[j])
                    # dosage is proportional to likelihood * prior * posterior of specific state
                    cur_emissions[j] = (
                        emission_baf(
                            bafs[i],
                            m_ij,
                            p_ij,
                            pi0=pi0,
                            std_dev=std_dev,
                            k=ks[j],
                        )
                        + emission_lrr(lrrs[i], k=ks[j], std_dev=sigmas[i])
                        + np.log(p)
                        + gammas[j, i]
                    )
                geno_dosage[idx, i] = logsumexp(cur_emissions)
        # Now rescale the dosage estimates here ...
        geno_dosage_rev = np.zeros(shape=(3, n))
        for i in range(n):
            tot = logsumexp_sp(geno_dosage[:, i])
            geno_dosage_rev[0, i] = geno_dosage[0, i] - tot
            geno_dosage_rev[1, i] = (
                logaddexp(geno_dosage[1, i], geno_dosage[2, i]) - tot
            )
            geno_dosage_rev[2, i] = geno_dosage[3, i] - tot
        return geno_dosage_rev

    def infer_missing_af(self, bafs, geno, eps=0.1):
        """Estimate the opposite parents allele frequency conditional on BAF."""
        raise NotImplementedError("Method is not currently implemented!")


class MccEst:
    """Class containing estimator of maternal-cell contamination."""

    def __init__(self):
        """Initialize the maternal cell contamination estimates."""
        pass

    def loglik_mcc_poc(self, bafs, mat_haps, freqs, c=0.0, std_dev=0.1):
        """Calculate the log-likelihood of MCC for the POC samples."""
        assert bafs.ndim == 1
        assert mat_haps.ndim == 2
        assert bafs.size == mat_haps.shape[1]
        assert freqs.ndim == 1
        assert freqs.size == bafs.size
        assert np.all((bafs >= 0) & (bafs <= 1))
        assert np.all((freqs >= 0) & (freqs <= 1))
        assert (c >= 0) and (c <= 0.5)
        assert std_dev > 0
        mat_geno = np.sum(mat_haps, axis=0).astype(np.int32)
        assert np.all(np.isin(mat_geno, [0, 1, 2]))
        logll = 0.0
        for i in range(bafs.size):
            ll_p0 = 2 * np.log(1.0 - freqs[i]) + loglik_mcc(
                baf=bafs[i], mg=mat_geno[i], pg=0, c=c, std_dev=std_dev
            )
            ll_p1 = np.log(2 * freqs[i] * (1 - freqs[i])) + loglik_mcc(
                baf=bafs[i], mg=mat_geno[i], pg=1, c=c, std_dev=std_dev
            )
            ll_p2 = 2 * np.log(freqs[i]) + loglik_mcc(
                baf=bafs[i], mg=mat_geno[i], pg=2, c=c, std_dev=std_dev
            )
            logll += logsumexp(np.array([ll_p0, ll_p1, ll_p2]))
        return logll

    def loglik_mcc_trio(self, bafs, mat_haps, pat_haps, c=0.0, std_dev=0.1):
        """Calculate the log-likelihood of MCC for the POC samples."""
        assert bafs.ndim == 1
        assert mat_haps.ndim == 2
        assert pat_haps.ndim == 2
        assert bafs.size == mat_haps.shape[1]
        assert bafs.size == pat_haps.shape[1]
        assert np.all((bafs >= 0) & (bafs <= 1))
        assert (c >= 0) and (c <= 0.5)
        assert std_dev > 0
        mat_geno = np.sum(mat_haps, axis=0)
        pat_geno = np.sum(pat_haps, axis=0)
        assert np.all(np.isin(mat_geno, [0, 1, 2]))
        assert np.all(np.isin(pat_geno, [0, 1, 2]))
        logll = 0.0
        for i in range(bafs.size):
            logll += loglik_mcc(
                baf=bafs[i],
                mg=mat_geno[i],
                pg=pat_geno[i],
                c=c,
                std_dev=std_dev,
            )
        return logll

    def est_mcc_poc(self, bafs, mat_haps, freqs, algo="Nelder-Mead", **kwargs):
        """Estimate maternal cell-contamination using MLE within mother-child duos."""
        opt_res = minimize(
            lambda x: (
                -self.loglik_mcc_poc(
                    bafs=bafs,
                    mat_haps=mat_haps,
                    freqs=freqs,
                    c=x[0],
                    std_dev=x[1],
                )
            ),
            x0=[0.05, 0.1],
            method=algo,
            bounds=[(0, 0.5), (1e-3, 0.3)],
            **kwargs,
        )
        c_est = opt_res.x[0]
        sigma_est = opt_res.x[1]
        return c_est, sigma_est

    def est_mcc_trio(self, bafs, mat_haps, pat_haps, algo="Nelder-Mead", **kwargs):
        """Estimate maternal cell-contamination using MLE within full trios."""
        opt_res = minimize(
            lambda x: (
                -self.loglik_mcc_trio(
                    bafs=bafs,
                    mat_haps=mat_haps,
                    pat_haps=pat_haps,
                    c=x[0],
                    std_dev=x[1],
                )
            ),
            x0=[0.05, 0.1],
            method=algo,
            bounds=[(0, 0.5), (1e-3, 0.3)],
            **kwargs,
        )
        c_est = opt_res.x[0]
        sigma_est = opt_res.x[1]
        return c_est, sigma_est

    def mcc_ci_poc(
        self, bafs, mat_haps, freqs, c_hat=0.0, std_dev=0.1, alpha=0.95, df=1
    ):
        """Obtain a confidence interval for the MCC estimates in the POC model using a profile-likelihood."""
        assert (c_hat >= 0) and (c_hat <= 0.5)
        assert std_dev > 0
        assert (alpha > 0) and (alpha < 1)
        wilks = lambda x: (
            2
            * (
                self.loglik_mcc_poc(
                    bafs=bafs, mat_haps=mat_haps, freqs=freqs, c=c_hat, std_dev=std_dev
                )
                - self.loglik_mcc_poc(
                    bafs=bafs, mat_haps=mat_haps, freqs=freqs, c=x, std_dev=std_dev
                )
            )
        )
        qval = chi2.ppf(alpha, df=df)
        try:
            lower_CI = brentq(lambda x: wilks(x) - qval, 1e-4, c_hat)
        except ValueError:
            lower_CI = 0.0
        try:
            upper_CI = brentq(lambda x: wilks(x) - qval, c_hat, 0.5)
        except ValueError:
            upper_CI = 0.5
        return (lower_CI, c_hat, upper_CI)

    def mcc_ci_trio(
        self, bafs, mat_haps, pat_haps, c_hat=0.0, std_dev=0.1, h=1e-5, alpha=0.95, df=1
    ):
        """Obtain the CI for the estimated contamination in trios using a profile-likelihood."""
        assert (c_hat >= 0) and (c_hat <= 0.5)
        assert std_dev > 0
        assert (alpha > 0) and (alpha < 1)
        assert h > 0
        wilks = lambda x: (
            2
            * (
                self.loglik_mcc_trio(
                    bafs=bafs,
                    mat_haps=mat_haps,
                    pat_haps=pat_haps,
                    c=c_hat,
                    std_dev=std_dev,
                )
                - self.loglik_mcc_trio(
                    bafs=bafs,
                    mat_haps=mat_haps,
                    pat_haps=pat_haps,
                    c=x,
                    std_dev=std_dev,
                )
            )
        )
        qval = chi2.ppf(alpha, df=df)
        try:
            lower_CI = brentq(lambda x: wilks(x) - qval, 1e-4, c_hat)
        except ValueError:
            lower_CI = 0.0
        try:
            upper_CI = brentq(lambda x: wilks(x) - qval, c_hat, 0.5)
        except ValueError:
            upper_CI = 0.5
        return (lower_CI, c_hat, upper_CI)


class MosaicEst:
    """Class to perform estimation of mosaic rates."""

    def __init__(self, mat_haps, pat_haps, bafs, pos):
        """Initialize the class."""
        assert mat_haps.ndim == 2
        assert pat_haps.ndim == 2
        assert bafs.ndim == 1
        assert bafs.size == mat_haps.shape[1]
        assert bafs.size == pat_haps.shape[1]
        assert pos.size == bafs.size
        self.mat_haps = mat_haps
        self.pat_haps = pat_haps
        self.bafs = bafs
        self.pos = pos
        self.het_bafs = None
        self.A = None
        self.mle_theta = None

    def baf_hets(self):
        """Compute the BAF at expected heterozygotes in the embryo."""
        mat_geno = np.sum(self.mat_haps, axis=0)
        pat_geno = np.sum(self.pat_haps, axis=0)
        exp_het_idx = ((mat_geno == 0) & (pat_geno == 2)) | (
            (mat_geno == 2) & (pat_geno == 0)
        )
        self.het_bafs = self.bafs[exp_het_idx]
        self.n_het = self.het_bafs.size
        self.signed_baf = False
        if self.n_het < 10:
            raise ValueError("Fewer than 10 expected heterozygotes!")
        assert exp_het_idx.ndim == 1
        self.het_idx = np.flatnonzero(exp_het_idx)

    def viterbi_hets(self, **kwargs):
        """Predict the embryo genotype using viterbi traceback from parental haplotypes (under disomy)."""
        meta_hmm = MetaHMM(disomy=True)
        path, karyo, _, _ = meta_hmm.viterbi_algorithm(
            pos=self.pos,
            mat_haps=self.mat_haps,
            pat_haps=self.pat_haps,
            bafs=self.bafs,
            **kwargs,
        )
        embryo_geno = np.zeros(path.size, dtype=np.int32)
        m = self.pos.size
        assert embryo_geno.size == m
        for i in range(m):
            embryo_geno[i] = (
                self.mat_haps[karyo[path[i]][0], i]
                + self.pat_haps[karyo[path[i]][2], i]
            )
        pred_het_idx = embryo_geno == 1
        self.het_bafs = self.bafs[pred_het_idx]
        assert pred_het_idx.ndim == 1
        self.het_idx = np.flatnonzero(pred_het_idx)
        self.n_het = self.het_bafs.size
        if self.n_het < 10:
            raise ValueError("Fewer than 10 expected heterozygotes!")
        baf_signs = np.zeros(self.het_idx.size)
        for i, j in enumerate(self.het_idx):
            # If the alt allele is maternal we give this a positive sign ...
            if self.mat_haps[karyo[path[j]][0], j] == 1:
                baf_signs[i] = 1
            else:
                baf_signs[i] = -1
        self.signed_baf = True
        self.phased_baf = baf_signs * np.abs(self.het_bafs - 0.5)

    def phase_hets(self, **kwargs):
        """Inferring the phase of heterozygotes for signed BAF."""
        meta_hmm = MetaHMM(disomy=True)
        path, karyo, _, _ = meta_hmm.viterbi_algorithm(
            pos=self.pos,
            mat_haps=self.mat_haps,
            pat_haps=self.pat_haps,
            bafs=self.bafs,
            **kwargs,
        )
        # Conduct the phasing of heterozygotes here
        baf_signs = np.zeros(self.het_idx.size)
        for i, j in enumerate(self.het_idx):
            # If the alt allele is maternal we give this a positive sign ...
            if self.mat_haps[karyo[path[j]][0], j] == 1:
                baf_signs[i] = 1
            else:
                baf_signs[i] = -1
        self.signed_baf = True
        self.phased_baf = baf_signs * np.abs(self.het_bafs - 0.5)

    def create_transition_matrix(self, switch_err=0.01, t_rate=1e-4):
        """Create the transition matrix.

        NOTE: should allow for asymmetry in the t_rates ...

        Arguments:
            - switch_err (`float`): rate parameter sibling copying model...
            - t_rate (`float`): the actual transition rate probability

        Returns:
            - A (`np.array`): state x state matrix of log-transition rates

        """
        assert (switch_err > 0) and (switch_err <= 0.05)
        assert (t_rate > 0) and (t_rate < 0.5)
        A = np.zeros(shape=(3, 3))
        # Just make this as a kind of switch error rate or something?
        # 1. Transition rates here are the switch error rate
        A[0, 2] = switch_err
        A[2, 0] = switch_err
        # 2. Transition rates
        A[0, 1] = t_rate
        A[2, 1] = t_rate
        A[1, 0] = t_rate
        A[1, 2] = t_rate
        A[0, 0] = 1.0 - np.sum(A[0, :])
        A[1, 1] = 1.0 - np.sum(A[1, :])
        A[2, 2] = 1.0 - np.sum(A[2, :])
        self.A = np.log(A)

    def forward_algo_mix(self, theta=0.0, std_dev=0.1):
        """Implement the forward-algorithm for the 3-state HMM under the Loh et al model."""
        assert theta >= 0
        assert std_dev > 0.0
        assert self.A is not None
        assert self.signed_baf
        n = self.het_bafs.size
        m = 3
        alphas = np.zeros(shape=(m, n))
        alphas[:, 0] = np.log(1.0 / m)
        zs = [-theta, 0, theta]
        # NOTE: I wonder if you don't have to use the truncation here and can instead just use the baf - 0.5f
        for j in range(3):
            alphas[j, 0] += norm_logl(self.phased_baf[0], zs[j], std_dev)
        scaler = np.zeros(n)
        scaler[0] = logsumexp(alphas[:, 0])
        alphas[:, 0] -= scaler[0]
        for i in range(1, n):
            for j in range(3):
                alphas[j, i] += norm_logl(self.phased_baf[i], zs[j], std_dev)
                alphas[j, i] += logsumexp(self.A[:, j] + alphas[:, (i - 1)])
            scaler[i] = logsumexp(alphas[:, i])
            alphas[:, i] -= scaler[i]
        return alphas, scaler, np.sum(scaler)

    def viterbi_algo_mix(self, theta=0.0, std_dev=0.1):
        """Viterbi algorithm implementation for phased BAF decoding."""
        assert theta >= 0
        assert std_dev >= 0.0
        assert self.A is not None
        assert self.signed_baf
        n = self.het_bafs.size
        m = 3
        deltas = np.zeros(shape=(m, n))
        deltas[:, 0] = np.log(1.0 / m)
        psi = np.zeros(shape=(m, n), dtype=int)
        zs = [-theta, 0.0, theta]
        for i in range(1, n):
            for j in range(3):
                deltas[j, i] = np.max(deltas[:, i - 1] + self.A[:, j])
                deltas[j, i] += norm_logl(self.phased_baf[i], zs[j], std_dev)
                psi[j, i] = np.argmax(deltas[:, i - 1] + self.A[:, j]).astype(int)
        path = np.zeros(n, dtype=int)
        path[-1] = np.argmax(deltas[:, -1]).astype(int)
        for i in range(n - 2, -1, -1):
            path[i] = psi[path[i + 1], i]
        path[0] = psi[path[1], 1]
        return path, zs, deltas, psi

    def lrt_theta(self, std_dev=0.1):
        """LRT of Theta not being 0."""
        assert std_dev > 0.0
        ll_h0 = self.forward_algo_mix(theta=0.0, std_dev=std_dev)[2]
        if self.mle_theta is None:
            self.est_mle_theta(std_dev=std_dev)
        if ~np.isnan(self.mle_theta):
            ll_h1 = self.forward_algo_mix(theta=self.mle_theta, std_dev=std_dev)[2]
            return -2 * (ll_h0 - ll_h1)
        else:
            return np.nan

    def est_mle_theta(self, std_dev=0.1):
        """Estimate the MLE estimate of theta."""
        try:
            f = lambda x: -self.forward_algo_mix(theta=x, std_dev=std_dev)[2]
            opt_res = minimize(
                f, x0=[0.1], method="Powell", tol=1e-4, bounds=[(0.0, 0.5)]
            )
            self.mle_theta = opt_res.x[0]
        except ValueError:
            self.mle_theta = np.nan

    def ci_mle_theta(self, std_dev=0.1, h=1e-6):
        """Estimate the 95% confidence interval of the MLE-theta estimate.

        NOTE: this uses a symmetric second derivative appx to the Fisher Information.
        """
        assert (self.mle_theta is not None) and (~np.isnan(self.mle_theta))
        assert (h >= 0) and (h <= 0.01)
        ci_mle_theta = [np.nan, np.nan, np.nan]
        try:
            f = lambda x: self.forward_algo_mix(theta=x, std_dev=std_dev)[2]
            if self.mle_theta < h:
                logI = (
                    f(self.mle_theta + h) - 2 * f(self.mle_theta) + f(self.mle_theta)
                ) / (h**2)
            elif self.mle_theta > (0.5 - h):
                logI = (
                    f(self.mle_theta) - 2 * f(self.mle_theta) + f(self.mle_theta - h)
                ) / (h**2)
            else:
                logI = (
                    f(self.mle_theta + h)
                    - 2 * f(self.mle_theta)
                    + f(self.mle_theta - h)
                ) / (h**2)
            fisher_I_inv = 1.0 / -logI

            ci_mle_theta[0] = self.mle_theta - 1.96 * np.sqrt(
                1.0 / self.n_het * fisher_I_inv
            )
            ci_mle_theta[1] = self.mle_theta
            ci_mle_theta[2] = self.mle_theta + 1.96 * np.sqrt(
                1.0 / self.n_het * fisher_I_inv
            )
            ci_mle_theta[0] = max(0.0, ci_mle_theta[0])
            ci_mle_theta[2] = min(0.5, ci_mle_theta[2])
        except ValueError:
            pass
        return ci_mle_theta

    def est_cf(self, theta=0.0, gain=True):
        """Estimate mosaic cell fraction from allelic intensity mean shift."""
        assert (theta >= 0.0) and (theta <= 0.5)
        cn_est = lambda cn: np.abs(1 / cn - 0.5)
        cf_est = lambda cn: np.abs(2.0 - cn)
        if np.isnan(theta):
            return np.nan
        else:
            if gain:
                try:
                    return cf_est(brentq(lambda x: cn_est(x) - theta, 2.0, 3.0))
                except ValueError:
                    return 1.0
            else:
                return cf_est(brentq(lambda x: cn_est(x) - theta, 1.0, 2.0))


class PhaseCorrect:
    """Module for implementing Mendelian phase correction using BAF data.

    Implements the key methods for phasing parental haplotypes using Mendelian phasing via a LOD-score

    """

    def __init__(self, mat_haps, pat_haps, pos):
        """Intialize the class for phase correction.

        Arguments:
            - mat_haps (`np.array`): a 2 x m array of 0/1 maternal haplotypes
            - pat_haps (`np.array`): a 2 x m array of 0/1 paternal haplotypes
            - pos (`np.array`): a m-length array of bp-position

        Returns:
            - PhaseCorrect Object

        """
        assert mat_haps.shape[0] == pat_haps.shape[0]
        assert mat_haps.shape[1] == pat_haps.shape[1]
        assert np.all(np.isin(mat_haps, [0, 1]))
        assert np.all(np.isin(mat_haps, [0, 1]))
        assert pos.ndim == 1
        assert pos.size == mat_haps.shape[1]
        self.mat_haps = mat_haps
        self.pat_haps = pat_haps
        self.pos = pos
        self.mat_haps_true = None
        self.pat_haps_true = None
        self.mat_haps_fixed = None
        self.pat_haps_fixed = None
        self.embryo_bafs = None
        self.embryo_pi0s = None
        self.embryo_sigmas = None

    def add_true_haps(self, true_mat_haps, true_pat_haps):
        """Add in true haplotypes if available from a simulation.

        Arguments:
            - true_mat_haps (`np.array`): a 2 x m array of 0/1 maternal haplotypes
            - true_pat_haps (`np.array`): a 2 x m array of 0/1 paternal haplotypes

        """
        assert true_mat_haps.ndim == 2
        assert true_pat_haps.ndim == 2
        assert true_mat_haps.shape[0] == self.mat_haps.shape[0]
        assert true_mat_haps.shape[1] == self.mat_haps.shape[1]
        assert true_pat_haps.shape[0] == self.pat_haps.shape[0]
        assert true_pat_haps.shape[1] == self.pat_haps.shape[1]
        self.mat_haps_true = true_mat_haps
        self.pat_haps_true = true_pat_haps

    def add_baf(self, embryo_bafs=[]):
        """Add in BAF estimates for each embryo.

        Arguments:
            - embryo_bafs (`list`): a list of embryo BAF from the same parental individuals

        """
        assert len(embryo_bafs) > 0
        self.embryo_bafs = []
        for baf in embryo_bafs:
            assert baf.size == self.mat_haps.shape[1]
            assert baf.size == self.pat_haps.shape[1]
            self.embryo_bafs.append(baf)

    def est_sigma_pi0s(self, **kwargs):
        """Estimate the noise parameters under disomy for each sibling embryo."""
        assert self.embryo_bafs is not None
        hmm = MetaHMM(disomy=True)
        m = self.pos.size
        lrrs_missing = np.full(m, -9.0)
        sigmas_lrr = np.ones(m)
        pi0_est_acc = np.zeros(len(self.embryo_bafs))
        sigma_est_acc = np.zeros(len(self.embryo_bafs))
        for i, baf in enumerate(self.embryo_bafs):
            pi0_est, sigma_est = hmm.est_sigma_pi0(
                baf,
                self.pos,
                self.mat_haps,
                self.pat_haps,
                lrrs=lrrs_missing,
                sigmas=sigmas_lrr,
                **kwargs,
            )
            pi0_est_acc[i] = pi0_est
            sigma_est_acc[i] = sigma_est
        self.embryo_pi0s = pi0_est_acc
        self.embryo_sigmas = sigma_est_acc

    def correct_haps_viterbi(self, haps, paths):
        """Correct haplotypes using multiple copying paths + majority rule."""
        assert paths.shape[1] == haps.shape[1]
        assert paths.shape[0] > 1
        n_sib = paths.shape[0]
        m = haps.shape[1]
        n_mis = np.zeros(m - 1)
        for i, j in zip(np.arange(m - 1), np.arange(1, m)):
            n_mis[i] = np.sum((paths[:, i] != paths[:, j]))
        hap_idx1 = np.zeros(haps.shape[1], dtype=np.uint16)
        hap_idx2 = np.ones(haps.shape[1], dtype=np.uint16)
        for i in np.arange(m - 1):
            # If majority have the same switch - we swap haplotypes ...
            if n_mis[i] > (n_sib / 2):
                hap_idx1[:i] = 1 - hap_idx1[:i]
                hap_idx2[:i] = 1 - hap_idx2[:i]
        fixed_hap1 = [haps[x, i] for i, x in enumerate(hap_idx1)]
        fixed_hap2 = [haps[x, i] for i, x in enumerate(hap_idx2)]
        fixed_haps = np.vstack([fixed_hap1, fixed_hap2])
        assert fixed_haps.shape == haps.shape
        return n_mis, fixed_haps

    def viterbi_phase_correct(self, niter=5, r=1e-8):
        """Use the Viterbi decoding for phase correction in multiple iterations.

        Arguments:
            - niter (`int`): number of iterations to go through for viterbi-based phase correction
            - r (`float`): recombination rate per basepair

        """
        assert niter > 0
        hmm = MetaHMM(disomy=True)
        mat_haps = self.mat_haps
        pat_haps = self.pat_haps
        n_sibs = len(self.embryo_bafs)
        m = self.pos.size
        lrrs_missing = np.full(m, -9.0)
        sigmas_lrr = np.ones(m)
        n_mis_mat_tot = np.zeros(niter)
        n_mis_pat_tot = np.zeros(niter)
        for i in range(niter):
            mat_paths = np.zeros(shape=(n_sibs, m))
            pat_paths = np.zeros(shape=(n_sibs, m))
            for j, baf in enumerate(self.embryo_bafs):
                path, _, _, _ = hmm.viterbi_algorithm(
                    baf,
                    lrrs_missing,
                    sigmas_lrr,
                    self.pos,
                    mat_haps,
                    pat_haps,
                    r=r,
                    pi0=self.embryo_pi0s[j],
                    std_dev=self.embryo_sigmas[j],
                )
                # Collect indicators of both copying the same haplotype
                mat_paths[j, :] = np.isin(path, [0, 1])
                pat_paths[j, :] = np.isin(path, [0, 2])
            # now we apply the minimum recombination path-switching setting ...
            n_mis_mat, fixed_mat_haps = self.correct_haps_viterbi(mat_haps, mat_paths)
            n_mis_pat, fixed_pat_haps = self.correct_haps_viterbi(pat_haps, pat_paths)
            n_mis_mat_tot[i] = np.sum(n_mis_mat)
            n_mis_pat_tot[i] = np.sum(n_mis_pat)
            # NOTE: print out that the mismatches have been minimized on both paternal & maternal chromosomes this way?
            mat_haps = fixed_mat_haps
            pat_haps = fixed_pat_haps
        # Set the corrected haplotypes here ...
        self.mat_haps_fixed = mat_haps
        self.pat_haps_fixed = pat_haps
        return mat_haps, pat_haps, n_mis_mat_tot, n_mis_pat_tot

    def flag_parental_genotype_errors(self, use_fixed=False, r=1e-8):
        """Flag potential parental genotype errors using multiple euploid siblings.

        Runs forward-backward under disomy for each sibling embryo and accumulates
        the posterior-weighted log Bayes factor error scores across all siblings.
        Genuine genotype errors are consistently flagged across siblings (same
        parental genotype), while per-embryo noise is uncorrelated and averages out.

        Arguments:
            - use_fixed (`bool`): use phase-corrected haplotypes if available (default: False)
            - r (`float`): recombination rate per basepair

        Returns:
            - mat_err_scores (`np.array`): m-length array of summed maternal error scores
            - pat_err_scores (`np.array`): m-length array of summed paternal error scores

        """
        assert self.embryo_bafs is not None
        assert self.embryo_pi0s is not None, "Call est_sigma_pi0s first"
        if use_fixed:
            assert self.mat_haps_fixed is not None, "Call viterbi_phase_correct first"
            mat_haps = self.mat_haps_fixed
            pat_haps = self.pat_haps_fixed
        else:
            mat_haps = self.mat_haps
            pat_haps = self.pat_haps
        m = self.pos.size
        lrrs_missing = np.full(m, -9.0)
        sigmas_lrr = np.ones(m)
        hmm = MetaHMM(disomy=True)
        mat_err_scores = np.zeros(m)
        pat_err_scores = np.zeros(m)
        for j, baf in enumerate(self.embryo_bafs):
            gammas, states, _ = hmm.forward_backward(
                baf,
                lrrs_missing,
                sigmas_lrr,
                self.pos,
                mat_haps,
                pat_haps,
                pi0=self.embryo_pi0s[j],
                std_dev=self.embryo_sigmas[j],
                r=r,
            )
            mat_err_j, pat_err_j = hmm.flag_parental_genotype_errors(
                gammas,
                states,
                baf,
                mat_haps,
                pat_haps,
                pi0=self.embryo_pi0s[j],
                std_dev=self.embryo_sigmas[j],
            )
            mat_err_scores += mat_err_j
            pat_err_scores += pat_err_j
        return mat_err_scores, pat_err_scores

    def estimate_switch_err_true(self, maternal=True, fixed=False):
        """Estimate the switch error from true and inferred haplotypes.

        The switch error is defined as consecutive heterozygotes that are
        in the incorrect orientation.

        Arguments:
            - maternal (`bool`): apply the function to the maternal haplotypes (default: True)
            - fixed (`bool`): apply the function to the fixed haplotypes (default: False)

        Returns:
            - n_switches (`int`): number of switches between consecutive heterozygotes
            - n_consecutive_hets (`int`): number of consecutive heterozygotes
            - switch_err_rate (`float`): number of switches per consecutive heterozygote
            - switch_idx (`np.array`): snps where the variant is out of phase with its predecessor
            - het_idx (`np.array`): locations/indexs of the heterozygotes
            - lods (`np.array`) : lod-scores for the estimated switches (default: None)

        """
        assert self.mat_haps_true is not None
        assert self.pat_haps_true is not None
        if maternal:
            true_haps = self.mat_haps_true
            if fixed:
                assert self.mat_haps_fixed is not None
                inf_haps = self.mat_haps_fixed
            else:
                inf_haps = self.mat_haps
        else:
            true_haps = self.pat_haps_true
            if fixed:
                assert self.pat_haps_fixed is not None
                inf_haps = self.pat_haps_fixed
            else:
                inf_haps = self.pat_haps
        # NOTE: this is just between all consecutive hets ...
        geno = true_haps.sum(axis=0)
        het_idxs = np.where(geno == 1)[0]
        n_switches = 0
        n_consecutive_hets = 0
        switch_idxs = []
        for i, j in zip(het_idxs[:-1], het_idxs[1:]):
            assert inf_haps[:, i].sum() == 1
            assert inf_haps[:, j].sum() == 1
            n_consecutive_hets += 1
            true_hap = true_haps[:, [i, j]]
            inf_hap = inf_haps[:, [i, j]]
            # Check if the heterozygotes are oriented appropriately
            if ~(
                np.all(true_hap[0, :] == inf_hap[0, :])
                or np.all(true_hap[0, :] == inf_hap[1, :])
            ):
                n_switches += 1
                switch_idxs.append((i, j))
        return (
            n_switches,
            n_consecutive_hets,
            n_switches / n_consecutive_hets,
            switch_idxs,
            het_idxs,
            None,
        )


class RecombEst(PhaseCorrect):
    """Class implementing the simplified algorithm for detection of crossovers of Coop et al 2007."""

    def __init__(self, **kwargs):
        """Use a superclass to preserve structure for sibling embryos."""
        super(RecombEst, self).__init__(**kwargs)

    def informative_markers(self, maternal=True):
        """Identify markers where one parent is heterozygous and the other parent is homozygous."""
        mat_geno = np.sum(self.mat_haps, axis=0)
        pat_geno = np.sum(self.pat_haps, axis=0)
        if maternal:
            info_idx = ((mat_geno == 1) & (pat_geno == 0)) | (
                (mat_geno == 1) & (pat_geno == 2)
            )
        else:
            info_idx = ((pat_geno == 1) & (mat_geno == 0)) | (
                (pat_geno == 1) & (mat_geno == 2)
            )
        return info_idx

    def refine_recomb_events(self, potential_switches, npad=5):
        """Refine recombination estimation using the switch-clusters approach of Coop et al 2007.

        Arguments:
            - potential_switches (`np.array`): array of potential switches at informative markers
            - npad (`int`): integer value of adjacent informative snps to consider as a switch cluster.

        Returns:
            - subset_co (`list`): only the isolated crossovers across all non-template embryos

        """
        assert npad > 1
        if potential_switches.size > 0:
            subset_co = []
            for i in potential_switches:
                # Count the number of switches within 5 informative SNPs
                n = np.sum(
                    (potential_switches < i + npad) & (potential_switches > i - npad)
                )
                # Even number of crossovers suggest it is likely not a well-supported CO
                if n % 2 == 1:
                    subset_co.append(i)
            return subset_co
        else:
            return []

    def isolate_recomb_events(
        self, template_embryo=0, maternal=True, ll_thresh=0, npad=5
    ):
        """Isolate specific recombination events.

        NOTE: this uses paternal by default!

        Arguments:
            - template_embryo (`int`): index of the template embryo
            - maternal (`bool`): indicator of estimating maternal crossovers.
            - ll_thresh (`float`): indicator of the likelihood threshold for transmission.
            - npad (`int`): integer value of adjacent informative snps to consider as a switch cluster.

        Returns:
            - Z (`np.array`): (m-1) x L array of switch indicators for allele copying in non-template embryos
            - llr_z (`np.array`): raw llr values for switch indicators for allele-copying in non-template embryos
            - potential_switches_filt (`np.array`): list of potential switches in template embryo.

        """
        assert self.embryo_bafs is not None
        assert self.embryo_pi0s is not None
        assert self.embryo_sigmas is not None
        m = len(self.embryo_bafs)
        assert m > 1
        assert template_embryo < m
        assert ll_thresh >= 0
        non_template_ids = [i for i in range(m) if i != template_embryo]
        # Get the informative snps for the specific parent
        info_snps = self.informative_markers(maternal=maternal)
        llr_z = np.zeros(shape=(len(non_template_ids), np.sum(info_snps)))
        for j, k in enumerate(non_template_ids):
            z1s = np.zeros(np.sum(info_snps))
            z2s = np.zeros(np.sum(info_snps))
            # NOTE: we use a signed-version of the log-likelihood ratio ...
            for p, i in enumerate(np.where(info_snps)[0]):
                z_same = np.zeros(2)
                z_diff = np.zeros(2)
                if maternal:
                    # Same allele is transmitted
                    z_same[0] = emission_baf(
                        self.embryo_bafs[template_embryo][i],
                        m=0,
                        p=self.pat_haps[0, i],
                        pi0=self.embryo_pi0s[template_embryo],
                        std_dev=self.embryo_sigmas[template_embryo],
                    ) + emission_baf(
                        self.embryo_bafs[k][i],
                        m=0,
                        p=self.pat_haps[0, i],
                        pi0=self.embryo_pi0s[k],
                        std_dev=self.embryo_sigmas[k],
                    )
                    z_same[1] = emission_baf(
                        self.embryo_bafs[template_embryo][i],
                        m=1,
                        p=self.pat_haps[0, i],
                        pi0=self.embryo_pi0s[template_embryo],
                        std_dev=self.embryo_sigmas[template_embryo],
                    ) + emission_baf(
                        self.embryo_bafs[k][i],
                        m=1,
                        p=self.pat_haps[0, i],
                        pi0=self.embryo_pi0s[k],
                        std_dev=self.embryo_sigmas[k],
                    )
                    # Same allele is not transmitted ...
                    z_diff[0] = emission_baf(
                        self.embryo_bafs[template_embryo][i],
                        m=0,
                        p=self.pat_haps[0, i],
                        pi0=self.embryo_pi0s[template_embryo],
                        std_dev=self.embryo_sigmas[template_embryo],
                    ) + emission_baf(
                        self.embryo_bafs[k][i],
                        m=1,
                        p=self.pat_haps[0, i],
                        pi0=self.embryo_pi0s[k],
                        std_dev=self.embryo_sigmas[k],
                    )
                    z_diff[1] = emission_baf(
                        self.embryo_bafs[template_embryo][i],
                        m=1,
                        p=self.pat_haps[0, i],
                        pi0=self.embryo_pi0s[template_embryo],
                        std_dev=self.embryo_sigmas[template_embryo],
                    ) + emission_baf(
                        self.embryo_bafs[k][i],
                        m=0,
                        p=self.pat_haps[0, i],
                        pi0=self.embryo_pi0s[k],
                        std_dev=self.embryo_sigmas[k],
                    )
                else:
                    # Same allele is transmitted
                    z_same[0] = emission_baf(
                        self.embryo_bafs[template_embryo][i],
                        m=self.mat_haps[0, i],
                        p=0,
                        pi0=self.embryo_pi0s[template_embryo],
                        std_dev=self.embryo_sigmas[template_embryo],
                    ) + emission_baf(
                        self.embryo_bafs[k][i],
                        m=self.mat_haps[0, i],
                        p=0,
                        pi0=self.embryo_pi0s[k],
                        std_dev=self.embryo_sigmas[k],
                    )
                    z_same[1] = emission_baf(
                        self.embryo_bafs[template_embryo][i],
                        m=self.mat_haps[0, i],
                        p=1,
                        pi0=self.embryo_pi0s[template_embryo],
                        std_dev=self.embryo_sigmas[template_embryo],
                    ) + emission_baf(
                        self.embryo_bafs[k][i],
                        m=self.mat_haps[0, i],
                        p=1,
                        pi0=self.embryo_pi0s[k],
                        std_dev=self.embryo_sigmas[k],
                    )
                    # Same allele is not transmitted ...
                    z_diff[0] = emission_baf(
                        self.embryo_bafs[template_embryo][i],
                        m=self.mat_haps[0, i],
                        p=0,
                        pi0=self.embryo_pi0s[template_embryo],
                        std_dev=self.embryo_sigmas[template_embryo],
                    ) + emission_baf(
                        self.embryo_bafs[k][i],
                        m=self.mat_haps[0, i],
                        p=1,
                        pi0=self.embryo_pi0s[k],
                        std_dev=self.embryo_sigmas[k],
                    )
                    z_diff[1] = emission_baf(
                        self.embryo_bafs[template_embryo][i],
                        m=self.mat_haps[0, i],
                        p=1,
                        pi0=self.embryo_pi0s[template_embryo],
                        std_dev=self.embryo_sigmas[template_embryo],
                    ) + emission_baf(
                        self.embryo_bafs[k][i],
                        m=self.mat_haps[0, i],
                        p=0,
                        pi0=self.embryo_pi0s[k],
                        std_dev=self.embryo_sigmas[k],
                    )
                z1s[p] = logsumexp(z_same)
                z2s[p] = logsumexp(z_diff)
            llrs = np.array(z1s) - np.array(z2s)
            llr_z[j, :] = llrs

        # Now we make a rough decision rule here for the likelihood-ratio supporting one or the other class ...
        # 1 indicates copying the same allele, -1 indicates that copying different alleles
        # 0 indicates a "missing" variant
        Z = np.zeros(shape=llr_z.shape)
        Z[llr_z < ll_thresh] = -1
        Z[llr_z > ll_thresh] = 1

        # # For each sibling embryo check its "switch-clusters"
        # isolated_switches = []
        # for i in range(len(non_template_ids)):
        #     # Check the switch cluster for sibling i ....
        #     potential_switches = np.where(Z[i, :-1] != Z[i, 1:])[0]
        #     potential_switches_filt = self.refine_recomb_events(
        #         potential_switches, npad=npad
        #     )
        #     isolated_switches.append(potential_switches_filt)

        isolated_switches = self.identify_switch_intervals(Z, npad=npad)
        # Check the total isolated switches ...
        isolated_switches = np.hstack(isolated_switches)
        switch_idx, cnts = np.unique(isolated_switches, return_counts=True)
        # Choose the potential switches by the majority rule ...
        potential_switches_filt = switch_idx[cnts > (m - 1) / 2].astype(int)
        switch_cnts_filt = cnts[cnts > (m - 1) / 2].astype(int)
        return Z, llr_z, potential_switches_filt, switch_cnts_filt

    def identify_switch_intervals(self, Z, npad=5):
        """Identify windows of switch intervals - accounting for missing/poor genotypes."""
        assert Z.ndim == 2
        assert Z.shape[1] > 0
        assert Z.shape[0] > 1
        assert npad > 1
        nsib = Z.shape[0]
        isolated_switches = []
        for i in range(nsib):
            zs = Z[i, :]
            asign = np.sign(zs)
            sz = asign == 0
            while sz.any():
                asign[sz] = np.roll(asign, 1)[sz]
                sz = asign == 0
            signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
            # NOTE: we don't consider the first signchange index...
            potential_switches = np.where(signchange[1:])[0]
            potential_switches_filt = self.refine_recomb_events(
                potential_switches, npad=npad
            )
            isolated_switches.append(potential_switches_filt)
        return isolated_switches

    def second_refine_recomb(
        self, template_embryo=0, maternal=True, start=None, end=None
    ):
        """Second attempt to refine the recombination locations."""
        assert (start is not None) and (end is not None)
        assert start < self.pos.size
        assert end < self.pos.size
        assert start < end
        embryo_ids = np.arange(len(self.embryo_bafs))
        assert template_embryo < embryo_ids.size
        # Make sure there are intermediate SNPs for refinement if necessary ...
        if end - start > 2:
            quad_hmm = QuadHMM()
            paths = []
            for j in embryo_ids:
                if j != template_embryo:
                    bafs = [
                        self.embryo_bafs[template_embryo][start:end],
                        self.embryo_bafs[j][start:end],
                    ]
                    # Use the sibling embryo paths here ...
                    v_path = quad_hmm.viterbi_path(
                        bafs=bafs,
                        pos=self.pos[start:end],
                        mat_haps=self.mat_haps[:, start:end],
                        pat_haps=self.pat_haps[:, start:end],
                        pi0=(self.embryo_pi0s[template_embryo], self.embryo_pi0s[j]),
                        std_dev=(
                            self.embryo_sigmas[template_embryo],
                            self.embryo_sigmas[j],
                        ),
                    )
                    paths.append(v_path)
            # Identify places where there are switches in the local copying paths?
            # Note that these indexes are within the shortened range now ...
            _, _, mat_rec_dict, pat_rec_dict = quad_hmm.isolate_recomb(
                paths[0], paths[1:], window=5
            )
            # They have to be at the same position across all the siblings ...
            mat_rec = [k for k in mat_rec_dict if mat_rec_dict[k] == len(paths)]
            pat_rec = [k for k in pat_rec_dict if pat_rec_dict[k] == len(paths)]
            if maternal:
                if (len(mat_rec) == 1) and (len(pat_rec) == 0):
                    het_idx = np.where((self.mat_haps.sum(axis=0) == 1))[0]
                    pos_het = self.pos[het_idx]
                    pos = self.pos[start:end][mat_rec[0]]
                    start = self.pos[np.argmax(pos_het > pos)]
                    end = self.pos[np.argmin(pos <= pos_het)]
            else:
                if (len(pat_rec) == 1) and (len(mat_rec) == 0):
                    het_idx = np.where((self.pat_haps.sum(axis=0) == 1))[0]
                    pos_het = self.pos[het_idx]
                    pos = self.pos[start:end][mat_rec[0]]
                    start = self.pos[np.argmax(pos_het > pos)]
                    end = self.pos[np.argmin(pos <= pos_het)]
        return self.pos[start], self.pos[end]

    def finalize_recomb_events(
        self, potential_switches, template_embryo=0, maternal=True
    ):
        """See if the location of the potential switches can be further localized."""
        if potential_switches == []:
            return []
        else:
            rec_locations = []
            # Get the locations of the informative markers
            info_idx = np.where(self.informative_markers(maternal=maternal))[0]
            assert info_idx.size > 0
            for p in potential_switches:
                # This is the position of the transition at the current resolution ...
                idx1, idx2 = info_idx[p], info_idx[p + 1]
                assert idx1 <= idx2
                (p1, p2) = self.second_refine_recomb(
                    template_embryo=template_embryo,
                    maternal=maternal,
                    start=idx1,
                    end=idx2,
                )
                rec_locations.append((p1, p2))
            return rec_locations

    def estimate_crossovers(
        self, template_embryo=0, ll_thresh=0, maternal=True, npad=5
    ):
        """Routine that actually does the FULL crossover estimation + interval refinement."""
        assert self.embryo_bafs is not None
        assert self.embryo_pi0s is not None
        assert self.embryo_sigmas is not None
        Z, llr_z, potential_switches, switch_cnts = self.isolate_recomb_events(
            template_embryo=template_embryo,
            maternal=maternal,
            ll_thresh=ll_thresh,
            npad=npad,
        )
        rec_loc = self.finalize_recomb_events(
            potential_switches, template_embryo=template_embryo, maternal=maternal
        )
        # We should have the same number of recombination events ...
        assert len(rec_loc) == potential_switches.size
        return rec_loc, switch_cnts
