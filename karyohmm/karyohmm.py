"""
Karyohmm is an HMM-based model for aneuploidy detection.

Karyohmm implements methods for haplotype based analyses
of copy number changes from array intensity data when
parental genotypes are available.

Modules available are:

- MetaHMM: module for whole chromosome aneuploidy determination via HMMs
- QuadHMM: module leveraging multi-sibling design for evaluating crossover recombination estimation
- PhaseCorrect: module which implements Mendelian phase correction for parental haplotypes.

"""

import numpy as np
from karyohmm_utils import (
    backward_algo,
    backward_algo_sibs,
    emission_baf,
    forward_algo,
    forward_algo_sibs,
    lod_phase,
    logsumexp,
    mat_dosage,
    mix_loglik,
    norm_logl,
    pat_dosage,
    viterbi_algo,
    viterbi_algo_sibs,
)
from scipy.optimize import brentq, brute, minimize
from scipy.special import logsumexp as logsumexp_sp


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
        algo="Powell",
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
            lambda x: -self.forward_algorithm(
                bafs=bafs,
                pos=pos,
                mat_haps=mat_haps,
                pat_haps=pat_haps,
                pi0=x[0],
                std_dev=x[1],
                **kwargs,
            )[4],
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
        algo="Nelder-Mead",
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

    def __init__(self, disomy=False):
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
        self.isodisomy_states = [(0, 1, -1, -1), (-1, -1, 0, 1)]
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
        if disomy:
            self.aploid = "disomy"
            self.states = self.disomy_states
            self.karyotypes = np.array(["2", "2", "2", "2"], dtype=str)
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

    def forward_algorithm(
        self,
        bafs,
        pos,
        mat_haps,
        pat_haps,
        pi0=0.5,
        std_dev=0.25,
        r=1e-8,
        a=1e-2,
        unphased=False,
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

        Returns:
            - alphas (`np.array`): forward variable from hmm across k states
            - scaler (`np.array`): m-length array of scale parameters
            - states (`list`): tuple representation of states
            - karyotypes (`np.array`):  array of karyotypes in the MetaHMM model
            - loglik (`float`): total log-likelihood of B-allele frequency

        """
        assert bafs.ndim == 1
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
        alphas, scaler, _, _, loglik = forward_algo(
            bafs,
            pos,
            mat_haps,
            pat_haps,
            self.states,
            self.karyotypes,
            r=r,
            a=a,
            pi0=pi0,
            std_dev=std_dev,
        )
        return alphas, scaler, self.states, self.karyotypes, loglik

    def backward_algorithm(
        self,
        bafs,
        pos,
        mat_haps,
        pat_haps,
        pi0=0.5,
        std_dev=0.25,
        r=1e-8,
        a=1e-2,
        unphased=False,
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

        Returns:
            - betas (`np.array`): backward variables from hmm across the k states
            - scaler (`np.array`): m-length array of scale parameters
            - states (`list`): tuple representation of states
            - karyotypes (`np.array`):  array of karyotypes in the MetaHMM model
            - loglik (`float`): total log-likelihood of B-allele frequency

        """
        assert bafs.ndim == 1
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
        betas, scaler, _, _, loglik = backward_algo(
            bafs,
            pos,
            mat_haps,
            pat_haps,
            self.states,
            self.karyotypes,
            r=r,
            a=a,
            pi0=pi0,
            std_dev=std_dev,
        )
        return betas, scaler, self.states, self.karyotypes, loglik

    def forward_backward(
        self,
        bafs,
        pos,
        mat_haps,
        pat_haps,
        pi0=0.2,
        std_dev=0.25,
        r=1e-8,
        a=1e-2,
        unphased=False,
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

        Returns:
            - gammas (`np.array`): log posterior density of being in each of k hidden states
            - states (`list`): tuple representation of states
            - karyotypes (`np.array`):  array of karyotypes in the model

        """
        alphas, _, states, karyotypes, _ = self.forward_algorithm(
            bafs,
            pos,
            mat_haps,
            pat_haps,
            pi0=pi0,
            std_dev=std_dev,
            r=r,
            a=a,
            unphased=unphased,
        )
        betas, _, _, _, _ = self.backward_algorithm(
            bafs,
            pos,
            mat_haps,
            pat_haps,
            pi0=pi0,
            std_dev=std_dev,
            r=r,
            a=a,
            unphased=unphased,
        )
        gammas = (alphas + betas) - logsumexp_sp(alphas + betas, axis=0)
        return gammas, states, karyotypes

    def viterbi_algorithm(
        self,
        bafs,
        pos,
        mat_haps,
        pat_haps,
        pi0=0.2,
        std_dev=0.25,
        r=1e-8,
        a=1e-2,
        unphased=False,
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

        Returns:
            - path (`np.array`): most likely copying path through k states in the model
            - states (`list`): tuple representation of states
            - deltas (`np.array`): delta variable (maximum path probability at step m)
            - psi (`np.array`): storage vector for psi variable

        """
        assert bafs.ndim == 1
        assert pos.ndim == 1
        assert (mat_haps.ndim == 2) & (pat_haps.ndim == 2)
        assert (pi0 > 0) & (pi0 < 1.0)
        assert std_dev > 0
        assert bafs.size == pos.size
        assert bafs.size == mat_haps.shape[1]
        assert mat_haps.shape == pat_haps.shape
        assert np.all(pos[1:] > pos[:-1])
        path, states, deltas, psi = viterbi_algo(
            bafs,
            pos,
            mat_haps,
            pat_haps,
            self.states,
            self.karyotypes,
            r=r,
            a=a,
            pi0=pi0,
            std_dev=std_dev,
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
                    mat_dosage(mat_haps[:, i], x) + pat_dosage(pat_haps[:, i], x)
                )
                assert (cur_geno >= 0) and (cur_geno <= 2)
                dosages[cur_geno, i] = 1.0
        else:
            gammas, states, _ = self.forward_backward(
                bafs=bafs,
                pos=pos,
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
        if i == 0 and j == 1:
            # maternal haploidentity -> paternal haploidentity
            m = 0
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
            # maternal haploidentity -> identity
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
        if self.n_het < 10:
            raise ValueError("Fewer than 10 expected heterozygotes!")

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
        self.n_het = self.het_bafs.size
        if self.n_het < 10:
            raise ValueError("Fewer than 10 expected heterozygotes!")

    def create_transition_matrix(self, switch_err=0.01, t_rate=1e-4):
        """Create the transition matrix.

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
        # 1. Transition rates here are the switch error rate effectively
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
        n = self.het_bafs.size
        m = 3
        alphas = np.zeros(shape=(m, n))
        alphas[:, 0] = np.log(1.0 / m)
        # NOTE: I wonder if you don't have to use the truncation here and can instead just use the baf - 0.5f
        alphas[0, 0] += norm_logl(self.het_bafs[0] - 0.5, -theta, std_dev)
        alphas[1, 0] += norm_logl(self.het_bafs[0] - 0.5, 0.0, std_dev)
        alphas[2, 0] += norm_logl(self.het_bafs[0] - 0.5, +theta, std_dev)
        scaler = np.zeros(n)
        scaler[0] = logsumexp(alphas[:, 0])
        alphas[:, 0] -= scaler[0]
        for i in range(1, n):
            for j in range(3):
                # NOTE: there has to be a better way to do this ...
                if j == 0:
                    alphas[j, i] += norm_logl(self.het_bafs[i] - 0.5, -theta, std_dev)
                elif j == 1:
                    alphas[j, i] += norm_logl(self.het_bafs[i] - 0.5, 0.0, std_dev)
                else:
                    alphas[j, i] += norm_logl(self.het_bafs[i] - 0.5, theta, std_dev)
                alphas[j, i] += logsumexp(self.A[:, j] + alphas[:, (i - 1)])
            scaler[i] = logsumexp(alphas[:, i])
            alphas[:, i] -= scaler[i]
        return alphas, scaler, np.sum(scaler)

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
            x0 = brute(func=f, ranges=[[0.0, 0.5]], disp=True, finish=None, Ns=25)
            opt_res = minimize(f, x0=[x0], bounds=[(0.0, 0.5)])
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
        pi0_est_acc = np.zeros(len(self.embryo_bafs))
        sigma_est_acc = np.zeros(len(self.embryo_bafs))
        for i, baf in enumerate(self.embryo_bafs):
            pi0_est, sigma_est = hmm.est_sigma_pi0(
                baf, self.pos, self.mat_haps, self.pat_haps, **kwargs
            )
            pi0_est_acc[i] = pi0_est
            sigma_est_acc[i] = sigma_est
        self.embryo_pi0s = pi0_est_acc
        self.embryo_sigmas = sigma_est_acc

    def lod_phase(self, haps1, haps2, baf, **kwargs):
        """Compute the log-likelihood of being in the phase orientation.

        NOTE: this marginalizes over all the possible phases

        Arguments:
            - haps1 (`np.array`): focal haplotypes to be mendelian-phased
            - haps2 (`np.array`): ancillary haplotypes from other parent (to be marginalized over)
            - baf (`list`): list of BAF values for embryos

        Returns:
            - phase_orientation (`float`): log-density of haps1 being in the correct phase
            - antiphase_orientation (`float`): log-density of haps1 being in the opposite phase

        """
        assert (haps1.shape[0] == 2) and (haps1.shape[1] == 2)
        assert (haps2.shape[0] == 2) and (haps2.shape[1] == 2)
        assert np.all(np.sum(haps1, axis=0) == 1)
        phase_orientation, antiphase_orientation = lod_phase(
            haps1, haps2, baf, **kwargs
        )
        return phase_orientation, antiphase_orientation

    def lod_phase_correct(self, maternal=True, lod_thresh=-1.0, **kwargs):
        """Apply a phase correction for the specified parental haplotype.

        Arguments:
            - maternal (`bool`): apply phase correction on the maternal haplotypes
            - lod_thresh (`float`): Log-odds threshold on the phase/antiphase ratio between SNPs to flip snps

        """
        assert self.embryo_bafs is not None
        if maternal:
            haps1 = self.mat_haps
            haps2 = self.pat_haps
        else:
            haps1 = self.pat_haps
            haps2 = self.mat_haps
        assert (haps1.ndim == 2) and (haps2.ndim == 2)
        assert (haps1.shape[0] == 2) and (haps2.shape[0] == 2)
        assert haps1.shape[1] == haps2.shape[1]
        geno1 = haps1.sum(axis=0)
        idx_het1 = np.where(geno1 == 1)[0]
        hap_idx1 = np.zeros(haps1.shape[1], dtype=np.uint16)
        hap_idx2 = np.ones(haps1.shape[1], dtype=np.uint16)
        for i, j in zip(idx_het1[:-1], idx_het1[1:]):
            tot_phase = 0.0
            tot_antiphase = 0.0
            cur_hap = np.vstack(
                [haps1[hap_idx1[i], [i, j]], haps1[hap_idx2[i], [i, j]]]
            )
            for baf in self.embryo_bafs:
                # now we have to use the current phasing approach ...
                cur_phase, cur_antiphase = self.lod_phase(
                    haps1=cur_hap, haps2=haps2[:, [i, j]], baf=baf[[i, j]], **kwargs
                )
                tot_phase += cur_phase
                tot_antiphase += cur_antiphase
            # If we are below the log-odds threshold then we create a switch
            if tot_phase - tot_antiphase < lod_thresh:
                hap_idx1[j:] = 1 - hap_idx1[i]
                hap_idx2[j:] = 1 - hap_idx2[i]
        # Getting each index sequentially
        fixed_hap1 = [haps1[x, i] for i, x in enumerate(hap_idx1)]
        fixed_hap2 = [haps1[x, i] for i, x in enumerate(hap_idx2)]
        fixed_haps = np.vstack([fixed_hap1, fixed_hap2])
        assert fixed_haps.shape[1] == haps1.shape[1]
        assert fixed_haps.shape[0] == 2
        if maternal:
            self.mat_haps_fixed = fixed_haps
        else:
            self.pat_haps_fixed = fixed_haps

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
        n_mis_mat_tot = np.zeros(niter)
        n_mis_pat_tot = np.zeros(niter)
        for i in range(niter):
            mat_paths = np.zeros(shape=(n_sibs, m))
            pat_paths = np.zeros(shape=(n_sibs, m))
            for j, baf in enumerate(self.embryo_bafs):
                path, _, _, _ = hmm.viterbi_algorithm(
                    baf,
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
        # Set the corrected haplotypes here
        self.mat_haps_fixed = mat_haps
        self.pat_haps_fixed = pat_haps
        return mat_haps, pat_haps, n_mis_mat_tot, n_mis_pat_tot

    def estimate_switch_err_true(self, maternal=True, fixed=False):
        """Estimate the switch error from true and inferred haplotypes.

        The switch error is defined as consecutive heterozygotes that are
        in the incorrect orientation.

        Arguments:
            - maternal (`bool`): apply the function to the maternal chromosome (default: True)
            - fixed (`bool`): apply the function to the fixed chromosome (default: False)

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
        # NOTE: this is just between all consecutive hets, not
        geno = true_haps.sum(axis=0)
        het_idxs = np.where(geno == 1)[0]
        n_switches = 0
        n_consecutive_hets = 0
        switch_idxs = []
        for (i, j) in zip(het_idxs[:-1], het_idxs[1:]):
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

    def estimate_switch_err_empirical(
        self, maternal=True, fixed=False, truth=False, lod_thresh=-1, **kwargs
    ):
        """Use the empirical embryo BAF data to determine the switch-error rate via LOD score for B-allele Frequency.

        Arguments:
            - maternal (`bool`): estimate switch-errors from the maternal haplotypes
            - fixed (`bool`): estimate switch-errors from the fixed haplotypes
            - truth (`bool`): use the true haplotypes if available
            - lod_thresh (`float`): any log-density ratio below this value will be called as a switch

        Returns:
            - n_switches (`int`): number of switches between consecutive heterozygotes
            - n_consecutive_hets (`int`): number of consecutive heterozygotes
            - switch_err_rate (`float`): number of switches per consecutive heterozygote
            - switch_idx (`np.array`): snps where the variant is out of phase with its predecessor
            - het_idx (`np.array`): locations/indexs of the heterozygotes
            - lods (`np.array`) : lod-scores for the estimated switches (default: None)

        """
        assert self.embryo_bafs is not None
        # haps1 is the individual that we are evaluating the switch errors for
        haps1 = None
        haps2 = None
        # Make sure that only one of fixed or truth are available
        assert (not fixed) or (not truth)
        if maternal:
            if fixed:
                assert self.mat_haps_fixed is not None
                haps1 = self.mat_haps_fixed
                haps2 = self.pat_haps
            elif truth:
                assert self.mat_haps_true is not None
                haps1 = self.mat_haps_true
                haps2 = self.pat_haps_true
            else:
                haps1 = self.mat_haps
                haps2 = self.pat_haps
        else:
            if fixed:
                assert self.pat_haps_fixed is not None
                haps1 = self.pat_haps_fixed
                haps2 = self.mat_haps
            elif truth:
                assert self.pat_haps_true is not None
                haps1 = self.pat_haps_true
                haps2 = self.mat_haps_true
            else:
                haps1 = self.pat_haps
                haps2 = self.mat_haps
        # The variables where we store the switch information
        n_switches = 0
        n_consecutive_hets = 0
        switch_idxs = []
        lods = []
        # NOTE: here we restrict to "phase-informative" switches ...
        geno1 = haps1[0, :] + haps1[1, :]
        het_idx = np.where(geno1 == 1)[0]
        for i, j in zip(het_idx[:-1], het_idx[1:]):
            n_consecutive_hets += 1
            tot_phase = 0.0
            tot_antiphase = 0.0
            for baf in self.embryo_bafs:
                # now we have to use the current phasing approach ...
                cur_phase, cur_antiphase = self.lod_phase(
                    haps1=haps1[:, [i, j]],
                    haps2=haps2[:, [i, j]],
                    baf=baf[[i, j]],
                    **kwargs,
                )
                tot_phase += cur_phase
                tot_antiphase += cur_antiphase
            lods.append(tot_phase - tot_antiphase)
            # This is the ratio between the two probability densities
            if tot_phase - tot_antiphase < lod_thresh:
                n_switches += 1
                switch_idxs.append((i, j))
        lods = np.array(lods)
        return (
            n_switches,
            n_consecutive_hets,
            n_switches / n_consecutive_hets,
            switch_idxs,
            het_idx,
            lods,
        )

    def solveTrio(self, cg=0, fg=0, mg=0):
        """Solve the trio setup to phase the parents.

        Code taken from: https://github.com/odelaneau/makeScaffold/blob/master/src/data_mendel.cpp
        """
        phased = 0
        mendel = 0
        if fg == 0 & mg == 0 & cg == 0:
            f0 = 0
            f1 = 0
            m0 = 0
            m1 = 0
            c0 = 0
            c1 = 0
            mendel = 0
            phased = 1
        if fg == 0 & mg == 0 & cg == 1:
            f0 = 0
            f1 = 0
            m0 = 0
            m1 = 0
            c0 = 0
            c1 = 1
            mendel = 1
            phased = 0
        if fg == 0 & mg == 0 & cg == 2:
            f0 = 0
            f1 = 0
            m0 = 0
            m1 = 0
            c0 = 1
            c1 = 1
            mendel = 1
            phased = 0
        if fg == 0 & mg == 1 & cg == 0:
            f0 = 0
            f1 = 0
            m0 = 1
            m1 = 0
            c0 = 0
            c1 = 0
            mendel = 0
            phased = 1
        if fg == 0 & mg == 1 & cg == 1:
            f0 = 0
            f1 = 0
            m0 = 0
            m1 = 1
            c0 = 0
            c1 = 1
            mendel = 0
            phased = 1
        if fg == 0 & mg == 1 & cg == 2:
            f0 = 0
            f1 = 0
            m0 = 0
            m1 = 1
            c0 = 1
            c1 = 1
            mendel = 1
            phased = 0
        if fg == 0 & mg == 2 & cg == 0:
            f0 = 0
            f1 = 0
            m0 = 1
            m1 = 1
            c0 = 0
            c1 = 0
            mendel = 1
            phased = 0
        if fg == 0 & mg == 2 & cg == 1:
            f0 = 0
            f1 = 0
            m0 = 1
            m1 = 1
            c0 = 0
            c1 = 1
            mendel = 0
            phased = 1
        if fg == 0 & mg == 2 & cg == 2:
            f0 = 0
            f1 = 0
            m0 = 1
            m1 = 1
            c0 = 1
            c1 = 1
            mendel = 1
            phased = 0
        if fg == 1 & mg == 0 & cg == 0:
            f0 = 0
            f1 = 1
            m0 = 0
            m1 = 0
            c0 = 0
            c1 = 0
            mendel = 0
            phased = 1
        if fg == 1 & mg == 0 & cg == 1:
            f0 = 1
            f1 = 0
            m0 = 0
            m1 = 0
            c0 = 1
            c1 = 0
            mendel = 0
            phased = 1
        if fg == 1 & mg == 0 & cg == 2:
            f0 = 1
            f1 = 0
            m0 = 0
            m1 = 0
            c0 = 1
            c1 = 1
            mendel = 1
            phased = 0
        if fg == 1 & mg == 1 & cg == 0:
            f0 = 0
            f1 = 1
            m0 = 1
            m1 = 0
            c0 = 0
            c1 = 0
            mendel = 0
            phased = 1
        if fg == 1 & mg == 1 & cg == 1:
            f0 = 0
            f1 = 1
            m0 = 0
            m1 = 1
            c0 = 0
            c1 = 1
            mendel = 0
            phased = 0
        if fg == 1 & mg == 1 & cg == 2:
            f0 = 1
            f1 = 0
            m0 = 0
            m1 = 1
            c0 = 1
            c1 = 1
            mendel = 0
            phased = 1
        if fg == 1 & mg == 2 & cg == 0:
            f0 = 0
            f1 = 1
            m0 = 1
            m1 = 1
            c0 = 0
            c1 = 0
            mendel = 1
            phased = 0
        if fg == 1 & mg == 2 & cg == 1:
            f0 = 0
            f1 = 1
            m0 = 1
            m1 = 1
            c0 = 0
            c1 = 1
            mendel = 0
            phased = 1
        if fg == 1 & mg == 2 & cg == 2:
            f0 = 1
            f1 = 0
            m0 = 1
            m1 = 1
            c0 = 1
            c1 = 1
            mendel = 0
            phased = 1
        if fg == 2 & mg == 0 & cg == 0:
            f0 = 1
            f1 = 1
            m0 = 0
            m1 = 0
            c0 = 0
            c1 = 0
            mendel = 1
            phased = 0
        if fg == 2 & mg == 0 & cg == 1:
            f0 = 1
            f1 = 1
            m0 = 0
            m1 = 0
            c0 = 1
            c1 = 0
            mendel = 0
            phased = 1
        if fg == 2 & mg == 0 & cg == 2:
            f0 = 1
            f1 = 1
            m0 = 0
            m1 = 0
            c0 = 1
            c1 = 1
            mendel = 1
            phased = 0
        if fg == 2 & mg == 1 & cg == 0:
            f0 = 1
            f1 = 1
            m0 = 0
            m1 = 1
            c0 = 0
            c1 = 0
            mendel = 1
            phased = 0
        if fg == 2 & mg == 1 & cg == 1:
            f0 = 1
            f1 = 1
            m0 = 1
            m1 = 0
            c0 = 1
            c1 = 0
            mendel = 0
            phased = 1
        if fg == 2 & mg == 1 & cg == 2:
            f0 = 1
            f1 = 1
            m0 = 0
            m1 = 1
            c0 = 1
            c1 = 1
            mendel = 0
            phased = 1
        if fg == 2 & mg == 2 & cg == 0:
            f0 = 1
            f1 = 1
            m0 = 1
            m1 = 1
            c0 = 0
            c1 = 0
            mendel = 1
            phased = 0
        if fg == 2 & mg == 2 & cg == 1:
            f0 = 1
            f1 = 1
            m0 = 1
            m1 = 1
            c0 = 0
            c1 = 1
            mendel = 1
            phased = 0
        if fg == 2 & mg == 2 & cg == 2:
            f0 = 1
            f1 = 1
            m0 = 1
            m1 = 1
            c0 = 1
            c1 = 1
            mendel = 0
            phased = 1
        if mendel:
            return True
        else:
            return False

    # solveTrio(cg=1, fg=0, mg=2)


class RecombEst(PhaseCorrect):
    """Class implementing the simplified algorithm for detection of crossovers of Coop et al 2007."""

    def __init__(self, **kwargs):
        """Use a superclass to preserve structure for sibling embryos."""
        super(RecombEst, self).__init__(**kwargs)

    def informative_markers(self, paternal=True):
        """Identify markers where one parent is heterozygous and the other parent is homozygous."""
        mat_geno = np.sum(self.mat_haps, axis=0)
        pat_geno = np.sum(self.pat_haps, axis=0)
        if paternal:
            info_idx = ((pat_geno == 1) & (mat_geno == 0)) | (
                (pat_geno == 1) & (mat_geno == 2)
            )
        else:
            info_idx = ((mat_geno == 1) & (pat_geno == 0)) | (
                (mat_geno == 1) & (pat_geno == 2)
            )
        return info_idx

    def isolate_recomb_events(self, template_embryo=0, paternal=True):
        """Isolate specific recombination events."""
        assert self.embryo_bafs is not None
        assert self.embryo_pi0s is not None
        assert self.embryo_sigmas is not None
        m = len(self.embryo_bafs)
        assert m > 1
        assert template_embryo < m
        non_template_ids = [i for i in range(m) if i != template_embryo]
        # Get the informative snps for the specific parent
        info_snps = self.informative_markers(paternal=paternal)
        llr_z = np.zeros(shape=(len(non_template_ids), np.sum(info_snps)))
        for j, k in enumerate(non_template_ids):
            z1s = np.zeros(np.sum(info_snps))
            z2s = np.zeros(np.sum(info_snps))
            # NOTE: we use a signed-version of the log-likelihood ratio ...
            for p, i in enumerate(np.where(info_snps)[0]):
                z_same = np.zeros(2)
                z_diff = np.zeros(2)
                if paternal:
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
                else:
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
                z1s[p] = logsumexp(z_same)
                z2s[p] = logsumexp(z_diff)
            llrs = np.array(z1s) - np.array(z2s)
            llr_z[j, :] = llrs

        # Now we make a rough decision rule here for the likelihood-ratio supporting one or the other class ...
        # 1 indicates copying the same allele, 2 indicates that copying different alleles.
        Z = np.zeros(shape=llr_z.shape)
        Z[llr_z < 0] = 2
        Z[llr_z > 0] = 1
        # Now isolate crossovers in the specific parent by the majority rule ...
        potential_switches = np.where(Z[:, 1:] != Z[:, :-1])[1]
        switch_idx, cnts = np.unique(potential_switches, return_counts=True)
        # Choose the potential switches by the majority rule ...
        potential_switches_filt = switch_idx[cnts > (m - 1) / 2]
        return Z, llr_z, potential_switches_filt

    def refine_recomb_events(self, potential_switches_filt, npad=5):
        """Refine recombination estimation using the switch-clusters approach of Coop et al 2007.

        Arguments:
            - potential_switches_filt (`np.array`): array of potential switches at informative markers
            - npad (`int`): integer value of adjacent informative snps to consider as a switch cluster.

        """
        assert npad > 1
        if potential_switches_filt.size > 0:
            subset_co = []
            for i in potential_switches_filt:
                # Count the number of switches within 5 informative SNPs
                n = np.sum(
                    (potential_switches_filt < i + npad)
                    & (potential_switches_filt > i - npad)
                )
                # Even number of crossovers suggest it is likely not a well-supported CO
                if n % 2 == 1:
                    subset_co.append(i)
            return subset_co
        else:
            return []
