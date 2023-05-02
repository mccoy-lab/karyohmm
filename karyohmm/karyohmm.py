import numpy as np
import pyximport

pyximport.install(language_level=3)
from scipy.optimize import minimize
from scipy.special import logsumexp as logsumexp_sp
from tqdm import tqdm

from .hmm_utils import *


class AneuploidyHMM:
    """Base class for defining all the aneuploidy HMM."""

    def __init__(self):
        self.ploidy = 2
        self.aploid = None

    def emission_prob(self, baf, m, p, pi0=0.2, std_dev=0.3, eps=1e-4, err=1e-20, k=2):
        """Emission density of embryo BAF at site i."""
        assert (pi0 < 1.0) & (pi0 > 0.0)
        assert (eps > 0.0) & (eps < 1.0)
        assert std_dev > 0
        return emission_help(baf, m, p, pi0=pi0, std_dev=std_dev, eps=eps, err=err, k=k)

    def transition_prob(self, zprime, zi, r=1e-2, k=4):
        """Transition probability between hidden state zprime and zi."""
        if zprime == zi:
            return 1 - r / (k - 1)
        else:
            return r / (k - 1)

    def get_state_str(self, state):
        """NOTE: this might have to be slightly revised ..."""
        if len(state) == 2:
            return f"m{state[0]}p{state[1]}"
        else:
            t = []
            m = sum([s >= 0 for s in state])
            for i, s in enumerate(state):
                if s < 0:
                    if m == 2:
                        if i == 0:
                            t.append(f"m{s}")
                        if i == 1:
                            t.append(f"p{s}")
                    if m == 4:
                        if i < 2:
                            t.append(f"m{s}")
                        if i >= 2:
                            t.append(f"p{s}")
            return "".join(t)

    def est_sigma_pi0(self, bafs, mat_haps, pat_haps, **kwargs):
        """Estimate sigma and pi0 using the forward algorithm for the HMM."""

        def f(x):
            loglik = -self.forward_algorithm(
                bafs=bafs,
                mat_haps=mat_haps,
                pat_haps=pat_haps,
                pi0=x[0],
                std_dev=x[1],
                **kwargs,
            )[4]
            print(x, loglik)
            return loglik

        opt_res = minimize(
            f,
            x0=[0.2, 0.2],
            method="L-BFGS-B",
            bounds=[(0.05, 0.95), (0.1, 0.5)],
            tol=1e-3,
        )
        return opt_res

    def est_lrr_sd(
        self,
        lrrs,
        lrr_mu=np.array([-9, -4, np.log2(0.5), 0.0, np.log2(1.5)]),
        a=-4,
        b=1.0,
        **kwargs,
    ):
        """Estimate the underlying approach to estimate variances in LRR distributions."""
        lrrs_clip = np.clip(lrrs, a, b)
        pis, mus, std, logliks = est_gmm_variance(
            lrrs_clip, mus=lrr_mu, a=a, b=b, **kwargs
        )
        pi0 = pis[0]
        return pi0, mus[1:], std[1:], logliks


class EuploidyHMM(AneuploidyHMM):
    """HMM to estimate haplotype traceback in the euploid context."""

    def __init__(self, aploid="2"):
        """Implement the euploidy HMM (mostly for crossover detection)"""
        super().__init__()
        assert aploid in ["2"]
        self.ploidy = 2
        self.aploid = aploid
        self.states = [
            (0, -1, 0, -1),
            (0, -1, 1, -1),
            (1, -1, 0, -1),
            (1, -1, 1, -1),
        ]

    def create_transition_matrix(self, r=1e-4):
        """Create the full transition matrix for this set of samples."""
        assert (r < 1) and (r > 0)
        A = np.zeros(shape=(4, 4))
        for i in range(4):
            A[i, :] = r / 3.0
            A[i, i] = 1.0 - r
        return np.log(A)

    def forward_algorithm(
        self, bafs, mat_haps, pat_haps, pi0=0.2, std_dev=0.25, r=1e-4, eps=1e-8
    ):
        """Forward HMM algorithm under a specified statespace model."""
        assert bafs.ndim == 1
        assert (mat_haps.ndim == 2) & (pat_haps.ndim == 2)
        assert (pi0 > 0) & (pi0 < 1.0)
        assert std_dev > 0
        assert (eps > 0) & (eps < 1e-2)
        assert bafs.size == mat_haps.shape[1]
        assert mat_haps.shape == pat_haps.shape

        A = self.create_transition_matrix(r=r)
        alphas, scaler, _, _, loglik = forward_algo(
            bafs,
            np.ones(bafs.size),
            mat_haps,
            pat_haps,
            self.states,
            A,
            pi0=pi0,
            std_dev=std_dev,
            eps=eps,
            logr=False,
        )
        return alphas, scaler, self.states, None, loglik

    def backward_algorithm(
        self, bafs, mat_haps, pat_haps, pi0=0.2, std_dev=0.25, r=1e-4, eps=1e-8
    ):
        """Backward HMM algorithm under a given statespace model."""
        assert bafs.ndim == 1
        assert (mat_haps.ndim == 2) & (pat_haps.ndim == 2)
        assert (pi0 > 0) & (pi0 < 1.0)
        assert std_dev > 0
        assert (eps > 0) & (eps < 1e-2)
        assert bafs.size == mat_haps.shape[1]
        assert mat_haps.shape == pat_haps.shape
        A = self.create_transition_matrix(r=r)
        betas, scaler, _, _, loglik = backward_algo(
            bafs,
            np.ones(bafs.size),
            mat_haps,
            pat_haps,
            self.states,
            A,
            pi0=pi0,
            std_dev=std_dev,
            eps=eps,
            logr=False,
        )
        return betas, scaler, self.states, None, loglik

    def forward_backward(
        self, bafs, mat_haps, pat_haps, pi0=0.2, std_dev=0.25, r=1e-4, eps=1e-8
    ):
        """Apply the forward-backward algorithm."""
        alphas, _, states, _, _ = self.forward_algorithm(
            bafs,
            mat_haps,
            pat_haps,
            pi0=pi0,
            std_dev=std_dev,
            eps=eps,
            r=r,
        )
        betas, _, _, _, _ = self.backward_algorithm(
            bafs,
            mat_haps,
            pat_haps,
            pi0=pi0,
            std_dev=std_dev,
            eps=eps,
            r=r,
        )
        gammas = (alphas + betas) - logsumexp_sp(alphas + betas, axis=0)
        return gammas, states, None

    def viterbi_algorithm(
        self, bafs, mat_haps, pat_haps, pi0=0.2, std_dev=0.25, r=1e-4, eps=1e-8
    ):
        """Implements the Viterbi algorithm."""
        assert bafs.ndim == 1
        assert (mat_haps.ndim == 2) & (pat_haps.ndim == 2)
        assert (pi0 > 0) & (pi0 < 1.0)
        assert std_dev > 0
        assert (eps > 0) & (eps < 1e-2)
        assert bafs.size == mat_haps.shape[1]
        assert mat_haps.shape == pat_haps.shape
        A = self.create_transition_matrix(r=r)
        path, states, deltas, psi = viterbi_algo(
            bafs,
            np.ones(bafs.size),
            mat_haps,
            pat_haps,
            self.states,
            A,
            pi0=pi0,
            std_dev=std_dev,
            eps=eps,
            logr=False,
        )
        return path, states, deltas, psi

    def assign_recomb(self, states, path):
        """Obtain the indices and sex-specific recombination events.

        NOTE: to obtain true intervals we will have to look at distance to nearest heterozygote from the changept
        """
        assert len(states) > 0
        assert np.min(path) >= 0
        assert np.max(path) <= len(states)
        changepts = np.where(path[:-1] != path[1:])[0]
        paternal = []
        maternal = []
        for c in changepts:
            if states[path[c]][2] != states[path[c + 1]][2]:
                paternal.append(c)
            else:
                maternal.append(c)
        maternal = np.array(maternal, dtype=int)
        paternal = np.array(paternal, dtype=int)
        assert changepts.size == (maternal.size + paternal.size)
        return maternal, paternal, changepts

    def assign_co_windows(self, changepts, haps):
        """Assign crossover windows based on distance to nearest heterozygote.

        Args:
            - changepts (`np.array`): inferred changepts for specific parent
            - haps (`np.array`): haplotypes for parental individuals.

        NOTE: how do we establish this for faulty haplotypes?
        """
        windows = []
        for c in changepts:
            pass
        return windows

    def filter_switch_errors(self, intervals, p=0.5):
        """Filtering putative switch errors.

        Args:
         - intervals (`list`): list
         - p (`float`): proportion of embryos for window to be representative.

        NOTE: this will likely use an interval tree for comparisons.
        """
        pass


class MetaHMM(AneuploidyHMM):
    """A meta-HMM that attempts to evaluate all possible ploidy states at once."""

    def __init__(self, logr=True):
        super().__init__()
        self.ploidy = 0
        self.aploid = "meta"
        self.logr = logr
        self.nullisomy_state = [(-1, -1, -1, -1)]
        self.p_monosomy_states = [(-1, 1, -1, -1), (-1, 0, -1, -1)]
        self.m_monosomy_states = [(0, -1, -1, -1), (1, -1, -1, -1)]
        self.isodisomy_states = [(0, 1, -1, -1), (-1, -1, 0, 1)]
        self.euploid_states = [
            (0, -1, 0, -1),
            (0, -1, 1, -1),
            (1, -1, 0, -1),
            (1, -1, 1, -1),
        ]
        # First two are the female states ...
        self.m_trisomy_states = [
            (0, 0, 0, -1),
            (1, 0, 0, -1),
            (1, 1, 0, -1),
            (0, 0, 1, -1),
            (1, 0, 1, -1),
            (1, 1, 1, -1),
        ]
        self.p_trisomy_states = [
            (0, -1, 0, 0),
            (1, -1, 0, 0),
            (1, -1, 1, 0),
            (0, -1, 0, 1),
            (1, -1, 0, 1),
            (1, -1, 1, 1),
        ]
        # NOTE: all of these states are combined here ...
        if self.logr:
            self.states = (
                self.nullisomy_state
                + self.m_monosomy_states
                + self.p_monosomy_states
                + self.isodisomy_states
                + self.euploid_states
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
                    "2m",
                    "2p",
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
        else:
            self.states = (
                self.nullisomy_state
                + self.m_monosomy_states
                + self.p_monosomy_states
                + self.euploid_states
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

    def create_transition_matrix(self, karyotypes, r=1e-4, a=1e-7, unphased=False):
        m = karyotypes.size
        assert r <= (1 / m)
        assert a <= (1 / m)
        A = np.zeros(shape=(m, m))
        for i in range(m):
            for j in range(m):
                if i != j:
                    if karyotypes[i] == karyotypes[j]:
                        k = np.sum(karyotypes == karyotypes[i])
                        if unphased:
                            A[i, j] = 1 / k
                        else:
                            A[i, j] = r
                    else:
                        A[i, j] = a
        for i in range(m):
            A[i, i] = 1.0 - np.sum(A[i, :])
        return np.log(A)

    def forward_algorithm(
        self,
        bafs,
        lrrs,
        mat_haps,
        pat_haps,
        lrr_mu=None,
        lrr_sd=None,
        pi0=0.2,
        std_dev=0.25,
        pi0_lrr=0.2,
        r=1e-4,
        a=1e-7,
        eps=1e-4,
        unphased=False,
        logr=False,
    ):
        """Forward HMM algorithm under a multi-ploidy model."""
        assert bafs.ndim == 1
        assert (mat_haps.ndim == 2) & (pat_haps.ndim == 2)
        assert (pi0 > 0) & (pi0 < 1.0)
        assert std_dev > 0
        assert (eps > 0) & (eps < 1e-1)
        assert bafs.size == mat_haps.shape[1]
        assert mat_haps.shape == pat_haps.shape
        n = bafs.size
        m = len(self.states)
        A = self.create_transition_matrix(self.karyotypes, r=r, a=a, unphased=unphased)
        alphas, scaler, _, _, loglik = forward_algo(
            bafs,
            lrrs,
            mat_haps,
            pat_haps,
            self.states,
            A,
            lrr_mu=lrr_mu,
            lrr_sd=lrr_sd,
            pi0=pi0,
            std_dev=std_dev,
            pi0_lrr=pi0_lrr,
            eps=eps,
            logr=logr,
        )
        return alphas, scaler, self.states, self.karyotypes, loglik

    def backward_algorithm(
        self,
        bafs,
        lrrs,
        mat_haps,
        pat_haps,
        lrr_mu=None,
        lrr_sd=None,
        pi0=0.2,
        std_dev=0.25,
        pi0_lrr=0.2,
        r=1e-4,
        a=1e-7,
        eps=1e-4,
        unphased=False,
        logr=False,
    ):
        """Backward HMM algorithm under a given statespace model."""
        assert bafs.ndim == 1
        assert (mat_haps.ndim == 2) & (pat_haps.ndim == 2)
        assert (pi0 > 0) & (pi0 < 1.0)
        assert std_dev > 0
        assert (eps > 0) & (eps < 1e-1)
        assert bafs.size == mat_haps.shape[1]
        assert mat_haps.shape == pat_haps.shape
        n = bafs.size
        m = len(self.states)
        A = self.create_transition_matrix(self.karyotypes, r=r, a=a, unphased=unphased)
        betas, scaler, _, _, loglik = backward_algo(
            bafs,
            lrrs,
            mat_haps,
            pat_haps,
            self.states,
            A,
            lrr_mu=lrr_mu,
            lrr_sd=lrr_sd,
            pi0=pi0,
            std_dev=std_dev,
            pi0_lrr=pi0_lrr,
            eps=eps,
            logr=logr,
        )
        return betas, scaler, self.states, self.karyotypes, loglik

    def forward_backward(
        self,
        bafs,
        lrrs,
        mat_haps,
        pat_haps,
        lrr_mu=None,
        lrr_sd=None,
        pi0=0.2,
        std_dev=0.25,
        pi0_lrr=0.2,
        r=1e-4,
        a=1e-7,
        eps=1e-4,
        unphased=False,
        logr=False,
    ):
        """Run the forward-backward algorithm across all states."""
        alphas, _, states, karyotypes, _ = self.forward_algorithm(
            bafs,
            lrrs,
            mat_haps,
            pat_haps,
            lrr_mu=lrr_mu,
            lrr_sd=lrr_sd,
            pi0=pi0,
            std_dev=std_dev,
            pi0_lrr=pi0_lrr,
            eps=eps,
            r=r,
            a=a,
            unphased=unphased,
            logr=logr,
        )
        betas, _, _, _, _ = self.backward_algorithm(
            bafs,
            lrrs,
            mat_haps,
            pat_haps,
            lrr_mu=lrr_mu,
            lrr_sd=lrr_sd,
            pi0=pi0,
            std_dev=std_dev,
            pi0_lrr=pi0_lrr,
            eps=eps,
            r=r,
            a=a,
            unphased=unphased,
            logr=logr,
        )
        gammas = (alphas + betas) - logsumexp_sp(alphas + betas, axis=0)
        return gammas, states, karyotypes

    def viterbi_algorithm(
        self,
        bafs,
        lrrs,
        mat_haps,
        pat_haps,
        lrr_mu=None,
        lrr_sd=None,
        pi0=0.2,
        std_dev=0.25,
        pi0_lrr=0.2,
        r=1e-4,
        a=1e-7,
        eps=1e-4,
        unphased=False,
        logr=False,
    ):
        assert bafs.ndim == 1
        assert (mat_haps.ndim == 2) & (pat_haps.ndim == 2)
        assert (pi0 > 0) & (pi0 < 1.0)
        assert std_dev > 0
        assert (eps > 0) & (eps < 1e-1)
        assert bafs.size == mat_haps.shape[1]
        assert mat_haps.shape == pat_haps.shape

        A = self.create_transition_matrix(self.karyotypes, r=r, a=a, unphased=unphased)
        path, states, deltas, psi = viterbi_algo(
            bafs,
            lrrs,
            mat_haps,
            pat_haps,
            self.states,
            A,
            pi0=pi0,
            std_dev=std_dev,
            pi0_lrr=pi0_lrr,
            eps=eps,
            logr=logr,
        )
        return path, states, deltas, psi

    def marginal_posterior_karyotypes(self, gammas, karyotypes):
        """Obtain the marginal posterior (not logged) probability over karyotypic states."""
        assert gammas.ndim == 2
        assert gammas.shape[0] == karyotypes.size
        k, m = gammas.shape
        nk = np.unique(karyotypes).size
        gamma_karyo = np.zeros(shape=(nk, m))
        for i, k in enumerate(np.unique(karyotypes)):
            # This is just the summed version of the posteriors ...
            gamma_karyo[i, :] = np.sum(np.exp(gammas[(karyotypes == k), :]), axis=0)
        return gamma_karyo

    def posterior_karyotypes(self, gammas, karyotypes):
        """Obtain full posterior on karyotypes chromosome-wide.

        NOTE: this is the weighted proportion of time spent in each karyotypic state-space.
        """
        assert gammas.ndim == 2
        assert gammas.shape[0] == karyotypes.size
        k, m = gammas.shape
        kar_prob = {}
        for k in np.unique(karyotypes):
            kar_prob[k] = np.sum(np.exp(gammas[(karyotypes == k), :])) / m
        return kar_prob
