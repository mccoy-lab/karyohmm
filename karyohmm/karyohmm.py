import numpy as np
from karyohmm_utils import (backward_algo, backward_algo_sibs, emission_baf,
                            est_gmm_variance, forward_algo, forward_algo_sibs,
                            mat_dosage, pat_dosage, viterbi_algo,
                            viterbi_algo_sibs)
from scipy.optimize import minimize
from scipy.special import logsumexp as logsumexp_sp
from tqdm import tqdm


class AneuploidyHMM:
    """Base class for defining all the aneuploidy HMM."""

    def __init__(self):
        """Initialize the base aneuploidy HMM class."""
        self.ploidy = 2
        self.aploid = None

    def get_state_str(self, state):
        """NOTE: this might have to be slightly revised ..."""
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
            return loglik

        opt_res = minimize(
            f,
            x0=[0.2, 0.2],
            method="L-BFGS-B",
            bounds=[(0.05, 0.95), (0.1, 0.5)],
            tol=1e-3,
            options={"disp": False},
        )
        pi0_est = opt_res.x[0]
        sigma_est = opt_res.x[1]
        return pi0_est, sigma_est

    def est_lrr_sd(
        self,
        lrrs,
        lrr_mu=np.array([-9, -4, np.log2(0.5), 0.0, np.log2(1.5)]),
        a=-4,
        b=1.0,
        **kwargs,
    ):
        """Estimate the variances in the LRR distribution."""
        lrrs_clip = np.clip(lrrs, a, b)
        pis, mus, std, logliks = est_gmm_variance(
            lrrs_clip, mus=lrr_mu, a=a, b=b, **kwargs
        )
        pi0 = pis[0]
        return pi0, mus[1:], std[1:], logliks


class MetaHMM(AneuploidyHMM):
    """A meta-HMM that attempts to evaluate all possible ploidy states at once."""

    def __init__(self, logr=True):
        super().__init__()
        self.ploidy = 0
        self.aploid = "meta"
        self.logr = logr
        self.nullisomy_state = [(-1, -1, -1, -1)]
        self.p_monosomy_states = [(-1, -1, 1, -1), (-1, -1, 0, -1)]
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
        """Create an inter-karyotype transition matrix."""
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
        """Implements a viterbi traceback through the various files."""
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


class QuadHMM(AneuploidyHMM):
    """Updated HMM for sibling embryos based on the model of Roach et al 2010 but designed for BAF data."""

    def __init__(self):
        """"""
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

    def create_transition_matrix(self, r=1e-4):
        """Create the transition matrix here."""
        m = len(self.states)
        A = np.zeros(shape=(m, m))
        A[:, :] = r / m
        for i in range(m):
            A[i, i] = 0.0
            A[i, i] = 1.0 - np.sum(A[i, :])
        return np.log(A)

    def forward_algorithm(
        self, bafs, mat_haps, pat_haps, pi0=0.2, std_dev=0.1, r=1e-15, eps=1e-4
    ):
        A = self.create_transition_matrix(r=r)
        alphas, scaler, states, karyotypes, loglik = forward_algo_sibs(
            bafs,
            mat_haps,
            pat_haps,
            states=self.states,
            A=A,
            pi0=pi0,
            std_dev=std_dev,
            eps=eps,
        )
        return alphas, scaler, states, karyotypes, loglik

    def forward_backward(
        self, bafs, mat_haps, pat_haps, pi0=0.2, std_dev=0.1, r=1e-15, eps=1e-4
    ):
        A = self.create_transition_matrix(r=r)
        alphas, _, states, _, _ = forward_algo_sibs(
            bafs,
            mat_haps,
            pat_haps,
            states=self.states,
            A=A,
            pi0=pi0,
            std_dev=std_dev,
            eps=eps,
        )
        betas, _, _, _, _ = backward_algo_sibs(
            bafs,
            mat_haps,
            pat_haps,
            states=self.states,
            A=A,
            pi0=pi0,
            std_dev=std_dev,
            eps=eps,
        )
        gammas = (alphas + betas) - logsumexp_sp(alphas + betas, axis=0)
        return gammas, states, None

    def viterbi_algorithm(
        self, bafs, mat_haps, pat_haps, pi0=0.2, std_dev=0.1, r=1e-15, eps=1e-4
    ):
        """Viterbi algorithm definition in a quad-context."""
        A = self.create_transition_matrix(r=r)
        path, states, deltas, psi = viterbi_algo_sibs(
            bafs,
            mat_haps,
            pat_haps,
            states=self.states,
            A=A,
            pi0=pi0,
            std_dev=std_dev,
            eps=eps,
        )
        return path, states, deltas, psi

    def restrict_states(self):
        """Break down states into the same categories as Roach et al for determining recombinations."""
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

    def restrict_path(self, path):
        """Break down states into the same categories as Roach et al for determining recombinations."""
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

    def det_recomb_sex(self, i, j):
        """Determine the parental origin of the recombination event."""
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

    def isolate_recomb(self, path_xy, path_xz, window=100):
        """Isolate key recombination events from a pair of refined viterbi paths.

        NOTE: we will be comparing the viterbi path to the nearest recombinaton event
        """
        assert path_xy.size == path_xz.size
        transitions_01 = np.where(path_xy[:-1] != path_xy[1:])[0]
        transitions_02 = np.where(path_xz[:-1] != path_xz[1:])[0]
        mat_recomb = []
        pat_recomb = []
        for r in transitions_01:
            # This is the targetted transition that we want to assign to maternal or paternal position.
            i0, j0 = path_xy[r], path_xy[r + 1]
            dists = np.sqrt((transitions_02 - r) ** 2)
            if np.any(dists < window):
                min_dist = np.min(dists)
                r2 = transitions_02[np.argmin(dists)]
                i1, j1 = path_xz[r2], path_xz[r2 + 1]
                m = self.det_recomb_sex(i0, j0)
                m2 = self.det_recomb_sex(i1, j1)
                if m == 1 and (m2 == 1 or m2 == -1):
                    pat_recomb.append((r, min_dist))
                elif m == 0 and (m2 == 0 or m2 == -1):
                    mat_recomb.append((r, min_dist))
        # This returns the list of tuples on the recombination positions and minimum distances across the traces.
        return mat_recomb, pat_recomb
