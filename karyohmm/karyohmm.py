"""Main implementation of karyohmm classes."""

import numpy as np
from karyohmm_utils import (backward_algo, backward_algo_sibs, emission_baf,
                            forward_algo, forward_algo_sibs, viterbi_algo,
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
        """Obtain the state-string from the HMM."""
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

    def est_sigma_pi0(self, bafs, mat_haps, pat_haps, algo="Nelder-Mead", **kwargs):
        """Estimate sigma and pi0 using numerical optimization of forward algorithm likelihood."""
        assert algo in ["Nelder-Mead", "L-BFGS-B"]
        opt_res = minimize(
            lambda x: -self.forward_algorithm(
                bafs=bafs,
                mat_haps=mat_haps,
                pat_haps=pat_haps,
                pi0=x[0],
                std_dev=x[1],
                **kwargs,
            )[4],
            x0=[0.6, 0.2],
            method=algo,
            bounds=[(0.1, 0.99), (0.05, 0.4)],
            tol=1e-6,
            options={"disp": True},
        )
        pi0_est = opt_res.x[0]
        sigma_est = opt_res.x[1]
        return pi0_est, sigma_est


class MetaHMM(AneuploidyHMM):
    """A meta-HMM that evaluates all possible ploidy states for allele intensity data."""

    def __init__(self):
        """Initialize the MetaHMM class."""
        super().__init__()
        self.ploidy = 0
        self.aploid = "meta"
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
        mat_haps,
        pat_haps,
        pi0=0.2,
        std_dev=0.25,
        r=1e-4,
        a=1e-7,
        unphased=False,
    ):
        """Forward HMM algorithm under a multi-ploidy model."""
        assert bafs.ndim == 1
        assert (mat_haps.ndim == 2) & (pat_haps.ndim == 2)
        assert (pi0 > 0) & (pi0 < 1.0)
        assert std_dev > 0
        assert bafs.size == mat_haps.shape[1]
        assert mat_haps.shape == pat_haps.shape
        A = self.create_transition_matrix(self.karyotypes, r=r, a=a, unphased=unphased)
        alphas, scaler, _, _, loglik = forward_algo(
            bafs,
            mat_haps,
            pat_haps,
            self.states,
            A,
            pi0=pi0,
            std_dev=std_dev,
        )
        return alphas, scaler, self.states, self.karyotypes, loglik

    def backward_algorithm(
        self,
        bafs,
        mat_haps,
        pat_haps,
        pi0=0.2,
        std_dev=0.25,
        r=1e-4,
        a=1e-7,
        unphased=False,
        logr=False,
    ):
        """Backward HMM algorithm under a given statespace model."""
        assert bafs.ndim == 1
        assert (mat_haps.ndim == 2) & (pat_haps.ndim == 2)
        assert (pi0 > 0) & (pi0 < 1.0)
        assert std_dev > 0
        assert bafs.size == mat_haps.shape[1]
        assert mat_haps.shape == pat_haps.shape
        A = self.create_transition_matrix(self.karyotypes, r=r, a=a, unphased=unphased)
        betas, scaler, _, _, loglik = backward_algo(
            bafs,
            mat_haps,
            pat_haps,
            self.states,
            A,
            pi0=pi0,
            std_dev=std_dev,
        )
        return betas, scaler, self.states, self.karyotypes, loglik

    def forward_backward(
        self,
        bafs,
        mat_haps,
        pat_haps,
        pi0=0.2,
        std_dev=0.25,
        r=1e-4,
        a=1e-7,
        unphased=False,
    ):
        """Run the forward-backward algorithm across all states."""
        alphas, _, states, karyotypes, _ = self.forward_algorithm(
            bafs,
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
        mat_haps,
        pat_haps,
        pi0=0.2,
        std_dev=0.25,
        r=1e-4,
        a=1e-7,
        unphased=False,
    ):
        """Implement the viterbi traceback through karyotypic states."""
        assert bafs.ndim == 1
        assert (mat_haps.ndim == 2) & (pat_haps.ndim == 2)
        assert (pi0 > 0) & (pi0 < 1.0)
        assert std_dev > 0
        assert bafs.size == mat_haps.shape[1]
        assert mat_haps.shape == pat_haps.shape

        A = self.create_transition_matrix(self.karyotypes, r=r, a=a, unphased=unphased)
        path, states, deltas, psi = viterbi_algo(
            bafs,
            mat_haps,
            pat_haps,
            self.states,
            A,
            pi0=pi0,
            std_dev=std_dev,
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

    def create_transition_matrix(self, r=1e-16):
        """Create the transition matrix here."""
        m = len(self.states)
        A = np.zeros(shape=(m, m))
        A[:, :] = r / m
        for i in range(m):
            A[i, i] = 0.0
            A[i, i] = 1.0 - np.sum(A[i, :])
        return np.log(A)

    def forward_algorithm(
        self, bafs, mat_haps, pat_haps, pi0=0.2, std_dev=0.1, r=1e-16
    ):
        """Implement the forward algorithm for QuadHMM model."""
        A = self.create_transition_matrix(r=r)
        alphas, scaler, states, karyotypes, loglik = forward_algo_sibs(
            bafs,
            mat_haps,
            pat_haps,
            states=self.states,
            A=A,
            pi0=pi0,
            std_dev=std_dev,
        )
        return alphas, scaler, states, karyotypes, loglik

    def forward_backward(self, bafs, mat_haps, pat_haps, pi0=0.2, std_dev=0.1, r=1e-16):
        """Implement the forward-backward algorithm for the QuadHMM model."""
        A = self.create_transition_matrix(r=r)
        alphas, _, states, _, _ = forward_algo_sibs(
            bafs,
            mat_haps,
            pat_haps,
            states=self.states,
            A=A,
            pi0=pi0,
            std_dev=std_dev,
        )
        betas, _, _, _, _ = backward_algo_sibs(
            bafs,
            mat_haps,
            pat_haps,
            states=self.states,
            A=A,
            pi0=pi0,
            std_dev=std_dev,
        )
        gammas = (alphas + betas) - logsumexp_sp(alphas + betas, axis=0)
        return gammas, states, None

    def viterbi_algorithm(
        self, bafs, mat_haps, pat_haps, pi0=0.2, std_dev=0.1, r=1e-16
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
        )
        return path, states, deltas, psi

    def viterbi_path(self, bafs, mat_haps, pat_haps, pi0=0.2, std_dev=0.1, r=1e-16):
        """Obtain the restricted viterbi path for traceback."""
        path, _, _, _ = self.viterbi_algorithm(
            bafs, mat_haps, pat_haps, pi0=pi0, std_dev=std_dev, r=r
        )
        res_path = self.restrict_path(path)
        return res_path

    def map_path(self, bafs, mat_haps, pat_haps, pi0=0.2, std_dev=0.1, r=1e-16):
        """Obtain the Maximum A-Posteriori Path across restricted states."""
        gammas, _, _ = self.forward_backward(
            bafs, mat_haps, pat_haps, pi0=pi0, std_dev=std_dev, r=r
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
        return np.argmax(red_gammas, axis=0)

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

    def isolate_recomb(self, path_xy, path_xzs, window=20):
        """Isolate key recombination events from a pair of refined viterbi paths.

        Args:
        - path_xy: numpy array of path through specific focal pair of individuals
        - path_xzs: list of numpy arrays of
        - window: number of SNPs that the closest transition must be in (e.g. minimum resolution)

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
        mat_recomb_lst = [k for k in mat_recomb if mat_recomb[k] >= len(path_xzs) / 2]
        pat_recomb_lst = [k for k in pat_recomb if pat_recomb[k] >= len(path_xzs) / 2]
        # This returns the list of tuples on the recombination positions and minimum distances across the traces.
        return mat_recomb_lst, pat_recomb_lst, mat_recomb, pat_recomb


class PhaseCorrect:
    """Module for implementing Mendelian phase correction using BAF data."""

    def __init__(self, mat_haps, pat_haps):
        """Intialize the class for phase correction."""
        assert mat_haps.shape[0] == pat_haps.shape[0]
        assert mat_haps.shape[1] == pat_haps.shape[1]
        assert np.all(np.isin(mat_haps, [0, 1]))
        assert np.all(np.isin(mat_haps, [0, 1]))
        self.mat_haps = mat_haps
        self.pat_haps = pat_haps
        self.mat_haps_true = None
        self.pat_haps_true = None
        self.embryo_bafs = None

    def add_true_haps(self, true_mat_haps, true_pat_haps):
        """Add in true haplotypes if available from a simulation."""
        assert true_mat_haps.shape[0] == self.mat_haps.shape[0]
        assert true_mat_haps.shape[1] == self.mat_haps.shape[1]
        assert true_pat_haps.shape[0] == self.pat_haps.shape[0]
        assert true_pat_haps.shape[1] == self.pat_haps.shape[1]
        self.mat_haps_true = true_mat_haps
        self.pat_haps_true = true_pat_haps

    def add_baf(self, embryo_bafs):
        """Add in BAF estimates for each embryo."""
        pass

    def estimate_switch_err(self, maternal=True):
        """Estimate the switch error from true and inferred haplotypes.

        The switch error is defined as consecutive heterozygotes that are
        in the incorrect orientation.

        Returns:
            - `n_switches`: number of switches between consecutive heterozygotes
            - `n_consecutive_hets`: number of consecutive heterozygotes
            - `switch_err_rate`: number of switches per consecutive heterozygote
        """
        assert self.mat_hap_true is not None
        assert self.pat_hap_true is not None
        if maternal:
            true_haps = self.mat_haps_true
            inf_haps = self.mat_haps
        else:
            true_haps = self.pat_haps_true
            inf_haps = self.pat_haps
        geno = true_haps.sum(axis=0)
        het_idxs = np.where(geno == 1)[0]
        n_switches = 0
        n_consecutive_hets = 0
        for (i, j) in zip(het_idxs[:-1], het_idxs[1:]):
            n_consecutive_hets += 1
            true_hap = true_haps[:, [i, j]]
            inf_hap = inf_haps[:, [i, j]]
            # Check if the heterozygotes are oriented appropriately
            if np.any(true_hap[0, :] != inf_hap[0, :]) and np.any(
                (true_hap[0, :] != inf_hap[1, :])
            ):
                n_switches += 1
        return n_switches, n_consecutive_hets, n_switches / n_consecutive_hets

    def calculate_logll(self, mat_haps, pat_haps, baf, **kwargs):
        """Helper function for calculating the log-likelihood of being in phase or anti-phase orientation."""
        assert mat_haps.shape[1] == 2
        assert mat_haps.shape[0] == 2
        assert pat_haps.shape[1] == 2
        assert pat_haps.shape[0] == 2
        assert baf.size == 2
        # Calculate the likelihood of this embryo BAF in the phase orientation
        phase_orientation1 = emission_baf(
            baf=baf[0], m=mat_haps[0, 0], p=pat_haps[0, 0], **kwargs
        ) + emission_baf(baf=baf[1], m=mat_haps[0, 1], p=pat_haps[0, 1], **kwargs)
        phase_orientation2 = emission_baf(
            baf=baf[0], m=mat_haps[1, 0], p=pat_haps[0, 0], **kwargs
        ) + emission_baf(baf=baf[1], m=mat_haps[1, 1], p=pat_haps[0, 1], **kwargs)
        # Calculate the likelihood of this embryo BAF in the `antiphase` orientation
        antiphase_orientation1 = emission_baf(
            baf=baf[0], m=mat_haps[0, 0], p=pat_haps[0, 0], **kwargs
        ) + emission_baf(baf=baf[1], m=mat_haps[1, 1], p=pat_haps[0, 1], **kwargs)
        antiphase_orientation2 = emission_baf(
            baf=baf[0], m=mat_haps[1, 0], p=pat_haps[0, 0], **kwargs
        ) + emission_baf(baf=baf[1], m=mat_haps[0, 1], p=pat_haps[0, 1], **kwargs)
        # This gets the maximum likelihood out of the phase and antiphase configurations ...
        phase_orientation = logsumexp([phase_orientation1, phase_orientation2])
        antiphase_orientation = logsumexp(
            [antiphase_orientation1, antiphase_orientation2]
        )
        return phase_orientation, antiphase_orientation

    def phase_correct(self, lod_score=-0.5, **kwargs):
        """Implement a phase correction for both maternal and paternal haplotypes."""
        assert self.embryo_bafs is not None
        mat_geno = self.mat_haps.sum(axis=0)
        pat_geno = self.pat_haps.sum(axis=0)
        # 1. Identify the switch points for the maternal switches
        idx_het_mat = np.where((mat_geno == 1) & (pat_geno != 1))[0]
        idx_het_pat = np.where((pat_geno == 1) & (mat_geno != 1))[0]
        mat_switch_inf = []
        pat_switch_inf = []
        for i, j in tqdm(zip(idx_het_mat[:-1], idx_het_mat[1:])):
            phase = []
            antiphase = []
            for baf in sibling_bafs:
                cur_baf = baf[[i, j]]
                cur_phase, cur_antiphase = self.calculate_logll(
                    mat_haps[:, [i, j]], pat_haps[:, [i, j]], baf=cur_baf, **kwargs
                )
                phase.append(cur_phase)
                antiphase.append(cur_antiphase)
            tot_phase = logsumexp_sp(phase)
            tot_antiphase = logsumexp_sp(antiphase)
            scalar = logsumexp_sp([tot_phase, tot_antiphase])
            mat_switch_inf.append([i, j, tot_phase - scalar])
        # 2. Infer the switch-points for the putative paternal switches
        pat_switch_inf = []
        for i, j in tqdm(zip(idx_het_pat[:-1], idx_het_pat[1:])):
            phase = []
            antiphase = []
            for baf in sibling_bafs:
                cur_baf = baf[[i, j]]
                cur_phase, cur_antiphase = self.calculate_logll(
                    pat_haps[:, [i, j]], mat_haps[:, [i, j]], baf=cur_baf, **kwargs
                )
                phase.append(cur_phase)
                antiphase.append(cur_antiphase)
            tot_phase = logsumexp_sp(phase)
            tot_antiphase = logsumexp_sp(antiphase)
            scalar = logsumexp_sp([tot_phase, tot_antiphase])
            pat_switch_inf.append([i, j, tot_phase - scalar])
        mat_switch_inf = np.array(mat_switch_inf)
        pat_switch_inf = np.array(pat_switch_inf)
        return mat_switch_inf, pat_switch_inf
