"""
Karyohmm: HMM-based inference of chromosomal copy-number state from array intensity data.

Karyohmm implements haplotype-aware hidden Markov models for detecting aneuploidies,
mosaic copy-number changes, and maternal-cell contamination from B-allele frequency
(BAF) and log-R ratio (LRR) data when parental genotypes are available.  All models
condition on phased parental haplotypes and exploit Mendelian transmission to
distinguish true copy-number signal from genotyping noise.

Available classes
-----------------
- :class:`MetaHMM`: Whole-chromosome aneuploidy detection across a full ploidy state space
  (nullisomy, monosomy, disomy, trisomy, and optional UPD states).
- :class:`QuadHMM`: Joint HMM for two sibling embryos; used to call inter-sibling haplotype
  sharing and to detect crossover recombination events (Roach et al. 2010).
- :class:`PocHMM`: Aneuploidy inference in products-of-conception with only one observed parent
  (mother–child duo or father–child duo); marginalises over the unobserved parent using
  population allele frequencies.
- :class:`MccEst`: Maximum-likelihood estimator of maternal-cell contamination (MCC) fraction
  from BAF data, with both unphased and phase-aware HMM variants and genome-wide
  multi-chromosome support.
- :class:`MosaicEst`: Estimator of mosaic cell fraction from BAF imbalance at heterozygous
  sites using a 3-state (gain / neutral / loss) HMM (Loh et al. model).
- :class:`PhaseCorrect`: Mendelian phase correction for parental haplotypes using BAF data
  from multiple sibling embryos.
- :class:`RecombEst`: Simplified crossover detection following Coop et al. 2007, extended
  with QuadHMM-based interval refinement.

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
    forward_mcc_phased_poc,
    forward_mcc_phased_trio,
    logaddexp,
    logsumexp,
    loglik_mcc,
    loglik_mcc_phased,
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
    """Abstract base class shared by all karyohmm HMM models.

    Defines the common interface (``forward_algorithm``, ``backward_algorithm``,
    ``forward_backward``, ``viterbi_algorithm``) and shared parameter-estimation
    helpers.  Concrete subclasses (:class:`MetaHMM`, :class:`QuadHMM`,
    :class:`PocHMM`) implement the state space and emission model appropriate for
    their data type.

    Attributes:
        ploidy (int): Expected ploidy of the embryo (0 = meta/unknown, 2 = disomy).
        aploid (str or None): String tag identifying the active model variant
            (e.g. ``"meta"``, ``"disomy"``, ``"meta+upd"``).
    """

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

    # Per-algorithm tolerance keys recognized by scipy.optimize.minimize.
    # Nelder-Mead uses xatol/fatol; L-BFGS-B uses ftol/gtol; Powell uses xtol/ftol.
    _ALGO_DEFAULT_OPTIONS = {
        "Nelder-Mead": {"xatol": 1e-4, "fatol": 1e-4, "disp": False},
        "L-BFGS-B": {"ftol": 1e-4, "gtol": 1e-5, "disp": False},
        "Powell": {"xtol": 1e-4, "ftol": 1e-4, "disp": False},
    }

    def est_sigma_pi0(
        self,
        bafs,
        pos,
        mat_haps,
        pat_haps,
        algo="L-BFGS-B",
        pi0_bounds=(0.01, 0.99),
        sigma_bounds=(1e-2, 1.0),
        opt_tol=1e-4,
        opt_options=None,
        **kwargs,
    ):
        """Estimate sigma and pi0 under the B-Allele Frequency model using optimization of forward algorithm likelihood.

        Arguments:
            - bafs (`np.array`): B-allele frequencies across the all m sites
            - pos (`np.array`): basepair positions of the SNPs
            - mat_haps (`np.array`): a 2 x m array of 0/1 maternal haplotypes
            - pat_haps (`np.array`): a 2 x m array of 0/1 paternal haplotypes
            - algo (`str`): one of Nelder-Mead, L-BFGS-B, or Powell algorithms for optimization
            - opt_tol (`float`): master tolerance passed to scipy minimize (default 1e-4)
            - opt_options (`dict` or None): algorithm-specific options passed to scipy minimize;
              merged over per-algorithm defaults, so only keys you want to override are needed

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
        options = {**self._ALGO_DEFAULT_OPTIONS[algo], **(opt_options or {})}
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
            tol=opt_tol,
            options=options,
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
    """HMM for whole-chromosome aneuploidy detection in trio PGT data.

    Evaluates a joint state space covering nullisomy (0), maternal and paternal
    monosomy (1m / 1p), disomy (2), maternal and paternal trisomy (3m / 3p),
    and optionally uniparental disomy (UPD) states.  Each hidden state encodes
    which pair of parental haplotypes were transmitted to the embryo; transitions
    follow an exponential kernel parameterised by an intra-karyotype recombination
    rate ``r`` and an inter-karyotype switching rate ``a``.

    The BAF emission at each site is a Gaussian mixture centred on the expected
    allele dosage for the current copying state, controlled by ``pi0`` (fraction
    of sites with no signal) and ``std_dev`` (noise).  LRR data are incorporated
    through a Gaussian emission centred on the expected copy-number fold change.

    Parameters
    ----------
    disomy : bool, optional
        Restrict the state space to disomy-only (4 states).  Useful for genotyping
        embryos without calling aneuploidies.  Default ``False``.
    upd : bool, optional
        Add uniparental disomy states to the full state space.  Requires LRR data
        to distinguish UPD from normal disomy.  Default ``False``.

    Attributes
    ----------
    states : list of tuple
        Active HMM states; each tuple is ``(m0, m1, p0, p1)`` where values are
        haplotype indices (0 or 1) or -1 when that haplotype is absent.
    karyotypes : np.ndarray of str
        Karyotype label for each state (e.g. ``"2"``, ``"3m"``, ``"1p"``).
    aploid : str
        One of ``"meta"``, ``"disomy"``, or ``"meta+upd"``.
    """

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
    """Joint HMM for two sibling embryos used to detect inter-sibling haplotype sharing.

    Models the joint copying state of two euploid siblings as a pair of independent
    disomy states drawn from the same parental haplotypes.  The 16-state product
    space (4 × 4 single-embryo disomy states) is collapsed into four Roach et al.
    2010 identity-by-descent classes:

    - 0: maternally haploidentical (siblings share the same maternal haplotype)
    - 1: paternally haploidentical (siblings share the same paternal haplotype)
    - 2: fully identical (same maternal *and* paternal haplotype)
    - 3: non-identical (different maternal and paternal haplotypes)

    Transitions between copying states are governed by a single recombination rate
    parameter ``r``; the BAF emission for each sibling follows the same Gaussian
    mixture model as :class:`MetaHMM` but with per-sibling ``pi0`` and ``std_dev``.

    Attributes
    ----------
    states : list of tuple of tuple
        16 joint copying states, each ``((m0, _, p0, _), (m1, _, p1, _))``.
    karyotypes : np.ndarray of str
        All entries are ``"2"`` (disomy assumed for both siblings).
    """

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
    """Aneuploidy HMM for products-of-conception with a single observed parent.

    Extends :class:`MetaHMM` to handle mother–child or father–child duos in which
    only one parent's haplotypes are available.  The unobserved parent's genotype is
    marginalised out at each site using population allele frequencies supplied as
    ``freqs``.  When ``freqs`` is ``None`` the model treats every unobserved-parent
    site as a 50 / 50 heterozygote.

    The forward, backward, and Viterbi algorithms are re-implemented here via a
    dedicated Cython kernel (``forward_algo_duo`` / ``backward_algo_duo``) that
    handles the marginalisation efficiently.

    Parameters
    ----------
    disomy : bool, optional
        Restrict the state space to disomy (ignored in current implementation;
        the full MetaHMM state space is always used).  Default ``False``.
    """

    def __init__(self, disomy=False):
        """Initialize the PocHMM with the full MetaHMM state space."""
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
        """Compute posterior genotype dosages for the unobserved parent.

        For each site integrates the BAF and LRR likelihood over all four possible
        unobserved-parent genotypes (AA, AB, BA, BB), weighted by Hardy–Weinberg
        priors from ``freqs`` and the HMM posterior ``gammas``.

        Arguments:
            - bafs (`np.array`): m-length array of B-allele frequencies
            - lrrs (`np.array`): m-length array of log-R ratios
            - sigmas (`np.array`): m-length array of per-site LRR noise estimates
            - haps (`np.array`): 2 x m array of 0/1 haplotypes for the *observed* parent
            - gammas (`np.array`): k x m log-posterior array from :meth:`forward_backward`
            - freqs (`np.array`, optional): m-length array of population allele frequencies;
              defaults to 0.5 at every site
            - maternal (`bool`): if ``True`` (default), ``haps`` are the maternal haplotypes
              and the paternal genotype is imputed; set ``False`` for the reverse
            - pi0 (`float`): sparsity parameter for the BAF emission model
            - std_dev (`float`): noise parameter for the BAF emission model

        Returns:
            - geno_dosage_rev (`np.array`): 3 x m array of posterior genotype probabilities
              (log scale) for the unobserved parent; rows correspond to homozygous-ref (0),
              heterozygous (1), and homozygous-alt (2)

        """
        assert bafs.ndim == 1
        assert lrrs.ndim == 1
        assert sigmas.ndim == 1
        assert bafs.size == lrrs.size
        assert sigmas.size == lrrs.size
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

    def infer_missing_af(self, bafs, geno, hap_matrix, pos, eps=0.2):
        """Estimate allele frequencies for the unobserved parent using a haplotype reference panel.

        Identifies sites of opposite homozygosity — where the observed parent is homozygous
        and BAF indicates the unobserved parent carries the opposite allele — then tags the
        corresponding haplotypes in the reference panel. Frequencies at non-anchor sites are
        estimated via distance-weighted interpolation between adjacent anchor sites (Eq. 9 in
        the PocHMM methods document). Falls back to the reference-panel mean at sites with no
        nearby anchors.

        Arguments:
            - bafs (`np.array`): m-length B-allele frequency array
            - geno (`np.array`): m-length observed-parent genotype dosage array (0, 1, or 2)
            - hap_matrix (`np.array`): (2N, m) haplotype reference panel matrix (0/1 encoded)
            - pos (`np.array`): m-length array of genomic positions in basepairs
            - eps (`float`): BAF threshold for calling opposite homozygosity (default 0.2)

        Returns:
            - freqs (`np.array`): m-length array of estimated allele frequencies for the
              unobserved parent

        """
        assert bafs.ndim == 1
        assert geno.ndim == 1
        assert hap_matrix.ndim == 2
        assert pos.ndim == 1
        m = bafs.size
        assert geno.size == m
        assert hap_matrix.shape[1] == m
        assert pos.size == m
        assert 0.0 < eps < 0.5
        assert np.all(np.isin(geno, [0, 1, 2]))
        assert np.all((bafs >= 0.0) & (bafs <= 1.0))
        assert np.all(pos[1:] > pos[:-1])

        # Fallback: global mean alt allele frequency from the reference panel
        global_freq = hap_matrix.mean(axis=0)

        # Identify anchor sites of opposite homozygosity
        anchor_mask = ((geno == 0) & (bafs >= eps)) | (
            (geno == 2) & (bafs <= 1.0 - eps)
        )
        anchor_indices = np.where(anchor_mask)[0]

        if len(anchor_indices) == 0:
            return global_freq

        # Tag reference haplotypes at each anchor by the inferred unobserved-parent allele:
        #   geno==0 (observed hom-ref) → unobserved carries alt → tag alt haplotypes
        #   geno==2 (observed hom-alt) → unobserved carries ref → tag ref haplotypes
        tagged = {}
        for idx in anchor_indices:
            allele = 1 if geno[idx] == 0 else 0
            x = np.where(hap_matrix[:, idx] == allele)[0]
            if len(x) > 0:
                tagged[int(idx)] = x

        valid_anchors = sorted(tagged.keys())
        if len(valid_anchors) == 0:
            return global_freq

        va = np.array(valid_anchors)

        def _af(anchor_site, query_site):
            return float(hap_matrix[tagged[anchor_site], query_site].mean())

        freqs = np.empty(m)
        for k in range(m):
            ins = int(np.searchsorted(va, k, side="right"))
            left, right = ins - 1, ins

            if left < 0:
                # Before the first anchor: use first anchor's tagged haplotypes
                freqs[k] = _af(int(va[0]), k)
            elif right >= len(va):
                # After the last anchor: use last anchor's tagged haplotypes
                freqs[k] = _af(int(va[-1]), k)
            elif int(va[left]) == k:
                # At an anchor site: use direct estimate from tagged haplotypes
                freqs[k] = _af(k, k)
            else:
                # Between two anchors: linear distance-weighted interpolation
                i, j = int(va[left]), int(va[right])
                d_ij = float(pos[j] - pos[i])
                w_i = 1.0 - float(pos[k] - pos[i]) / d_ij
                w_j = 1.0 - float(pos[j] - pos[k]) / d_ij
                freqs[k] = w_i * _af(i, k) + w_j * _af(j, k)

        return freqs


class MccEst:
    """Maximum-likelihood estimator of maternal-cell contamination (MCC) from BAF data.

    Maternal-cell contamination occurs when maternal DNA co-purifies with the embryo
    sample, biasing B-allele frequencies towards the maternal genotype.  This class
    provides log-likelihood functions and MLE routines for estimating the contamination
    fraction ``c`` (the proportion of signal attributable to the mother) under two
    data configurations:

    - **Trio** (``_trio`` suffix): both maternal and paternal haplotypes are known, so
      the paternal genotype at each site is observed directly.
    - **POC / duo** (``_poc`` suffix): only maternal haplotypes are available; the
      paternal genotype is marginalised using Hardy–Weinberg allele frequencies.

    Each configuration is further available in an **unphased** form (sites modelled
    independently) and a **phase-aware** form (a 2-state HMM tracks which maternal
    haplotype was transmitted to the embryo).

    Genome-wide inference across multiple chromosomes is supported through
    ``loglik_mcc_genome_*`` and ``est_mcc_genome_*`` methods; per-chromosome
    estimates — useful for identifying aneuploid chromosomes before a genome-wide
    fit — are provided by ``est_mcc_per_chrom_*`` methods.

    All estimation methods jointly optimise the contamination fraction ``c`` and the
    BAF noise standard deviation ``std_dev``.
    """

    def __init__(self):
        """Initialize the MCC estimator (stateless; all state is passed per call)."""
        pass

    def loglik_mcc_poc(self, bafs, mat_haps, freqs, c=0.0, std_dev=0.1):
        """Log-likelihood of MCC for a single chromosome in a mother–child duo (POC).

        At each site the paternal genotype is unknown and is marginalised over
        Hardy–Weinberg frequencies derived from ``freqs``.  The contamination
        fraction ``c`` shifts the expected BAF mean towards the maternal genotype.

        Arguments:
            - bafs (`np.array`): m-length array of B-allele frequencies in [0, 1]
            - mat_haps (`np.array`): 2 x m array of 0/1 maternal haplotypes
            - freqs (`np.array`): m-length array of population allele frequencies in [0, 1]
            - c (`float`): maternal contamination fraction in [0, 0.5]
            - std_dev (`float`): BAF noise standard deviation (> 0)

        Returns:
            - logll (`float`): total log-likelihood summed across all m sites

        """
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
        """Log-likelihood of MCC for a single chromosome in a full trio.

        With both parents phased the paternal genotype at every site is known,
        giving a simpler per-site likelihood with no HWE marginalisation.

        Arguments:
            - bafs (`np.array`): m-length array of B-allele frequencies in [0, 1]
            - mat_haps (`np.array`): 2 x m array of 0/1 maternal haplotypes
            - pat_haps (`np.array`): 2 x m array of 0/1 paternal haplotypes
            - c (`float`): maternal contamination fraction in [0, 0.5]
            - std_dev (`float`): BAF noise standard deviation (> 0)

        Returns:
            - logll (`float`): total log-likelihood summed across all m sites

        """
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
        """Estimate MCC by MLE for a single chromosome in a mother–child duo.

        Jointly optimises contamination fraction ``c`` and noise ``std_dev`` by
        maximising :meth:`loglik_mcc_poc`.

        Arguments:
            - bafs (`np.array`): m-length array of B-allele frequencies
            - mat_haps (`np.array`): 2 x m array of 0/1 maternal haplotypes
            - freqs (`np.array`): m-length array of population allele frequencies
            - algo (`str`): scipy minimisation algorithm; default ``"Nelder-Mead"``
            - **kwargs: additional keyword arguments forwarded to ``scipy.optimize.minimize``

        Returns:
            - c_est (`float`): MLE of the contamination fraction
            - sigma_est (`float`): MLE of the BAF noise standard deviation

        """
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
        """Estimate MCC by MLE for a single chromosome in a full trio.

        Jointly optimises contamination fraction ``c`` and noise ``std_dev`` by
        maximising :meth:`loglik_mcc_trio`.

        Arguments:
            - bafs (`np.array`): m-length array of B-allele frequencies
            - mat_haps (`np.array`): 2 x m array of 0/1 maternal haplotypes
            - pat_haps (`np.array`): 2 x m array of 0/1 paternal haplotypes
            - algo (`str`): scipy minimisation algorithm; default ``"Nelder-Mead"``
            - **kwargs: additional keyword arguments forwarded to ``scipy.optimize.minimize``

        Returns:
            - c_est (`float`): MLE of the contamination fraction
            - sigma_est (`float`): MLE of the BAF noise standard deviation

        """
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
        """Profile-likelihood confidence interval for MCC in the POC model.

        Uses Wilks' theorem: the CI boundary is the set of ``c`` values where
        twice the log-likelihood drop from the MLE equals the ``alpha`` quantile
        of a chi-squared distribution with ``df`` degrees of freedom.

        Arguments:
            - bafs (`np.array`): m-length array of B-allele frequencies
            - mat_haps (`np.array`): 2 x m array of 0/1 maternal haplotypes
            - freqs (`np.array`): m-length array of population allele frequencies
            - c_hat (`float`): MLE contamination fraction (the point estimate)
            - std_dev (`float`): MLE BAF noise standard deviation (held fixed during profile)
            - alpha (`float`): confidence level in (0, 1); default 0.95
            - df (`int`): degrees of freedom for chi-squared quantile; default 1

        Returns:
            - lower_CI (`float`): lower confidence bound (falls back to 0 if optimiser fails)
            - c_hat (`float`): the supplied point estimate
            - upper_CI (`float`): upper confidence bound (falls back to 0.5 if optimiser fails)

        """
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
        """Profile-likelihood confidence interval for MCC in the trio model.

        Uses Wilks' theorem: the CI boundary is the set of ``c`` values where
        twice the log-likelihood drop from the MLE equals the ``alpha`` quantile
        of a chi-squared distribution with ``df`` degrees of freedom.

        Arguments:
            - bafs (`np.array`): m-length array of B-allele frequencies
            - mat_haps (`np.array`): 2 x m array of 0/1 maternal haplotypes
            - pat_haps (`np.array`): 2 x m array of 0/1 paternal haplotypes
            - c_hat (`float`): MLE contamination fraction (the point estimate)
            - std_dev (`float`): MLE BAF noise standard deviation (held fixed during profile)
            - h (`float`): finite-difference step (unused; retained for API compatibility)
            - alpha (`float`): confidence level in (0, 1); default 0.95
            - df (`int`): degrees of freedom for chi-squared quantile; default 1

        Returns:
            - lower_CI (`float`): lower confidence bound (falls back to 0 if optimiser fails)
            - c_hat (`float`): the supplied point estimate
            - upper_CI (`float`): upper confidence bound (falls back to 0.5 if optimiser fails)

        """
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

    def loglik_mcc_phased_trio(
        self, bafs, mat_haps, pat_haps, pos, c=0.0, std_dev=0.1, r=1e-8
    ):
        """Phase-aware log-likelihood of MCC for a single chromosome in a trio.

        A 2-state HMM tracks which maternal haplotype (0 or 1) was transmitted to
        the embryo.  At each site the emission is a Gaussian centred on the BAF
        expected for the copied maternal allele plus the contamination contribution.
        Transitions follow an exponential recombination kernel:
        ``rho = 1 - exp(-r * d)`` where ``d`` is the inter-site distance in base pairs.

        Arguments:
            - bafs (`np.array`): m-length array of B-allele frequencies in [0, 1]
            - mat_haps (`np.array`): 2 x m array of 0/1 maternal haplotypes
            - pat_haps (`np.array`): 2 x m array of 0/1 paternal haplotypes
            - pos (`np.array`): m-length strictly increasing array of base-pair positions
            - c (`float`): maternal contamination fraction in [0, 0.5]
            - std_dev (`float`): BAF noise standard deviation (> 0)
            - r (`float`): per-base-pair recombination rate (> 0); default 1e-8

        Returns:
            - loglik (`float`): total forward-algorithm log-likelihood across m sites

        """
        assert bafs.ndim == 1
        assert mat_haps.ndim == 2 and mat_haps.shape == (2, bafs.size)
        assert pat_haps.ndim == 2 and pat_haps.shape == (2, bafs.size)
        assert pos.ndim == 1 and pos.size == bafs.size
        assert np.all(pos[1:] > pos[:-1]), "pos must be strictly increasing"
        assert np.all((bafs >= 0) & (bafs <= 1))
        assert (c >= 0) and (c <= 0.5)
        assert std_dev > 0
        assert r > 0
        return forward_mcc_phased_trio(
            bafs.astype(np.float64),
            mat_haps.astype(np.int32),
            pat_haps.astype(np.int32),
            pos.astype(np.float64),
            c=c,
            std_dev=std_dev,
            r=r,
        )

    def loglik_mcc_phased_poc(
        self, bafs, mat_haps, freqs, pos, c=0.0, std_dev=0.1, r=1e-8
    ):
        """Phase-aware log-likelihood of MCC for a single chromosome in a duo (POC).

        Identical to :meth:`loglik_mcc_phased_trio` but marginalises over the
        unobserved paternal genotype at each site using Hardy–Weinberg weights
        derived from ``freqs``.

        Arguments:
            - bafs (`np.array`): m-length array of B-allele frequencies in [0, 1]
            - mat_haps (`np.array`): 2 x m array of 0/1 maternal haplotypes
            - freqs (`np.array`): m-length array of population allele frequencies in [0, 1]
            - pos (`np.array`): m-length strictly increasing array of base-pair positions
            - c (`float`): maternal contamination fraction in [0, 0.5]
            - std_dev (`float`): BAF noise standard deviation (> 0)
            - r (`float`): per-base-pair recombination rate (> 0); default 1e-8

        Returns:
            - loglik (`float`): total forward-algorithm log-likelihood across m sites

        """
        assert bafs.ndim == 1
        assert mat_haps.ndim == 2 and mat_haps.shape == (2, bafs.size)
        assert freqs.ndim == 1 and freqs.size == bafs.size
        assert pos.ndim == 1 and pos.size == bafs.size
        assert np.all(pos[1:] > pos[:-1]), "pos must be strictly increasing"
        assert np.all((bafs >= 0) & (bafs <= 1))
        assert np.all((freqs >= 0) & (freqs <= 1))
        assert (c >= 0) and (c <= 0.5)
        assert std_dev > 0
        assert r > 0
        return forward_mcc_phased_poc(
            bafs.astype(np.float64),
            mat_haps.astype(np.int32),
            freqs.astype(np.float64),
            pos.astype(np.float64),
            c=c,
            std_dev=std_dev,
            r=r,
        )

    def est_mcc_phased_trio(
        self, bafs, mat_haps, pat_haps, pos, r=1e-8, algo="Nelder-Mead", **kwargs
    ):
        """Estimate MCC by MLE using the phase-aware HMM for a single trio chromosome.

        Jointly optimises ``c`` and ``std_dev`` by maximising
        :meth:`loglik_mcc_phased_trio`.

        Arguments:
            - bafs (`np.array`): m-length array of B-allele frequencies
            - mat_haps (`np.array`): 2 x m array of 0/1 maternal haplotypes
            - pat_haps (`np.array`): 2 x m array of 0/1 paternal haplotypes
            - pos (`np.array`): m-length strictly increasing array of base-pair positions
            - r (`float`): per-base-pair recombination rate; default 1e-8
            - algo (`str`): scipy minimisation algorithm; default ``"Nelder-Mead"``
            - **kwargs: additional keyword arguments forwarded to ``scipy.optimize.minimize``

        Returns:
            - c_est (`float`): MLE of the contamination fraction
            - sigma_est (`float`): MLE of the BAF noise standard deviation

        """
        opt_res = minimize(
            lambda x: (
                -self.loglik_mcc_phased_trio(
                    bafs=bafs,
                    mat_haps=mat_haps,
                    pat_haps=pat_haps,
                    pos=pos,
                    c=x[0],
                    std_dev=x[1],
                    r=r,
                )
            ),
            x0=[0.05, 0.1],
            method=algo,
            bounds=[(0, 0.5), (1e-3, 0.3)],
            **kwargs,
        )
        return opt_res.x[0], opt_res.x[1]

    def est_mcc_phased_poc(
        self, bafs, mat_haps, freqs, pos, r=1e-8, algo="Nelder-Mead", **kwargs
    ):
        """Estimate MCC by MLE using the phase-aware HMM for a single duo (POC) chromosome.

        Jointly optimises ``c`` and ``std_dev`` by maximising
        :meth:`loglik_mcc_phased_poc`.

        Arguments:
            - bafs (`np.array`): m-length array of B-allele frequencies
            - mat_haps (`np.array`): 2 x m array of 0/1 maternal haplotypes
            - freqs (`np.array`): m-length array of population allele frequencies
            - pos (`np.array`): m-length strictly increasing array of base-pair positions
            - r (`float`): per-base-pair recombination rate; default 1e-8
            - algo (`str`): scipy minimisation algorithm; default ``"Nelder-Mead"``
            - **kwargs: additional keyword arguments forwarded to ``scipy.optimize.minimize``

        Returns:
            - c_est (`float`): MLE of the contamination fraction
            - sigma_est (`float`): MLE of the BAF noise standard deviation

        """
        opt_res = minimize(
            lambda x: (
                -self.loglik_mcc_phased_poc(
                    bafs=bafs,
                    mat_haps=mat_haps,
                    freqs=freqs,
                    pos=pos,
                    c=x[0],
                    std_dev=x[1],
                    r=r,
                )
            ),
            x0=[0.05, 0.1],
            method=algo,
            bounds=[(0, 0.5), (1e-3, 0.3)],
            **kwargs,
        )
        return opt_res.x[0], opt_res.x[1]

    def mcc_ci_phased_trio(
        self,
        bafs,
        mat_haps,
        pat_haps,
        pos,
        c_hat=0.0,
        std_dev=0.1,
        r=1e-8,
        alpha=0.95,
        df=1,
    ):
        """Profile-likelihood CI for MCC using the phase-aware trio HMM.

        Arguments:
            - bafs (`np.array`): m-length array of B-allele frequencies
            - mat_haps (`np.array`): 2 x m array of 0/1 maternal haplotypes
            - pat_haps (`np.array`): 2 x m array of 0/1 paternal haplotypes
            - pos (`np.array`): m-length strictly increasing array of base-pair positions
            - c_hat (`float`): MLE contamination fraction (the point estimate)
            - std_dev (`float`): MLE BAF noise standard deviation (held fixed during profile)
            - r (`float`): per-base-pair recombination rate; default 1e-8
            - alpha (`float`): confidence level in (0, 1); default 0.95
            - df (`int`): degrees of freedom for chi-squared quantile; default 1

        Returns:
            - lower_CI (`float`): lower confidence bound (falls back to 0 if optimiser fails)
            - c_hat (`float`): the supplied point estimate
            - upper_CI (`float`): upper confidence bound (falls back to 0.5 if optimiser fails)

        """
        assert (c_hat >= 0) and (c_hat <= 0.5)
        assert std_dev > 0
        assert (alpha > 0) and (alpha < 1)
        ll_hat = self.loglik_mcc_phased_trio(
            bafs=bafs,
            mat_haps=mat_haps,
            pat_haps=pat_haps,
            pos=pos,
            c=c_hat,
            std_dev=std_dev,
            r=r,
        )
        wilks = lambda x: (
            2
            * (
                ll_hat
                - self.loglik_mcc_phased_trio(
                    bafs=bafs,
                    mat_haps=mat_haps,
                    pat_haps=pat_haps,
                    pos=pos,
                    c=x,
                    std_dev=std_dev,
                    r=r,
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

    def mcc_ci_phased_poc(
        self,
        bafs,
        mat_haps,
        freqs,
        pos,
        c_hat=0.0,
        std_dev=0.1,
        r=1e-8,
        alpha=0.95,
        df=1,
    ):
        """Profile-likelihood CI for MCC using the phase-aware POC (duo) HMM.

        Arguments:
            - bafs (`np.array`): m-length array of B-allele frequencies
            - mat_haps (`np.array`): 2 x m array of 0/1 maternal haplotypes
            - freqs (`np.array`): m-length array of population allele frequencies
            - pos (`np.array`): m-length strictly increasing array of base-pair positions
            - c_hat (`float`): MLE contamination fraction (the point estimate)
            - std_dev (`float`): MLE BAF noise standard deviation (held fixed during profile)
            - r (`float`): per-base-pair recombination rate; default 1e-8
            - alpha (`float`): confidence level in (0, 1); default 0.95
            - df (`int`): degrees of freedom for chi-squared quantile; default 1

        Returns:
            - lower_CI (`float`): lower confidence bound (falls back to 0 if optimiser fails)
            - c_hat (`float`): the supplied point estimate
            - upper_CI (`float`): upper confidence bound (falls back to 0.5 if optimiser fails)

        """
        assert (c_hat >= 0) and (c_hat <= 0.5)
        assert std_dev > 0
        assert (alpha > 0) and (alpha < 1)
        ll_hat = self.loglik_mcc_phased_poc(
            bafs=bafs,
            mat_haps=mat_haps,
            freqs=freqs,
            pos=pos,
            c=c_hat,
            std_dev=std_dev,
            r=r,
        )
        wilks = lambda x: (
            2
            * (
                ll_hat
                - self.loglik_mcc_phased_poc(
                    bafs=bafs,
                    mat_haps=mat_haps,
                    freqs=freqs,
                    pos=pos,
                    c=x,
                    std_dev=std_dev,
                    r=r,
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

    # ------------------------------------------------------------------
    # Genome-wide log-likelihoods (sum across chromosomes)
    # ------------------------------------------------------------------

    def loglik_mcc_genome_poc(self, baf_list, mat_haps_list, freqs_list, c=0.0, std_dev=0.1):
        """Genome-wide log-likelihood of MCC summed across all chromosomes (POC model).

        Chromosomes are conditionally independent given ``c`` and ``std_dev``, so
        the genome-wide log-likelihood is the sum of per-chromosome likelihoods from
        :meth:`loglik_mcc_poc`.  Sex chromosomes should be excluded before calling.

        Implemented by concatenating all chromosome arrays into a single call to
        :meth:`loglik_mcc_poc`, which is equivalent to summing independent
        per-chromosome log-likelihoods but avoids repeated Python function-call
        overhead and assertion checks during optimisation.

        Arguments:
            - baf_list (`list` of `np.array`): per-chromosome BAF arrays
            - mat_haps_list (`list` of `np.array`): per-chromosome 2 x m maternal haplotype arrays
            - freqs_list (`list` of `np.array`): per-chromosome population allele frequency arrays
            - c (`float`): maternal contamination fraction in [0, 0.5]
            - std_dev (`float`): BAF noise standard deviation (> 0)

        Returns:
            - loglik (`float`): total log-likelihood summed across all chromosomes

        """
        return self.loglik_mcc_poc(
            np.concatenate(baf_list),
            np.concatenate(mat_haps_list, axis=1),
            np.concatenate(freqs_list),
            c=c,
            std_dev=std_dev,
        )

    def loglik_mcc_genome_trio(self, baf_list, mat_haps_list, pat_haps_list, c=0.0, std_dev=0.1):
        """Genome-wide log-likelihood of MCC summed across all chromosomes (trio model).

        Implemented by concatenating all chromosome arrays into a single call to
        :meth:`loglik_mcc_trio` (valid because sites are independent across
        chromosomes in the unphased model).

        Arguments:
            - baf_list (`list` of `np.array`): per-chromosome BAF arrays
            - mat_haps_list (`list` of `np.array`): per-chromosome 2 x m maternal haplotype arrays
            - pat_haps_list (`list` of `np.array`): per-chromosome 2 x m paternal haplotype arrays
            - c (`float`): maternal contamination fraction in [0, 0.5]
            - std_dev (`float`): BAF noise standard deviation (> 0)

        Returns:
            - loglik (`float`): total log-likelihood summed across all chromosomes

        """
        return self.loglik_mcc_trio(
            np.concatenate(baf_list),
            np.concatenate(mat_haps_list, axis=1),
            np.concatenate(pat_haps_list, axis=1),
            c=c,
            std_dev=std_dev,
        )

    def loglik_mcc_genome_phased_poc(
        self, baf_list, mat_haps_list, freqs_list, pos_list, c=0.0, std_dev=0.1, r=1e-8
    ):
        """Genome-wide phase-aware log-likelihood of MCC summed across all chromosomes (POC model).

        Arguments:
            - baf_list (`list` of `np.array`): per-chromosome BAF arrays
            - mat_haps_list (`list` of `np.array`): per-chromosome 2 x m maternal haplotype arrays
            - freqs_list (`list` of `np.array`): per-chromosome population allele frequency arrays
            - pos_list (`list` of `np.array`): per-chromosome strictly increasing position arrays
            - c (`float`): maternal contamination fraction in [0, 0.5]
            - std_dev (`float`): BAF noise standard deviation (> 0)
            - r (`float`): per-base-pair recombination rate; default 1e-8

        Returns:
            - loglik (`float`): total forward-algorithm log-likelihood summed across all chromosomes

        """
        return sum(
            self.loglik_mcc_phased_poc(b, m, f, pos, c=c, std_dev=std_dev, r=r)
            for b, m, f, pos in zip(baf_list, mat_haps_list, freqs_list, pos_list)
        )

    def loglik_mcc_genome_phased_trio(
        self, baf_list, mat_haps_list, pat_haps_list, pos_list, c=0.0, std_dev=0.1, r=1e-8
    ):
        """Genome-wide phase-aware log-likelihood of MCC summed across all chromosomes (trio model).

        Arguments:
            - baf_list (`list` of `np.array`): per-chromosome BAF arrays
            - mat_haps_list (`list` of `np.array`): per-chromosome 2 x m maternal haplotype arrays
            - pat_haps_list (`list` of `np.array`): per-chromosome 2 x m paternal haplotype arrays
            - pos_list (`list` of `np.array`): per-chromosome strictly increasing position arrays
            - c (`float`): maternal contamination fraction in [0, 0.5]
            - std_dev (`float`): BAF noise standard deviation (> 0)
            - r (`float`): per-base-pair recombination rate; default 1e-8

        Returns:
            - loglik (`float`): total forward-algorithm log-likelihood summed across all chromosomes

        """
        return sum(
            self.loglik_mcc_phased_trio(b, m, p, pos, c=c, std_dev=std_dev, r=r)
            for b, m, p, pos in zip(baf_list, mat_haps_list, pat_haps_list, pos_list)
        )

    # ------------------------------------------------------------------
    # Genome-wide MLE (single c estimated across all chromosomes jointly)
    # ------------------------------------------------------------------

    def est_mcc_genome_poc(self, baf_list, mat_haps_list, freqs_list, algo="Nelder-Mead", **kwargs):
        """Estimate MCC by MLE using all chromosomes jointly (POC model).

        Maximises :meth:`loglik_mcc_genome_poc` to obtain a single estimate of
        ``c`` and ``std_dev`` shared across all supplied chromosomes.  Aneuploid
        chromosomes should be excluded from ``baf_list`` before calling; use
        :meth:`est_mcc_per_chrom_poc` to identify outliers first.

        Arguments:
            - baf_list (`list` of `np.array`): per-chromosome BAF arrays
            - mat_haps_list (`list` of `np.array`): per-chromosome 2 x m maternal haplotype arrays
            - freqs_list (`list` of `np.array`): per-chromosome population allele frequency arrays
            - algo (`str`): scipy minimisation algorithm; default ``"Nelder-Mead"``
            - **kwargs: additional keyword arguments forwarded to ``scipy.optimize.minimize``

        Returns:
            - c_est (`float`): MLE of the contamination fraction
            - sigma_est (`float`): MLE of the BAF noise standard deviation

        """
        opt_res = minimize(
            lambda x: -self.loglik_mcc_genome_poc(
                baf_list, mat_haps_list, freqs_list, c=x[0], std_dev=x[1]
            ),
            x0=[0.05, 0.1],
            method=algo,
            bounds=[(0, 0.5), (1e-3, 0.3)],
            **kwargs,
        )
        return opt_res.x[0], opt_res.x[1]

    def est_mcc_genome_trio(
        self, baf_list, mat_haps_list, pat_haps_list, algo="Nelder-Mead", **kwargs
    ):
        """Estimate MCC by MLE using all chromosomes jointly (trio model).

        Maximises :meth:`loglik_mcc_genome_trio` to obtain a single estimate of
        ``c`` and ``std_dev`` shared across all supplied chromosomes.

        Arguments:
            - baf_list (`list` of `np.array`): per-chromosome BAF arrays
            - mat_haps_list (`list` of `np.array`): per-chromosome 2 x m maternal haplotype arrays
            - pat_haps_list (`list` of `np.array`): per-chromosome 2 x m paternal haplotype arrays
            - algo (`str`): scipy minimisation algorithm; default ``"Nelder-Mead"``
            - **kwargs: additional keyword arguments forwarded to ``scipy.optimize.minimize``

        Returns:
            - c_est (`float`): MLE of the contamination fraction
            - sigma_est (`float`): MLE of the BAF noise standard deviation

        """
        opt_res = minimize(
            lambda x: -self.loglik_mcc_genome_trio(
                baf_list, mat_haps_list, pat_haps_list, c=x[0], std_dev=x[1]
            ),
            x0=[0.05, 0.1],
            method=algo,
            bounds=[(0, 0.5), (1e-3, 0.3)],
            **kwargs,
        )
        return opt_res.x[0], opt_res.x[1]

    def est_mcc_genome_phased_poc(
        self, baf_list, mat_haps_list, freqs_list, pos_list, r=1e-8, algo="Nelder-Mead", **kwargs
    ):
        """Estimate MCC by MLE using all chromosomes jointly (phase-aware POC model).

        Maximises :meth:`loglik_mcc_genome_phased_poc`.

        Arguments:
            - baf_list (`list` of `np.array`): per-chromosome BAF arrays
            - mat_haps_list (`list` of `np.array`): per-chromosome 2 x m maternal haplotype arrays
            - freqs_list (`list` of `np.array`): per-chromosome population allele frequency arrays
            - pos_list (`list` of `np.array`): per-chromosome strictly increasing position arrays
            - r (`float`): per-base-pair recombination rate; default 1e-8
            - algo (`str`): scipy minimisation algorithm; default ``"Nelder-Mead"``
            - **kwargs: additional keyword arguments forwarded to ``scipy.optimize.minimize``

        Returns:
            - c_est (`float`): MLE of the contamination fraction
            - sigma_est (`float`): MLE of the BAF noise standard deviation

        """
        opt_res = minimize(
            lambda x: -self.loglik_mcc_genome_phased_poc(
                baf_list, mat_haps_list, freqs_list, pos_list, c=x[0], std_dev=x[1], r=r
            ),
            x0=[0.05, 0.1],
            method=algo,
            bounds=[(0, 0.5), (1e-3, 0.3)],
            **kwargs,
        )
        return opt_res.x[0], opt_res.x[1]

    def est_mcc_genome_phased_trio(
        self,
        baf_list,
        mat_haps_list,
        pat_haps_list,
        pos_list,
        r=1e-8,
        algo="Nelder-Mead",
        **kwargs,
    ):
        """Estimate MCC by MLE using all chromosomes jointly (phase-aware trio model).

        Maximises :meth:`loglik_mcc_genome_phased_trio`.

        Arguments:
            - baf_list (`list` of `np.array`): per-chromosome BAF arrays
            - mat_haps_list (`list` of `np.array`): per-chromosome 2 x m maternal haplotype arrays
            - pat_haps_list (`list` of `np.array`): per-chromosome 2 x m paternal haplotype arrays
            - pos_list (`list` of `np.array`): per-chromosome strictly increasing position arrays
            - r (`float`): per-base-pair recombination rate; default 1e-8
            - algo (`str`): scipy minimisation algorithm; default ``"Nelder-Mead"``
            - **kwargs: additional keyword arguments forwarded to ``scipy.optimize.minimize``

        Returns:
            - c_est (`float`): MLE of the contamination fraction
            - sigma_est (`float`): MLE of the BAF noise standard deviation

        """
        opt_res = minimize(
            lambda x: -self.loglik_mcc_genome_phased_trio(
                baf_list, mat_haps_list, pat_haps_list, pos_list, c=x[0], std_dev=x[1], r=r
            ),
            x0=[0.05, 0.1],
            method=algo,
            bounds=[(0, 0.5), (1e-3, 0.3)],
            **kwargs,
        )
        return opt_res.x[0], opt_res.x[1]

    # ------------------------------------------------------------------
    # Per-chromosome MLE (useful for flagging aneuploid chromosomes)
    # ------------------------------------------------------------------

    def est_mcc_per_chrom_poc(
        self, baf_list, mat_haps_list, freqs_list, algo="Nelder-Mead", **kwargs
    ):
        """Estimate MCC independently per chromosome (POC model).

        Runs :meth:`est_mcc_poc` separately on each chromosome and returns a
        per-chromosome estimate.  Chromosomes whose ``c`` estimate is an outlier
        relative to the genome-wide median are candidates for aneuploidy and should
        be excluded before a joint genome-wide fit with :meth:`est_mcc_genome_poc`.

        Arguments:
            - baf_list (`list` of `np.array`): per-chromosome BAF arrays
            - mat_haps_list (`list` of `np.array`): per-chromosome 2 x m maternal haplotype arrays
            - freqs_list (`list` of `np.array`): per-chromosome population allele frequency arrays
            - algo (`str`): scipy minimisation algorithm; default ``"Nelder-Mead"``
            - **kwargs: additional keyword arguments forwarded to ``scipy.optimize.minimize``

        Returns:
            - results (`list` of `tuple`): list of ``(c_hat, std_dev_hat)`` tuples, one per chromosome

        """
        return [
            self.est_mcc_poc(b, m, f, algo=algo, **kwargs)
            for b, m, f in zip(baf_list, mat_haps_list, freqs_list)
        ]

    def est_mcc_per_chrom_trio(
        self, baf_list, mat_haps_list, pat_haps_list, algo="Nelder-Mead", **kwargs
    ):
        """Estimate MCC independently per chromosome (trio model).

        Runs :meth:`est_mcc_trio` separately on each chromosome.

        Arguments:
            - baf_list (`list` of `np.array`): per-chromosome BAF arrays
            - mat_haps_list (`list` of `np.array`): per-chromosome 2 x m maternal haplotype arrays
            - pat_haps_list (`list` of `np.array`): per-chromosome 2 x m paternal haplotype arrays
            - algo (`str`): scipy minimisation algorithm; default ``"Nelder-Mead"``
            - **kwargs: additional keyword arguments forwarded to ``scipy.optimize.minimize``

        Returns:
            - results (`list` of `tuple`): list of ``(c_hat, std_dev_hat)`` tuples, one per chromosome

        """
        return [
            self.est_mcc_trio(b, m, p, algo=algo, **kwargs)
            for b, m, p in zip(baf_list, mat_haps_list, pat_haps_list)
        ]

    def est_mcc_per_chrom_phased_poc(
        self, baf_list, mat_haps_list, freqs_list, pos_list, r=1e-8, algo="Nelder-Mead", **kwargs
    ):
        """Estimate MCC independently per chromosome (phase-aware POC model).

        Runs :meth:`est_mcc_phased_poc` separately on each chromosome.

        Arguments:
            - baf_list (`list` of `np.array`): per-chromosome BAF arrays
            - mat_haps_list (`list` of `np.array`): per-chromosome 2 x m maternal haplotype arrays
            - freqs_list (`list` of `np.array`): per-chromosome population allele frequency arrays
            - pos_list (`list` of `np.array`): per-chromosome strictly increasing position arrays
            - r (`float`): per-base-pair recombination rate; default 1e-8
            - algo (`str`): scipy minimisation algorithm; default ``"Nelder-Mead"``
            - **kwargs: additional keyword arguments forwarded to ``scipy.optimize.minimize``

        Returns:
            - results (`list` of `tuple`): list of ``(c_hat, std_dev_hat)`` tuples, one per chromosome

        """
        return [
            self.est_mcc_phased_poc(b, m, f, pos, r=r, algo=algo, **kwargs)
            for b, m, f, pos in zip(baf_list, mat_haps_list, freqs_list, pos_list)
        ]

    def est_mcc_per_chrom_phased_trio(
        self,
        baf_list,
        mat_haps_list,
        pat_haps_list,
        pos_list,
        r=1e-8,
        algo="Nelder-Mead",
        **kwargs,
    ):
        """Estimate MCC independently per chromosome (phase-aware trio model).

        Runs :meth:`est_mcc_phased_trio` separately on each chromosome.

        Arguments:
            - baf_list (`list` of `np.array`): per-chromosome BAF arrays
            - mat_haps_list (`list` of `np.array`): per-chromosome 2 x m maternal haplotype arrays
            - pat_haps_list (`list` of `np.array`): per-chromosome 2 x m paternal haplotype arrays
            - pos_list (`list` of `np.array`): per-chromosome strictly increasing position arrays
            - r (`float`): per-base-pair recombination rate; default 1e-8
            - algo (`str`): scipy minimisation algorithm; default ``"Nelder-Mead"``
            - **kwargs: additional keyword arguments forwarded to ``scipy.optimize.minimize``

        Returns:
            - results (`list` of `tuple`): list of ``(c_hat, std_dev_hat)`` tuples, one per chromosome

        """
        return [
            self.est_mcc_phased_trio(b, m, p, pos, r=r, algo=algo, **kwargs)
            for b, m, p, pos in zip(baf_list, mat_haps_list, pat_haps_list, pos_list)
        ]


class MosaicEst:
    """Estimator of mosaic copy-number cell fraction from BAF imbalance.

    Models the distribution of BAF at expected heterozygous sites using a
    3-state HMM following Loh et al.:

    - State 0 (loss): signed BAF shift of ``-theta`` (one copy lost)
    - State 1 (neutral): no shift
    - State 2 (gain): signed BAF shift of ``+theta``

    The signed BAF is computed by phasing heterozygotes relative to the
    transmitted maternal haplotype, so gains and losses produce opposite signs.
    The mosaic cell fraction ``theta`` is estimated by MLE over the forward
    algorithm log-likelihood, and is converted to a cell fraction via
    :meth:`est_cf`.

    Parameters
    ----------
    mat_haps : np.ndarray
        2 x m array of 0/1 maternal haplotypes.
    pat_haps : np.ndarray
        2 x m array of 0/1 paternal haplotypes.
    bafs : np.ndarray
        m-length array of B-allele frequencies.
    pos : np.ndarray
        m-length array of base-pair positions.
    """

    def __init__(self, mat_haps, pat_haps, bafs, pos):
        """Initialize the MosaicEst object and validate input dimensions."""
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
        """Identify expected heterozygous sites and store their BAF values.

        A site is expected to be heterozygous in the embryo when one parent is
        homozygous-ref (genotype 0) and the other is homozygous-alt (genotype 2).
        Sets ``self.het_bafs``, ``self.het_idx``, and ``self.n_het``; does *not*
        phase the BAF (``signed_baf`` is set to ``False``).

        Raises:
            ValueError: if fewer than 10 expected heterozygotes are found.
        """
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
        """Identify heterozygous sites via Viterbi decoding and phase their BAF.

        Runs the disomy :class:`MetaHMM` Viterbi algorithm to call the embryo
        genotype at each site; sites called as heterozygous (dosage 1) are
        retained.  The BAF at each retained site is phased relative to the
        transmitted maternal haplotype and stored as ``self.phased_baf``
        (``self.signed_baf`` is set to ``True``).

        Arguments:
            - **kwargs: passed through to :meth:`MetaHMM.viterbi_algorithm`
              (e.g. ``pi0``, ``std_dev``, ``r``)

        Raises:
            ValueError: if fewer than 10 Viterbi-called heterozygotes are found.
        """
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
        """Phase the BAF at existing heterozygous sites using Viterbi copying path.

        Requires :meth:`baf_hets` to have been called first so that ``self.het_idx``
        is populated.  Runs the disomy :class:`MetaHMM` Viterbi algorithm and uses
        the resulting copying path to assign a sign (+1 / -1) to each heterozygous
        site based on which maternal haplotype was transmitted.  Stores the result
        in ``self.phased_baf`` and sets ``self.signed_baf = True``.

        Arguments:
            - **kwargs: passed through to :meth:`MetaHMM.viterbi_algorithm`
        """
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
        """Forward algorithm for the 3-state mosaic HMM (Loh et al. model).

        Requires :meth:`create_transition_matrix` and either :meth:`viterbi_hets`
        or :meth:`phase_hets` to have been called first.

        Arguments:
            - theta (`float`): mean absolute BAF shift parameter (≥ 0); the
              three states emit from N(-theta, σ²), N(0, σ²), and N(+theta, σ²)
              on the signed phased BAF
            - std_dev (`float`): emission noise standard deviation (> 0)

        Returns:
            - alphas (`np.array`): 3 x n log-scaled forward variable at each of the n het sites
            - scaler (`np.array`): n-length array of per-site log normalisation constants
            - loglik (`float`): total log-likelihood (sum of scalers)

        """
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
        """Viterbi decoding for the 3-state mosaic HMM.

        Arguments:
            - theta (`float`): mean absolute BAF shift parameter (≥ 0)
            - std_dev (`float`): emission noise standard deviation (≥ 0)

        Returns:
            - path (`np.array`): n-length most-likely state sequence (values in {0, 1, 2})
            - zs (`list`): emission means [-theta, 0, theta]
            - deltas (`np.array`): 3 x n Viterbi score array
            - psi (`np.array`): 3 x n back-pointer array

        """
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
        """Likelihood-ratio test statistic for theta > 0 vs theta = 0.

        Computes -2 × (loglik(theta=0) - loglik(theta_MLE)).  Under the null
        (no mosaicism) this is approximately chi-squared with 1 df.

        Arguments:
            - std_dev (`float`): BAF noise standard deviation used in both models

        Returns:
            - lrt (`float`): LRT statistic, or ``nan`` if :meth:`est_mle_theta` fails

        """
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
        """Estimate the MLE of the BAF shift parameter theta.

        Maximises the forward-algorithm log-likelihood over theta in [0, 0.5]
        using Powell's method.  Stores the result in ``self.mle_theta`` (set to
        ``nan`` if optimisation fails).

        Arguments:
            - std_dev (`float`): BAF noise standard deviation (held fixed during optimisation)
        """
        try:
            f = lambda x: -self.forward_algo_mix(theta=x, std_dev=std_dev)[2]
            opt_res = minimize(
                f, x0=[0.1], method="Powell", tol=1e-4, bounds=[(0.0, 0.5)]
            )
            self.mle_theta = opt_res.x[0]
        except ValueError:
            self.mle_theta = np.nan

    def ci_mle_theta(self, std_dev=0.1, h=1e-6):
        """95% confidence interval for theta using observed Fisher information.

        Approximates the Fisher information via a symmetric finite-difference second
        derivative of the log-likelihood at the MLE, then applies the standard
        ``±1.96 / sqrt(n × I)`` formula.  Requires :meth:`est_mle_theta` to have
        been called first.

        Arguments:
            - std_dev (`float`): BAF noise standard deviation (held fixed)
            - h (`float`): finite-difference step size; default 1e-6

        Returns:
            - ci_mle_theta (`list`): ``[lower_95, mle_theta, upper_95]``, clamped to [0, 0.5];
              entries are ``nan`` if the Fisher information computation fails

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
        """Convert BAF shift theta to a mosaic cell fraction.

        Uses the relationship between BAF shift and copy number to solve for the
        fraction of cells carrying the mosaic event.  For a gain the copy number
        is in (2, 3]; for a loss it is in [1, 2).

        Arguments:
            - theta (`float`): estimated BAF shift parameter in [0, 0.5]
            - gain (`bool`): if ``True`` (default) interpret theta as a gain (copy
              number > 2); if ``False`` interpret as a loss (copy number < 2)

        Returns:
            - cell_fraction (`float`): estimated fraction of cells carrying the mosaic event
              in [0, 1]; returns 1.0 if theta exceeds the gain boundary

        """
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
        """Estimate per-embryo BAF noise parameters under a disomy assumption.

        Runs :meth:`MetaHMM.est_sigma_pi0` (disomy model) on each embryo BAF
        array and stores the results in ``self.embryo_pi0s`` and
        ``self.embryo_sigmas``.  Must be called before :meth:`viterbi_phase_correct`
        or :meth:`flag_parental_genotype_errors`.

        Arguments:
            - **kwargs: forwarded to :meth:`MetaHMM.est_sigma_pi0`
              (e.g. ``pi0_bounds``, ``sigma_bounds``, ``algo``)
        """
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
        """Correct haplotype phase using majority-rule consensus across copying paths.

        At each inter-site transition the number of siblings showing a haplotype
        switch is counted.  If a majority of siblings switch, the haplotype
        orientations prior to that site are flipped for both strands.

        Arguments:
            - haps (`np.array`): 2 x m array of 0/1 haplotypes to be corrected
            - paths (`np.array`): n_sib x m boolean array where entry ``[j, i]`` is
              1 if sibling j copies haplotype 0 at site i (output of Viterbi decoding)

        Returns:
            - n_mis (`np.array`): (m-1)-length array of per-interval mismatch counts
            - fixed_haps (`np.array`): 2 x m phase-corrected haplotype array

        """
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
    """Crossover detection via allele-transmission comparison across sibling embryos.

    Implements the simplified crossover-detection approach of Coop et al. (2007),
    extended with :class:`QuadHMM`-based interval refinement.  For each pair of
    (template, non-template) sibling embryos, the log-likelihood ratio of copying
    the *same* vs *different* parental allele is computed at informative markers
    (one parent heterozygous, the other homozygous).  Crossover locations are
    identified as positions where the sign of this ratio changes consistently
    across sibling pairs.

    Inherits the embryo storage, noise-parameter estimation, and phase-correction
    infrastructure from :class:`PhaseCorrect`.
    """

    def __init__(self, **kwargs):
        """Initialise RecombEst inheriting from PhaseCorrect.

        Arguments:
            - **kwargs: forwarded to :meth:`PhaseCorrect.__init__`
              (``mat_haps``, ``pat_haps``, ``pos``)
        """
        super(RecombEst, self).__init__(**kwargs)

    def informative_markers(self, maternal=True):
        """Return a boolean mask of allele-transmission–informative sites.

        A site is informative for the maternal haplotype when the mother is
        heterozygous (genotype 1) and the father is homozygous (genotype 0 or 2),
        so the transmitted maternal allele is unambiguously determined by the
        embryo BAF.  The paternal case is the mirror image.

        Arguments:
            - maternal (`bool`): if ``True`` (default) return sites informative for
              maternal crossovers; if ``False`` return sites informative for paternal crossovers

        Returns:
            - info_idx (`np.array`): m-length boolean mask; ``True`` at informative sites

        """
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
        """Identify sign-change intervals in transmission indicator matrix Z.

        Missing or uncertain sites (Z = 0) are imputed by forward-propagating the
        last non-zero sign before applying :meth:`refine_recomb_events` to each row.

        Arguments:
            - Z (`np.array`): n_sib x n_info matrix of transmission indicators
              (+1 same allele, -1 different allele, 0 uncertain)
            - npad (`int`): half-width of switch cluster used by :meth:`refine_recomb_events`

        Returns:
            - isolated_switches (`list` of `list`): per-sibling lists of refined switch indices

        """
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
        """Refine a crossover interval using QuadHMM on the bracketing SNP window.

        Uses :class:`QuadHMM` pairwise Viterbi paths within the interval
        [start, end] to narrow the crossover location to the tightest flanking
        heterozygous-site pair that is consistent across all sibling pairs.

        Arguments:
            - template_embryo (`int`): index of the reference embryo; default 0
            - maternal (`bool`): if ``True`` refine maternal crossovers; default ``True``
            - start (`int`): left SNP index (inclusive) of the coarse interval
            - end (`int`): right SNP index (inclusive) of the coarse interval

        Returns:
            - p1 (`float`): refined left boundary position in base pairs
            - p2 (`float`): refined right boundary position in base pairs

        """
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
        """Localise potential crossovers to the tightest flanking informative-marker interval.

        For each entry in ``potential_switches`` calls :meth:`second_refine_recomb`
        on the flanking informative-marker pair to obtain a refined base-pair interval.

        Arguments:
            - potential_switches (`list`): list of informative-marker indices at which a
              crossover is suspected (output of :meth:`isolate_recomb_events`)
            - template_embryo (`int`): index of the reference embryo; default 0
            - maternal (`bool`): if ``True`` refine maternal crossovers; default ``True``

        Returns:
            - rec_locations (`list` of `tuple`): list of ``(p1, p2)`` base-pair intervals
              bounding each crossover; empty list if ``potential_switches`` is empty

        """
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
        """Full crossover detection pipeline: isolation, majority-rule filtering, and refinement.

        Calls :meth:`isolate_recomb_events` to identify candidate switch positions,
        applies majority-rule filtering, then passes surviving candidates to
        :meth:`finalize_recomb_events` for QuadHMM-based interval refinement.

        Requires :meth:`add_baf`, :meth:`est_sigma_pi0s` to have been called first.

        Arguments:
            - template_embryo (`int`): index of the reference embryo; default 0
            - ll_thresh (`float`): log-likelihood ratio threshold separating "same" from
              "different" allele transmission; default 0 (LLR sign flip)
            - maternal (`bool`): if ``True`` detect maternal crossovers; default ``True``
            - npad (`int`): half-width of the switch-cluster filter; default 5

        Returns:
            - Z (`np.array`): n_sib x n_info transmission-indicator matrix
            - llr_z (`np.array`): n_sib x n_info raw LLR matrix
            - potential_switches_filt (`np.array`): majority-vote filtered switch indices at informative markers
            - switch_cnts_filt (`np.array`): support counts for each filtered switch

        .. note::
            To obtain refined base-pair intervals pass ``potential_switches_filt``
            to :meth:`finalize_recomb_events`.

        """
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
