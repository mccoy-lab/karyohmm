"""Simulation utilities for unit-testing suite."""

import numpy as np
from scipy.stats import (
    beta,
    betabinom,
    binom,
    norm,
    lognorm,
    poisson,
    randint,
    rv_histogram,
    truncnorm,
    uniform,
)


class PGTSimBase:
    """Base-class for simulation of PGT-A data."""

    def __init__(self):
        """Initialize the PGT-A Simulator."""
        # NOTE: the initial SDs are derived from the PENNCNV source code - but can be scaled effectively
        self.lrr_mu = {0: -3.527211, 1: np.log2(0.5), 2: 0.0, 3: np.log2(1.5)}
        self.lrr_sd = {0: 1.329152, 1: 0.284338, 2: 0.159645, 3: 0.209089}

    def draw_parental_genotypes(self, afs=None, m=100, seed=42):
        """Draw parental genotypes from a beta distribution.

        Args:
            afs (`np.array`): realized allele frequencies.
            m (`int`): number of variants to simulate.
            seed (`int`): random number seed.
        Output:
            maternal_haps (`np.array`): maternal haplotypes
            paternal_haps (`np.array`): paternal haplotypes
            ps (`np.array`): allele frequency of variants

        """
        assert m > 0
        assert seed > 0
        np.random.seed(seed)
        if afs is None:
            # Draw from a uniform distribution ...
            ps = beta.rvs(1.0, 1.0, size=m)
        else:
            # This is the case where we actually have an AFS ...
            assert afs.size > 10
            rv = rv_histogram(
                np.histogram(afs, bins=np.min([100, afs.size / 20]).astype(np.int32))
            )
            ps = rv.rvs(size=m)
        # Simulate diploid parental haplotypes
        mat_h1 = binom.rvs(1, ps)
        mat_h2 = binom.rvs(1, ps)
        pat_h1 = binom.rvs(1, ps)
        pat_h2 = binom.rvs(1, ps)
        # NOTE: assuming diploid here ...
        return np.vstack([mat_h1, mat_h2]), np.vstack([pat_h1, pat_h2]), ps

    def create_genotyping_errors(self, haps, err_rate=1e-3, seed=42):
        """Create genotyping errors in haplotypes sampled.

        Args:
            haps (`np.array`): haplotype array from either maternal or paternal
            err_rate (`float`): site-level error rate in genotyping
            seed (`int`): random seed

        Output:
            err_idx (`np.array`): m-length array indicating simulated errors
            err_haps (`np.array`): haplotypes which reflect newly simulated errors

        """
        assert haps.ndim == 2
        assert haps.shape[0] == 2
        assert (err_rate >= 0) and (err_rate <= 0.5)
        assert seed > 0
        m = haps.shape[1]
        err_idx = np.random.uniform(size=m) <= err_rate
        err_haps = haps.copy()
        for j in np.where(err_idx)[0]:
            if np.random.uniform() <= 0.5:
                err_haps[0, j] = np.abs(haps[0, j] - 1)
            else:
                err_haps[1, j] = np.abs(haps[1, j] - 1)
        return err_idx, err_haps

    def create_switch_errors_help(self, haps, err_rate=1e-3, seed=42):
        """Revised method to create switch errors.

        Args:
            haps (`np.array`): 2 x M numpy array of haplotypes for bi-allelic variants.
            err_rate (`float`): switch error rate between heterozygotes.
            seed (`int`): random number seed.

        Output:
            haps_prime (`np.array`): switched haplotypes.
            switches (`np.array`): snp-indices of haplotype switches.

        """
        np.random.seed(seed)
        m = haps.shape[1]
        geno = haps.sum(axis=0)
        n_hets = np.sum(geno == 1)
        us = np.random.uniform(size=n_hets)
        haps_prime = np.zeros(shape=haps.shape, dtype=int)
        switches = []
        i0, i1 = 0, 1
        j = 0
        for i in range(m):
            # We only create switches between heterozygotes ...
            if geno[i] == 1:
                if us[j] < err_rate:
                    i0 = 1 - i0
                    i1 = 1 - i1
                    switches.append(i)
                j += 1
            haps_prime[0, i] = haps[i0, i]
            haps_prime[1, i] = haps[i1, i]
        return haps_prime, np.array(switches)

    def create_switch_errors(self, mat_haps, pat_haps, err_rate=1e-3, seed=42):
        """Create switch errors to evaluate impact of poor phasing.

        Args:
            mat_haps (`np.array`): 2 x M numpy array for maternal haplotypes.
            pat_haps (`np.array`): 2 x M numpy array for paternal haplotypes.
            err_rate (`float`): switch error rate between heterozygotes.
            seed (`int`): random number seed.

        Output:
            mat_haps_prime (`np.array`): switched maternal haplotypes.
            pat_haps_prime (`np.array`): switched paternal haplotypes.
            m_switches (`np.array`): snp-indices of maternal haplotype switches.
            p_switches (`np.array`): snp-indices of maternal haplotype switches.

        """
        assert mat_haps.size == pat_haps.size
        assert mat_haps.ndim == 2
        assert mat_haps.ndim == 2
        assert (err_rate >= 0) and (err_rate < 1)
        assert seed > 0
        mat_haps_prime, m_switch = self.create_switch_errors_help(
            mat_haps, err_rate=err_rate, seed=seed
        )
        pat_haps_prime, p_switch = self.create_switch_errors_help(
            pat_haps, err_rate=err_rate, seed=seed
        )
        return mat_haps_prime, pat_haps_prime, m_switch, p_switch

    def sim_haplotype_paths(
        self, mat_haps, pat_haps, pos, ploidy=2, rec_rate=1e-8, mat_skew=0.5, seed=42
    ):
        """Simulate copying paths through the maternal and paternal haplotypes.

        Args:
            mat_haps (`np.array`): 2 x M numpy array for maternal haplotypes.
            pat_haps (`np.array`): 2 x M numpy array for paternal haplotypes.
            pos (`np.array`): position of individual variants.
            ploidy (`int`): integer of number of chromosomes being simulated.
            rec_rate (`float`): uniform recombination rate per basepair.
            mat_skew (`float`): probability of maternal-origin aneuploidy.
            seed (`int`): random number seed.

        Output:
            zs_maternal (`np.array`): copying path through maternal haplotypes.
            zs_paternal (`np.array`): copying path through paternal haplotypes.
            mat_real_hap (`np.array`): simulated embryo maternal haplotype(s).
            pat_real_hap (`np.array`): simulated embryo paternal haplotype(s).
            aploid (`str`): indicator of aneuploidy status.

        """
        assert (ploidy <= 3) & (ploidy >= 0)
        assert (mat_skew >= 0) and (mat_skew <= 1.0)
        assert mat_haps.size == pat_haps.size
        assert (pos.ndim == 1) and (pos.size == mat_haps.shape[1])
        assert np.all(np.diff(pos) > 0)
        np.random.seed(seed)
        m = mat_haps.shape[1]
        # Simulating the hidden variables ...
        zs_maternal = np.zeros(m, dtype=np.uint16)
        zs_paternal = np.zeros(m, dtype=np.uint16)
        zs0_maternal = np.zeros(m, dtype=np.uint16)
        zs1_maternal = np.zeros(m, dtype=np.uint16)
        zs0_paternal = np.zeros(m, dtype=np.uint16)
        zs1_paternal = np.zeros(m, dtype=np.uint16)
        if ploidy == 0:
            # Drawing a nullisomy ...
            mat_real_hap = np.zeros(m, dtype=np.uint16)
            pat_real_hap = np.zeros(m, dtype=np.uint16)
            aploid = "0"
        elif ploidy == 1:
            # Drawing a maternal or paternal monosomy
            pat = binom.rvs(1, mat_skew)
            if pat:
                # We have maternal monosomy (loss of hromosome)
                zs_maternal = None
                zs_paternal[0] = binom.rvs(1, 0.5)
                for i in range(1, m):
                    d = pos[i] - pos[i - 1]
                    zs_paternal[i] = (
                        1 - zs_paternal[i - 1]
                        if uniform.rvs() <= (1 - np.exp(-rec_rate * d))
                        else zs_paternal[i - 1]
                    )
                pat_real_hap = np.array(
                    [pat_haps[i, p] for p, i in enumerate(zs_paternal)]
                )
                mat_real_hap = np.zeros(pat_real_hap.size)
                aploid = "1p"
            else:
                # We have a paternal monosomy (loss of paternal chrom)
                zs_paternal = None
                zs_maternal[0] = binom.rvs(1, 0.5)
                for i in range(1, m):
                    d = pos[i] - pos[i - 1]
                    zs_maternal[i] = (
                        1 - zs_maternal[i - 1]
                        if uniform.rvs() <= (1 - np.exp(-rec_rate * d))
                        else zs_maternal[i - 1]
                    )
                mat_real_hap = np.array(
                    [mat_haps[i, p] for p, i in enumerate(zs_maternal)]
                )
                pat_real_hap = np.zeros(mat_real_hap.size)
                aploid = "1m"
        elif ploidy == 3:
            # Drawing a maternal or paternal trisomy
            pat = binom.rvs(1, 1.0 - mat_skew)
            if pat:
                # Simulate a paternal trisomy
                zs_maternal[0] = binom.rvs(0, 0.5)
                zs0_paternal[0] = binom.rvs(0, 0.5)
                zs1_paternal[0] = binom.rvs(0, 0.5)

                for i in range(1, m):
                    d = pos[i] - pos[i - 1]
                    zs_maternal[i] = (
                        1 - zs_maternal[i - 1]
                        if uniform.rvs() <= (1 - np.exp(-rec_rate * d))
                        else zs_maternal[i - 1]
                    )
                    zs0_paternal[i] = (
                        1 - zs0_paternal[i - 1]
                        if uniform.rvs() <= (1 - np.exp(-rec_rate * d))
                        else zs0_paternal[i - 1]
                    )
                    zs1_paternal[i] = (
                        1 - zs1_paternal[i - 1]
                        if uniform.rvs() <= (1 - np.exp(-rec_rate * d))
                        else zs1_paternal[i - 1]
                    )
                # Simulate a duplicate configuration of the paternal alleles ...
                zs_paternal = np.vstack([zs0_paternal, zs1_paternal])
                mat_real_hap = np.array(
                    [mat_haps[i, p] for p, i in enumerate(zs_maternal)]
                )
                pat_real_hap = np.array(
                    [
                        pat_haps[i, p] + pat_haps[j, p]
                        for p, (i, j) in enumerate(zip(zs0_paternal, zs1_paternal))
                    ]
                )
                aploid = "3p"
            else:
                # Simulate a maternal trisomy
                zs_paternal[0] = binom.rvs(0, 0.5)
                zs0_maternal[0] = binom.rvs(0, 0.5)
                zs1_maternal[0] = binom.rvs(0, 0.5)
                for i in range(1, m):
                    d = pos[i] - pos[i - 1]
                    zs_paternal[i] = (
                        1 - zs_paternal[i - 1]
                        if uniform.rvs() <= (1 - np.exp(-rec_rate * d))
                        else zs_paternal[i - 1]
                    )
                    zs0_maternal[i] = (
                        1 - zs0_maternal[i - 1]
                        if uniform.rvs() <= (1 - np.exp(-rec_rate * d))
                        else zs0_maternal[i - 1]
                    )
                    zs1_maternal[i] = (
                        1 - zs1_maternal[i - 1]
                        if uniform.rvs() <= (1 - np.exp(-rec_rate * d))
                        else zs1_maternal[i - 1]
                    )
                # Simulate a duplicate configuration of the paternal alleles ...
                zs_maternal = np.vstack([zs0_maternal, zs1_maternal])
                mat_real_hap = np.array(
                    [
                        mat_haps[i, p] + mat_haps[j, p]
                        for p, (i, j) in enumerate(zip(zs0_maternal, zs1_maternal))
                    ]
                )
                pat_real_hap = np.array(
                    [pat_haps[i, p] for p, i in enumerate(zs_paternal)]
                )
                aploid = "3m"
        elif ploidy == 2:
            # A typical euploid sample ...
            zs_maternal[0] = binom.rvs(1, 0.5)
            zs_paternal[0] = binom.rvs(1, 0.5)
            for i in range(1, m):
                # We switch with a specific probability
                d = pos[i] - pos[i - 1]
                zs_maternal[i] = (
                    1 - zs_maternal[i - 1]
                    if uniform.rvs() <= (1 - np.exp(-rec_rate * d))
                    else zs_maternal[i - 1]
                )
                zs_paternal[i] = (
                    1 - zs_paternal[i - 1]
                    if uniform.rvs() <= (1 - np.exp(-rec_rate * d))
                    else zs_paternal[i - 1]
                )
            mat_real_hap = np.array([mat_haps[i, p] for p, i in enumerate(zs_maternal)])
            pat_real_hap = np.array([pat_haps[i, p] for p, i in enumerate(zs_paternal)])
            aploid = "2"
        else:
            raise ValueError(f"{ploidy} should be between 0 and 3!")
        return zs_maternal, zs_paternal, mat_real_hap, pat_real_hap, aploid

    def sim_haplotype_paths_upd(
        self,
        mat_haps,
        pat_haps,
        pos,
        ploidy=2,
        aploid="2p0",
        rec_rate=1e-8,
        mat_skew=0.5,
        seed=42,
    ):
        """Extension to simulate haplotype-copying paths under uniparental disomy."""
        assert mat_haps.size == pat_haps.size
        assert (pos.ndim == 1) and (pos.size == mat_haps.shape[1])
        assert np.all(np.diff(pos) > 0)
        assert aploid in ["2p0", "2p1", "2m0", "2m1"]
        if ploidy != 2:
            raise ValueError(f"{ploidy} should be 2 when simulating a UPD event!")
        np.random.seed(seed)
        m = mat_haps.shape[1]
        # Simulating the hidden variables ...
        zs_maternal = np.zeros(m, dtype=np.uint16)
        zs_paternal = np.zeros(m, dtype=np.uint16)
        zs0_maternal = np.zeros(m, dtype=np.uint16)
        zs1_maternal = np.zeros(m, dtype=np.uint16)
        zs0_paternal = np.zeros(m, dtype=np.uint16)
        zs1_paternal = np.zeros(m, dtype=np.uint16)
        if aploid == "2p0":
            # Case: paternal isodisomy
            zs_paternal[0] = binom.rvs(0, 0.5)
            for i in range(1, m):
                # We switch with a specific probability
                d = pos[i] - pos[i - 1]
                zs_paternal[i] = (
                    1 - zs_paternal[i - 1]
                    if uniform.rvs() <= (1 - np.exp(-rec_rate * d))
                    else zs_paternal[i - 1]
                )
            pat_real_hap = np.array(
                [
                    pat_haps[i, p] + pat_haps[j, p]
                    for p, (i, j) in enumerate(zip(zs_paternal, zs_paternal))
                ]
            )
            mat_real_hap = np.zeros(m, dtype=np.uint16)
        elif aploid == "2p1":
            # Case: paternal heterodisomy
            zs0_paternal[0] = binom.rvs(0, 0.5)
            zs1_paternal[0] = binom.rvs(0, 0.5)
            # We switch with a specific probability
            for i in range(1, m):
                d = pos[i] - pos[i - 1]
                zs0_paternal[i] = (
                    1 - zs0_paternal[i - 1]
                    if uniform.rvs() <= (1 - np.exp(-rec_rate * d))
                    else zs0_paternal[i - 1]
                )
                zs1_paternal[i] = (
                    1 - zs1_paternal[i - 1]
                    if uniform.rvs() <= (1 - np.exp(-rec_rate * d))
                    else zs1_paternal[i - 1]
                )
            zs_paternal = np.vstack([zs0_paternal, zs1_paternal])
            pat_real_hap = np.array(
                [
                    pat_haps[i, p] + pat_haps[j, p]
                    for p, (i, j) in enumerate(zip(zs0_paternal, zs1_paternal))
                ]
            )
            mat_real_hap = np.zeros(m, dtype=np.uint16)
        elif aploid == "2m0":
            # Case: maternal isodisomy
            zs_maternal[0] = binom.rvs(0, 0.5)
            for i in range(1, m):
                # We switch with a specific probability
                d = pos[i] - pos[i - 1]
                zs_maternal[i] = (
                    1 - zs_maternal[i - 1]
                    if uniform.rvs() <= (1 - np.exp(-rec_rate * d))
                    else zs_maternal[i - 1]
                )
            mat_real_hap = np.array(
                [
                    mat_haps[i, p] + mat_haps[j, p]
                    for p, (i, j) in enumerate(zip(zs_maternal, zs_maternal))
                ]
            )
            pat_real_hap = np.zeros(m, dtype=np.uint16)
        else:
            # Case: maternal heterodisomy
            zs0_maternal[0] = binom.rvs(0, 0.5)
            zs1_maternal[0] = binom.rvs(0, 0.5)
            # We switch with a specific probability
            for i in range(1, m):
                d = pos[i] - pos[i - 1]
                zs0_maternal[i] = (
                    1 - zs0_paternal[i - 1]
                    if uniform.rvs() <= (1 - np.exp(-rec_rate * d))
                    else zs0_maternal[i - 1]
                )
                zs1_paternal[i] = (
                    1 - zs1_maternal[i - 1]
                    if uniform.rvs() <= (1 - np.exp(-rec_rate * d))
                    else zs1_maternal[i - 1]
                )
            zs_maternal = np.vstack([zs0_maternal, zs1_maternal])
            mat_real_hap = np.array(
                [
                    pat_haps[i, p] + pat_haps[j, p]
                    for p, (i, j) in enumerate(zip(zs0_maternal, zs1_maternal))
                ]
            )
            pat_real_hap = np.zeros(m, dtype=np.uint16)
            aploid = "2m1"
        return zs_maternal, zs_paternal, mat_real_hap, pat_real_hap, aploid

    def sim_b_allele_freq(
        self, mat_hap, pat_hap, ploidy=2, std_dev=0.2, mix_prop=0.3, seed=42
    ):
        """Simulate of B-allele frequency."""
        np.random.seed(seed)
        assert (ploidy <= 3) & (ploidy >= 0)
        assert mat_hap.size == pat_hap.size
        true_geno = mat_hap + pat_hap
        baf = np.zeros(true_geno.size)
        for i in range(baf.size):
            if ploidy == 0:
                # I think that this might have to change ...
                a, b = (0 - 0.5) / std_dev, (1 - 0.5) / std_dev
                baf[i] = truncnorm.rvs(a, b, loc=0.5, scale=std_dev)
            else:
                mu_i = true_geno[i] / ploidy
                a, b = (0 - mu_i) / std_dev, (1 - mu_i) / std_dev
                if mu_i == 0:
                    baf[i] = (
                        0.0
                        if uniform.rvs() < mix_prop
                        else truncnorm.rvs(a, b, loc=mu_i, scale=std_dev)
                    )
                elif mu_i == 1:
                    baf[i] = (
                        1.0
                        if uniform.rvs() < mix_prop
                        else truncnorm.rvs(a, b, loc=mu_i, scale=std_dev)
                    )
                else:
                    baf[i] = truncnorm.rvs(a, b, loc=mu_i, scale=std_dev)
        return true_geno, baf

    def sim_logR_ratio(self, mat_hap, pat_hap, ploidy=2, alpha=0.5, seed=42):
        """Simulate logR-ratio conditional on ploidy.

        Alpha is the degree to which the variance is increased for the LRR.
        """
        assert seed > 0
        assert ploidy in [0, 1, 2, 3]
        assert alpha > 0
        assert mat_hap.size == pat_hap.size
        np.random.seed(seed)
        m = mat_hap.size
        # Actually simulate per-variant noise in LRR and varying
        alphas = lognorm.rvs(s=alpha, scale=1.0, size=m)
        sigmas = self.lrr_sd[ploidy] * np.sqrt(alphas)
        lrr = norm.rvs(self.lrr_mu[ploidy], scale=sigmas, size=m)
        return lrr, sigmas

    def sim_read_counts(
        self, mat_hap, pat_hap, ploidy, coverage=1.0, a=10.0, b=10.0, eps=1e-2, seed=42
    ):
        """Simulate read counts for embryo biopsies.

        Args:
            mat_hap (`np.array`): 1 x M numpy array for maternal haplotype.
            pat_hap (`np.array`): 1 x M numpy array for paternal haplotype.
            pos (`np.array`): position of individual variants.
            ploidy (`int`): integer of number of chromosomes being simulated.
            coverage (`float`): mean coverage for a standard diploid individual.
            a (`float`): shape parameter for beta-binomial.
            b (`float`): scale parameter for beta-binomial.
            eps (`float`): small noise parameter.
            seed (`int`): random number seed.

        Output:
            true_geno (`np.array`): the true genotype of the embryo.
            alt_reads (`np.array`): the number of reads with alt. allele.
            ref_reads (`np.array`): the number of reads with ref. allele.

        """
        assert seed > 0
        assert coverage > 0.0
        assert mat_hap.size == pat_hap.size
        assert (a > 0) & (b > 0)
        assert ploidy in [0, 1, 2, 3]
        assert (0 < eps) & (eps <= 0.5)
        np.random.seed(seed)
        true_geno = mat_hap + pat_hap
        alt_read_cnt, ref_read_cnt = (
            np.zeros(true_geno.size, dtype=np.uint16),
            np.zeros(true_geno.size, dtype=np.uint16),
        )
        for i, g in enumerate(true_geno):
            # 1. Simulate the total number of reads at the site
            tot_reads = poisson.rvs(mu=coverage * ((ploidy + eps) / 2.0))
            # 2. Simulate the total allelic balance as from a beta-binomial random-variable
            if tot_reads > 0:
                p = (g + eps) / ploidy if ploidy > 0 else eps
                p = (p - 2 * eps) if p > 1 else p
                assert (p > 0) & (p < 1)
                alt_read_cnt[i] = betabinom.rvs(
                    n=tot_reads, a=p * (a + b), b=(1.0 - p) * (a + b)
                )
                ref_read_cnt[i] = tot_reads - alt_read_cnt[i]
        return true_geno, alt_read_cnt, ref_read_cnt

    def sim_joint_het(self, switch=False, nsibs=1, meta_seed=42, **kwargs):
        """Simulate a joint heterozygote and potential switch error."""
        assert nsibs > 0
        assert meta_seed > 0
        true_haps1 = np.array([[1, 1], [0, 0]])
        if switch:
            haps1 = np.array([[1, 0], [0, 1]])
        else:
            haps1 = true_haps1
        true_haps2 = np.array([[0, 0], [0, 0]])
        haps2 = true_haps2
        bafs = []
        genos = []
        for i in range(nsibs):
            x = binom.rvs(1, 0.5)
            geno, b = self.sim_b_allele_freq(
                true_haps1[x, :], haps2[0, :], seed=(i + 1) + meta_seed, **kwargs
            )
            bafs.append(b)
            genos.append(geno)
        return true_haps1, true_haps2, haps1, haps2, bafs, genos

    def sim_haplotype_ref_panel(self, haps, pos, panel_size=10, seed=42, **kwargs):
        """Simulate a haplotype reference panel from haplotypes."""
        assert panel_size > 0
        assert haps.ndim == 2
        assert haps.shape[1] > 0
        assert haps.shape[1] == pos.size
        ref_panel = np.zeros(shape=(panel_size, pos.size))
        for i in range(panel_size):
            # NOTE: this just shuffles up the current parental haplotypes ...
            _, _, sim_hap1, _, _ = self.sim_haplotype_paths(
                mat_haps=haps, pat_haps=haps, pos=pos, ploidy=2, **kwargs
            )
            ref_panel[i, :] = sim_hap1
        return ref_panel


class PGTSim(PGTSimBase):
    """Simulation of meiotic-origin whole chromosome ploidy changes."""

    def __init__(self):
        """Initialize the full Aneuploidy Simulator.

        Returns: a PGTSim object

        """
        super().__init__()

    def full_ploidy_sim(
        self,
        afs=None,
        reads=False,
        ploidy=2,
        m=10000,
        upd=None,
        length=1e7,
        rec_rate=1e-4,
        mat_skew=0.5,
        std_dev=0.15,
        mix_prop=0.3,
        alpha=1.0,
        coverage=5.0,
        a=10.0,
        b=10.0,
        switch_err_rate=1e-2,
        err_rate=1e-3,
        seed=42,
    ):
        """Simulate a single embryo with a given ploidy status."""
        np.random.seed(seed)
        mat_haps, pat_haps, ps = self.draw_parental_genotypes(afs=afs, m=m, seed=seed)
        pos = np.sort(np.random.uniform(high=length, size=m))
        if (ploidy == 2) and (upd is not None):
            assert upd in ["2p0", "2p1", "2m0", "2m1"]
            zs_maternal, zs_paternal, mat_hap1, pat_hap1, aploid = (
                self.sim_haplotype_paths_upd(
                    mat_haps,
                    pat_haps,
                    pos,
                    ploidy=ploidy,
                    aploid=upd,
                    rec_rate=rec_rate,
                    seed=seed,
                )
            )
        else:
            zs_maternal, zs_paternal, mat_hap1, pat_hap1, aploid = (
                self.sim_haplotype_paths(
                    mat_haps,
                    pat_haps,
                    pos,
                    ploidy=ploidy,
                    mat_skew=mat_skew,
                    rec_rate=rec_rate,
                    seed=seed,
                )
            )
        (
            mat_haps_prime,
            pat_haps_prime,
            mat_switch,
            pat_switch,
        ) = self.create_switch_errors(
            mat_haps, pat_haps, err_rate=switch_err_rate, seed=seed
        )
        mat_err_idx, mat_haps_err = self.create_genotyping_errors(
            mat_haps_prime, err_rate=err_rate, seed=seed
        )
        pat_err_idx, pat_haps_err = self.create_genotyping_errors(
            pat_haps_prime, err_rate=err_rate, seed=seed
        )

        if reads:
            geno, alt_reads, ref_reads = self.sim_read_counts(
                mat_hap1,
                pat_hap1,
                ploidy=ploidy,
                coverage=coverage,
                a=a,
                b=b,
                seed=seed,
            )
            baf = None
            lrr = None
            sigmas = None
            assert geno.size == m
            assert alt_reads.size == m
            assert ref_reads.size == m
        else:
            geno, baf = self.sim_b_allele_freq(
                mat_hap1,
                pat_hap1,
                ploidy=ploidy,
                std_dev=std_dev,
                mix_prop=mix_prop,
                seed=seed,
            )
            lrr, sigmas = self.sim_logR_ratio(
                mat_hap1, pat_hap1, ploidy=ploidy, alpha=alpha, seed=seed
            )
            alt_reads = None
            ref_reads = None
            assert geno.size == m
            assert baf.size == m
        assert pos.size == m
        res_table = {
            "mat_haps": mat_haps,
            "pat_haps": pat_haps,
            "mat_haps_prime": mat_haps_prime,
            "pat_haps_prime": pat_haps_prime,
            "mat_haps_err": mat_haps_err,
            "pat_haps_err": pat_haps_err,
            "mat_switch": mat_switch,
            "pat_switch": pat_switch,
            "mat_err": mat_err_idx,
            "pat_err": pat_err_idx,
            "zs_maternal": zs_maternal,
            "zs_paternal": zs_paternal,
            "geno_embryo": geno,
            "baf": baf,
            "lrr": lrr,
            "sigmas": sigmas,
            "alt_reads": alt_reads,
            "ref_reads": ref_reads,
            "pos": pos,
            "af": ps,
            "m": m,
            "length": length,
            "aploid": aploid,
            "ploidy": ploidy,
            "rec_rate": rec_rate,
            "std_dev": std_dev,
            "mix_prop": mix_prop,
            "alpha": alpha,
            "seed": seed,
        }
        return res_table

    def sibling_euploid_sim(
        self,
        afs=None,
        reads=False,
        ploidy=2,
        m=10000,
        length=1e7,
        nsibs=3,
        rec_rate=1e-8,
        std_dev=0.15,
        mix_prop=0.3,
        alpha=1.0,
        coverage=5.0,
        a=10.0,
        b=10.0,
        switch_err_rate=1e-2,
        err_rate=1e-3,
        seed=42,
    ):
        """Simulate euploid sibling embryos."""
        assert ploidy == 2
        assert m > 0
        assert seed > 0
        assert nsibs > 0
        assert length > 0
        np.random.seed(seed)

        res_table = {}
        mat_haps, pat_haps, ps = self.draw_parental_genotypes(afs=afs, m=m, seed=seed)
        (
            mat_haps_prime,
            pat_haps_prime,
            mat_switch,
            pat_switch,
        ) = self.create_switch_errors(
            mat_haps, pat_haps, err_rate=switch_err_rate, seed=seed
        )
        pos = np.sort(np.random.uniform(high=length, size=m))
        mat_err_idx, mat_haps_err = self.create_genotyping_errors(
            mat_haps_prime, err_rate=err_rate, seed=seed
        )
        pat_err_idx, pat_haps_err = self.create_genotyping_errors(
            pat_haps_prime, err_rate=err_rate, seed=seed
        )
        res_table["mat_haps_true"] = mat_haps
        res_table["pat_haps_true"] = pat_haps
        res_table["mat_haps_real"] = mat_haps_prime
        res_table["pat_haps_real"] = pat_haps_prime
        res_table["mat_haps_err"] = mat_haps_err
        res_table["pat_haps_err"] = pat_haps_err
        res_table["mat_switch"] = mat_switch
        res_table["pat_switch"] = pat_switch
        res_table["mat_err"] = mat_err_idx
        res_table["pat_err"] = pat_err_idx
        res_table["af"] = ps
        res_table["m"] = m
        res_table["nsibs"] = nsibs
        res_table["aploid"] = "2"
        res_table["pos"] = pos
        res_table["length"] = length
        for i in range(nsibs):
            (
                zs_maternal,
                zs_paternal,
                mat_hap1,
                pat_hap1,
                aploid,
            ) = self.sim_haplotype_paths(
                mat_haps,
                pat_haps,
                pos,
                ploidy=ploidy,
                rec_rate=rec_rate,
                seed=seed + i,
            )
            if reads:
                geno, alt_reads, ref_reads = self.sim_read_counts(
                    mat_hap1,
                    pat_hap1,
                    ploidy=ploidy,
                    coverage=coverage,
                    a=a,
                    b=b,
                    seed=seed,
                )
                baf = None
                lrr = None
                sigmas = None
                assert geno.size == m
                assert alt_reads.size == m
                assert ref_reads.size == m
            else:
                geno, baf = self.sim_b_allele_freq(
                    mat_hap1,
                    pat_hap1,
                    ploidy=ploidy,
                    std_dev=std_dev,
                    mix_prop=mix_prop,
                    seed=seed + i,
                )
                lrr, sigmas = self.sim_logR_ratio(
                    mat_hap1, pat_hap1, ploidy=ploidy, alpha=alpha, seed=seed
                )
                alt_reads = None
                ref_reads = None
                assert geno.size == m
                assert baf.size == m
            res_table[f"baf_embryo{i}"] = baf
            res_table[f"lrr_embryo{i}"] = lrr
            res_table[f"sigma_embryo{i}"] = sigmas
            res_table[f"geno_embryo{i}"] = geno
            res_table[f"alt_reads{i}"] = alt_reads
            res_table[f"ref_reads{i}"] = ref_reads
            res_table[f"zs_maternal{i}"] = zs_maternal
            res_table[f"zs_paternal{i}"] = zs_paternal
        return res_table

    def sim_from_haps(
        self,
        mat_haps,
        pat_haps,
        pos,
        reads=False,
        afs=None,
        ploidy=2,
        rec_rate=1e-8,
        mat_skew=0.5,
        std_dev=0.15,
        mix_prop=0.3,
        alpha=0.5,
        coverage=5.0,
        a=10.0,
        b=10.0,
        switch_err_rate=1e-2,
        err_rate=1e-3,
        seed=42,
    ):
        """Simulate data from pre-existing haplotype data."""
        zs_maternal, zs_paternal, mat_hap1, pat_hap1, aploid = self.sim_haplotype_paths(
            mat_haps,
            pat_haps,
            pos,
            ploidy=ploidy,
            mat_skew=mat_skew,
            rec_rate=rec_rate,
            seed=seed,
        )
        (
            mat_haps_prime,
            pat_haps_prime,
            mat_switch,
            pat_switch,
        ) = self.create_switch_errors(
            mat_haps, pat_haps, err_rate=switch_err_rate, seed=seed
        )
        mat_err_idx, mat_haps_err = self.create_genotyping_errors(
            mat_haps_prime, err_rate=err_rate, seed=seed
        )
        pat_err_idx, pat_haps_err = self.create_genotyping_errors(
            pat_haps_prime, err_rate=err_rate, seed=seed
        )
        if reads:
            geno, alt_reads, ref_reads = self.sim_read_counts(
                mat_hap1,
                pat_hap1,
                ploidy=ploidy,
                coverage=coverage,
                a=a,
                b=b,
                seed=seed,
            )
            baf = None
            lrr = None
            sigmas = None
            assert geno.size == pos.size
            assert alt_reads.size == pos.size
            assert ref_reads.size == pos.size
        else:
            geno, baf = self.sim_b_allele_freq(
                mat_hap1,
                pat_hap1,
                ploidy=ploidy,
                std_dev=std_dev,
                mix_prop=mix_prop,
                seed=seed,
            )
            lrr, sigmas = self.sim_logR_ratio(
                mat_hap1, pat_hap1, ploidy=ploidy, alpha=alpha, seed=seed
            )
            alt_reads = None
            ref_reads = None
            assert geno.size == baf.size
            assert baf.size == pos.size
        res_table = {
            "mat_haps": mat_haps,
            "pat_haps": pat_haps,
            "mat_haps_prime": mat_haps_prime,
            "pat_haps_prime": pat_haps_prime,
            "mat_haps_err": mat_haps_err,
            "pat_haps_err": pat_haps_err,
            "mat_switch": mat_switch,
            "pat_switch": pat_switch,
            "mat_err": mat_err_idx,
            "pat_err": pat_err_idx,
            "zs_maternal": zs_maternal,
            "zs_paternal": zs_paternal,
            "geno_embryo": geno,
            "baf": baf,
            "lrr": lrr,
            "sigmas": sigmas,
            "alt_reads": alt_reads,
            "ref_reads": ref_reads,
            "pos": pos,
            "af": np.ones(baf.size) * np.nan if afs is None else afs,
            "aploid": aploid,
            "ploidy": ploidy,
            "rec_rate": rec_rate,
            "std_dev": std_dev,
            "mix_prop": mix_prop,
            "alpha": alpha,
            "seed": seed,
        }
        return res_table

    def sim_cell_contamination(self, baf, haps, fraction=0.01, seed=42):
        """
        Simulate parental cell contamination (typically maternal)
        """
        assert fraction >= 0.0
        assert fraction < 1.0
        assert baf.ndim == 1
        assert haps.ndim == 2
        assert baf.size == haps.shape[1]
        assert seed > 0
        np.random.seed(seed)
        m = baf.size
        cc_baf = np.zeros(m)
        for i in range(m):
            # Dosage of the contaminating parental individual
            geno = np.sum(haps[:, i]) / 2.0
            # Mixture of the existing BAF + genotype-specific BAF ...
            cc_baf[i] = (1.0 - fraction) * baf[i] + fraction * geno
        return cc_baf


class PGTSimMosaic(PGTSimBase):
    """Simulator for mosaic aneuploidies."""

    def __init__(self):
        """Initialize the mosaic aneuploidy simulator.

        Returns: a PGTSim object

        """
        super().__init__()

    def mixed_ploidy_sim(
        self,
        afs=None,
        ploidies=np.array([0, 1, 2, 3]),
        props=np.array([0, 0, 1.0, 0.0]),
        ncells=10,
        m=10000,
        length=1e7,
        rec_rate=1e-8,
        mat_skew=0.5,
        std_dev=0.15,
        mix_prop=0.3,
        alpha=1.0,
        switch_err_rate=1e-2,
        err_rate=1e-3,
        seed=42,
    ):
        """Simulate BAF from a mixture of ploidies."""
        assert ploidies.size == props.size
        assert m > 0
        assert ncells > 0
        assert length > 0
        if ~np.isclose(np.sum(props), 1.0):
            props /= np.sum(props)
        # 1. Simulate the parental haplotypes
        np.random.seed(seed)
        pos = np.sort(np.random.uniform(high=length, size=m))
        mat_haps, pat_haps, ps = self.draw_parental_genotypes(afs=afs, m=m, seed=seed)
        mat_haps_prime, pat_haps_prime, _, _ = self.create_switch_errors(
            mat_haps, pat_haps, err_rate=switch_err_rate, seed=seed
        )
        mat_err_idx, mat_haps_err = self.create_genotyping_errors(
            mat_haps_prime, err_rate=err_rate, seed=seed
        )
        pat_err_idx, pat_haps_err = self.create_genotyping_errors(
            pat_haps_prime, err_rate=err_rate, seed=seed
        )
        # 2. Draw cells from a distribution of ploidies
        mix_ploidies = np.random.choice(ploidies, p=props, size=ncells)
        bafs = np.zeros(shape=(ncells, m))
        lrrs = np.zeros(shape=(ncells, m))
        sigmas_bulk = np.zeros(shape=(ncells, m))
        genos = np.zeros(shape=(ncells, m))
        aploids = []
        # NOTE: only simulate unique ploidies once and then take the weighted mean across them?
        for i, p in enumerate(np.unique(mix_ploidies)):
            (
                zs_maternal,
                zs_paternal,
                mat_hap1,
                pat_hap1,
                aploid,
            ) = self.sim_haplotype_paths(
                mat_haps,
                pat_haps,
                pos,
                ploidy=p,
                mat_skew=mat_skew,
                rec_rate=rec_rate,
                seed=i + 1,
            )
            geno, baf = self.sim_b_allele_freq(
                mat_hap1,
                pat_hap1,
                ploidy=p,
                std_dev=std_dev,
                mix_prop=mix_prop,
                seed=i + 1,
            )
            lrr, sigmas = self.sim_logR_ratio(
                mat_hap1, pat_hap1, ploidy=p, alpha=alpha, seed=i + 1
            )
            assert baf.size == m
            assert lrr.size == m
            assert sigmas.size == m
            for j in np.where(mix_ploidies == p):
                bafs[j, :] = baf
                lrrs[j, :] = lrr
                sigmas_bulk[j, :] = sigmas
            aploids.append(aploid)
        # Take the mean BAF + LRR + sigma estimates across the bulk samples
        baf_embryo = np.mean(bafs, axis=0)
        lrr_embryo = np.mean(lrrs, axis=0)
        sigmas_embryo = np.mean(sigmas_bulk, axis=0)
        assert baf_embryo.size == m
        res_table = {
            "mat_haps": mat_haps,
            "pat_haps": pat_haps,
            "mat_haps_prime": mat_haps_prime,
            "pat_haps_prime": pat_haps_prime,
            "mat_haps_err": mat_haps_err,
            "pat_haps_err": pat_haps_err,
            "mat_err": mat_err_idx,
            "pat_err": pat_err_idx,
            "geno_embryo_bulk": genos,
            "baf_embryo_bulk": bafs,
            "lrr_embryo_bulk": lrrs,
            "sigmas_embryo_bulk": sigmas_bulk,
            "baf": baf_embryo,
            "lrr": lrr_embryo,
            "sigmas": sigmas_embryo,
            "m": m,
            "af": ps,
            "pos": pos,
            "length": length,
            "aploid": aploids,
            "ploidies": mix_ploidies,
            "rec_rate": rec_rate,
            "std_dev": std_dev,
            "mix_prop": mix_prop,
            "seed": seed,
            "alpha": alpha,
        }
        return res_table

    def sim_from_haps(
        self,
        mat_haps,
        pat_haps,
        pos,
        afs=None,
        ploidies=np.array([0, 1, 2, 3]),
        props=np.array([0, 0, 1.0, 0.0]),
        ncells=10,
        rec_rate=1e-8,
        mat_skew=0.5,
        std_dev=0.15,
        mix_prop=0.3,
        alpha=1.0,
        switch_err_rate=1e-2,
        err_rate=1e-3,
        seed=42,
    ):
        """Simulate mosaic aneuploidies from known parental haplotypes."""
        assert mat_haps.ndim == 2
        assert pat_haps.ndim == 2
        assert mat_haps.size == pat_haps.size
        assert mat_haps.shape[1] == pos.size
        assert pat_haps.shape[1] == pos.size
        m = pos.size
        mat_haps_prime, pat_haps_prime, _, _ = self.create_switch_errors(
            mat_haps, pat_haps, err_rate=switch_err_rate, seed=seed
        )
        mat_err_idx, mat_haps_err = self.create_genotyping_errors(
            mat_haps_prime, err_rate=err_rate, seed=seed
        )
        pat_err_idx, pat_haps_err = self.create_genotyping_errors(
            pat_haps_prime, err_rate=err_rate, seed=seed
        )
        # 2. Draw cells from a distribution of ploidies
        mix_ploidies = np.random.choice(ploidies, p=props, size=ncells)
        bafs = np.zeros(shape=(ncells, m))
        lrrs = np.zeros(shape=(ncells, m))
        sigmas_bulk = np.zeros(shape=(ncells, m))
        genos = np.zeros(shape=(ncells, m))
        aploids = []
        # NOTE: only simulate unique ploidies once and then take the weighted mean across them?
        for i, p in enumerate(np.unique(mix_ploidies)):
            (
                zs_maternal,
                zs_paternal,
                mat_hap1,
                pat_hap1,
                aploid,
            ) = self.sim_haplotype_paths(
                mat_haps,
                pat_haps,
                pos,
                ploidy=p,
                mat_skew=mat_skew,
                rec_rate=rec_rate,
                seed=i + 1,
            )
            geno, baf = self.sim_b_allele_freq(
                mat_hap1,
                pat_hap1,
                ploidy=p,
                std_dev=std_dev,
                mix_prop=mix_prop,
                seed=i + 1,
            )
            lrr, sigmas = self.sim_logR_ratio(
                mat_hap1, pat_hap1, ploidy=p, alpha=alpha, seed=i + 1
            )
            assert baf.size == m
            assert lrr.size == m
            assert sigmas.size == m
            for j in np.where(mix_ploidies == p):
                bafs[j, :] = baf
                lrrs[j, :] = lrr
                sigmas_bulk[j, :] = sigmas
            aploids.append(aploid)
        # Take the mean BAF + LRR + sigma estimates across the bulk samples
        baf_embryo = np.mean(bafs, axis=0)
        lrr_embryo = np.mean(lrrs, axis=0)
        sigmas_embryo = np.mean(sigmas_bulk, axis=0)
        assert baf_embryo.size == m
        res_table = {
            "mat_haps": mat_haps,
            "pat_haps": pat_haps,
            "mat_haps_prime": mat_haps_prime,
            "pat_haps_prime": pat_haps_prime,
            "mat_haps_err": mat_haps_err,
            "pat_haps_err": pat_haps_err,
            "mat_err": mat_err_idx,
            "pat_err": pat_err_idx,
            "geno_embryo_bulk": genos,
            "baf_embryo_bulk": bafs,
            "lrr_embryo_bulk": lrrs,
            "sigmas_embryo_bulk": sigmas_bulk,
            "baf": baf_embryo,
            "lrr": lrr_embryo,
            "sigmas": sigmas_embryo,
            "af": np.ones(baf.size) * np.nan if afs is None else afs,
            "pos": pos,
            "aploid": aploids,
            "ploidies": mix_ploidies,
            "rec_rate": rec_rate,
            "std_dev": std_dev,
            "mix_prop": mix_prop,
            "seed": seed,
            "alpha": alpha,
        }
        return res_table


class PGTSimSegmental(PGTSimBase):
    """Simulator for segmental aneuploidies."""

    def __init__(self):
        """Initialize simulation of segmental data."""
        super().__init__()

    def seg_aneuploidy(self, m=100, mean_size=10, ploidy=3, mat_skew=0.5, seed=42):
        """Choose the position and type of the segmental aneuploidy."""
        assert mean_size > 0
        assert seed > 0
        assert ploidy in [0, 1, 2, 3]
        assert (mat_skew >= 0) and (mat_skew <= 1.0)
        # 1. Identify the number of SNPs that the segmental aneuploidy occupies
        np.random.seed(seed)
        seg_l = m
        while seg_l >= m:
            seg_l = poisson.rvs(mean_size, loc=0, size=1)
        assert m > seg_l
        # 1. Define the start and ending position of the aneuploidy
        start = randint.rvs(low=0, high=m - seg_l, size=1)
        end = start + seg_l
        # 2. Determine the type of the aneuploidy as a random choice
        if ploidy == 0:
            aneu_type = "0"
        elif ploidy == 1:
            pat = binom.rvs(1, mat_skew)
            if pat:
                aneu_type = "1p"
            else:
                aneu_type = "1m"
        elif ploidy == 2:
            aneu_type = "2"
        elif ploidy == 3:
            pat = binom.rvs(1, 1.0 - mat_skew)
            if pat:
                aneu_type = "3p"
            else:
                aneu_type = "3m"
        return aneu_type, (start, end)

    def sim_haplotype_paths_segmental(
        self,
        mat_haps,
        pat_haps,
        pos,
        start,
        end,
        aneu_type="3m",
        rec_rate=1e-4,
        seed=42,
    ):
        """Simulate haplotypes and segmental aneuploidies.

        This simulates a local segmental aneuploidy in an otherwise disomic background.

        """
        assert mat_haps.size == pat_haps.size
        assert seed > 0
        assert (rec_rate > 0) and (rec_rate <= 1)
        assert start <= end
        assert start <= mat_haps.shape[1]
        assert end <= mat_haps.shape[1]
        assert aneu_type in ["0", "1m", "1p", "2", "3m", "3p"]
        np.random.seed(seed)
        m = mat_haps.shape[1]
        zs_maternal = np.zeros(m)
        zs_paternal = np.zeros(m)
        zs1_maternal = np.repeat(np.nan, m)
        zs1_paternal = np.repeat(np.nan, m)
        ploidies = np.zeros(m, dtype=np.uint16)
        zs_maternal[0] = binom.rvs(1, 0.5)
        zs_paternal[0] = binom.rvs(1, 0.5)
        ploidies[0] = 2
        if start == 0:
            if aneu_type == "0":
                ploidies[0] = 0
            elif aneu_type in ["1m", "1p"]:
                ploidies[0] = 1
            elif aneu_type in ["3m", "3p"]:
                ploidies[0] = 3
        for i in range(1, m):
            d = pos[i] - pos[i - 1]
            if (i >= start) and (i <= end):
                # NOTE: we're in the aneuploidy state here ...
                if aneu_type == "0":
                    ploidies[i] = 0
                    zs_maternal[i] = np.nan
                    zs_paternal[i] = np.nan
                elif aneu_type == "1m":
                    # Only sample from the maternal haplotypes
                    ploidies[i] = 1
                    zs_maternal[i] = (
                        1 - zs_maternal[i - 1]
                        if uniform.rvs() <= (1 - np.exp(-rec_rate * d))
                        else zs_maternal[i - 1]
                    )
                    zs_paternal[i] = np.nan
                elif aneu_type == "1p":
                    # Only sample from the paternal haplotypes
                    ploidies[i] = 1
                    zs_paternal[i] = (
                        1 - zs_paternal[i - 1]
                        if uniform.rvs() <= (1 - np.exp(-rec_rate * d))
                        else zs_paternal[i - 1]
                    )
                    zs_maternal[i] = np.nan
                elif aneu_type == "2":
                    ploidies[i] = 2
                    zs_paternal[i] = (
                        1 - zs_paternal[i - 1]
                        if uniform.rvs() <= (1 - np.exp(-rec_rate * d))
                        else zs_paternal[i - 1]
                    )
                    zs_maternal[i] = (
                        1 - zs_maternal[i - 1]
                        if uniform.rvs() <= (1 - np.exp(-rec_rate * d))
                        else zs_maternal[i - 1]
                    )
                elif aneu_type == "3m":
                    ploidies[i] = 3
                    if (i == 0) or np.isnan(zs1_maternal[(i - 1)]):
                        zs1_maternal[i] = binom.rvs(1, 0.5)
                    else:
                        zs1_maternal[i] = (
                            1 - zs1_maternal[i - 1]
                            if uniform.rvs() <= (1 - np.exp(-rec_rate * d))
                            else zs1_maternal[i - 1]
                        )
                    zs_maternal[i] = (
                        1 - zs_maternal[i - 1]
                        if uniform.rvs() <= (1 - np.exp(-rec_rate * d))
                        else zs_maternal[i - 1]
                    )
                    zs_paternal[i] = (
                        1 - zs_paternal[i - 1]
                        if uniform.rvs() <= (1 - np.exp(-rec_rate * d))
                        else zs_paternal[i - 1]
                    )
                elif aneu_type == "3p":
                    ploidies[i] = 3
                    if (i == 0) or np.isnan(zs1_paternal[(i - 1)]):
                        zs1_paternal[i] = binom.rvs(1, 0.5)
                    else:
                        zs1_paternal[i] = (
                            1 - zs1_maternal[i - 1]
                            if uniform.rvs() <= (1 - np.exp(-rec_rate * d))
                            else zs1_paternal[i - 1]
                        )
                    zs_maternal[i] = (
                        1 - zs_maternal[i - 1]
                        if uniform.rvs() <= (1 - np.exp(-rec_rate * d))
                        else zs_maternal[i - 1]
                    )
                    zs_paternal[i] = (
                        1 - zs_paternal[i - 1]
                        if uniform.rvs() <= (1 - np.exp(-rec_rate * d))
                        else zs_paternal[i - 1]
                    )
                else:
                    raise ValueError(
                        f"{aneu_type} is an invalid segmental aneuploidy status!"
                    )
            else:
                ploidies[i] = 2
                # NOTE: what happens if you are coming back from a null zs_maternal?
                if ~np.isnan(zs_maternal[i - 1]):
                    zs_maternal[i] = (
                        1 - zs_maternal[i - 1]
                        if uniform.rvs() <= (1 - np.exp(-rec_rate * d))
                        else zs_maternal[i - 1]
                    )
                else:
                    pass
                if ~np.isnan(zs_paternal[i - 1]):
                    zs_paternal[i] = (
                        1 - zs_paternal[i - 1]
                        if uniform.rvs() <= (1 - np.exp(-rec_rate * d))
                        else zs_paternal[i - 1]
                    )
                else:
                    pass

        # Now sample through the underlying haplotypes
        mat_haps1 = np.zeros(m, dtype=np.uint16)
        pat_haps1 = np.zeros(m, dtype=np.uint16)
        for i in range(m):
            # Sample the alleles for both maternal and paternal haplotypes ...
            if ~np.isnan(zs_maternal[i]):
                mat_haps1[i] = mat_haps[int(zs_maternal[i]), i]
            if ~np.isnan(zs_paternal[i]):
                pat_haps1[i] = pat_haps[int(zs_paternal[i]), i]
            if ~np.isnan(zs1_maternal[i]):
                mat_haps1[i] += mat_haps[int(zs1_maternal[i]), i]
            if ~np.isnan(zs1_paternal[i]):
                pat_haps1[i] += pat_haps[int(zs1_paternal[i]), i]
        return (
            mat_haps1,
            pat_haps1,
            aneu_type,
            start,
            end,
            zs_maternal,
            zs_paternal,
            zs1_maternal,
            zs1_paternal,
            ploidies,
        )

    def sim_b_allele_freq_segmental(
        self, mat_hap, pat_hap, ploidies, std_dev=0.1, mix_prop=0.6, seed=42
    ):
        """Simulate B-allele frequency conditional on parental haplotypes."""
        np.random.seed(seed)
        assert mat_hap.size == pat_hap.size
        assert ploidies.size == mat_hap.size
        true_geno = mat_hap + pat_hap
        baf = np.zeros(true_geno.size)
        for i in range(baf.size):
            if ploidies[i] == 0:
                a, b = (0 - 0.5) / std_dev, (1 - 0.5) / std_dev
                baf[i] = truncnorm.rvs(a, b, loc=0.5, scale=std_dev)
            else:
                mu_i = true_geno[i] / ploidies[i]
                a, b = (0 - mu_i) / std_dev, (1 - mu_i) / std_dev
                if mu_i == 0:
                    baf[i] = (
                        0.0
                        if uniform.rvs() < mix_prop
                        else truncnorm.rvs(a, b, loc=mu_i, scale=std_dev)
                    )
                elif mu_i == 1:
                    baf[i] = (
                        1.0
                        if uniform.rvs() < mix_prop
                        else truncnorm.rvs(a, b, loc=mu_i, scale=std_dev)
                    )
                else:
                    baf[i] = truncnorm.rvs(a, b, loc=mu_i, scale=std_dev)
        return true_geno, baf, ploidies

    def sim_read_counts_segmental(
        self,
        mat_hap,
        pat_hap,
        ploidies,
        coverage=1.0,
        a=10.0,
        b=10.0,
        eps=1e-2,
        seed=42,
    ):
        """Simulate read counts for segmental embryo biopsies.

        Args:
            mat_hap (`np.array`): 1 x M numpy array for maternal haplotype.
            pat_hap (`np.array`): 1 x M numpy array for paternal haplotype.
            ploidies (`np.array`): ploidy count of number of chromosomes being simulated.
            coverage (`float`): mean coverage for a standard diploid individual.
            a (`float`): shape parameter for beta-binomial.
            b (`float`): scale parameter for beta-binomial.
            eps (`float`): small noise parameter.
            seed (`int`): random number seed.

        Output:
            true_geno (`np.array`): the true genotype of the embryo.
            alt_reads (`np.array`): the number of reads with alt. allele.
            ref_reads (`np.array`): the number of reads with ref. allele.

        """
        assert seed > 0
        assert coverage > 0.0
        assert mat_hap.size == pat_hap.size
        assert ploidies.size == mat_hap.size
        assert (a > 0) & (b > 0)
        assert np.all(np.isin(ploidies, [0, 1, 2, 3]))
        assert (0 < eps) & (eps <= 0.5)
        np.random.seed(seed)
        true_geno = mat_hap + pat_hap
        alt_read_cnt, ref_read_cnt = (
            np.zeros(true_geno.size, dtype=np.uint16),
            np.zeros(true_geno.size, dtype=np.uint16),
        )
        for i, g in enumerate(true_geno):
            # 1. Simulate the total number of reads at the site
            tot_reads = poisson.rvs(mu=coverage * ((ploidies[i] + eps) / 2.0))
            # 2. Simulate the total allelic balance as from a beta-binomial random-variable
            if tot_reads > 0:
                p = (g + eps) / ploidies[i] if ploidies[i] > 0 else eps
                p = (p - 2 * eps) if p > 1 else p
                assert (p > 0) & (p < 1)
                alt_read_cnt[i] = betabinom.rvs(
                    n=tot_reads, a=p * (a + b), b=(1.0 - p) * (a + b)
                )
                ref_read_cnt[i] = tot_reads - alt_read_cnt[i]
        return true_geno, alt_read_cnt, ref_read_cnt

    def full_segmental_sim(
        self,
        reads=False,
        afs=None,
        ploidy=2,
        m=10000,
        length=50e6,
        mat_skew=0.5,
        rec_rate=1e-8,
        std_dev=0.1,
        mix_prop=0.7,
        mean_size=10,
        coverage=5.0,
        a=10.0,
        b=10.0,
        switch_err_rate=1e-2,
        err_rate=1e-3,
        seed=42,
    ):
        """Conduct a full simulation of segmental aneuploidies conditional on parental haplotypes."""
        np.random.seed(seed)
        pos = np.sort(np.random.uniform(high=length, size=m))
        mat_haps, pat_haps, ps = self.draw_parental_genotypes(afs=afs, m=m, seed=seed)
        aneu_type, (start, end) = self.seg_aneuploidy(
            m=m, mean_size=mean_size, ploidy=ploidy, mat_skew=mat_skew, seed=seed
        )
        (
            mat_hap1,
            pat_hap1,
            aneu_type,
            start,
            end,
            zs_maternal,
            zs_paternal,
            zs1_maternal,
            zs1_paternal,
            ploidies,
        ) = self.sim_haplotype_paths_segmental(
            mat_haps,
            pat_haps,
            pos,
            start=start,
            end=end,
            aneu_type=aneu_type,
            rec_rate=rec_rate,
            seed=seed,
        )
        if reads:
            geno, alt_reads, ref_reads = self.sim_read_counts(
                mat_hap1,
                pat_hap1,
                ploidy=ploidies,
                coverage=coverage,
                a=a,
                b=b,
                seed=seed,
            )
            baf = None
            assert geno.size == m
            assert alt_reads.size == m
            assert ref_reads.size == m
        else:
            geno, baf, _ = self.sim_b_allele_freq_segmental(
                mat_hap1,
                pat_hap1,
                ploidies,
                std_dev=std_dev,
                mix_prop=mix_prop,
                seed=seed,
            )
            alt_reads = None
            ref_reads = None
        (
            mat_haps_prime,
            pat_haps_prime,
            mat_switch,
            pat_switch,
        ) = self.create_switch_errors(
            mat_haps, pat_haps, err_rate=switch_err_rate, seed=seed
        )
        mat_err_idx, mat_haps_err = self.create_genotyping_errors(
            mat_haps_prime, err_rate=err_rate, seed=seed
        )
        pat_err_idx, pat_haps_err = self.create_genotyping_errors(
            pat_haps_prime, err_rate=err_rate, seed=seed
        )
        assert geno.size == m
        assert baf.size == m
        res_table = {
            "mat_haps": mat_haps,
            "pat_haps": pat_haps,
            "mat_haps_prime": mat_haps_prime,
            "pat_haps_prime": pat_haps_prime,
            "mat_haps_err": mat_haps_err,
            "pat_haps_err": pat_haps_err,
            "mat_err": mat_err_idx,
            "pat_err": pat_err_idx,
            "ploidies": ploidies,
            "seg_aneuploidy_type": aneu_type,
            "seg_start": start,
            "seg_end": end,
            "geno_embryo": geno,
            "baf": baf,
            "alt_reads": alt_reads,
            "ref_reads": ref_reads,
            "af": ps,
            "pos": pos,
            "m": m,
            "rec_rate": rec_rate,
            "std_dev": std_dev,
            "mix_prop": mix_prop,
            "seed": seed,
        }
        return res_table

    def sim_from_haps(
        self,
        mat_haps,
        pat_haps,
        pos,
        reads=False,
        afs=None,
        rec_rate=1e-4,
        std_dev=0.1,
        mix_prop=0.7,
        mean_size=10,
        coverage=5.0,
        a=10.0,
        b=10.0,
        switch_err_rate=1e-2,
        err_rate=1e-3,
        seed=42,
    ):
        """Simulate a segmental aneuploidy from known haplotypes."""
        assert mat_haps.ndim == 2
        assert pat_haps.ndim == 2
        assert mat_haps.size == pat_haps.size
        assert mat_haps.shape[1] == pos.size
        assert pat_haps.shape[1] == pos.size
        m = pos.size
        (
            mat_hap1,
            pat_hap1,
            aneu_type,
            start,
            end,
            zs_maternal,
            zs_paternal,
            zs1_maternal,
            zs1_paternal,
            ploidies,
        ) = self.sim_haplotype_paths_segmental(
            mat_haps,
            pat_haps,
            pos,
            rec_rate=rec_rate,
            mean_size=mean_size,
            seed=seed,
        )
        if reads:
            geno, alt_reads, ref_reads = self.sim_read_counts(
                mat_hap1,
                pat_hap1,
                ploidy=ploidies,
                coverage=coverage,
                a=a,
                b=b,
                seed=seed,
            )
            baf = None
            assert geno.size == m
            assert alt_reads.size == m
            assert ref_reads.size == m
        else:
            geno, baf, ploidies = self.sim_b_allele_freq_segmental(
                mat_hap1,
                pat_hap1,
                ploidies,
                std_dev=std_dev,
                mix_prop=mix_prop,
                seed=seed,
            )
        (
            mat_haps_prime,
            pat_haps_prime,
            mat_switch,
            pat_switch,
        ) = self.create_switch_errors(
            mat_haps, pat_haps, err_rate=switch_err_rate, seed=seed
        )
        mat_err_idx, mat_haps_err = self.create_genotyping_errors(
            mat_haps_prime, err_rate=err_rate, seed=seed
        )
        pat_err_idx, pat_haps_err = self.create_genotyping_errors(
            pat_haps_prime, err_rate=err_rate, seed=seed
        )
        res_table = {
            "mat_haps": mat_haps,
            "pat_haps": pat_haps,
            "mat_haps_prime": mat_haps_prime,
            "pat_haps_prime": pat_haps_prime,
            "mat_haps_err": mat_haps_err,
            "pat_haps_err": pat_haps_err,
            "mat_err": mat_err_idx,
            "pat_err": pat_err_idx,
            "ploidies": ploidies,
            "seg_aneuploidy_type": aneu_type,
            "seg_start": start,
            "seg_end": end,
            "geno_embryo": geno,
            "baf": baf,
            "alt_reads": alt_reads,
            "ref_reads": ref_reads,
            "af": np.ones(baf.size) * np.nan if afs is None else afs,
            "pos": pos,
            "rec_rate": rec_rate,
            "std_dev": std_dev,
            "mix_prop": mix_prop,
            "seed": seed,
        }
        return res_table


class PGTSimVCF(PGTSim):
    """Implements PGT-simulation from parental haplotypes."""

    def __init__(self):
        """Initialize the class."""
        super().__init__()

    def gen_parental_haplotypes(
        self, vcf_fp, maternal_id=None, paternal_id=None, **kwargs
    ):
        """Generate parental haplotypes from an actual VCF.

        Args:
            vcf_fp (`str`): path to an input VCF file.
            maternal_id (`str`): ID of maternal individual.
            paternal_id (`str`): ID of paternal individual.
        Output:
            mat_haps (`np.array`): paternal haplotypes.
            pat_haps (`np.array`): maternal haplotypes.
            pos (`np.array`): basepair positions of variants.
            afs (`np.array`): alternative allele frequency.
            df (`pd.DataFrame`): pandas dataframe of cleaned options.

        """
        from cyvcf2 import VCF

        assert isinstance(maternal_id, str)
        assert isinstance(paternal_id, str)
        vcf = VCF(vcf_fp, **kwargs)
        if maternal_id not in vcf.samples:
            raise ValueError(f"{maternal_id} is not in {vcf_fp}!")
        if paternal_id not in vcf.samples:
            raise ValueError(f"{paternal_id} is not in {vcf_fp}!")
        assert maternal_id in vcf.samples
        assert paternal_id in vcf.samples
        pos = []
        mat_haps = []
        for var in VCF(vcf_fp, samples=[maternal_id], **kwargs):
            if len(var.ALT) == 1:
                pos.append(var.POS)
                mat_haps.append([var.genotypes[0][0], var.genotypes[0][1]])
        pat_haps = []
        for var in VCF(vcf_fp, samples=[paternal_id], **kwargs):
            if len(var.ALT) == 1:
                pat_haps.append([var.genotypes[0][0], var.genotypes[0][1]])
        afs = []
        for var in vcf:
            if len(var.ALT) == 1:
                afs.append(var.aaf)
        mat_haps = np.array(mat_haps).astype(np.uint8).T
        pat_haps = np.array(pat_haps).astype(np.uint8).T
        pos = np.array(pos, dtype=np.float64)
        afs = np.array(afs, dtype=np.float64)
        assert pos.size == afs.size
        assert mat_haps.size == pat_haps.size
        return mat_haps, pat_haps, pos, afs
