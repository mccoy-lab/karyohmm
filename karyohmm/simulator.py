"""Simulation utilities for unit-testing suite."""

import numpy as np
from scipy.stats import beta, binom, norm, rv_histogram, truncnorm, uniform

# These are the different classes of aneuploidy that we can putatively simulate from


class PGTSimBase:
    """Base-class for simulation of PGT-A data."""

    def __init__(self):
        """Initialize the PGT-A Simulator."""
        # NOTE: the initial SDs are derived from the PENNCNV source code
        self.lrr_mu = {0: -3.527211, 1: np.log2(0.5), 2: np.log2(1.0), 3: np.log2(1.5)}
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
        return [np.vstack([mat_h1, mat_h2]), np.vstack([pat_h1, pat_h2])]

    def create_switch_errors_help(self, haps, err_rate=1e-3, seed=42):
        """Revised method to create switch errors."""
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
        """Create switch errors to evaluate impact of poor phasing."""
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
        self, mat_haps, pat_haps, ploidy=2, rec_prob=1e-2, mat_skew=0.5, seed=42
    ):
        """Simulate copying paths through the maternal and paternal haplotypes."""
        assert (ploidy <= 3) & (ploidy >= 0)
        assert (mat_skew >= 0) and (mat_skew <= 1.0)
        assert mat_haps.size == pat_haps.size
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
                # We have a paternal monosomy ...
                zs_maternal = None
                zs_paternal[0] = binom.rvs(1, 0.5)
                for i in range(1, m):
                    zs_paternal[i] = (
                        1 - zs_paternal[i - 1]
                        if uniform.rvs() <= rec_prob
                        else zs_paternal[i - 1]
                    )
                pat_real_hap = np.array(
                    [pat_haps[i, p] for p, i in enumerate(zs_paternal)]
                )
                mat_real_hap = np.zeros(pat_real_hap.size)
                aploid = "1p"
            else:
                # We have a maternal monosomy ...
                zs_paternal = None
                zs_maternal[0] = binom.rvs(1, 0.5)
                for i in range(1, m):
                    zs_maternal[i] = (
                        1 - zs_maternal[i - 1]
                        if uniform.rvs() <= rec_prob
                        else zs_maternal[i - 1]
                    )
                mat_real_hap = np.array(
                    [mat_haps[i, p] for p, i in enumerate(zs_maternal)]
                )
                pat_real_hap = np.zeros(mat_real_hap.size)
                aploid = "1m"
        elif ploidy == 3:
            # Drawing a maternal or paternal trisomy
            pat = binom.rvs(1, mat_skew)
            if pat:
                # Simulate a paternal trisomy
                zs_maternal[0] = binom.rvs(0, 0.5)
                zs0_paternal[0] = binom.rvs(0, 0.5)
                zs1_paternal[0] = binom.rvs(0, 0.5)

                for i in range(1, m):
                    zs_maternal[i] = (
                        1 - zs_maternal[i - 1]
                        if uniform.rvs() <= rec_prob
                        else zs_maternal[i - 1]
                    )
                    zs0_paternal[i] = (
                        1 - zs0_paternal[i - 1]
                        if uniform.rvs() <= rec_prob
                        else zs0_paternal[i - 1]
                    )
                    zs1_paternal[i] = (
                        1 - zs1_paternal[i - 1]
                        if uniform.rvs() <= rec_prob
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
                    zs_paternal[i] = (
                        1 - zs_paternal[i - 1]
                        if uniform.rvs() <= rec_prob
                        else zs_paternal[i - 1]
                    )
                    zs0_maternal[i] = (
                        1 - zs0_maternal[i - 1]
                        if uniform.rvs() <= rec_prob
                        else zs0_maternal[i - 1]
                    )
                    zs1_maternal[i] = (
                        1 - zs1_maternal[i - 1]
                        if uniform.rvs() <= rec_prob
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
                    [
                        pat_haps[i, p]
                        for p, (i, j) in enumerate(zip(zs0_paternal, zs1_paternal))
                    ]
                )
                aploid = "3m"
        elif ploidy == 2:
            # A typical euploid sample ...
            zs_maternal[0] = binom.rvs(1, 0.5)
            zs_paternal[0] = binom.rvs(1, 0.5)
            for i in range(1, m):
                # We switch with a specific probability
                zs_maternal[i] = (
                    1 - zs_maternal[i - 1]
                    if uniform.rvs() <= rec_prob
                    else zs_maternal[i - 1]
                )
                zs_paternal[i] = (
                    1 - zs_paternal[i - 1]
                    if uniform.rvs() <= rec_prob
                    else zs_paternal[i - 1]
                )
            mat_real_hap = np.array([mat_haps[i, p] for p, i in enumerate(zs_maternal)])
            pat_real_hap = np.array([pat_haps[i, p] for p, i in enumerate(zs_paternal)])
            aploid = "2"
        else:
            raise ValueError(f"{ploidy} should be between 0 and 3!")
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

    def sim_logR_ratio(self, mat_hap, pat_hap, ploidy=2, alpha=1.0, seed=42):
        """Simulate logR-ratio conditional on ploidy.

        Alpha is the degree to which the variance is increased for the LRR.
        """
        assert seed > 0
        assert ploidy in [0, 1, 2, 3]
        assert mat_hap.size == pat_hap.size
        np.random.seed(seed)
        m = mat_hap.size
        lrr = norm.rvs(self.lrr_mu[ploidy], scale=self.lrr_sd[ploidy] * alpha, size=m)
        return lrr

    def sim_read_counts(self, mat_hap, pat_hap, coverage=1.0):
        """Simulate read counts for embryos."""
        raise NotImplementedError("NGS-based simulation is not available yet!")

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
        ploidy=2,
        m=10000,
        length=1e7,
        rec_prob=1e-4,
        mat_skew=0.5,
        std_dev=0.15,
        mix_prop=0.3,
        alpha=1.0,
        switch_err_rate=1e-2,
        seed=42,
    ):
        """Simulate a single embryo with a given ploidy status."""
        np.random.seed(seed)
        mat_haps, pat_haps = self.draw_parental_genotypes(afs=afs, m=m, seed=seed)
        zs_maternal, zs_paternal, mat_hap1, pat_hap1, aploid = self.sim_haplotype_paths(
            mat_haps,
            pat_haps,
            ploidy=ploidy,
            mat_skew=mat_skew,
            rec_prob=rec_prob,
            seed=seed,
        )
        geno, baf = self.sim_b_allele_freq(
            mat_hap1,
            pat_hap1,
            ploidy=ploidy,
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
        pos = np.sort(np.random.uniform(high=length, size=m))
        assert geno.size == m
        assert baf.size == m
        assert pos.size == m
        res_table = {
            "mat_haps": mat_haps,
            "pat_haps": pat_haps,
            "mat_haps_prime": mat_haps_prime,
            "pat_haps_prime": pat_haps_prime,
            "mat_switch": mat_switch,
            "pat_switch": pat_switch,
            "zs_maternal": zs_maternal,
            "zs_paternal": zs_paternal,
            "geno_embryo": geno,
            "baf_embryo": baf,
            "pos": pos,
            "m": m,
            "length": length,
            "aploid": aploid,
            "ploidy": ploidy,
            "rec_prob": rec_prob,
            "std_dev": std_dev,
            "mix_prop": mix_prop,
            "alpha": alpha,
            "seed": seed,
        }
        return res_table

    def sibling_euploid_sim(
        self,
        afs=None,
        ploidy=2,
        m=10000,
        length=1e7,
        nsibs=3,
        rec_prob=1e-4,
        std_dev=0.15,
        mix_prop=0.3,
        switch_err_rate=1e-2,
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
        mat_haps, pat_haps = self.draw_parental_genotypes(afs=None, m=m, seed=seed)
        (
            mat_haps_prime,
            pat_haps_prime,
            mat_switch,
            pat_switch,
        ) = self.create_switch_errors(
            mat_haps, pat_haps, err_rate=switch_err_rate, seed=seed
        )
        pos = np.sort(np.random.uniform(high=length, size=m))
        res_table["mat_haps_true"] = mat_haps
        res_table["pat_haps_true"] = pat_haps
        res_table["mat_haps_real"] = mat_haps_prime
        res_table["pat_haps_real"] = pat_haps_prime
        res_table["mat_switch"] = mat_switch
        res_table["pat_switch"] = pat_switch
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
                ploidy=ploidy,
                rec_prob=rec_prob,
                seed=seed + i,
            )
            geno, baf = self.sim_b_allele_freq(
                mat_hap1,
                pat_hap1,
                ploidy=ploidy,
                std_dev=std_dev,
                mix_prop=mix_prop,
                seed=seed + i,
            )

            assert geno.size == m
            assert baf.size == m
            res_table[f"baf_embryo{i}"] = baf
            res_table[f"geno_embryo{i}"] = geno
            res_table[f"zs_maternal{i}"] = zs_maternal
            res_table[f"zs_paternal{i}"] = zs_paternal
        return res_table


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
        rec_prob=1e-4,
        mat_skew=0.5,
        std_dev=0.15,
        mix_prop=0.3,
        alpha=1.0,
        switch_err_rate=1e-2,
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
        mat_haps, pat_haps = self.draw_parental_genotypes(afs=afs, m=m, seed=seed)
        mat_haps_prime, pat_haps_prime, _, _ = self.create_switch_errors(
            mat_haps, pat_haps, err_rate=switch_err_rate, seed=seed
        )
        # 2. Draw cells from a distribution of ploidies
        mix_ploidies = np.random.choice(ploidies, p=props, size=ncells)
        bafs = np.zeros(shape=(ncells, m))
        lrrs = np.zeros(shape=(ncells, m))
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
                ploidy=p,
                mat_skew=mat_skew,
                rec_prob=rec_prob,
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
            lrr = self.sim_logR_ratio(
                mat_hap1, pat_hap1, ploidy=p, alpha=alpha, seed=i + 1
            )
            assert baf.size == m
            assert lrr.size == m
            for j in np.where(mix_ploidies == p):
                bafs[j, :] = baf
                lrrs[j, :] = lrr
            aploids.append(aploid)
        # Take the mean BAF + LRR estimates across the bulk samples
        baf_embryo = np.mean(bafs, axis=0)
        lrr_embryo = np.mean(lrrs, axis=0)
        pos = np.sort(np.random.uniform(high=length, size=m))
        assert baf_embryo.size == m
        res_table = {
            "mat_haps": mat_haps,
            "pat_haps": pat_haps,
            "mat_haps_prime": mat_haps_prime,
            "pat_haps_prime": pat_haps_prime,
            "geno_embryo_bulk": genos,
            "baf_embryo_bulk": bafs,
            "lrr_embryo_bulk": lrrs,
            "baf_embryo": baf_embryo,
            "lrr_embryo": lrr_embryo,
            "m": m,
            "pos": pos,
            "length": length,
            "aploid": aploids,
            "ploidies": mix_ploidies,
            "rec_prob": rec_prob,
            "std_dev": std_dev,
            "mix_prop": mix_prop,
            "seed": seed,
            "alpha": alpha,
        }
        return res_table
