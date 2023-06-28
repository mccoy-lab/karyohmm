"""Simulation utilities for unit-testing suite."""

import numpy as np
from scipy.stats import beta, binom, norm, rv_histogram, truncnorm, uniform

# These are the different classes of aneuploidy that we can putatively simulate from
sim_ploidy_values = ["0", "1m", "1p", "2", "3m", "3p"]


def draw_parental_genotypes(afs=None, m=100, seed=42):
    """Draw parental genotypes from a beta distribution.

    Args:
        afs (`np.`): alpha parameter of a beta distribution.
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


def create_switch_errors(mat_haps, pat_haps, err_rate=1e-3, seed=42):
    """Create switch errors to evaluate impact of poor phasing."""
    assert mat_haps.size == pat_haps.size
    assert mat_haps.ndim == 2
    assert mat_haps.ndim == 2
    assert (err_rate > 0) and (err_rate < 1)
    assert seed > 0
    # Create the shuffled maternal haplotypes
    m = mat_haps.shape[1]
    mat_haps_prime = np.zeros(shape=mat_haps.shape, dtype=int)
    pat_haps_prime = np.zeros(shape=mat_haps.shape, dtype=int)
    mi0, mi1 = 0, 1
    pi0, pi1 = 0, 1
    us1 = np.random.uniform(size=m)
    us2 = np.random.uniform(size=m)
    m_switch = np.where(us1 < err_rate)[0]
    p_switch = np.where(us2 < err_rate)[0]
    for i in range(m):
        if us1[i] < err_rate:
            mi0 = 1 - mi0
            mi1 = 1 - mi1
        mat_haps_prime[0, i] = mat_haps[mi0, i]
        mat_haps_prime[1, i] = mat_haps[mi1, i]
        if us2[i] < err_rate:
            pi0 = 1 - pi0
            pi1 = 1 - pi1
        pat_haps_prime[0, i] = pat_haps[pi0, i]
        pat_haps_prime[1, i] = pat_haps[pi1, i]
    # Create the shuffled paternal haplotypes
    return mat_haps_prime, pat_haps_prime, m_switch, p_switch


def sim_haplotype_paths(
    mat_haps, pat_haps, ploidy=2, rec_prob=1e-2, mat_skew=0.5, seed=42
):
    """Simulate paths through the maternal and paternal haplotypes."""
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
            pat_real_hap = np.array([pat_haps[i, p] for p, i in enumerate(zs_paternal)])
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
            mat_real_hap = np.array([mat_haps[i, p] for p, i in enumerate(zs_maternal)])
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
            mat_real_hap = np.array([mat_haps[i, p] for p, i in enumerate(zs_maternal)])
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


def sim_b_allele_freq(mat_hap, pat_hap, ploidy=2, std_dev=0.2, mix_prop=0.3, seed=42):
    """Simulate of B-allele frequency."""
    np.random.seed(seed)
    assert (ploidy <= 3) & (ploidy >= 0)
    assert mat_hap.size == pat_hap.size
    true_geno = mat_hap + pat_hap
    baf = np.zeros(true_geno.size)
    for i in range(baf.size):
        if ploidy == 0:
            baf[i] = np.random.uniform()
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


def full_ploidy_sim(
    afs=None,
    ploidy=2,
    m=10000,
    rec_prob=1e-4,
    mat_skew=0.5,
    std_dev=0.15,
    mix_prop=0.7,
    alpha=1.0,
    switch_err_rate=1e-2,
    seed=42,
):
    """Composite helper function to aid with total simulation runs."""
    np.random.seed(seed)
    mat_haps, pat_haps = draw_parental_genotypes(afs=afs, m=m, seed=seed)
    zs_maternal, zs_paternal, mat_hap1, pat_hap1, aploid = sim_haplotype_paths(
        mat_haps,
        pat_haps,
        ploidy=ploidy,
        mat_skew=mat_skew,
        rec_prob=rec_prob,
        seed=seed,
    )
    geno, baf = sim_b_allele_freq(
        mat_hap1, pat_hap1, ploidy=ploidy, std_dev=std_dev, mix_prop=mix_prop, seed=seed
    )
    mat_haps_prime, pat_haps_prime, mat_switch, pat_switch = create_switch_errors(
        mat_haps, pat_haps, err_rate=switch_err_rate, seed=seed
    )
    assert geno.size == m
    assert baf.size == m
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
        "m": m,
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
    afs=None,
    ploidy=2,
    m=10000,
    nsibs=5,
    rec_prob=1e-4,
    std_dev=0.2,
    mix_prop=0.3,
    alpha=1.0,
    switch_err_rate=1e-2,
    seed=42,
):
    """Simulate euploid embryos that are siblings."""
    assert ploidy == 2
    assert m > 0
    assert seed > 0
    assert nsibs > 0
    np.random.seed(seed)

    res_table = {}
    mat_haps, pat_haps = draw_parental_genotypes(afs=None, m=m, seed=seed)
    mat_haps_prime, pat_haps_prime, mat_switch, pat_switch = create_switch_errors(
        mat_haps, pat_haps, err_rate=switch_err_rate, seed=seed
    )
    res_table["mat_haps_true"] = mat_haps
    res_table["pat_haps_true"] = pat_haps
    res_table["mat_haps_real"] = mat_haps_prime
    res_table["pat_haps_real"] = pat_haps_prime
    res_table["mat_switch"] = mat_switch
    res_table["pat_switch"] = pat_switch
    res_table["nsibs"] = nsibs
    res_table["aploid"] = "2"
    for i in range(nsibs):
        zs_maternal, zs_paternal, mat_hap1, pat_hap1, aploid = sim_haplotype_paths(
            mat_haps,
            pat_haps,
            ploidy=ploidy,
            rec_prob=rec_prob,
            seed=seed + i,
        )
        # print(ploidy, aploid, mat_hap1, pat_hap1)
        geno, baf = sim_b_allele_freq(
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
        res_table[f"zs_maternal{i}"] = zs_maternal
        res_table[f"zs_paternal{i}"] = zs_paternal
    return res_table
