from libc.math cimport erf, exp, expm1, log, log1p, pi, sqrt
import numpy as np


cdef double sqrt2 = sqrt(2.)
cdef double sqrt2pi = sqrt(2*pi)
cdef double logsqrt2pi = log(1/sqrt2pi)

cpdef double logsumexp(double[:] x):
    """Cython implementation of the logsumexp trick"""
    cdef int i, n
    cdef double m = -1e32
    cdef double c = 0.0
    n = x.size
    for i in range(n):
        m = max(m, x[i])
    for i in range(n):
        c += exp(x[i] - m)
    return m + log(c)

cdef double logdiffexp(double a, double b):
    """Log-sum-exp trick but for differences."""
    return log(exp(a) - exp(b) + 1e-124)

cpdef double logaddexp(double a, double b):
    cdef double m = -1e32
    cdef double c = 0.0
    m = max(a, b)
    c = exp(a - m) + exp(b - m)
    return m + log(c)

cpdef double logmeanexp(double a, double b):
    """Apply a logmeanexp routine for two numbers."""
    cdef double m = -1e32
    cdef double c = 0.0
    m = max(a, b)
    c = exp(a - m) + exp(b - m)
    return m + log(c) - 2.0

cdef double log1mexp(double a):
    """Log of 1 - e^-x."""
    if a < 0.693:
        return log(-expm1(-a))
    else:
        return log1p(-exp(-a))

cdef double psi(double x):
    """CDF for a normal distribution function in log-space."""
    if x < -4:
        return logsqrt2pi - 0.5*(x**2) - log(-x)
    else:
        return log((1.0 + erf(x / sqrt2))) - log(2.0)

cdef double logbinomial(int alt, int ref, double p):
    """Log-probability mass function of the binomial distribution."""
    return alt * log(p) + ref*log(1. - p)

cdef double norm_pdf(double x):
    """PDF for the normal distribution function in log-space.

    NOTE: generalize this to include mean-shift and stddev.
    """
    return logsqrt2pi - 0.5*(x**2)

cpdef double norm_logl(double x, double m, double s):
    """Normal log-likelihood function."""
    return logsqrt2pi - 0.5*log(s) - 0.5*((x - m) / s)**2

cpdef double truncnorm_pdf(double x, double a, double b, double mu=0.5, double sigma=0.2):
    """Custom definition of the log of the truncated normal pdf."""
    cdef double p, z, alpha, beta, eta
    beta = (b - mu) / sigma
    alpha = (a - mu) / sigma
    eta = (max(min(x, b), a) - mu) / sigma
    z = logdiffexp(psi(beta), psi(alpha))
    p = norm_pdf(eta) - log(sigma) - z
    return p

cpdef double mat_dosage(mat_hap, state):
    """Obtain the maternal dosage."""
    cdef int i, j, k
    cdef double m
    k = 0
    m = 0.
    j = len(state)
    for i in range(j):
        k += (state[i] >= 0)
    if k == 0:
        m = -1
    elif k == 1:
        if state[0] != -1:
            m = mat_hap[state[0]]
    elif k == 2:
        if state[1] != -1:
            m = mat_hap[state[0]] + mat_hap[state[1]]
        else:
            m = mat_hap[state[0]]
    elif k == 3:
        if state[1] != -1:
            m = mat_hap[state[0]] + mat_hap[state[1]]
        else:
            m = mat_hap[state[0]]
    return m

cpdef double pat_dosage(pat_hap, state):
    cdef int i, j, k
    cdef double p
    k = 0
    p = 0.
    j = len(state)
    for i in range(j):
        k += (state[i] >= 0)
    if k == 0:
        p = -1
    elif k == 1:
        if state[2] != -1:
            p = pat_hap[state[2]]
    elif k == 2:
        if state[3] != -1:
            p = pat_hap[state[2]] + pat_hap[state[3]]
        else:
            p = pat_hap[state[2]]
    elif k == 3:
        if state[3] != -1:
            p = pat_hap[state[2]] + pat_hap[state[3]]
        else:
            p = pat_hap[state[2]]
    return p

cpdef double emission_baf(double baf, double m, double p, double pi0=0.2, double std_dev=0.2, int k=2, double eps=1e-3):
    """Emission distribution function for B-allele frequency in the sample."""
    cdef double mu_i, x, x0, x1
    if (m == -1) & (p == -1):
        x = truncnorm_pdf(baf, 0.0, 1.0, mu=0.5, sigma=std_dev)
        return x
    mu_i = (m + p) / k
    x = truncnorm_pdf(baf, 0.0, 1.0, mu=mu_i, sigma=std_dev)
    x0 = truncnorm_pdf(baf, 0.0, 1.0, mu=0.0, sigma=2e-3)
    x1 = truncnorm_pdf(baf, 0.0, 1.0, mu=1.0, sigma=2e-3)
    if mu_i == 0.0:
        return logaddexp(log(pi0) + x0, log((1 - pi0)) + x)
    if mu_i == 1.0:
        return logaddexp(log(pi0) + x1, log((1 - pi0)) + x)
    else:
        return x

cpdef double emission_baf_parent_err(
    double baf,
    mat_hap_i,
    pat_hap_i,
    state,
    double pi0=0.2,
    double std_dev=0.2,
    int k=2,
    double mat_err=0.0,
    double pat_err=0.0,
):
    """Emission probability marginalizing over parental genotype errors.

    For each parent, enumerates alternative genotypes weighted by the error
    probability and marginalizes.  Called het → alts are both homs (weight
    mat_err/2 each); called hom → alt is the het keeping the called allele in
    position 0 (weight mat_err).  Both parents are treated independently.
    """
    cdef int mat_h0, mat_h1, pat_h0, pat_h1
    cdef double m_ij, p_ij

    mat_h0 = int(mat_hap_i[0])
    mat_h1 = int(mat_hap_i[1])
    pat_h0 = int(pat_hap_i[0])
    pat_h1 = int(pat_hap_i[1])

    mat_alts = [[mat_h0, mat_h1]]
    mat_weights = [1.0 - mat_err]
    if mat_h0 != mat_h1:
        mat_alts.append([0, 0])
        mat_weights.append(mat_err / 2.0)
        mat_alts.append([1, 1])
        mat_weights.append(mat_err / 2.0)
    else:
        mat_alts.append([mat_h0, 1 - mat_h0])
        mat_weights.append(mat_err)

    pat_alts = [[pat_h0, pat_h1]]
    pat_weights = [1.0 - pat_err]
    if pat_h0 != pat_h1:
        pat_alts.append([0, 0])
        pat_weights.append(pat_err / 2.0)
        pat_alts.append([1, 1])
        pat_weights.append(pat_err / 2.0)
    else:
        pat_alts.append([pat_h0, 1 - pat_h0])
        pat_weights.append(pat_err)

    components = []
    for mat_g, w_m in zip(mat_alts, mat_weights):
        for pat_g, w_p in zip(pat_alts, pat_weights):
            m_ij = mat_dosage(mat_g, state)
            p_ij = pat_dosage(pat_g, state)
            components.append(
                log(w_m) + log(w_p) + emission_baf(baf, m_ij, p_ij, pi0=pi0, std_dev=std_dev, k=k)
            )
    return logsumexp(np.array(components, dtype=np.float64))


cpdef double emission_baf_sibs_parent_err(
    double baf0,
    double baf1,
    mat_hap_i,
    pat_hap_i,
    state0,
    state1,
    double pi0_0=0.2,
    double pi0_1=0.2,
    double std_dev_0=0.2,
    double std_dev_1=0.2,
    double mat_err=0.0,
    double pat_err=0.0,
):
    """Joint sibling emission marginalizing over shared parental genotype errors.

    Both siblings observe the SAME parental genotype, so the error is applied
    jointly rather than independently per sibling.
    """
    cdef int mat_h0, mat_h1, pat_h0, pat_h1
    cdef double m_ij0, p_ij0, m_ij1, p_ij1

    mat_h0 = int(mat_hap_i[0])
    mat_h1 = int(mat_hap_i[1])
    pat_h0 = int(pat_hap_i[0])
    pat_h1 = int(pat_hap_i[1])

    mat_alts = [[mat_h0, mat_h1]]
    mat_weights = [1.0 - mat_err]
    if mat_h0 != mat_h1:
        mat_alts.append([0, 0])
        mat_weights.append(mat_err / 2.0)
        mat_alts.append([1, 1])
        mat_weights.append(mat_err / 2.0)
    else:
        mat_alts.append([mat_h0, 1 - mat_h0])
        mat_weights.append(mat_err)

    pat_alts = [[pat_h0, pat_h1]]
    pat_weights = [1.0 - pat_err]
    if pat_h0 != pat_h1:
        pat_alts.append([0, 0])
        pat_weights.append(pat_err / 2.0)
        pat_alts.append([1, 1])
        pat_weights.append(pat_err / 2.0)
    else:
        pat_alts.append([pat_h0, 1 - pat_h0])
        pat_weights.append(pat_err)

    components = []
    for mat_g, w_m in zip(mat_alts, mat_weights):
        for pat_g, w_p in zip(pat_alts, pat_weights):
            m_ij0 = mat_dosage(mat_g, state0)
            p_ij0 = pat_dosage(pat_g, state0)
            m_ij1 = mat_dosage(mat_g, state1)
            p_ij1 = pat_dosage(pat_g, state1)
            components.append(
                log(w_m) + log(w_p)
                + emission_baf(baf0, m_ij0, p_ij0, pi0=pi0_0, std_dev=std_dev_0, k=2)
                + emission_baf(baf1, m_ij1, p_ij1, pi0=pi0_1, std_dev=std_dev_1, k=2)
            )
    return logsumexp(np.array(components, dtype=np.float64))


cpdef double emission_lrr(double lrr, int k=2, double std_dev=0.2):
    """
    Emission distribution for LRR conditional on ploidy.

    NOTE: missing LRR is denoted as -9
    """
    mu_i = {0: -3, 1: -1, 2: 0, 3: np.log2(1.5)}
    if lrr == -9:
        return -1
    else:
        return norm_logl(lrr, mu_i[k], std_dev)


cpdef double loglik_mcc(double baf, int mg, int pg, double std_dev=0.1, double c=0.1):
    """Log-likelihood of bafs under maternal cell contamination in the full-conditional model."""
    assert std_dev > 0.0
    assert (c >= 0) and (c <= 0.5)
    if (pg == 0):
        if (mg == 0):
            return truncnorm_pdf(baf, 0.0, 1.0, mu=0.0, sigma=std_dev)
        if (mg == 1):
            return logaddexp(np.log(0.5)+truncnorm_pdf(baf, 0.0, 1.0, mu=0.0, sigma=std_dev), np.log(0.5)+truncnorm_pdf(baf, 0.0, 1.0, mu=0.5+(c/2), sigma=std_dev))
        if (mg == 2):
            return truncnorm_pdf(baf, 0.0, 1.0, mu=(0.5+c), sigma=std_dev)
    elif (pg == 1):
        if (mg == 0):
            return logaddexp(np.log(0.5)+truncnorm_pdf(baf, 0.0, 1.0, mu=0.0, sigma=std_dev), np.log(0.5)+truncnorm_pdf(baf, 0.0, 1.0, mu=0.5-(c/2), sigma=std_dev))
        if (mg == 1):
            return logaddexp(logaddexp(np.log(0.25)+truncnorm_pdf(baf, 0.0, 1.0, mu=0.0+c/2, sigma=std_dev), np.log(0.25)+truncnorm_pdf(baf, 0.0, 1.0, mu=0.5-(c/2), sigma=std_dev)), np.log(0.5) + truncnorm_pdf(baf, 0.0, 1.0, mu=0.5, sigma=std_dev))
        if (mg == 2):
            return logaddexp(np.log(0.5)+truncnorm_pdf(baf, 0.0, 1.0, mu=1.0, sigma=std_dev), np.log(0.5)+truncnorm_pdf(baf, 0.0, 1.0, mu=0.5+c, sigma=std_dev))
    if (pg == 2):
        if (mg == 0):
            return truncnorm_pdf(baf, 0.0, 1.0, mu=(0.5-c), sigma=std_dev)
        if (mg == 1):
            return logaddexp(np.log(0.5)+truncnorm_pdf(baf, 0.0, 1.0, mu=1.0-c/2, sigma=std_dev), np.log(0.5)+truncnorm_pdf(baf, 0.0, 1.0, mu=0.5+c/2, sigma=std_dev))
        if (mg == 2):
            return truncnorm_pdf(baf, 0.0, 1.0, mu=1.0, sigma=std_dev)
    return 0.0

cpdef double emission_readcounts(int alt, int ref, double m, double p, int k=2, double eps=1e-6):
    """Emission distribution for read counts at specific SNVs.

    NOTE: currently this only includes the emission for the balance between alleles vs. total coverage
    """
    cdef double mu_i
    if (m == -1) & (p == -1):
        # This is the nullisomy case for allelic balance
        return logbinomial(alt, ref, p=eps)
    mu_i = (m + p) / k
    return logbinomial(alt, ref, p=mu_i)

cpdef double mix_loglik(double[:] bafs, double pi0=0.5, double theta=0.1, double std_dev=0.2):
    """Mixture log-likelihood for expected heterozygotes to estimate baf-deviation."""
    cdef double logll = 0.0
    cdef int i, n
    cdef double ll[3]
    n = bafs.size
    for i in range(n):
        ll[0] = log(pi0/2.0) + truncnorm_pdf(bafs[i], 0.0, 1.0, mu=0.5+theta, sigma=std_dev)
        ll[1] = log(pi0/2.0) + truncnorm_pdf(bafs[i], 0.0, 1.0, mu=0.5-theta, sigma=std_dev)
        ll[2] = log(1.0 - pi0) + truncnorm_pdf(bafs[i], 0.0, 1.0, mu=0.5, sigma=std_dev)
        logll += logsumexp(ll)
    return logll

cpdef marginal_mix_loglik(double[:] bafs, double[:] mus, double pi0=0.5, double std_dev=0.2):
    """Estimate parameters under the marginal basis."""
    cdef double logll = 0.0
    cdef int i, j, k, m, n
    n = bafs.size
    m = mus.size
    k = 0
    for j in range(m):
        if (mus[j] == 0.0) or (mus[j] == 1.0):
            k += 1
    for i in range(n):
        ll = np.zeros(mus.size)
        for j in range(m):
            if (mus[j] == 0) or (mus[j] == 1.0):
                ll[j] = log(pi0/k) + truncnorm_pdf(bafs[i], 0.0, 1.0, mu=mus[j], sigma=std_dev)
            else:
                ll[j] = log(1.0 - pi0) + truncnorm_pdf(bafs[i], 0.0, 1.0, mu=mus[j], sigma=std_dev)
        logll += logsumexp(ll)
    return logll


def lod_phase(haps1, haps2, baf, **kwargs):
    """Estimate the log-likelihood of being the phase vs. antiphase orientation for heterozygotes."""
    cdef int i, j
    cdef float phase_orientation, antiphase_orientation
    phase_orientation = 0.0
    antiphase_orientation = 0.0
    # Marginalize over all phase
    # NOTE: for speed include this as a helper function in the utils
    for i in range(2):
        for j in range(2):
            # Compute the phase orientations
            phase0 = emission_baf(
                baf=baf[0], m=haps1[0, 0], p=haps2[i, 0], **kwargs
            ) + emission_baf(baf=baf[1], m=haps1[0, 1], p=haps2[j, 1], **kwargs)
            phase1 = emission_baf(
                baf=baf[0], m=haps1[1, 0], p=haps2[i, 0], **kwargs
            ) + emission_baf(baf=baf[1], m=haps1[1, 1], p=haps2[j, 1], **kwargs)
            # Compute the antiphase orientations
            antiphase0 = emission_baf(
                baf=baf[0], m=haps1[0, 0], p=haps2[i, 0], **kwargs
            ) + emission_baf(baf=baf[1], m=haps1[1, 1], p=haps2[j, 1], **kwargs)
            antiphase1 = emission_baf(
                baf=baf[0], m=haps1[1, 0], p=haps2[i, 0], **kwargs
            ) + emission_baf(baf=baf[1], m=haps1[0, 1], p=haps2[j, 1], **kwargs)
            phase_orientation = logaddexp(phase_orientation, phase0)
            phase_orientation = logaddexp(phase_orientation, phase1)
            antiphase_orientation = logaddexp(antiphase_orientation, antiphase0)
            antiphase_orientation = logaddexp(antiphase_orientation, antiphase1)
    return phase_orientation, antiphase_orientation


def create_index_arrays(karyotypes):
    """Create index-arrays for within and between karyotype class."""
    m = karyotypes.size
    K0 = np.zeros(shape=(m, m))
    K1 = np.zeros(shape=(m, m))
    ks = np.zeros(m)
    for i in range(m):
        ks[i] = np.sum(karyotypes == karyotypes[i])
        for j in range(m):
            if i != j:
                if karyotypes[i] == karyotypes[j]:
                    K0[i, j] = 1./ks[i]
                else:
                    K1[i, j] = 1./(m-ks[i])
    return K0, K1


cpdef transition_kernel(K0, K1, double d=1e3, double r=1e-8, double a=1e-2, int unphased=0):
    """Estimate distance-dependent transition probabilities."""
    cdef int i, m
    cdef double rho, alpha
    m = K0.shape[0]
    rho = 1.0 - exp(-r*d)
    alpha = 1.0 - exp(-r*a*d)
    if unphased == 0:
        A = K0*rho + K1*alpha
        for i in range(m):
            A[i, i] = 1. - np.sum(A[i, :])
    else:
        A = K0 + K1*alpha
        for i in range(m):
            A[i, i] = 1. - np.sum(A[i, :])
    return np.log(A)


def forward_algo(bafs, lrrs, sigmas, pos, mat_haps, pat_haps, states, karyotypes, double r=1e-8, double a=1e-2, double pi0=0.2, double std_dev=0.25, int unphased=0, double mat_err=0.0, double pat_err=0.0):
    """Helper function for forward algorithm loop-optimization."""
    cdef int i, j, n, m
    cdef float di
    n = bafs.size
    m = len(states)
    ks = [sum([s >= 0 for s in state]) for state in states]
    K0, K1 = create_index_arrays(karyotypes)
    alphas = np.zeros(shape=(m, n))
    alphas[:, 0] = log(1.0 / m)
    for j in range(m):
        if mat_err > 0.0 or pat_err > 0.0:
            cur_emission = emission_baf_parent_err(
                bafs[0], mat_haps[:, 0], pat_haps[:, 0], states[j],
                pi0=pi0, std_dev=std_dev, k=ks[j], mat_err=mat_err, pat_err=pat_err,
            ) + emission_lrr(lrrs[0], k=ks[j], std_dev=sigmas[0])
        else:
            m_ij = mat_dosage(mat_haps[:, 0], states[j])
            p_ij = pat_dosage(pat_haps[:, 0], states[j])
            cur_emission = emission_baf(
                    bafs[0], m_ij, p_ij, pi0=pi0, std_dev=std_dev, k=ks[j],
                ) + emission_lrr(lrrs[0], k=ks[j], std_dev=sigmas[0])
        alphas[j, 0] += cur_emission
    scaler = np.zeros(n)
    scaler[0] = logsumexp(alphas[:, 0])
    alphas[:, 0] -= scaler[0]
    for i in range(1, n):
        di = pos[i] - pos[i-1]
        A_hat = transition_kernel(K0, K1, d=di, r=r, a=a, unphased=unphased)
        for j in range(m):
            if mat_err > 0.0 or pat_err > 0.0:
                cur_emission = emission_baf_parent_err(
                    bafs[i], mat_haps[:, i], pat_haps[:, i], states[j],
                    pi0=pi0, std_dev=std_dev, k=ks[j], mat_err=mat_err, pat_err=pat_err,
                ) + emission_lrr(lrrs[i], k=ks[j], std_dev=sigmas[i])
            else:
                m_ij = mat_dosage(mat_haps[:, i], states[j])
                p_ij = pat_dosage(pat_haps[:, i], states[j])
                cur_emission = emission_baf(
                        bafs[i], m_ij, p_ij, pi0=pi0, std_dev=std_dev, k=ks[j],
                    ) + emission_lrr(lrrs[i], k=ks[j], std_dev=sigmas[i])
            alphas[j, i] = cur_emission + logsumexp(A_hat[:, j] + alphas[:, (i - 1)])
        scaler[i] = logsumexp(alphas[:, i])
        alphas[:, i] -= scaler[i]
    return alphas, scaler, states, None, sum(scaler)


def backward_algo(bafs, lrrs, sigmas, pos, mat_haps, pat_haps, states, karyotypes, double r=1e-8, double a=1e-2, double pi0=0.2, double std_dev=0.25, int unphased=0, double mat_err=0.0, double pat_err=0.0):
    """Helper function for backward algorithm loop-optimization."""
    cdef int i, j, n, m
    cdef float di
    n = bafs.size
    m = len(states)
    ks = [sum([s >= 0 for s in state]) for state in states]
    K0, K1 = create_index_arrays(karyotypes)
    betas = np.zeros(shape=(m, n))
    betas[:, -1] = log(1)
    scaler = np.zeros(n)
    scaler[-1] = logsumexp(betas[:, -1])
    betas[:, -1] -= scaler[-1]
    for i in range(n - 2, -1, -1):
        di = pos[i+1] - pos[i]
        A_hat = transition_kernel(K0, K1, d=di, r=r, a=a, unphased=unphased)
        cur_emissions = np.zeros(m)
        for j in range(m):
            if mat_err > 0.0 or pat_err > 0.0:
                cur_emissions[j] = emission_baf_parent_err(
                    bafs[i + 1], mat_haps[:, i+1], pat_haps[:, i+1], states[j],
                    pi0=pi0, std_dev=std_dev, k=ks[j], mat_err=mat_err, pat_err=pat_err,
                ) + emission_lrr(lrrs[i+1], k=ks[j], std_dev=sigmas[i+1])
            else:
                m_ij = mat_dosage(mat_haps[:, i+1], states[j])
                p_ij = pat_dosage(pat_haps[:, i+1], states[j])
                cur_emissions[j] = emission_baf(
                        bafs[i + 1], m_ij, p_ij, pi0=pi0, std_dev=std_dev, k=ks[j],
                    ) + emission_lrr(lrrs[i+1], k=ks[j], std_dev=sigmas[i+1])
        for j in range(m):
            betas[j, i] = logsumexp(A_hat[:, j] + cur_emissions + betas[:, (i + 1)])
        if i == 0:
            for j in range(m):
                if mat_err > 0.0 or pat_err > 0.0:
                    cur_emission = emission_baf_parent_err(
                        bafs[i], mat_haps[:, i], pat_haps[:, i], states[j],
                        pi0=pi0, std_dev=std_dev, k=ks[j], mat_err=mat_err, pat_err=pat_err,
                    ) + emission_lrr(lrrs[i], k=ks[j], std_dev=sigmas[i])
                else:
                    m_ij = mat_dosage(mat_haps[:, i], states[j])
                    p_ij = pat_dosage(pat_haps[:, i], states[j])
                    cur_emission = emission_baf(
                            bafs[i], m_ij, p_ij, pi0=pi0, std_dev=std_dev, k=ks[j],
                        ) + emission_lrr(lrrs[i], k=ks[j], std_dev=sigmas[i])
                betas[j, i] += log(1/m) + cur_emission
        scaler[i] = logsumexp(betas[:, i])
        betas[:, i] -= scaler[i]
    return betas, scaler, states, None, sum(scaler)


def viterbi_algo(bafs, lrrs, sigmas, pos, mat_haps, pat_haps, states, karyotypes, double r=1e-8, double a=1e-2, double pi0=0.2, double std_dev=0.25, int unphased=0, double mat_err=0.0, double pat_err=0.0):
    """Cython implementation of the Viterbi algorithm for MLE path estimation through states."""
    cdef int i, j, n, m
    cdef float di
    n = bafs.size
    m = len(states)
    deltas = np.zeros(shape=(m, n))
    deltas[:, 0] = log(1.0 / m)
    psi = np.zeros(shape=(m, n), dtype=int)
    ks = [sum([s >= 0 for s in state]) for state in states]
    K0, K1 = create_index_arrays(karyotypes)
    for i in range(1, n):
        di = pos[i] - pos[i-1]
        A_hat = transition_kernel(K0, K1, d=di, r=r, a=a, unphased=unphased)
        for j in range(m):
            deltas[j, i] = np.max(deltas[:, i-1] + A_hat[:, j])
            if mat_err > 0.0 or pat_err > 0.0:
                deltas[j, i] += emission_baf_parent_err(
                    bafs[i], mat_haps[:, i], pat_haps[:, i], states[j],
                    pi0=pi0, std_dev=std_dev, k=ks[j], mat_err=mat_err, pat_err=pat_err,
                ) + emission_lrr(lrrs[i], k=ks[j], std_dev=sigmas[i])
            else:
                m_ij = mat_dosage(mat_haps[:, i], states[j])
                p_ij = pat_dosage(pat_haps[:, i], states[j])
                deltas[j, i] += emission_baf(
                        bafs[i], m_ij, p_ij, pi0=pi0, std_dev=std_dev, k=ks[j],
                    ) + emission_lrr(lrrs[i], k=ks[j], std_dev=sigmas[i])
            psi[j, i] = np.argmax(deltas[:, i - 1] + A_hat[:, j]).astype(int)
    path = np.zeros(n, dtype=int)
    path[-1] = np.argmax(deltas[:, -1]).astype(int)
    for i in range(n - 2, -1, -1):
        path[i] = psi[path[i + 1], i]
    path[0] = psi[path[1], 1]
    return path, states, deltas, psi


def forward_algo_sibs(bafs, pos, mat_haps, pat_haps, states, karyotypes, double r=1e-8, double a=1e-2, (double, double) pi0=(0.2, 0.2), (double, double) std_dev=(0.1, 0.1), double mat_err=0.0, double pat_err=0.0):
    """Compute the forward algorithm for sibling embryo HMM."""
    cdef int i, j, n, m
    cdef float di
    assert len(bafs) == 2
    assert bafs[1].size == bafs[0].size
    n = bafs[0].size
    m = len(states)
    K0, K1 = create_index_arrays(karyotypes)
    alphas = np.zeros(shape=(m, n))
    alphas[:, 0] = log(1.0 / m)
    for j in range(m):
        if mat_err > 0.0 or pat_err > 0.0:
            cur_emission = emission_baf_sibs_parent_err(
                bafs[0][0], bafs[1][0],
                mat_haps[:, 0], pat_haps[:, 0],
                states[j][0], states[j][1],
                pi0_0=pi0[0], pi0_1=pi0[1],
                std_dev_0=std_dev[0], std_dev_1=std_dev[1],
                mat_err=mat_err, pat_err=pat_err,
            )
        else:
            m_ij0 = mat_dosage(mat_haps[:, 0], states[j][0])
            p_ij0 = pat_dosage(pat_haps[:, 0], states[j][0])
            m_ij1 = mat_dosage(mat_haps[:, 0], states[j][1])
            p_ij1 = pat_dosage(pat_haps[:, 0], states[j][1])
            cur_emission = emission_baf(
                    bafs[0][0], m_ij0, p_ij0, pi0=pi0[0], std_dev=std_dev[0], k=2,
                ) + emission_baf(
                    bafs[1][0], m_ij1, p_ij1, pi0=pi0[1], std_dev=std_dev[1], k=2,
                )
        alphas[j, 0] += cur_emission
    scaler = np.zeros(n)
    scaler[0] = logsumexp(alphas[:, 0])
    alphas[:, 0] -= scaler[0]
    for i in range(1, n):
        di = pos[i] - pos[i-1]
        A_hat = transition_kernel(K0, K1, d=di, r=r, a=a)
        for j in range(m):
            if mat_err > 0.0 or pat_err > 0.0:
                cur_emission = emission_baf_sibs_parent_err(
                    bafs[0][i], bafs[1][i],
                    mat_haps[:, i], pat_haps[:, i],
                    states[j][0], states[j][1],
                    pi0_0=pi0[0], pi0_1=pi0[1],
                    std_dev_0=std_dev[0], std_dev_1=std_dev[1],
                    mat_err=mat_err, pat_err=pat_err,
                )
            else:
                m_ij0 = mat_dosage(mat_haps[:, i], states[j][0])
                p_ij0 = pat_dosage(pat_haps[:, i], states[j][0])
                m_ij1 = mat_dosage(mat_haps[:, i], states[j][1])
                p_ij1 = pat_dosage(pat_haps[:, i], states[j][1])
                cur_emission = emission_baf(
                        bafs[0][i], m_ij0, p_ij0, pi0=pi0[0], std_dev=std_dev[0], k=2,
                    ) + emission_baf(
                        bafs[1][i], m_ij1, p_ij1, pi0=pi0[1], std_dev=std_dev[1], k=2,
                    )
            alphas[j, i] = cur_emission + logsumexp(A_hat[:, j] + alphas[:, i - 1])
        scaler[i] = logsumexp(alphas[:, i])
        alphas[:, i] -= scaler[i]
    return alphas, scaler, states, None, sum(scaler)


def backward_algo_sibs(bafs, pos, mat_haps, pat_haps, states, karyotypes, double r=1e-8, double a=1e-2, (double, double) pi0=(0.2, 0.2), (double, double) std_dev=(0.1, 0.1), double mat_err=0.0, double pat_err=0.0):
    """Compute the backward algorithm for the sibling embryo HMM."""
    cdef int i, j, n, m
    cdef float di
    assert len(bafs) == 2
    assert bafs[1].size == bafs[0].size
    n = bafs[0].size
    m = len(states)
    K0, K1 = create_index_arrays(karyotypes)
    betas = np.zeros(shape=(m, n))
    betas[:, -1] = log(1.0)
    scaler = np.zeros(n)
    scaler[-1] = logsumexp(betas[:, -1])
    betas[:, -1] -= scaler[-1]
    for i in range(n - 2, -1, -1):
        di = min(pos[i+1] - pos[i], 1e6)
        A_hat = transition_kernel(K0, K1, d=di, r=r, a=a)
        cur_emissions = np.zeros(m)
        for j in range(m):
            if mat_err > 0.0 or pat_err > 0.0:
                cur_emissions[j] = emission_baf_sibs_parent_err(
                    bafs[0][i + 1], bafs[1][i + 1],
                    mat_haps[:, i+1], pat_haps[:, i+1],
                    states[j][0], states[j][1],
                    pi0_0=pi0[0], pi0_1=pi0[1],
                    std_dev_0=std_dev[0], std_dev_1=std_dev[1],
                    mat_err=mat_err, pat_err=pat_err,
                )
            else:
                m_ij0 = mat_dosage(mat_haps[:, i+1], states[j][0])
                p_ij0 = pat_dosage(pat_haps[:, i+1], states[j][0])
                m_ij1 = mat_dosage(mat_haps[:, i+1], states[j][1])
                p_ij1 = pat_dosage(pat_haps[:, i+1], states[j][1])
                cur_emissions[j] = emission_baf(
                        bafs[0][i + 1], m_ij0, p_ij0, pi0=pi0[0], std_dev=std_dev[0], k=2,
                    ) + emission_baf(
                        bafs[1][i + 1], m_ij1, p_ij1, pi0=pi0[1], std_dev=std_dev[1], k=2,
                    )
        for j in range(m):
            betas[j, i] = logsumexp(A_hat[:, j] + cur_emissions + betas[:, (i + 1)])
        if i == 0:
            for j in range(m):
                if mat_err > 0.0 or pat_err > 0.0:
                    cur_emission = emission_baf_sibs_parent_err(
                        bafs[0][i], bafs[1][i],
                        mat_haps[:, i], pat_haps[:, i],
                        states[j][0], states[j][1],
                        pi0_0=pi0[0], pi0_1=pi0[1],
                        std_dev_0=std_dev[0], std_dev_1=std_dev[1],
                        mat_err=mat_err, pat_err=pat_err,
                    )
                else:
                    m_ij0 = mat_dosage(mat_haps[:, i], states[j][0])
                    p_ij0 = pat_dosage(pat_haps[:, i], states[j][0])
                    m_ij1 = mat_dosage(mat_haps[:, i], states[j][1])
                    p_ij1 = pat_dosage(pat_haps[:, i], states[j][1])
                    cur_emission = emission_baf(
                            bafs[0][i], m_ij0, p_ij0, pi0=pi0[0], std_dev=std_dev[0], k=2,
                        ) + emission_baf(
                            bafs[1][i], m_ij1, p_ij1, pi0=pi0[1], std_dev=std_dev[1], k=2,
                        )
                betas[j, i] += log(1/m) + cur_emission
        scaler[i] = logsumexp(betas[:, i])
        betas[:, i] -= scaler[i]
    return betas, scaler, states, None, sum(scaler)


def viterbi_algo_sibs(bafs, pos, mat_haps, pat_haps, states, karyotypes, double r=1e-8, double a=1e-2, (double, double) pi0=(0.2, 0.2), (double, double) std_dev=(0.1, 0.1), double mat_err=0.0, double pat_err=0.0):
    """Viterbi algorithm and path tracing through sibling embryos."""
    cdef int i, j, n, m
    cdef float di
    assert len(bafs) == 2
    assert bafs[1].size == bafs[0].size
    n = bafs[0].size
    m = len(states)
    K0, K1 = create_index_arrays(karyotypes)
    deltas = np.zeros(shape=(m, n))
    deltas[:, 0] = log(1.0 / m)
    psi = np.zeros(shape=(m, n), dtype=int)
    for i in range(1, n):
        di = pos[i] - pos[i-1]
        A_hat = transition_kernel(K0, K1, d=di, r=r, a=a)
        for j in range(m):
            deltas[j, i] = np.max(deltas[:, i-1] + A_hat[:, j])
            if mat_err > 0.0 or pat_err > 0.0:
                deltas[j, i] += emission_baf_sibs_parent_err(
                    bafs[0][i], bafs[1][i],
                    mat_haps[:, i], pat_haps[:, i],
                    states[j][0], states[j][1],
                    pi0_0=pi0[0], pi0_1=pi0[1],
                    std_dev_0=std_dev[0], std_dev_1=std_dev[1],
                    mat_err=mat_err, pat_err=pat_err,
                )
            else:
                m_ij0 = mat_dosage(mat_haps[:, i], states[j][0])
                p_ij0 = pat_dosage(pat_haps[:, i], states[j][0])
                m_ij1 = mat_dosage(mat_haps[:, i], states[j][1])
                p_ij1 = pat_dosage(pat_haps[:, i], states[j][1])
                deltas[j, i] += emission_baf(
                        bafs[0][i], m_ij0, p_ij0, pi0=pi0[0], std_dev=std_dev[0], k=2,
                    ) + emission_baf(
                        bafs[1][i], m_ij1, p_ij1, pi0=pi0[1], std_dev=std_dev[1], k=2,
                    )
            psi[j, i] = np.argmax(deltas[:, i - 1] + A_hat[:, j]).astype(int)
    path = np.zeros(n, dtype=int)
    path[-1] = np.argmax(deltas[:, -1]).astype(int)
    for i in range(n - 2, -1, -1):
        path[i] = psi[path[i + 1], i]
    path[0] = psi[path[1], 1]
    return path, states, deltas, psi


def forward_algo_duo(bafs, lrrs, sigmas, pos, haps, freqs, states, karyotypes, bint maternal=True, double r=1e-8, double a=1e-2, double pi0=0.8, double std_dev=0.2, double obs_err=0.0):
    """Helper function for optimization for forward algorithm in the duo setting."""
    cdef int i, j, idx, n, m
    cdef float di, f
    cdef int obs_h0, obs_h1
    n = bafs.size
    m = len(states)
    ks = [sum([s >= 0 for s in state]) for state in states]
    K0, K1 = create_index_arrays(karyotypes)
    alphas = np.zeros(shape=(m, n))
    alphas[:, 0] = log(1.0 / m)
    # NOTE: this restricts the phase orientation of the unobserved het haplotype ...
    geno = [[0, 0], [1, 0], [1, 1]]
    for j in range(m):
        f = freqs[0]
        if f < 0:
            geno_freq = (1/3, 1/3, 1/3)
        else:
            geno_freq = ((1 - f)**2, 2*f*(1-f), f**2)
        if obs_err > 0.0:
            obs_h0 = int(haps[0, 0])
            obs_h1 = int(haps[1, 0])
            obs_alts = [[obs_h0, obs_h1]]
            obs_weights = [1.0 - obs_err]
            if obs_h0 != obs_h1:
                obs_alts.append([0, 0])
                obs_weights.append(obs_err / 2.0)
                obs_alts.append([1, 1])
                obs_weights.append(obs_err / 2.0)
            else:
                obs_alts.append([obs_h0, 1 - obs_h0])
                obs_weights.append(obs_err)
            components = []
            for obs_g, w_o in zip(obs_alts, obs_weights):
                for x, p in zip(geno, geno_freq):
                    if maternal:
                        m_ij = mat_dosage(obs_g, states[j])
                        p_ij = pat_dosage(x, states[j])
                    else:
                        m_ij = mat_dosage(x, states[j])
                        p_ij = pat_dosage(obs_g, states[j])
                    components.append(
                        emission_baf(bafs[0], m_ij, p_ij, pi0=pi0, std_dev=std_dev, k=ks[j])
                        + emission_lrr(lrrs[0], k=ks[j], std_dev=sigmas[0]) + log(w_o) + log(p)
                    )
            alphas[j, 0] = logsumexp(np.array(components, dtype=np.float64))
        else:
            cur_emission = np.zeros(3)
            for idx, (x, p) in enumerate(zip(geno, geno_freq)):
                if maternal:
                    m_ij = mat_dosage(haps[:, 0], states[j])
                    p_ij = pat_dosage(x, states[j])
                else:
                    m_ij = mat_dosage(x, states[j])
                    p_ij = pat_dosage(haps[:, 0], states[j])
                cur_emission[idx] = emission_baf(
                        bafs[0], m_ij, p_ij, pi0=pi0, std_dev=std_dev, k=ks[j],
                    ) + emission_lrr(lrrs[0], k=ks[j], std_dev=sigmas[0]) + log(p)
            alphas[j, 0] = logsumexp(cur_emission)
    scaler = np.zeros(n)
    scaler[0] = logsumexp(alphas[:, 0])
    alphas[:, 0] -= scaler[0]
    for i in range(1, n):
        di = pos[i] - pos[i-1]
        A_hat = transition_kernel(K0, K1, d=di, r=r, a=a)
        for j in range(m):
            f = freqs[i]
            if f < 0:
                geno_freq = (1/3, 1/3, 1/3)
            else:
                geno_freq = ((1 - f)**2, 2*f*(1-f), f**2)
            if obs_err > 0.0:
                obs_h0 = int(haps[0, i])
                obs_h1 = int(haps[1, i])
                obs_alts = [[obs_h0, obs_h1]]
                obs_weights = [1.0 - obs_err]
                if obs_h0 != obs_h1:
                    obs_alts.append([0, 0])
                    obs_weights.append(obs_err / 2.0)
                    obs_alts.append([1, 1])
                    obs_weights.append(obs_err / 2.0)
                else:
                    obs_alts.append([obs_h0, 1 - obs_h0])
                    obs_weights.append(obs_err)
                components = []
                for obs_g, w_o in zip(obs_alts, obs_weights):
                    for x, p in zip(geno, geno_freq):
                        if maternal:
                            m_ij = mat_dosage(obs_g, states[j])
                            p_ij = pat_dosage(x, states[j])
                        else:
                            m_ij = mat_dosage(x, states[j])
                            p_ij = pat_dosage(obs_g, states[j])
                        components.append(
                            emission_baf(bafs[i], m_ij, p_ij, pi0=pi0, std_dev=std_dev, k=ks[j])
                            + emission_lrr(lrrs[i], k=ks[j], std_dev=sigmas[i]) + log(w_o) + log(p)
                        )
                alphas[j, i] = logsumexp(np.array(components, dtype=np.float64)) + logsumexp(A_hat[:, j] + alphas[:, (i - 1)])
            else:
                cur_emission = np.zeros(3)
                for idx, (x, p) in enumerate(zip(geno, geno_freq)):
                    if maternal:
                        m_ij = mat_dosage(haps[:, i], states[j])
                        p_ij = pat_dosage(x, states[j])
                    else:
                        m_ij = mat_dosage(x, states[j])
                        p_ij = pat_dosage(haps[:, i], states[j])
                    cur_emission[idx] = emission_baf(
                            bafs[i], m_ij, p_ij, pi0=pi0, std_dev=std_dev, k=ks[j],
                        ) + emission_lrr(lrrs[i], k=ks[j], std_dev=sigmas[i]) + log(p)
                alphas[j, i] = logsumexp(cur_emission) + logsumexp(A_hat[:, j] + alphas[:, (i - 1)])
        scaler[i] = logsumexp(alphas[:, i])
        alphas[:, i] -= scaler[i]
    return alphas, scaler, states, None, sum(scaler)


def backward_algo_duo(bafs, lrrs, sigmas, pos, haps, freqs, states, karyotypes, bint maternal=True, double r=1e-8, double a=1e-2, double pi0=0.2, double std_dev=0.25, double obs_err=0.0):
    """Helper function for backward algorithm loop-optimization."""
    cdef int i, j, idx, n, m
    cdef float di, f, p
    cdef int obs_h0, obs_h1
    n = bafs.size
    m = len(states)
    ks = [sum([s >= 0 for s in state]) for state in states]
    K0, K1 = create_index_arrays(karyotypes)
    betas = np.zeros(shape=(m, n))
    betas[:, -1] = log(1)
    scaler = np.zeros(n)
    scaler[-1] = logsumexp(betas[:, -1])
    betas[:, -1] -= scaler[-1]
    geno = [[0, 0], [1, 0], [1, 1]]
    for i in range(n - 2, -1, -1):
        f = freqs[i]
        if f < 0:
            geno_freq = (1/3, 1/3, 1/3)
        else:
            geno_freq = ((1 - f)**2, 2*f*(1-f), f**2)
        di = pos[i+1] - pos[i]
        A_hat = transition_kernel(K0, K1, d=di, r=r, a=a)
        cur_emissions = np.zeros(m)
        for j in range(m):
            if obs_err > 0.0:
                obs_h0 = int(haps[0, i+1])
                obs_h1 = int(haps[1, i+1])
                obs_alts = [[obs_h0, obs_h1]]
                obs_weights = [1.0 - obs_err]
                if obs_h0 != obs_h1:
                    obs_alts.append([0, 0])
                    obs_weights.append(obs_err / 2.0)
                    obs_alts.append([1, 1])
                    obs_weights.append(obs_err / 2.0)
                else:
                    obs_alts.append([obs_h0, 1 - obs_h0])
                    obs_weights.append(obs_err)
                components = []
                for obs_g, w_o in zip(obs_alts, obs_weights):
                    for x, p in zip(geno, geno_freq):
                        if maternal:
                            m_ij = mat_dosage(obs_g, states[j])
                            p_ij = pat_dosage(x, states[j])
                        else:
                            m_ij = mat_dosage(x, states[j])
                            p_ij = pat_dosage(obs_g, states[j])
                        components.append(
                            emission_baf(bafs[i+1], m_ij, p_ij, pi0=pi0, std_dev=std_dev, k=ks[j])
                            + emission_lrr(lrrs[i+1], k=ks[j], std_dev=sigmas[i+1]) + log(w_o) + log(p)
                        )
                cur_emissions[j] = logsumexp(np.array(components, dtype=np.float64))
            else:
                cur_emission = np.zeros(3)
                for idx, (x, p) in enumerate(zip(geno, geno_freq)):
                    if maternal:
                        m_ij = mat_dosage(haps[:, i+1], states[j])
                        p_ij = pat_dosage(x, states[j])
                    else:
                        m_ij = mat_dosage(x, states[j])
                        p_ij = pat_dosage(haps[:, i+1], states[j])
                    cur_emission[idx] = emission_baf(
                            bafs[i+1], m_ij, p_ij, pi0=pi0, std_dev=std_dev, k=ks[j],
                        ) + emission_lrr(lrrs[i+1], k=ks[j], std_dev=sigmas[i+1]) + log(p)
                cur_emissions[j] = logsumexp(cur_emission)
        for j in range(m):
            betas[j, i] = logsumexp(A_hat[:, j] + cur_emissions + betas[:, (i + 1)])
        if i == 0:
            f = freqs[i]
            if f < 0:
                geno_freq = (1/3, 1/3, 1/3)
            else:
                geno_freq = ((1 - f)**2, 2*f*(1-f), f**2)
            for j in range(m):
                if obs_err > 0.0:
                    obs_h0 = int(haps[0, i])
                    obs_h1 = int(haps[1, i])
                    obs_alts = [[obs_h0, obs_h1]]
                    obs_weights = [1.0 - obs_err]
                    if obs_h0 != obs_h1:
                        obs_alts.append([0, 0])
                        obs_weights.append(obs_err / 2.0)
                        obs_alts.append([1, 1])
                        obs_weights.append(obs_err / 2.0)
                    else:
                        obs_alts.append([obs_h0, 1 - obs_h0])
                        obs_weights.append(obs_err)
                    components = []
                    for obs_g, w_o in zip(obs_alts, obs_weights):
                        for x, p in zip(geno, geno_freq):
                            if maternal:
                                m_ij = mat_dosage(obs_g, states[j])
                                p_ij = pat_dosage(x, states[j])
                            else:
                                m_ij = mat_dosage(x, states[j])
                                p_ij = pat_dosage(obs_g, states[j])
                            components.append(
                                emission_baf(bafs[i], m_ij, p_ij, pi0=pi0, std_dev=std_dev, k=ks[j])
                                + emission_lrr(lrrs[i], k=ks[j], std_dev=sigmas[i]) + log(w_o) + log(p)
                            )
                    cur_emissions_j = logsumexp(np.array(components, dtype=np.float64))
                else:
                    cur_emission = np.zeros(3)
                    for idx, (x, p) in enumerate(zip(geno, geno_freq)):
                        if maternal:
                            m_ij = mat_dosage(haps[:, i], states[j])
                            p_ij = pat_dosage(x, states[j])
                        else:
                            m_ij = mat_dosage(x, states[j])
                            p_ij = pat_dosage(haps[:, i], states[j])
                        cur_emission[idx] = emission_baf(
                                bafs[i], m_ij, p_ij, pi0=pi0, std_dev=std_dev, k=ks[j],
                            ) + emission_lrr(lrrs[i], k=ks[j], std_dev=sigmas[i]) + log(p)
                    cur_emissions_j = logsumexp(cur_emission)
                betas[j, i] += log(1/m) + cur_emissions_j
        scaler[i] = logsumexp(betas[:, i])
        betas[:, i] -= scaler[i]
    return betas, scaler, states, None, sum(scaler) + scaler[-1]


# -------- DANGER ZONE ---------- #
def solve_trio(self, cg=0, fg=0, mg=0):
    """Solve the trio setup to phase the parents.

    Code originally from: https://github.com/odelaneau/makeScaffold/blob/master/src/data_mendel.cpp
    """
    phased = None
    mendel = None
    if (fg == 0) & (mg == 0) & (cg == 0):
        f0 = 0
        f1 = 0
        m0 = 0
        m1 = 0
        c0 = 0
        c1 = 0
        mendel = 0
        phased = 1
    if (fg == 0) & (mg == 0) & (cg == 1):
        f0 = 0
        f1 = 0
        m0 = 0
        m1 = 0
        c0 = 0
        c1 = 1
        mendel = 1
        phased = 0
    if (fg == 0) & (mg == 0) & (cg == 2):
        f0 = 0
        f1 = 0
        m0 = 0
        m1 = 0
        c0 = 1
        c1 = 1
        mendel = 1
        phased = 0
    if (fg == 0) & (mg == 1) & (cg == 0):
        f0 = 0
        f1 = 0
        m0 = 1
        m1 = 0
        c0 = 0
        c1 = 0
        mendel = 0
        phased = 1
    if (fg == 0) & (mg == 1) & (cg == 1):
        f0 = 0
        f1 = 0
        m0 = 0
        m1 = 1
        c0 = 0
        c1 = 1
        mendel = 0
        phased = 1
    if (fg == 0) & (mg == 1) & (cg == 2):
        f0 = 0
        f1 = 0
        m0 = 0
        m1 = 1
        c0 = 1
        c1 = 1
        mendel = 1
        phased = 0
    if (fg == 0) & (mg == 2) & (cg == 0):
        f0 = 0
        f1 = 0
        m0 = 1
        m1 = 1
        c0 = 0
        c1 = 0
        mendel = 1
        phased = 0
    if (fg == 0) & (mg == 2) & (cg == 1):
        f0 = 0
        f1 = 0
        m0 = 1
        m1 = 1
        c0 = 0
        c1 = 1
        mendel = 0
        phased = 1
    if (fg == 0) & (mg == 2) & (cg == 2):
        f0 = 0
        f1 = 0
        m0 = 1
        m1 = 1
        c0 = 1
        c1 = 1
        mendel = 1
        phased = 0
    if (fg == 1) & (mg == 0) & (cg == 0):
        f0 = 0
        f1 = 1
        m0 = 0
        m1 = 0
        c0 = 0
        c1 = 0
        mendel = 0
        phased = 1
    if (fg == 1) & (mg == 0) & (cg == 1):
        f0 = 1
        f1 = 0
        m0 = 0
        m1 = 0
        c0 = 1
        c1 = 0
        mendel = 0
        phased = 1
    if (fg == 1) & (mg == 0) & (cg == 2):
        f0 = 1
        f1 = 0
        m0 = 0
        m1 = 0
        c0 = 1
        c1 = 1
        mendel = 1
        phased = 0
    if (fg == 1) & (mg == 1) & (cg == 0):
        f0 = 0
        f1 = 1
        m0 = 1
        m1 = 0
        c0 = 0
        c1 = 0
        mendel = 0
        phased = 1
    if (fg == 1) & (mg == 1) & (cg == 1):
        f0 = 0
        f1 = 1
        m0 = 0
        m1 = 1
        c0 = 0
        c1 = 1
        mendel = 0
        phased = 0
    if (fg == 1) & (mg == 1) & (cg == 2):
        f0 = 1
        f1 = 0
        m0 = 0
        m1 = 1
        c0 = 1
        c1 = 1
        mendel = 0
        phased = 1
    if (fg == 1) & (mg == 2) & (cg == 0):
        f0 = 0
        f1 = 1
        m0 = 1
        m1 = 1
        c0 = 0
        c1 = 0
        mendel = 1
        phased = 0
    if (fg == 1) & (mg == 2) & (cg == 1):
        f0 = 0
        f1 = 1
        m0 = 1
        m1 = 1
        c0 = 0
        c1 = 1
        mendel = 0
        phased = 1
    if (fg == 1) & (mg == 2) & (cg == 2):
        f0 = 1
        f1 = 0
        m0 = 1
        m1 = 1
        c0 = 1
        c1 = 1
        mendel = 0
        phased = 1
    if (fg == 2) & (mg == 0) & (cg == 0):
        f0 = 1
        f1 = 1
        m0 = 0
        m1 = 0
        c0 = 0
        c1 = 0
        mendel = 1
        phased = 0
    if (fg == 2) & (mg == 0) & (cg == 1):
        f0 = 1
        f1 = 1
        m0 = 0
        m1 = 0
        c0 = 1
        c1 = 0
        mendel = 0
        phased = 1
    if (fg == 2) & (mg == 0) & (cg == 2):
        f0 = 1
        f1 = 1
        m0 = 0
        m1 = 0
        c0 = 1
        c1 = 1
        mendel = 1
        phased = 0
    if (fg == 2) & (mg == 1) & (cg == 0):
        f0 = 1
        f1 = 1
        m0 = 0
        m1 = 1
        c0 = 0
        c1 = 0
        mendel = 1
        phased = 0
    if (fg == 2) & (mg == 1) & (cg == 1):
        f0 = 1
        f1 = 1
        m0 = 1
        m1 = 0
        c0 = 1
        c1 = 0
        mendel = 0
        phased = 1
    if (fg == 2) & (mg == 1) & (cg == 2):
        f0 = 1
        f1 = 1
        m0 = 0
        m1 = 1
        c0 = 1
        c1 = 1
        mendel = 0
        phased = 1
    if (fg == 2) & (mg == 2) & (cg == 0):
        f0 = 1
        f1 = 1
        m0 = 1
        m1 = 1
        c0 = 0
        c1 = 0
        mendel = 1
        phased = 0
    if (fg == 2) & (mg == 2) & (cg == 1):
        f0 = 1
        f1 = 1
        m0 = 1
        m1 = 1
        c0 = 0
        c1 = 1
        mendel = 1
        phased = 0
    if (fg == 2) & (mg == 2) & (cg == 2):
        f0 = 1
        f1 = 1
        m0 = 1
        m1 = 1
        c0 = 1
        c1 = 1
        mendel = 0
        phased = 1
    if phased is None:
        raise ValueError("Not-disomic genotype!")
    return [f0, f1, m0, m1, c0, c1, mendel, phased]
