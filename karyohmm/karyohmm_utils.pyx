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
    """Emission distribution function for B-allele frequency in the sample.

    NOTE: this should have some approximate error potentially?
    """
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
    # NOTE: should see if we need to do this in log-space for numerics ...
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


def forward_algo(bafs, pos, mat_haps, pat_haps, states, karyotypes, double r=1e-8, double a=1e-2, double pi0=0.2, double std_dev=0.25, int unphased=0):
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
        m_ij = mat_dosage(mat_haps[:, 0], states[j])
        p_ij = pat_dosage(pat_haps[:, 0], states[j])
        # This is in log-space ...
        cur_emission = emission_baf(
                bafs[0],
                m_ij,
                p_ij,
                pi0=pi0,
                std_dev=std_dev,
                k=ks[j],
            )
        alphas[j, 0] += cur_emission
    scaler = np.zeros(n)
    scaler[0] = logsumexp(alphas[:, 0])
    alphas[:, 0] -= scaler[0]
    for i in range(1, n):
        di = pos[i] - pos[i-1]
        # This should get the distance dependent transition models ...
        A_hat = transition_kernel(K0, K1, d=di, r=r, a=a, unphased=unphased)
        for j in range(m):
            m_ij = mat_dosage(mat_haps[:, i], states[j])
            p_ij = pat_dosage(pat_haps[:, i], states[j])
            # This is in log-space ...
            cur_emission = emission_baf(
                    bafs[i],
                    m_ij,
                    p_ij,
                    pi0=pi0,
                    std_dev=std_dev,
                    k=ks[j],
                )
            alphas[j, i] = cur_emission + logsumexp(A_hat[:, j] + alphas[:, (i - 1)])
        scaler[i] = logsumexp(alphas[:, i])
        alphas[:, i] -= scaler[i]
    return alphas, scaler, states, None, sum(scaler)


def backward_algo(bafs, pos, mat_haps, pat_haps, states, karyotypes, double r=1e-8, double a=1e-2, double pi0=0.2, double std_dev=0.25, int unphased=0):
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
        # The matrices are element-wise multiplied so add in log-space ...
        di = pos[i+1] - pos[i]
        A_hat = transition_kernel(K0, K1, d=di, r=r, a=a, unphased=unphased)
        # Calculate the full set of emissions
        cur_emissions = np.zeros(m)
        for j in range(m):
            m_ij = mat_dosage(mat_haps[:, i+1], states[j])
            p_ij = pat_dosage(pat_haps[:, i+1], states[j])
            # This is in log-space as well ...
            cur_emissions[j] = emission_baf(
                    bafs[i + 1],
                    m_ij,
                    p_ij,
                    pi0=pi0,
                    std_dev=std_dev,
                    k=ks[j],
                )
        for j in range(m):
            # This should be the correct version here ...
            betas[j, i] = logsumexp(A_hat[:, j] + cur_emissions + betas[:, (i + 1)])
        if i == 0:
            for j in range(m):
                m_ij = mat_dosage(mat_haps[:, i], states[j])
                p_ij = pat_dosage(pat_haps[:, i], states[j])
                # This is in log-space as well ...
                cur_emission = emission_baf(
                        bafs[i],
                        m_ij,
                        p_ij,
                        pi0=pi0,
                        std_dev=std_dev,
                        k=ks[j],
                    )
                # Add in the initialization + first emission?
                betas[j, i] += log(1/m) + cur_emission
        # Do the rescaling here ...
        scaler[i] = logsumexp(betas[:, i])
        betas[:, i] -= scaler[i]
    return betas, scaler, states, None, sum(scaler)


def viterbi_algo(bafs, pos, mat_haps, pat_haps, states, karyotypes, double r=1e-8, double a=1e-2, double pi0=0.2, double std_dev=0.25, int unphased=0):
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
            m_ij = mat_dosage(mat_haps[:, i], states[j])
            p_ij = pat_dosage(pat_haps[:, i], states[j])
            deltas[j, i] = np.max(deltas[:, i-1] + A_hat[:, j])
            deltas[j, i] += emission_baf(
                    bafs[i],
                    m_ij,
                    p_ij,
                    pi0=pi0,
                    std_dev=std_dev,
                    k=ks[j],
                )
            psi[j, i] = np.argmax(deltas[:, i - 1] + A_hat[:, j]).astype(int)
    path = np.zeros(n, dtype=int)
    path[-1] = np.argmax(deltas[:, -1]).astype(int)
    for i in range(n - 2, -1, -1):
        path[i] = psi[path[i + 1], i]
    path[0] = psi[path[1], 1]
    return path, states, deltas, psi


def forward_algo_sibs(bafs, pos, mat_haps, pat_haps, states, karyotypes, double r=1e-8, double a=1e-2, (double, double) pi0=(0.2, 0.2), (double, double) std_dev=(0.1, 0.1)):
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
        # First sibling embryo
        m_ij0 = mat_dosage(mat_haps[:, 0], states[j][0])
        p_ij0 = pat_dosage(pat_haps[:, 0], states[j][0])
        # Second sibling embryo
        m_ij1 = mat_dosage(mat_haps[:, 0], states[j][1])
        p_ij1 = pat_dosage(pat_haps[:, 0], states[j][1])
        # This is in log-space ...
        cur_emission = emission_baf(
                bafs[0][0],
                m_ij0,
                p_ij0,
                pi0=pi0[0],
                std_dev=std_dev[0],
                k=2,
            ) + emission_baf(
                bafs[1][0],
                m_ij1,
                p_ij1,
                pi0=pi0[1],
                std_dev=std_dev[1],
                k=2,
            )
        alphas[j, 0] += cur_emission
    scaler = np.zeros(n)
    scaler[0] = logsumexp(alphas[:, 0])
    alphas[:, 0] -= scaler[0]
    for i in range(1, n):
        di = pos[i] - pos[i-1]
        A_hat = transition_kernel(K0, K1, d=di, r=r, a=a)
        for j in range(m):
            # First sibling embryo
            m_ij0 = mat_dosage(mat_haps[:, i], states[j][0])
            p_ij0 = pat_dosage(pat_haps[:, i], states[j][0])
            # Second sibling embryo
            m_ij1 = mat_dosage(mat_haps[:, i], states[j][1])
            p_ij1 = pat_dosage(pat_haps[:, i], states[j][1])
            # This is in log-space ...
            cur_emission = emission_baf(
                    bafs[0][i],
                    m_ij0,
                    p_ij0,
                    pi0=pi0[0],
                    std_dev=std_dev[0],
                    k=2,
                ) + emission_baf(
                    bafs[1][i],
                    m_ij1,
                    p_ij1,
                    pi0=pi0[1],
                    std_dev=std_dev[1],
                    k=2,
                )
            alphas[j, i] = cur_emission + logsumexp(A_hat[:, j] + alphas[:, i - 1])
        scaler[i] = logsumexp(alphas[:, i])
        alphas[:, i] -= scaler[i]
    return alphas, scaler, states, None, sum(scaler)


def backward_algo_sibs(bafs, pos, mat_haps, pat_haps, states, karyotypes, double r=1e-8, double a=1e-2, (double, double) pi0=(0.2, 0.2), (double, double) std_dev=(0.1, 0.1)):
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
            m_ij0 = mat_dosage(mat_haps[:, i+1], states[j][0])
            p_ij0 = pat_dosage(pat_haps[:, i+1], states[j][0])
            m_ij1 = mat_dosage(mat_haps[:, i+1], states[j][1])
            p_ij1 = pat_dosage(pat_haps[:, i+1], states[j][1])
            # This is in log-space as well
            cur_emissions[j] = emission_baf(
                    bafs[0][i + 1],
                    m_ij0,
                    p_ij0,
                    pi0=pi0[0],
                    std_dev=std_dev[0],
                    k=2,
                ) + emission_baf(
                    bafs[1][i + 1],
                    m_ij1,
                    p_ij1,
                    pi0=pi0[1],
                    std_dev=std_dev[1],
                    k=2,
                )
        for j in range(m):
            betas[j, i] = logsumexp(A_hat[:, j] + cur_emissions + betas[:, (i + 1)])
        if i == 0:
            for j in range(m):
                m_ij0 = mat_dosage(mat_haps[:, i], states[j][0])
                p_ij0 = pat_dosage(pat_haps[:, i], states[j][0])
                m_ij1 = mat_dosage(mat_haps[:, i], states[j][1])
                p_ij1 = pat_dosage(pat_haps[:, i], states[j][1])
                cur_emission = emission_baf(
                        bafs[0][i],
                        m_ij0,
                        p_ij0,
                        pi0=pi0[0],
                        std_dev=std_dev[0],
                        k=2,
                    ) + emission_baf(
                        bafs[1][i],
                        m_ij1,
                        p_ij1,
                        pi0=pi0[1],
                        std_dev=std_dev[1],
                        k=2,
                    )
                # Add in the initialization + first emission?
                betas[j, i] += log(1/m) + cur_emission
        scaler[i] = logsumexp(betas[:, i])
        betas[:, i] -= scaler[i]
    return betas, scaler, states, None, sum(scaler)


def viterbi_algo_sibs(bafs, pos, mat_haps, pat_haps, states, karyotypes, double r=1e-8, double a=1e-2, (double, double) pi0=(0.2, 0.2), (double, double) std_dev=(0.1, 0.1)):
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
            m_ij0 = mat_dosage(mat_haps[:, i], states[j][0])
            p_ij0 = pat_dosage(pat_haps[:, i], states[j][0])
            m_ij1 = mat_dosage(mat_haps[:, i], states[j][1])
            p_ij1 = pat_dosage(pat_haps[:, i], states[j][1])
            deltas[j, i] = np.max(deltas[:, i-1] + A_hat[:, j])
            deltas[j, i] += emission_baf(
                    bafs[0][i],
                    m_ij0,
                    p_ij0,
                    pi0=pi0[0],
                    std_dev=std_dev[0],
                    k=2,
                ) + emission_baf(
                    bafs[1][i],
                    m_ij1,
                    p_ij1,
                    pi0=pi0[1],
                    std_dev=std_dev[1],
                    k=2,
                )
            psi[j, i] = np.argmax(deltas[:, i - 1] + A_hat[:, j]).astype(int)
    path = np.zeros(n, dtype=int)
    path[-1] = np.argmax(deltas[:, -1]).astype(int)
    for i in range(n - 2, -1, -1):
        path[i] = psi[path[i + 1], i]
    path[0] = psi[path[1], 1]
    return path, states, deltas, psi


def forward_algo_duo(bafs, pos, haps, freqs, states, karyotypes, bint maternal=True, double r=1e-8, double a=1e-2, double pi0=0.8, double std_dev=0.2):
    """Helper function for optimization for forward algorithm in the duo setting."""
    cdef int i, j, idx, n, m
    cdef float di, f
    n = bafs.size
    m = len(states)
    ks = [sum([s >= 0 for s in state]) for state in states]
    K0, K1 = create_index_arrays(karyotypes)
    alphas = np.zeros(shape=(m, n))
    alphas[:, 0] = log(1.0 / m)
    geno = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for j in range(m):
        # Need to marginalize over all the haplotypes ...
        f = freqs[0]
        cur_emission = np.zeros(4)
        for idx, (x, p) in enumerate(zip(geno, ((1 - f)**2, f*(1-f), f*(1-f), f**2))):
            if maternal:
                m_ij = mat_dosage(haps[:, 0], states[j])
                p_ij = pat_dosage(x, states[j])
            else:
                m_ij = mat_dosage(x, states[j])
                p_ij = pat_dosage(haps[:, 0], states[j])
            cur_emission[idx] = emission_baf(
                    bafs[0],
                    m_ij,
                    p_ij,
                    pi0=pi0,
                    std_dev=std_dev,
                    k=ks[j],
                ) + log(p)
        alphas[j, 0] = logsumexp(cur_emission)
    scaler = np.zeros(n)
    scaler[0] = logsumexp(alphas[:, 0])
    alphas[:, 0] -= scaler[0]
    for i in range(1, n):
        di = pos[i] - pos[i-1]
        # This should get the distance dependent transition models ...
        A_hat = transition_kernel(K0, K1, d=di, r=r, a=a)
        for j in range(m):
            cur_emission = np.zeros(4)
            f = freqs[i]
            for idx, (x, p) in enumerate(zip(geno, ((1 - f)**2, f*(1 - f), f*(1-f), f**2))):
                if maternal:
                    m_ij = mat_dosage(haps[:, i], states[j])
                    p_ij = pat_dosage(x, states[j])
                else:
                    m_ij = mat_dosage(x, states[j])
                    p_ij = pat_dosage(haps[:, i], states[j])
                # Build up the summed emission model?
                cur_emission[idx] = emission_baf(
                        bafs[i],
                        m_ij,
                        p_ij,
                        pi0=pi0,
                        std_dev=std_dev,
                        k=ks[j],
                    ) + log(p)
            alphas[j, i] = logsumexp(cur_emission) + logsumexp(A_hat[:, j] + alphas[:, (i - 1)])
        scaler[i] = logsumexp(alphas[:, i])
        alphas[:, i] -= scaler[i]
    return alphas, scaler, states, None, sum(scaler)


def backward_algo_duo(bafs, pos, haps, freqs, states, karyotypes, bint maternal=True, double r=1e-8, double a=1e-2, double pi0=0.2, double std_dev=0.25):
    """Helper function for backward algorithm loop-optimization."""
    cdef int i, j, idx, n, m
    cdef float di, f, p
    n = bafs.size
    m = len(states)
    ks = [sum([s >= 0 for s in state]) for state in states]
    K0, K1 = create_index_arrays(karyotypes)
    betas = np.zeros(shape=(m, n))
    betas[:, -1] = log(1)
    scaler = np.zeros(n)
    scaler[-1] = logsumexp(betas[:, -1])
    betas[:, -1] -= scaler[-1]
    geno = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for i in range(n - 2, -1, -1):
        f = freqs[i+1]
        # The matrices are element-wise multiplied so add in log-space ...
        di = pos[i+1] - pos[i]
        A_hat = transition_kernel(K0, K1, d=di, r=r, a=a)
        # Calculate the full set of emissions
        cur_emissions = np.zeros(m)
        for j in range(m):
            cur_emission = np.zeros(4)
            for idx, (x, p) in enumerate(zip(geno, ((1 - f)**2, f*(1-f), f*(1-f), f**2))):
                if maternal:
                    m_ij = mat_dosage(haps[:, i+1], states[j])
                    p_ij = pat_dosage(x, states[j])
                else:
                    m_ij = mat_dosage(x, states[j])
                    p_ij = pat_dosage(haps[:, i+1], states[j])
                cur_emission[idx] = emission_baf(
                        bafs[i+1],
                        m_ij,
                        p_ij,
                        pi0=pi0,
                        std_dev=std_dev,
                        k=ks[j],
                    ) + log(p)
            cur_emissions[j] = logsumexp(cur_emission)
        for j in range(m):
            betas[j, i] = logsumexp(A_hat[:, j] + cur_emissions + betas[:, (i + 1)])
        if i == 0:
            f = freqs[i]
            for j in range(m):
                cur_emission = np.zeros(4)
                for idx, (x, p) in enumerate(zip(geno, ((1 - f)**2, f*(1-f), f*(1-f), f**2))):
                    if maternal:
                        m_ij = mat_dosage(haps[:, i], states[j])
                        p_ij = pat_dosage(x, states[j])
                    else:
                        m_ij = mat_dosage(x, states[j])
                        p_ij = pat_dosage(haps[:, i], states[j])
                    cur_emission[idx] = emission_baf(
                            bafs[i],
                            m_ij,
                            p_ij,
                            pi0=pi0,
                            std_dev=std_dev,
                            k=ks[j],
                        ) + log(p)
                # This is in log-space as well ...
                cur_emissions = logsumexp(cur_emission)
                # Add in the initialization + first emission?
                betas[j, i] += log(1/m) + cur_emissions
        # Do the rescaling here ...
        scaler[i] = logsumexp(betas[:, i])
        betas[:, i] -= scaler[i]
    return betas, scaler, states, None, sum(scaler) + scaler[-1]


def forward_algo_duo_panel(bafs, pos, haps, ref_panel, states, karyotypes, bint maternal=True, double r=1e-8, double a=1e-2, double pi0=0.8, double std_dev=0.2):
    """Helper function for optimization for forward algorithm in the duo setting with a reference panel.

    Ref panel is a K x M set of reference haplotypes
    """
    cdef int i, j, idx, n, m
    cdef int zi, zj, k
    cdef float di, cur_emission
    cdef double[:] transitions
    n = bafs.size
    m = len(states)
    k = ref_panel.shape[0]
    ks = [sum([s >= 0 for s in state]) for state in states]
    K0, K1 = create_index_arrays(karyotypes)
    alphas = np.zeros(shape=(m, k, k, n))
    alphas[:, :, :, 0] = log(1.0 / m)
    for j in range(m):
        # Iterating through the unobserved parental haplotypes in the panel ...
        for zi in range(k):
            for zj in range(k):
                x = [ref_panel[zi, 0], ref_panel[zj, 0]]
                if maternal:
                    m_ij = mat_dosage(haps[:, 0], states[j])
                    p_ij = pat_dosage(x, states[j])
                else:
                    m_ij = mat_dosage(x, states[j])
                    p_ij = pat_dosage(haps[:, 0], states[j])
                alphas[j, zi, zj, 0] = emission_baf(
                        bafs[0],
                        m_ij,
                        p_ij,
                        pi0=pi0,
                        std_dev=std_dev,
                        k=ks[j],
                    )
    scaler = np.zeros(n)
    scaler[0] = logsumexp(alphas[:, :, :, 0].flatten())
    alphas[:, :, :, 0] -= scaler[0]
    for i in range(1, n):
        di = pos[i] - pos[i-1]
        A_hat = transition_kernel(K0, K1, d=di, r=r, a=a)
        cur_emission = 0.0
        for j in range(m):
            for zi in range(k):
                for zj in range(k):
                    x = [ref_panel[zi, i], ref_panel[zj, i]]
                    if maternal:
                        m_ij = mat_dosage(haps[:, i], states[j])
                        p_ij = pat_dosage(x, states[j])
                    else:
                        m_ij = mat_dosage(x, states[j])
                        p_ij = pat_dosage(haps[:, i], states[j])
                    # Build up the summed emission model ...
                    cur_emission = emission_baf(
                            bafs[i],
                            m_ij,
                            p_ij,
                            pi0=pi0,
                            std_dev=std_dev,
                            k=ks[j],
                        )
                    idx = 0
                    transitions = np.zeros(m*k*k)
                    for j_ in range(m):
                        for zi_ in range(k):
                            for zj_ in range(k):
                                transitions[idx] = A_hat[j_, j] + log((1.0 - exp(-r*di))*(zi_ != zi) + (-r*di)*(zi_ == zi)) + log((1.0 - exp(-r*di))*(zj_ != zj) + (-r*di)*(zj_ == zj)) + alphas[j_, zi_, zj_, (i-1)]
                                idx += 1
                    alphas[j, zi, zj, i] = cur_emission + logsumexp(transitions)
        scaler[i] = logsumexp(alphas[:, :, :, i].flatten())
        alphas[:, :, :, i] -= scaler[i]
    return alphas, scaler, states, None, sum(scaler)


def backward_algo_duo_panel(bafs, pos, haps, ref_panel, states, karyotypes, bint maternal=True, double r=1e-8, double a=1e-2, double pi0=0.8, double std_dev=0.2):
    """Helper function for optimization for backward algorithm in the duo setting with a reference panel.

    Ref panel is a K x M set of reference haplotypes
    """
    cdef int i, j, idx, n, m
    cdef int zi, zj, k
    cdef float di, cur_emission
    cdef double[:] updates
    n = bafs.size
    m = len(states)
    k = ref_panel.shape[0]
    ks = [sum([s >= 0 for s in state]) for state in states]
    K0, K1 = create_index_arrays(karyotypes)
    betas = np.zeros(shape=(m, k, k, n))
    betas[:, :, :, 0] = log(1.0)
    scaler = np.zeros(n)
    scaler[-1] = logsumexp(betas[:, :, :, -1])
    betas[:, :, : -1] -= scaler[-1]
    for i in range(n - 2, -1, -1):
        di = pos[i+1] - pos[i]
        A_hat = transition_kernel(K0, K1, d=di, r=r, a=a)
        cur_emission = np.zeros(shape=(m, k, k))
        for j in range(m):
            for zi in range(k):
                for zj in range(k):
                    updates = np.zeros(m*k*k)
                    idx = 0
                    for j_ in range(m):
                        x = [ref_panel[zi, i], ref_panel[zj, i]]
                        if maternal:
                            m_ij = mat_dosage(haps[:, i], states[j_])
                            p_ij = pat_dosage(x, states[j_])
                        else:
                            m_ij = mat_dosage(x, states[j_])
                            p_ij = pat_dosage(haps[:, i], states[j_])
                        cur_emission = emission_baf(
                                bafs[i],
                                m_ij,
                                p_ij,
                                pi0=pi0,
                                std_dev=std_dev,
                                k=ks[j_])
                        for zi_ in range(k):
                            for zj_ in range(k):
                                transition = A_hat[j, j_] + log((1.0 - exp(-r*di))*(zi != zi_) + (-r*di)*(zi == zi_)) + log((1.0 - exp(-r*di))*(zj != zj_) + (-r*di)*(zj == zj_))
                                updates[idx] = transition + cur_emission + betas[j_, zi_, zj_, i+1]
                                idx += 1
                betas[j, zi, zj, i] = logsumexp(updates)
        if i == 0:
            for j in range(m):
                for zi in range(m):
                    for zj in range(m):
                        x = [ref_panel[zi, i], ref_panel[zj, i]]
                        if maternal:
                            m_ij = mat_dosage(haps[:, i], states[j])
                            p_ij = pat_dosage(x, states[j])
                        else:
                            m_ij = mat_dosage(x, states[j])
                            p_ij = pat_dosage(haps[:, i], states[j])
                        cur_emission = emission_baf(
                            bafs[i],
                            m_ij,
                            p_ij,
                            pi0=pi0,
                            std_dev=std_dev,
                            k=ks[j],
                        )
                        betas[j, zi, zj, i] += log(1./(m*k*k)) + cur_emission
        scaler[i] = logsumexp(betas[:, :, :, i])
        betas[:, :, :, i] -= scaler[i]
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
