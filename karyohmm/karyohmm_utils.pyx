from libc.math cimport erf, exp, log, pi, sqrt
import numpy as np


cdef double sqrt2 = sqrt(2.);
cdef double sqrt2pi = sqrt(2*pi);
cdef double logsqrt2pi = log(1/sqrt2pi)

cdef double logsumexp(double[:] x):
    """Cython implementation of the logsumexp trick"""
    cdef int i,n;
    cdef double m = -1e32;
    cdef double c = 0.0;
    n = x.size
    for i in range(n):
        m = max(m,x[i])
    for i in range(n):
        c += exp(x[i] - m)
    return m + log(c)

cpdef double mat_dosage(mat_hap, state):
    """Obtain the maternal dosage."""
    cdef int k,i,l;
    cdef double m;
    k = 0
    m = 0.
    l = len(state)
    for i in range(l):
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
    cdef int k,i,l;
    cdef double p;
    k = 0
    p = 0.
    l = len(state)
    for i in range(l):
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

cdef double psi(double x):
    """CDF for a normal distribution function in log-space."""
    if x < -4:
        return logsqrt2pi - 0.5*(x**2) - log(-x)
    else:
        return log((1.0 + erf(x / sqrt2))) - log(2.0)

cdef double norm_pdf(double x):
    """PDF for the normal distribution function in log-space."""
    return logsqrt2pi -  0.5*(x**2)

cdef double logdiffexp(double a, double b):
    """Log-sum-exp trick but for differences."""
    return log(exp(a) - exp(b) + 1e-124)

cdef double logaddexp(double a, double b):
    cdef double m = -1e32;
    cdef double c = 0.0;
    m = max(a,b)
    c = exp(a - m) + exp(b - m)
    return m + log(c)

cpdef double truncnorm_pdf(double x, double a, double b, double mu=0.5, double sigma=0.2):
    """Custom definition of the log of the truncated normal pdf."""
    cdef double p, z, alpha, beta, eta;
    beta = (b - mu) / sigma
    alpha = (a - mu) / sigma
    eta = (max(min(x,b),a) - mu) / sigma
    z = logdiffexp(psi(beta), psi(alpha))
    p = norm_pdf(eta) - log(sigma) - z
    return p

cpdef double emission_baf(double baf, double m, double p, double pi0=0.2, double std_dev=0.2, int k=2):
    """Emission distribution function for B-allele frequency in the sample."""
    cdef double mu_i, x, x0, x1;
    if (m == -1) & (p == -1):
        return 1.0
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

def lod_phase(haps1, haps2, baf, **kwargs):
    """Estimate the log-likelihood of being the phase vs. antiphase orientation for heterozygotes."""
    cdef int i,j;
    cdef float phase_orientation, antiphase_orientation;
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

def forward_algo(bafs,  mat_haps, pat_haps, states, A, double pi0=0.2, double std_dev=0.25):
    """Helper function for forward algorithm loop-optimization."""
    cdef int i,j,n,m;
    n = bafs.size
    m = len(states)
    ks = [sum([s >= 0 for s in state]) for state in states]
    alphas = np.zeros(shape=(m, n))
    alphas[:, 0] = 1.0 / m
    scaler = np.zeros(n)
    scaler[0] = logsumexp(alphas[:, 0])
    alphas[:, 0] -= scaler[0]
    for i in range(1, n):
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
            alphas[j, i] = cur_emission + logsumexp(A[j, :] + alphas[:, i - 1])
        scaler[i] = logsumexp(alphas[:, i])
        alphas[:, i] -= scaler[i]
    return alphas, scaler, states, None, sum(scaler)

def backward_algo(bafs, mat_haps, pat_haps, states, A, double pi0=0.2, double std_dev=0.25):
    """Helper function for backward algorithm loop-optimization."""
    cdef int i,j,n,m;
    n = bafs.size
    m = len(states)
    ks = [sum([s >= 0 for s in state]) for state in states]
    betas = np.zeros(shape=(m, n))
    betas[:,-1] = 0.0
    scaler = np.zeros(n)
    scaler[-1] = logsumexp(betas[:, -1])
    betas[:, -1] -= scaler[-1]
    for i in range(n - 2, -1, -1):
        # Calculate the full set of emissions
        for j in range(m):
            m_ij = mat_dosage(mat_haps[:, i+1], states[j])
            p_ij = pat_dosage(pat_haps[:, i+1], states[j])
            # This is in log-space as well ...
            cur_emission = emission_baf(
                    bafs[i + 1],
                    m_ij,
                    p_ij,
                    pi0=pi0,
                    std_dev=std_dev,
                    k=ks[j],
                )
            betas[j,i] = logsumexp(A[j, :] + cur_emission + betas[:, (i + 1)])
        # Do the rescaling here ...
        scaler[i] = logsumexp(betas[:, i])
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
                betas[j,i] += log(1/m) + cur_emission
        betas[:, i] -= scaler[i]

    return betas, scaler, states, None, sum(scaler)

def viterbi_algo(bafs, mat_haps, pat_haps, states, A, double pi0=0.2, double std_dev=0.25):
    """Cython implementation of the Viterbi algorithm for MLE path estimation through states."""
    cdef int i,j,n,m;
    n = bafs.size
    m = len(states)
    deltas = np.zeros(shape=(m, n))
    deltas[:, 0] = log(1.0 / m)
    psi = np.zeros(shape=(m, n), dtype=int)
    ks = [sum([s >= 0 for s in state]) for state in states]
    for i in range(1, n):
        for j in range(m):
            m_ij = mat_dosage(mat_haps[:, i], states[j])
            p_ij = pat_dosage(pat_haps[:, i], states[j])
            deltas[j,i] = np.max(deltas[:,i-1] + A[:,j])
            deltas[j,i] += emission_baf(
                    bafs[i],
                    m_ij,
                    p_ij,
                    pi0=pi0,
                    std_dev=std_dev,
                    k=ks[j],
                )
            psi[j, i] = np.argmax(deltas[:, i - 1] + A[:, j]).astype(int)
    path = np.zeros(n, dtype=int)
    path[-1] = np.argmax(deltas[:, -1]).astype(int)
    for i in range(n - 2, -1, -1):
        path[i] = psi[path[i + 1], i]
    path[0] = psi[path[1], 1]
    return path, states, deltas, psi


def forward_algo_sibs(bafs, mat_haps, pat_haps, states, A, double pi0=0.2, double std_dev=0.1):
    """Compute the forward algorithm for sibling embryo HMM."""
    cdef int i,j,n,m;
    assert len(bafs) == 2
    assert bafs[1].size == bafs[0].size
    n = bafs[0].size
    m = len(states)
    alphas = np.zeros(shape=(m, n))
    alphas[:, 0] = 1.0 / m
    scaler = np.zeros(n)
    scaler[0] = logsumexp(alphas[:, 0])
    alphas[:, 0] -= scaler[0]
    for i in range(1, n):
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
                    pi0=pi0,
                    std_dev=std_dev,
                    k=2,
                ) + emission_baf(
                    bafs[1][i],
                    m_ij1,
                    p_ij1,
                    pi0=pi0,
                    std_dev=std_dev,
                    k=2,
                )
            alphas[j, i] = cur_emission + logsumexp(A[j, :] + alphas[:, i - 1])
        scaler[i] = logsumexp(alphas[:, i])
        alphas[:, i] -= scaler[i]
    return alphas, scaler, states, None, sum(scaler)


def backward_algo_sibs(bafs, mat_haps, pat_haps, states, A, double pi0=0.2, double std_dev=0.1):
    """Compute the backward algorithm for the sibling embryo HMM."""
    cdef int i,j,n,m;
    assert len(bafs) == 2
    assert bafs[1].size == bafs[0].size
    n = bafs[0].size
    m = len(states)
    betas = np.zeros(shape=(m, n))
    betas[:,-1] = 1.0
    scaler = np.zeros(n)
    scaler[-1] = logsumexp(betas[:, -1])
    betas[:, -1] -= scaler[-1]
    for i in range(n - 2, -1, -1):
        for j in range(m):
            m_ij0 = mat_dosage(mat_haps[:, i+1], states[j][0])
            p_ij0 = pat_dosage(pat_haps[:, i+1], states[j][0])
            m_ij1 = mat_dosage(mat_haps[:, i+1], states[j][1])
            p_ij1 = pat_dosage(pat_haps[:, i+1], states[j][1])
            # This is in log-space as well
            cur_emission = emission_baf(
                    bafs[0][i + 1],
                    m_ij0,
                    p_ij0,
                    pi0=pi0,
                    std_dev=std_dev,
                    k=2,
                ) + emission_baf(
                    bafs[1][i + 1],
                    m_ij1,
                    p_ij1,
                    pi0=pi0,
                    std_dev=std_dev,
                    k=2,
                )
            betas[j,i] = logsumexp(A[j, :] + cur_emission + betas[:, (i + 1)])
        scaler[i] = logsumexp(betas[:, i])
        betas[:, i] -= scaler[i]
    return betas, scaler, states, None, sum(scaler)



def viterbi_algo_sibs(bafs, mat_haps, pat_haps, states, A, double pi0=0.2, double std_dev=0.1):
    """Viterbi algorithm and path tracing through sibling embryos."""
    cdef int i,j,n,m;
    assert len(bafs) == 2
    assert bafs[1].size == bafs[0].size
    n = bafs[0].size
    m = len(states)
    deltas = np.zeros(shape=(m, n))
    deltas[:, 0] = log(1.0 / m)
    psi = np.zeros(shape=(m, n), dtype=int)
    for i in range(1, n):
        for j in range(m):
            m_ij0 = mat_dosage(mat_haps[:, i], states[j][0])
            p_ij0 = pat_dosage(pat_haps[:, i], states[j][0])
            m_ij1 = mat_dosage(mat_haps[:, i], states[j][1])
            p_ij1 = pat_dosage(pat_haps[:, i], states[j][1])
            deltas[j,i] = np.max(deltas[:,i-1] + A[:,j])
            deltas[j,i] += emission_baf(
                    bafs[0][i],
                    m_ij0,
                    p_ij0,
                    pi0=pi0,
                    std_dev=std_dev,
                    k=2,
                ) + emission_baf(
                    bafs[1][i],
                    m_ij1,
                    p_ij1,
                    pi0=pi0,
                    std_dev=std_dev,
                    k=2,
                )
            psi[j, i] = np.argmax(deltas[:, i - 1] + A[:, j]).astype(int)
    path = np.zeros(n, dtype=int)
    path[-1] = np.argmax(deltas[:, -1]).astype(int)
    for i in range(n - 2, -1, -1):
        path[i] = psi[path[i + 1], i]
    path[0] = psi[path[1], 1]
    return path, states, deltas, psi
