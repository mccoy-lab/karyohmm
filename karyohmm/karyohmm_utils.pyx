from libc.math cimport erf, exp, log, pi, sqrt
from libcpp cimport bool
import numpy as np

# NOTE: these are fixed for K=0, 1, 2, 3 in terms of ploidy
lrr_mu = [-3.527211, -0.664184, 0.000000, 0.395621]
lrr_sd = [1.329152, 0.284338, 0.159645, 0.209089]

cdef double sqrt2 = sqrt(2.);
cdef double sqrtpi = sqrt(pi);

cdef double logsumexp(double[:] x):
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
        return log(1/(2*sqrtpi))- 0.5*(x**2) - log(-x)
    else:
        return log((1.0 + erf(x / sqrt2))) - log(2.0)

cdef double logdiffexp(double a, double b):
    """Log-sum-exp trick but for differences."""
    return log(exp(a) - exp(b) + 1e-323)

cdef double truncnorm_pf(double x, double a, double b, double mu=0.5, double sigma=0.2, double eps=1e-4):
    """Custom definition of the truncated normal probability."""
    cdef double p, z, alpha, beta, eps1, eps2;
    cdef double upper, lower;
    beta = (b - mu) / sigma
    alpha = (a - mu) / sigma
    eps1 = (min(x + eps, b) - mu) / sigma
    eps2 = (max(x - eps, a) - mu) / sigma
    z = logdiffexp(psi(beta), psi(alpha))
    if (eps1 > mu) and (eps2 > mu):
        upper = psi(-eps2)
        lower = psi(-eps1)
    else:
        upper = psi(eps1)
        lower = psi(eps2)
    p = logdiffexp(upper, lower) - z
    return exp(p)

cdef double loglik_gmm(double[:] lrrs, double[:] pis, double[:] mus, double[:] stds, double a=-2, double b=1, double eps=1e-6):
    """Estimate the log-likelihood of a given set of LRR values.

    NOTE: we truncate the domain to -2,1 for LRRs to improve specificity
    """
    assert b > a
    assert pis.size == mus.size
    assert stds.size == mus.size
    cdef int i;
    cdef double x, loglik;
    loglik = 0.0
    for x in lrrs:
        z = np.zeros(pis.size)
        for i in range(pis.size):
            if mus[i] == -9:
                z[i] = log(pis[i]) + log((((x + eps) - (x - eps)) / (b-a)))
            else:
                z[i] = log(pis[i]) + log(truncnorm_pf(x, a=a, b=b, mu=mus[i], sigma=stds[i], eps=eps))
        loglik += logsumexp(z)
    return loglik

def est_gmm_variance(lrrs, mus, a=-4, b=1.0, niter=30, eps=1e-6):
    """Estimate the variance terms for the LRR modeling."""
    assert mus.size > 0
    assert np.all(lrrs)
    pis = np.ones(mus.size)/mus.size
    std = np.ones(mus.size)
    logliks = np.zeros(niter+1)
    logliks[0] = loglik_gmm(lrrs, pis=pis, mus=mus, stds=std, a=a, b=b)
    for l in range(1, niter+1):
        gammas = np.zeros(shape=(mus.size, lrrs.size))
        for i,x in enumerate(lrrs):
            for j in range(mus.size):
                # -9 is the signal for the mu with the null likelihood ...
                if mus[j] == -9:
                    gammas[j,i] = pis[j]*(((x + eps) - (x - eps)) / (b-a))
                else:
                    gammas[j,i] = pis[j]*truncnorm_pf(x, a=a, b=b, mu=mus[j], sigma=std[j], eps=eps)
            gammas[:,i] /= np.sum(gammas[:,i])
        # Learn the nk values ...
        nk = np.sum(gammas, axis=1)
        pis = nk / lrrs.size
        for k in range(mus.size):
            std[k] = sqrt(np.sum(gammas[k,:]*(lrrs - mus[k])**2) / nk[k])
        logliks[l] = loglik_gmm(lrrs, pis=pis, mus=mus, stds=std, a=a, b=b)
    return pis, mus, std, logliks

cpdef double emission_baf(double baf, double m, double p, double pi0=0.2, double std_dev=0.2, double eps=1e-6, int k=2):
    """Emission distribution helper function ..."""
    cdef double mu_i, x;
    if (m == -1) & (p == -1):
        # NOTE: this only happens in the case of nullisomy ...
        return (baf + eps) - (baf - eps)
    mu_i = (m + p) / k
    x = truncnorm_pf(baf, 0.0, 1.0, mu=mu_i, sigma=std_dev, eps=eps)
    if mu_i == 0:
        return (
            pi0 * (baf == 0) + (1.0 - pi0)* x
        )
    elif mu_i == 1:
        return (
            pi0 * (baf == 1) + (1.0 - pi0) * x
        )
    else:
        return x

cpdef double emission_lrr(double lrr, int k, double[:] lrr_mu, double[:] lrr_sd, double a=-4.0, double b=1.0, double pi0=0.2, double eps=1e-6):
    """Emission function for LRR which is based on a truncated-normal distribution."""
    cdef double z, p, mu, sd;
    cdef double upper, lower;
    mu = lrr_mu[k]
    sd = lrr_sd[k]
    x = truncnorm_pf(lrr, a, b, mu=mu, sigma=sd, eps=eps)
    # NOTE: you can have pi0 just as the estimated proportion of uniform.
    return pi0 + (1 - pi0)*x

def forward_algo(bafs, lrrs, mat_haps, pat_haps, states, A, lrr_mu=lrr_mu, lrr_sd=lrr_sd, double pi0=0.2, double std_dev=0.25, double pi0_lrr=0.2, double eps=1e-6, int logr=True):
    """Helper function for forward algorithm loop-optimization."""
    cdef int i,j,n,m;
    assert bafs.size == lrrs.size
    lrrs_clip = np.clip(lrrs, -4, 1.0)
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
            cur_emission = log(
                emission_baf(
                    bafs[i],
                    m_ij,
                    p_ij,
                    pi0=pi0,
                    std_dev=std_dev,
                    eps=eps,
                    k=ks[j],
                )
            )
            if logr:
                cur_emission += log(emission_lrr(lrr=lrrs_clip[i], k=ks[j], a=-4, b=1.0, lrr_mu=lrr_mu, lrr_sd=lrr_sd, pi0=pi0_lrr, eps=eps))
            alphas[j, i] = cur_emission + logsumexp(A[j, :] + alphas[:, i - 1])
        scaler[i] = logsumexp(alphas[:, i])
        alphas[:, i] -= scaler[i]
    return alphas, scaler, states, None, sum(scaler)

def backward_algo(bafs, lrrs, mat_haps, pat_haps, states, A, lrr_mu=lrr_mu, lrr_sd=lrr_sd, double pi0=0.2, double std_dev=0.25, double pi0_lrr=0.2, double eps=1e-6, int logr=True):
    """Helper function for backward algorithm loop-optimization."""
    cdef int i,j,n,m;
    assert bafs.size == lrrs.size
    lrrs_clip = np.clip(lrrs, -4, 1.0)
    n = bafs.size
    m = len(states)
    ks = [sum([s >= 0 for s in state]) for state in states]
    betas = np.zeros(shape=(m, n))
    betas[:,-1] = 1.0
    scaler = np.zeros(n)
    scaler[-1] = logsumexp(betas[:, -1])
    betas[:, -1] -= scaler[-1]
    for i in range(n - 2, -1, -1):
        for j in range(m):
            m_ij = mat_dosage(mat_haps[:, i], states[j])
            p_ij = pat_dosage(pat_haps[:, i], states[j])
            # This is in log-space as well
            cur_emission = log(
                emission_baf(
                    bafs[i + 1],
                    m_ij,
                    p_ij,
                    pi0=pi0,
                    std_dev=std_dev,
                    eps=eps,
                    k=ks[j],
                )
            )
            if logr:
                cur_emission += log(emission_lrr(lrrs_clip[i+1], ks[j], a=-4, b=1.0, lrr_mu=lrr_mu, lrr_sd=lrr_sd, eps=eps))
            betas[j,i] = logsumexp(A[:, j] + cur_emission + betas[:, i + 1])
        scaler[i] = logsumexp(betas[:, i])
        betas[:, i] -= scaler[i]
    return betas, scaler, states, None, sum(scaler)

def viterbi_algo(bafs, lrrs, mat_haps, pat_haps, states, A, lrr_mu=lrr_mu, lrr_sd=lrr_sd, double pi0=0.2, double std_dev=0.25, double pi0_lrr=0.2, double eps=1e-6, int logr=True):
    cdef int i,j,n,m;
    assert bafs.size == lrrs.size
    lrrs_clip = np.clip(lrrs, -4, 1.0)
    n = bafs.size
    m = len(states)
    deltas = np.zeros(shape=(m, n))
    deltas[:, 0] = log(1 / m)
    psi = np.zeros(shape=(m, n), dtype=int)
    ks = [sum([s >= 0 for s in state]) for state in states]
    for i in range(1, n):
        for j in range(m):
            m_ij = mat_dosage(mat_haps[:, i], states[j])
            p_ij = pat_dosage(pat_haps[:, i], states[j])
            deltas[j,i] = np.max(deltas[:,i-1] + A[:,j])
            deltas[j,i] += log(
                emission_baf(
                    bafs[i],
                    m_ij,
                    p_ij,
                    pi0=pi0,
                    std_dev=std_dev,
                    eps=eps,
                    k=ks[j],
                )
            )
            if logr:
                deltas[j,i] += log(emission_lrr(lrrs_clip[i], ks[j], a=-4, b=1.0, lrr_mu=lrr_mu, lrr_sd=lrr_sd, eps=eps))
            psi[j, i] = np.argmax(deltas[:, i - 1] + A[:, j]).astype(int)
    path = np.zeros(n, dtype=int)
    path[-1] = np.argmax(deltas[:, -1]).astype(int)
    for i in range(n - 2, -1, -1):
        path[i] = psi[path[i + 1], i]
    path[0] = psi[path[1], 1]
    return path, states, deltas, psi
