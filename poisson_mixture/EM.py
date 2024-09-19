import pandas as pd
import numpy as np
import scipy.stats as st
import scipy.special as sp


def em_poisson_mixture(X, K = 2, max_iter = 100, tol = 1e-4):
    '''
    EM algorithm for a Poisson mixture model.
    X: Observed data points (a 1D numpy array)
    K: Number of latent classes
    '''
    # initialize parameters
    lambdas = np.linspace(np.min(X), np.max(X), K + 2)[1:-1]  # Poisson mean
    pis = np.ones(K) / K  # mixing proportions
    
    for iter in range(max_iter):
        # E-step, calculation of responsibilities
        log_likelihoods = np.log(pis) + np.array([ st.poisson.logpmf(X, l) for l in lambdas ]).T
        log_denom = sp.logsumexp(log_likelihoods, axis = 1)
        log_phi = log_likelihoods - log_denom[:, np.newaxis]
        phi = np.exp(log_phi)  # responsibilities
        
        # M-step, update parameters
        pis = np.mean(phi, axis = 0)  # mixing proportions
        lambdas = phi.T @ X / phi.sum(axis = 0)  # Poisson mean
        
        # calc log-likelihood of Poisson term
        log_likelihoods = np.log(pis) + np.array([ st.poisson.logpmf(X, l) for l in lambdas ]).T
        log_likelihood = sp.logsumexp(log_likelihoods)
        
        # Check for convergence
        if iter > 0 and np.abs(log_likelihood - prev_log_likelihood) < tol:
            break
        prev_log_likelihood = log_likelihood
        
    return lambdas, pis, iter


# load data
df = pd.read_table('data.tsv')
X = df['X'].to_numpy()
lambdas, pis, iter = em_poisson_mixture(X)

print(iter)
print('mixing proportion estimate:', pis)
print('lambda estimate:', lambdas)
print()
print('True mixing proportion:', [0.25, 0.75])
print('True lambda point estimate:', [2, 11])
