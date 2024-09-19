import pandas as pd
import numpy as np
import scipy.stats as st
import scipy.special as sp


def variational_bayes_poisson_mixture(X, K = 2, max_iter = 100, tol = 1e-4):
    '''
    Variational Bayes for a Poisson mixture model.
    X: Observed data points (a 1D numpy array)
    K: Number of latent classes
    '''
    N = len(X)
    
    # Initialize parameters
    alpha = np.ones(K)  # Dirichlet prior for class proportions
    mixing_proportion = alpha / alpha.sum()
    lambdas = np.linspace(np.min(X), np.max(X), K + 2)[1:-1]  # initial mean of Poisson
    beta = np.array([lambdas, np.ones(K)]).T  # Gamma prior for Poisson rates (shape, rate)
    phi = np.random.rand(N, K)
    phi /= phi.sum(axis = 1, keepdims = True)  # responsibilities
    
    # Variational Bayes iterations
    for iter in range(max_iter):
        # responsibilities (phi)
        phi = np.array([ st.poisson.logpmf(X, mu = lambdas[k]) for k in range(K) ]).T
        phi += np.log(mixing_proportion)
        denom = sp.logsumexp(phi, axis = 1, keepdims = True)
        phi = np.exp(phi - denom)
        
        # Update Dirichlet parameters (alpha)
        alpha = 1 + phi.sum(axis = 0)
        
        # Update Gamma parameters (beta)
        beta[:, 0] = 1 + phi.T @ X
        beta[:, 1] = 1 + phi.sum(axis = 0)
        lambdas = beta[:, 0] / beta[:, 1]  # point estimate of Poisson mu
        
        # Calculate Evidence Lower Bound (ELBO)
        mixing_proportion = alpha / alpha.sum()
        elbo = (
            # Dirichlet term
            st.dirichlet.logpdf(mixing_proportion, alpha)
            # Gamma term
            + np.sum([ st.gamma.logpdf(lambdas, a = lambdas[k]) for k in range(K) ])
            # weighted Poisson term
            + sp.logsumexp(
                np.array([ st.poisson.logpmf(X, mu = lambdas[k]) for k in range(K) ]).T
                + np.log(mixing_proportion)
            )
        )
        print('ELBO:', elbo)
        
        # Check for convergence
        if iter > 0 and abs(elbo - prev_elbo) < tol:
            break
        prev_elbo = elbo
    return alpha, beta, phi, iter



# load data
df = pd.read_table('data.tsv')
X = df['X'].to_numpy()
alpha, beta, phi, iter = variational_bayes_poisson_mixture(X)

print(iter)
print('mixing proportion point estimate:', alpha / alpha.sum())
print('lambda point estimate:', beta[:, 0] / beta[:, 1])
print()
print('True mixing proportion:', [0.25, 0.75])
print('True lambda point estimate:', [2, 11])

#with np.printoptions(precision=2, suppress=True):
#    print(phi)
