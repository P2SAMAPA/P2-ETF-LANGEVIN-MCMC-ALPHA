import numpy as np
from config import PRIOR_MEAN, PRIOR_VARIANCE

def log_prior(weights):
    """Log prior: independent normal N(0, prior_variance)"""
    return -0.5 * np.sum((weights - PRIOR_MEAN)**2 / PRIOR_VARIANCE)

def log_likelihood(weights, returns):
    """
    Log likelihood of portfolio returns assuming Gaussian.
    Returns: (weights^T returns) ~ N(mu_p, sigma_p^2)
    """
    port_returns = returns @ weights
    mu = np.mean(port_returns)
    sigma2 = np.var(port_returns)
    # Gaussian log-likelihood
    n = len(port_returns)
    return -0.5 * n * np.log(2 * np.pi * sigma2) - 0.5 * np.sum((port_returns - mu)**2 / sigma2)

def log_posterior(weights, returns):
    """Log posterior = log prior + log likelihood"""
    return log_prior(weights) + log_likelihood(weights, returns)

def gradient_log_posterior(weights, returns):
    """
    Gradient of log posterior w.r.t weights.
    For Gaussian likelihood: gradient = (returns.T @ (port_returns - mu)) / sigma2 - weights / prior_variance
    """
    port_returns = returns @ weights
    mu = np.mean(port_returns)
    sigma2 = np.var(port_returns)
    n = len(port_returns)
    
    # Gradient of likelihood
    dL = (returns.T @ (port_returns - mu)) / (sigma2 * n)
    # Gradient of prior
    dP = - (weights - PRIOR_MEAN) / PRIOR_VARIANCE
    return dL + dP
