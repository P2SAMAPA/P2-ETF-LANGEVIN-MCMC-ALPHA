import numpy as np
from posterior import log_posterior, gradient_log_posterior

def sgld_sampler(returns, initial_weights, step_size, temperature, n_steps, burnin, batch_size):
    """
    Stochastic Gradient Langevin Dynamics.
    Returns: samples (list of weight vectors after burnin)
    """
    weights = np.array(initial_weights, dtype=float)
    n_assets = len(weights)
    N = len(returns)
    samples = []
    
    for step in range(n_steps):
        # Mini-batch gradient
        idx = np.random.choice(N, size=min(batch_size, N), replace=False)
        batch_returns = returns.iloc[idx] if hasattr(returns, 'iloc') else returns[idx]
        grad = gradient_log_posterior(weights, batch_returns)
        
        # Add Langevin noise: sqrt(2 * step_size * temperature) * N(0,1)
        noise = np.sqrt(2 * step_size * temperature) * np.random.randn(n_assets)
        
        # Euler update
        weights = weights + 0.5 * step_size * grad + noise
        
        # Store after burn-in
        if step >= burnin:
            samples.append(weights.copy())
    
    return np.array(samples)
