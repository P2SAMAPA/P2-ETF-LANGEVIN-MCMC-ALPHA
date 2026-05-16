import numpy as np
from sgld_sampler import sgld_sampler
from config import SGLD_STEPS, SGLD_BURNIN, SGLD_STEP_SIZE, SGLD_TEMPERATURE, SGLD_GRADIENT_BATCH_SIZE

def optimize_portfolio(returns, initial_weights=None):
    """
    Run SGLD to obtain posterior samples of optimal weights.
    Returns: posterior_mean, posterior_std, samples
    """
    n_assets = returns.shape[1]
    if initial_weights is None:
        # Equal weight as starting point
        initial_weights = np.ones(n_assets) / n_assets
    
    samples = sgld_sampler(
        returns=returns,
        initial_weights=initial_weights,
        step_size=SGLD_STEP_SIZE,
        temperature=SGLD_TEMPERATURE,
        n_steps=SGLD_STEPS,
        burnin=SGLD_BURNIN,
        batch_size=SGLD_GRADIENT_BATCH_SIZE
    )
    posterior_mean = np.mean(samples, axis=0)
    posterior_std = np.std(samples, axis=0)
    return posterior_mean, posterior_std, samples
