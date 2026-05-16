# Langevin MCMC Alpha – Bayesian Portfolio Optimisation

Stochastic Gradient Langevin Dynamics (SGLD) for Bayesian ETF portfolio weights.  
Posterior mean weights are computed daily and uploaded to Hugging Face Hub.

## Features
- SGLD sampler with temperature control (T→0 gives MAP, T>0 gives uncertainty)
- Walk‑forward backtest (daily weights, monthly rebalancing)
- Automated daily runs via GitHub Actions
- Streamlit dashboard to visualise weights over time

## Setup

1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`
3. Set your `HF_TOKEN` environment variable or GitHub secret.
4. Run backtest locally: `python run_backtest.py`
5. Launch dashboard: `streamlit run app.py`

## Configuration

Edit `config.py` to change universe, SGLD hyperparameters, or backtest settings.

## Results

Posterior weights are stored in `P2SAMAPA/p2-etf-langevin-mcmc-alpha-results` as Parquet files.
