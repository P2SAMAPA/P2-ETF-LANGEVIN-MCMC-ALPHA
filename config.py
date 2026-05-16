"""
Configuration for P2-ETF-LANGEVIN-MCMC-ALPHA engine.
"""

import os
from datetime import datetime

# --- Hugging Face Repositories ---
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-langevin-mcmc-alpha-results"

# --- Universe Definitions (same as other engines) ---
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM", "IWD"
]
ALL_TICKERS = list(set(FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": ALL_TICKERS
}

# Which universe to use for optimisation
ACTIVE_UNIVERSE = "COMBINED"   # or "FI_COMMODITIES", "EQUITY_SECTORS"

# --- Macro Features (available but not used directly) ---
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# --- Portfolio optimisation parameters ---
RETURN_LOOKBACK = 252          # days for computing returns
PRIOR_MEAN = 0.0               # prior on weights (zero-mean normal)
PRIOR_VARIANCE = 0.01          # prior variance (shrinkage)

# --- SGLD Hyperparameters ---
SGLD_STEPS = 1000
SGLD_BURNIN = 500
SGLD_STEP_SIZE = 1e-4
SGLD_TEMPERATURE = 1.0         # T→0 gives MAP
SGLD_GRADIENT_BATCH_SIZE = 128

# --- Walk-forward backtest ---
TRAIN_WINDOW = 504             # days (2 years)
REBALANCE_FREQ = 21            # days (monthly rebalancing)
COMMISSION = 0.001             # 10bps per trade

# --- Results ---
LOCAL_RESULTS_DIR = "results"

# --- Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# --- Run ID ---
TODAY = datetime.now().strftime("%Y-%m-%d")
