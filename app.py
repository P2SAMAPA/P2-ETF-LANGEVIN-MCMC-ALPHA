import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from huggingface_hub import HfApi, hf_hub_download
import re
from config import HF_OUTPUT_REPO, UNIVERSES, ACTIVE_UNIVERSE, HF_TOKEN

st.set_page_config(layout="wide")

# --- Dashboard title and educational content ---
st.title("🧠 Langevin MCMC Alpha – Bayesian Portfolio Optimisation")
st.markdown("**Stochastic Gradient Langevin Dynamics (SGLD)** posterior weights for ETF portfolios")

with st.expander("📖 How to use this dashboard for trading ETFs", expanded=False):
    st.markdown("""
    **What are these “posterior mean weights”?**  
    - The model learns a probability distribution over optimal portfolio weights given past returns.  
    - **Posterior mean** = average weight from thousands of MCMC samples → your best estimate for each ETF.  
    - **Posterior standard deviation** (not shown in this dashboard, but available in the raw data) represents uncertainty.

    **How to trade using the chart:**  
    1. **Rebalance monthly** to the latest weights shown in the bar chart.  
    2. **If a weight becomes negative** → it indicates a short position (but our model currently allows shorts; you can set a lower bound of 0 for long‑only).  
    3. **Trend in weights over time** – increasing weight on an ETF suggests rising expected return (according to the model).  

    **Practical strategy:**  
    - At each rebalance date, allocate capital proportionally to the posterior mean weights (normalise to sum = 1 if needed).  
    - Use the posterior standard deviation to size positions: lower weight if uncertainty is high (e.g., weight / variance).  

    **Limitations:**  
    - The model assumes Gaussian returns and independent priors.  
    - Transaction costs are not yet incorporated into the optimisation (but can be added).  
    - Past performance does not guarantee future results.
    """)

# --- Sidebar with theory explanation ---
st.sidebar.header("Configuration")
st.sidebar.markdown("---")
with st.sidebar.expander("📐 Theory: Posterior Mean Weights", expanded=False):
    st.markdown("""
    **Bayesian posterior**  
    `P(weights | returns) ∝ P(returns | weights) * P(weights)`  
    - Prior `P(weights)` : normal distribution favouring small weights (shrinkage).  
    - Likelihood `P(returns | weights)` : Gaussian distribution of portfolio returns.  

    **SGLD sampler**  
    - Adds calibrated noise to gradient descent → draws samples from posterior.  
    - Temperature `T=1` gives full posterior; `T→0` collapses to MAP estimate.  

    **Posterior mean** = expected optimal portfolio under the model.  
    Use it as a Bayesian estimator that automatically regularises and quantifies uncertainty.
    """)

# --- Existing universe selection and data loading ---
universe_options = list(UNIVERSES.keys())
default_index = universe_options.index(ACTIVE_UNIVERSE) if ACTIVE_UNIVERSE in universe_options else 0
selected_universe = st.sidebar.selectbox("Select Universe", universe_options, index=default_index)
tickers = UNIVERSES[selected_universe]

@st.cache_data(ttl=3600)
def get_latest_run_folder():
    if not HF_TOKEN:
        return None
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=HF_OUTPUT_REPO, repo_type="dataset", token=HF_TOKEN)
        run_folders = set()
        for f in files:
            match = re.match(r"(\d{8}_\d{6})/", f)
            if match:
                run_folders.add(match.group(1))
        if not run_folders:
            return None
        return sorted(run_folders, reverse=True)[0]
    except Exception as e:
        st.warning(f"Could not list repo files: {e}")
        return None

@st.cache_data
def load_weights(run_folder):
    try:
        df = pd.read_parquet(
            hf_hub_download(
                repo_id=HF_OUTPUT_REPO,
                filename=f"{run_folder}/posterior_weights.parquet",
                repo_type="dataset",
                token=HF_TOKEN if HF_TOKEN else None
            )
        )
        return df
    except Exception as e:
        st.warning(f"Could not load weights: {e}")
        return None

latest_run = get_latest_run_folder()
if latest_run is None:
    st.info("No results found. Run the backtest first.")
    st.stop()

weights_df = load_weights(latest_run)
if weights_df is None:
    st.info("No weight data available.")
    st.stop()

available_tickers = [t for t in tickers if t in weights_df.columns]
if not available_tickers:
    st.error(f"No matching tickers between universe `{selected_universe}` and results. "
             f"Expected: {tickers[:5]}... Found columns: {list(weights_df.columns)[:5]}...")
    st.stop()

weights_subset = weights_df[available_tickers]

st.caption(f"Results from run: `{latest_run}` | Universe: `{selected_universe}` | {len(available_tickers)} assets | Dates: {weights_subset.index.min().date()} to {weights_subset.index.max().date()}")

# --- Plot weights over time ---
fig = go.Figure()
for t in available_tickers:
    fig.add_trace(go.Scatter(x=weights_subset.index, y=weights_subset[t], mode='lines', name=t))
fig.update_layout(title="Posterior Mean Weights Over Time", xaxis_title="Date", yaxis_title="Weight")
st.plotly_chart(fig, use_container_width=True)

# --- Latest weights bar chart ---
st.subheader("Latest Portfolio Weights")
latest_weights = weights_subset.iloc[-1].sort_values(ascending=False)
fig_bar = px.bar(x=latest_weights.index, y=latest_weights.values, title=f"Current Weights ({weights_subset.index[-1].date()})")
st.plotly_chart(fig_bar, use_container_width=True)

# --- Full history scrollable ---
st.subheader("Full History (scrollable)")
st.dataframe(weights_subset, use_container_width=True, height=500)

# --- Optional summary stats ---
with st.expander("Summary Statistics (annualised mean weight over period)"):
    annual_means = weights_subset.mean() * 252   # roughly annualised average weight
    st.dataframe(annual_means.to_frame(name="Average Weight (annualised factor)"))

# --- Footer ---
st.markdown("---")
st.caption("Posterior mean weights from SGLD. Rebalance monthly to these weights. Uncertainty not shown here but available in raw posterior samples.")
