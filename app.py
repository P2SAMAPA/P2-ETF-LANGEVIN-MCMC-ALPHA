import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from huggingface_hub import HfApi, hf_hub_download
import re
from config import HF_OUTPUT_REPO, UNIVERSES, ACTIVE_UNIVERSE, HF_TOKEN

st.set_page_config(layout="wide")
st.title("🧠 Langevin MCMC Alpha – Bayesian Portfolio Optimisation")
st.markdown("**Stochastic Gradient Langevin Dynamics (SGLD)** posterior weights for ETF portfolios")

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
    except:
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

def main():
    st.sidebar.header("Configuration")
    universe = st.sidebar.selectbox("Select Universe", list(UNIVERSES.keys()), index=0)
    tickers = UNIVERSES[universe]
    
    latest_run = get_latest_run_folder()
    if latest_run is None:
        st.info("No results found. Run the backtest first.")
        return
    
    weights_df = load_weights(latest_run)
    if weights_df is None:
        st.info("No weight data available.")
        return
    
    # Keep only tickers that exist in the results
    available_tickers = [t for t in tickers if t in weights_df.columns]
    if not available_tickers:
        st.error("No matching tickers in results.")
        return
    
    weights_subset = weights_df[available_tickers]
    
    # Plot portfolio weights over time
    fig = go.Figure()
    for t in available_tickers:
        fig.add_trace(go.Scatter(x=weights_subset.index, y=weights_subset[t], mode='lines', name=t))
    fig.update_layout(title="Posterior Mean Weights Over Time", xaxis_title="Date", yaxis_title="Weight")
    st.plotly_chart(fig, use_container_width=True)
    
    # Latest weights bar chart
    st.subheader("Latest Portfolio Weights")
    latest_weights = weights_subset.iloc[-1].sort_values(ascending=False)
    fig_bar = px.bar(x=latest_weights.index, y=latest_weights.values, title="Current Posterior Mean Weights")
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Full table
    st.subheader("Full History (first 5 rows)")
    st.dataframe(weights_subset.head())

if __name__ == "__main__":
    main()
