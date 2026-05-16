from backtest import run_backtest
from results_uploader import upload_results
from config import ACTIVE_UNIVERSE, UNIVERSES

def main():
    print(f"Running SGLD backtest for universe: {ACTIVE_UNIVERSE} ({len(UNIVERSES[ACTIVE_UNIVERSE])} tickers)")
    weights_df = run_backtest()
    if weights_df is not None and not weights_df.empty:
        print(f"Results shape: {weights_df.shape}")
        print(f"Columns (first 5): {list(weights_df.columns[:5])}")
        upload_results({"posterior_weights": weights_df})
        print("Backtest completed and uploaded.")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
