from backtest import run_backtest
from results_uploader import upload_results

def main():
    print("Running SGLD backtest...")
    trade_dates, portfolio_weights = run_backtest()
    # Convert to DataFrame
    import pandas as pd
    results = pd.DataFrame(portfolio_weights, index=trade_dates)
    upload_results({"posterior_weights": results})
    print("Done.")

if __name__ == "__main__":
    main()
