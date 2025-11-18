import numpy as np
import pandas as pd
import time
import datetime as dt
import yfinance as yf

# Import all custom modules from the project
from models.characteristic_funcs import heston_char_func, vg_char_func
from pricers.fft_pricer import price_options_fft
from pricers.mc_pricer import heston_mc_pricer, vg_mc_pricer
from utils.option_utils import bs_call_price, implied_vol_call
from utils.data_handler import fetch_option_chain, clean_option_data
from visuals.plotter import plot_vol_surface_3d

# Set numpy print options for better readability
np.set_printoptions(precision=4, suppress=True)

def main():
    """
    Main execution script to run the option pricing analysis.
    1. Runs a benchmark of FFT vs. MC for Heston and VG models.
    2. Generates and plots model-implied volatility surfaces.
    3. Fetches real market data and plots the market-implied volatility surface.
    """
    
    # --- 1. General Market and Model Parameters ---
    S0 = 100.0       # Initial asset price
    r = 0.03         # Risk-free rate (e.g., 3%)
    q = 0.01         # Dividend yield (e.g., 1%)
    T_bench = 1.0    # Benchmark maturity (1 year)
    
    # Heston parameters
    heston_params = {
        'v0': 0.04,      # Initial variance (sqrt(0.04) = 20% vol)
        'kappa': 2.0,    # Speed of mean reversion
        'theta': 0.04,   # Long-term variance
        'sigma': 0.3,    # Volatility of variance
        'rho': -0.7      # Correlation between asset and variance
    }
    
    # Variance Gamma parameters
    vg_params = {
        'sigma_vg': 0.2,   # Volatility of the Brownian motion component
        'nu': 0.1,         # Variance rate of the Gamma process (controls kurtosis)
        'theta_vg': -0.14  # Drift of the Brownian motion (controls skew)
    }
    
    # Simulation parameters
    MC_PATHS = 50000
    MC_STEPS = 252 # Daily steps for 1 year (for Heston)
    
    # FFT parameters
    N_FFT = 4096
    ETA_FFT = 0.1
    ALPHA_FFT = 1.5

    # --- 2. Benchmark: FFT vs. Monte Carlo ---
    print("--- Running Model Benchmark (FFT vs. MC) ---")
    benchmark_strikes = [80.0, 100.0, 120.0]

    # --- Heston Benchmark ---
    print(f"\n[Heston Model] (T={T_bench}, {MC_PATHS} paths, {MC_STEPS} steps)")
    
    # Heston FFT (prices all strikes at once)
    start_fft_h = time.time()
    heston_fft_strikes, heston_fft_prices = price_options_fft(
        S0, r, q, T_bench, heston_char_func, heston_params, N_FFT, ETA_FFT, ALPHA_FFT
    )
    end_fft_h = time.time()
    print(f"FFT Time: {end_fft_h - start_fft_h:.5f} s (for {N_FFT} strikes)")
    
    print("Strike | FFT Price | MC Price | MC Time (s)")
    print("---------------------------------------------")
    
    for K in benchmark_strikes:
        # Get matching FFT price by interpolation
        fft_price = np.interp(K, heston_fft_strikes, heston_fft_prices)
        
        # Run MC for this single strike
        start_mc_h = time.time()
        mc_price = heston_mc_pricer(
            S0, K, r, q, T_bench, **heston_params, 
            n_paths=MC_PATHS, n_steps=MC_STEPS
        )
        end_mc_h = time.time()
        print(f"{K:<6.1f} | {fft_price:<9.4f} | {mc_price:<9.4f} | {end_mc_h - start_mc_h:.4f}")

    # --- Variance Gamma Benchmark ---
    print(f"\n[Variance Gamma Model] (T={T_bench}, {MC_PATHS} paths, 1 step)")
    
    # VG FFT (prices all strikes at once)
    start_fft_vg = time.time()
    vg_fft_strikes, vg_fft_prices = price_options_fft(
        S0, r, q, T_bench, vg_char_func, vg_params, N_FFT, ETA_FFT, ALPHA_FFT
    )
    end_fft_vg = time.time()
    print(f"FFT Time: {end_fft_vg - start_fft_vg:.5f} s (for {N_FFT} strikes)")
    
    print("Strike | FFT Price | MC Price | MC Time (s)")
    print("---------------------------------------------")
    
    for K in benchmark_strikes:
        # Get matching FFT price by interpolation
        fft_price = np.interp(K, vg_fft_strikes, vg_fft_prices)
        
        # Run MC for this single strike
        start_mc_vg = time.time()
        mc_price = vg_mc_pricer(
            S0, K, r, q, T_bench, **vg_params, 
            n_paths=MC_PATHS, n_steps=1 # VG is an exact simulation, 1 step is fine
        )
        end_mc_vg = time.time()
        print(f"{K:<6.1f} | {fft_price:<9.4f} | {mc_price:<9.4f} | {end_mc_vg - start_mc_vg:.4f}")

    # --- 3. Generate Model-Implied Volatility Surfaces ---
    print("\n--- Generating Model-Implied Volatility Surfaces ---")
    
    T_vec = np.linspace(0.1, 2.0, 10) # 10 maturities from ~1 month to 2 years
    model_vol_data = # To store (model_name, T, strike, iv)

    for T in T_vec:
        # --- Heston Prices & IV ---
        heston_strikes, heston_prices = price_options_fft(
            S0, r, q, T, heston_char_func, heston_params, N_FFT, ETA_FFT, ALPHA_FFT
        )
        for K, price in zip(heston_strikes, heston_prices):
            if K > 0.7 * S0 and K < 1.3 * S0: # Filter for a reasonable moneyness range
                iv = implied_vol_call(price, S0, K, T, r, q)
                if not np.isnan(iv) and iv > 1e-4:
                    model_vol_data.append(('Heston', T, K, iv))
        
        # --- Variance Gamma Prices & IV ---
        vg_strikes, vg_prices = price_options_fft(
            S0, r, q, T, vg_char_func, vg_params, N_FFT, ETA_FFT, ALPHA_FFT
        )
        for K, price in zip(vg_strikes, vg_prices):
            if K > 0.7 * S0 and K < 1.3 * S0: # Filter
                iv = implied_vol_call(price, S0, K, T, r, q)
                if not np.isnan(iv) and iv > 1e-4:
                    model_vol_data.append(('VG', T, K, iv))

    # Create DataFrame and plot surfaces
    model_df = pd.DataFrame(model_vol_data, columns=)
    
    # Plot Heston Surface
    heston_df = model_df[model_df['model'] == 'Heston']
    print("Displaying Heston Model Volatility Surface...")
    plot_vol_surface_3d(heston_df, title='Heston Model-Implied Volatility Surface')
    
    # Plot VG Surface
    vg_df = model_df[model_df['model'] == 'VG']
    print("Displaying Variance Gamma Model Volatility Surface...")
    plot_vol_surface_3d(vg_df, title='Variance Gamma Model-Implied Volatility Surface')

    # --- 4. Generate Market-Implied Volatility Surface ---
    print("\n--- Generating Market-Implied Volatility Surface ---")
    TICKER = 'SPY' # S&P 500 ETF, very liquid option chain
    print(f"Fetching option data for {TICKER}...")
    
    try:
        # Get current market price for S0
        ticker_yf = yf.Ticker(TICKER)
        S_market = ticker_yf.history(period='1d')['Close'].iloc
        print(f"Current {TICKER} price: {S_market:.2f}")

        # Fetch and clean data using our data_handler
        raw_chain = fetch_option_chain(TICKER)
        cleaned_chain = clean_option_data(raw_chain, S_market)
        
        # We only want calls for this analysis
        market_calls = cleaned_chain[cleaned_chain['type'] == 'call'].copy()
        
        print(f"Calculating market IV for {len(market_calls)} liquid call options...")
        
        market_ivs =
        # Use a fixed rate for simplicity. 
        # A professional setup would use a risk-free rate curve.
        market_r = 0.03 
        market_q = 0.01

        for _, row in market_calls.iterrows():
            # Use our implied_vol_call solver
            iv = implied_vol_call(
                row['mid_price'], S_market, row['strike'], 
                row, market_r, market_q
            )
            if not np.isnan(iv) and iv > 1e-4 and iv < 2.0: # Filter outliers
                market_ivs.append((row, row['strike'], iv))
        
        market_df = pd.DataFrame(market_ivs, columns=)
        
        # Plot Market Surface
        print("Displaying Market-Implied Volatility Surface...")
        plot_vol_surface_3d(market_df, title=f'{TICKER} Market-Implied Volatility Surface')

    except Exception as e:
        print(f"Could not fetch or process market data. Error: {e}")
        print("Skipping market volatility surface generation.")
        print("Note: Market data fetching requires an active internet connection.")

if __name__ == "__main__":
    main()