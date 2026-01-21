import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import norm

# %% Monte Carlo Simulation for Discretely Monitored Fixed-Strike Lookback Options
# Based on Fusai, Germano and Marazzina (EJOR, 2016)
# Reference Values:
# Call on Max: 0.1832645983
# Put on Min:  0.1178715852

# 1. Contract & Model Parameters
# ---------------------------------------------------------
T = 0.5         # Maturity
S0 = 1.0        # Initial stock price
K = 1.0         # Strike price
r = 0.1         # Risk-free interest rate
q = 0.0         # Dividend rate
sigma = 0.3     # Volatility
N = 50          # Number of monitoring dates

# Derived parameters
dt = T / N
sqrt_dt = np.sqrt(dt)
mu_log = r - q - 0.5 * sigma**2
df = np.exp(-r * T)

# Targets (Fusai et al. 2016)
target_call = 0.1832645983
target_put = 0.1178715852

# 2. Simulation Configuration
# ---------------------------------------------------------
# Schedule: 10k -> 100k -> 1M -> 10M
path_steps = [10_000, 100_000, 1_000_000, 10_000_000]
BATCH_SIZE = 500_000  # Process paths in chunks to manage memory (prevents OOM on 10M)

print(f"{'--- Monte Carlo Convergence (Call & Put) ---':^80}")
print(f"{'N Paths':<10} | {'Call Price':<10} | {'Call Err':<9} | {'Put Price':<10} | {'Put Err':<9} | {'Time(s)':<7}")
print("-" * 85)

# Storage for plotting
results_call = []
results_put = []
errors_call = []
errors_put = []
se_call_list = []
se_put_list = []

# 3. Simulation Loop
# ---------------------------------------------------------
for n_sim in path_steps:
    start_time = time.time()

    # Initialize Accumulators for Mean and Variance
    sum_call = 0.0
    sum_sq_call = 0.0
    sum_put = 0.0
    sum_sq_put = 0.0

    # Determine number of batches needed
    # (If n_sim < BATCH_SIZE, we just run 1 batch of size n_sim)
    num_batches = int(np.ceil(n_sim / BATCH_SIZE))

    paths_processed = 0

    for b in range(num_batches):
        # Calculate current batch size (handle last partial batch)
        current_batch = min(BATCH_SIZE, n_sim - paths_processed)

        if current_batch <= 0:
            break

        # Generate Z (Standard Normal) for this batch
        # Shape: (current_batch, N)
        Z = np.random.randn(current_batch, N)

        # Calculate log-returns
        dX = mu_log * dt + sigma * sqrt_dt * Z

        # Accumulate log-returns to get path of X = log(S/S0)
        X_path = np.cumsum(dX, axis=1)

        # Convert to Stock Price paths
        S_paths = S0 * np.exp(X_path)

        # --- Payoff Calculation ---

        # Lookback Call: depends on Maximum over the path
        S_max = np.max(S_paths, axis=1)
        payoff_c = np.maximum(S_max - K, 0)

        # Lookback Put: depends on Minimum over the path
        S_min = np.min(S_paths, axis=1)
        payoff_p = np.maximum(K - S_min, 0)

        # Accumulate Statistics
        sum_call += np.sum(payoff_c)
        sum_sq_call += np.sum(payoff_c**2)

        sum_put += np.sum(payoff_p)
        sum_sq_put += np.sum(payoff_p**2)

        paths_processed += current_batch

    # --- Final Metrics for N_sim ---

    # Pricing
    avg_call = sum_call / n_sim
    avg_put = sum_put / n_sim

    price_c = df * avg_call
    price_p = df * avg_put

    # Errors
    err_c = abs(price_c - target_call)
    err_p = abs(price_p - target_put)

    # Standard Error Calculation
    # Var = E[X^2] - (E[X])^2
    var_c = (sum_sq_call / n_sim) - (avg_call**2)
    var_p = (sum_sq_put / n_sim) - (avg_put**2)

    se_c = df * np.sqrt(var_c / n_sim)
    se_p = df * np.sqrt(var_p / n_sim)

    end_time = time.time()
    elapsed = end_time - start_time

    # Store for plots
    results_call.append(price_c)
    results_put.append(price_p)
    errors_call.append(err_c)
    errors_put.append(err_p)
    se_call_list.append(se_c)
    se_put_list.append(se_p)

    # Print Row
    print(f"{n_sim:<10} | {price_c:.5f}    | {err_c:.1e}   | {price_p:.5f}    | {err_p:.1e}   | {elapsed:.4f}")

# 4. Final Verification & Plots
# ---------------------------------------------------------
print("-" * 85)
print(f"Final Call Price (N={path_steps[-1]}): {results_call[-1]:.8f} (Target: {target_call})")
print(f"Final Put Price  (N={path_steps[-1]}): {results_put[-1]:.8f} (Target: {target_put})")

# Convergence Plot
plt.figure(figsize=(12, 5))

# Plot 1: Error Convergence
plt.subplot(1, 2, 1)
plt.loglog(path_steps, errors_call, 'o-', color='b', label='Call Error')
plt.loglog(path_steps, errors_put, 's-', color='r', label='Put Error')

# Reference O(1/sqrt(N)) line
ref_x = np.array(path_steps)
ref_y = ref_x**(-0.5) * (errors_call[0] * np.sqrt(path_steps[0]))
plt.loglog(ref_x, ref_y, 'k:', label=r'$O(1/\sqrt{N})$')

plt.xlabel('Number of Paths (N)')
plt.ylabel('Absolute Error')
plt.title('Monte Carlo Error Convergence')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()

# Plot 2: Price Estimates with Confidence Intervals
plt.subplot(1, 2, 2)
# Call
plt.errorbar(path_steps, results_call, yerr=1.96*np.array(se_call_list), fmt='o-', color='b', label='MC Call')
plt.axhline(y=target_call, color='b', linestyle='--', alpha=0.5, label='Target Call')
# Put
plt.errorbar(path_steps, results_put, yerr=1.96*np.array(se_put_list), fmt='s-', color='r', label='MC Put')
plt.axhline(y=target_put, color='r', linestyle='--', alpha=0.5, label='Target Put')

plt.xscale('log')
plt.xlabel('Number of Paths (N)')
plt.ylabel('Option Price')
plt.title('Price Estimates vs Target')
plt.legend()
plt.grid(True, which="both", alpha=0.5)

plt.tight_layout()
plt.show()

