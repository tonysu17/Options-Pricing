import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define Parameters and Time Grid ---
npaths = 100000        # Number of paths
T = 1.0               # Time horizon
nsteps = 200          # Number of time steps
dt = T / nsteps       # Time step
t = np.linspace(0, T, nsteps + 1) # Observation times

# Model parameters
muS = 0.2             # Expected return (Drift of the Geometric Brownian Motion)
sigma = 0.3           # Volatility of diffusion
muJ = -0.1            # Mean of the jump size
sigmaJ = 0.15         # Volatility of the jump size
lam = 0.5             # Lambda: Jump intensity (jumps per year)
S0 = 1.0              # Initial stock price

# --- 2. Monte Carlo Simulation ---

# A. Compute increments of the ABM (Arithmetic Brownian Motion)
# dW = (drift - 0.5*sigma^2)*dt + sigma * sqrt(dt) * Z
drift_term = (muS - 0.5 * sigma**2) * dt
diffusion_term = sigma * np.sqrt(dt) * np.random.randn(nsteps, npaths)
dW = drift_term + diffusion_term

# B. Compute increments of the NCPP (Normal Compound Poisson Process)
# Poisson number of jumps in each step
dN = np.random.poisson(lam * dt, size=(nsteps, npaths))

# Jump sizes: For each step, sum of normal variables = Normal(N*mu, sqrt(N)*sigma)
# Note: we multiply sigmaJ by sqrt(dN) because the variance of sum of N variables is N*sigma^2
dJ = muJ * dN + sigmaJ * np.sqrt(dN) * np.random.randn(nsteps, npaths)

# C. Sum increments
dX = dW + dJ

# D. Accumulate increments to get process X
# We prepend a row of zeros (starting value X0 = 0 for log returns)
X = np.vstack([np.zeros((1, npaths)), np.cumsum(dX, axis=0)])

# Optional: Convert to Stock Price (commented out as in original)
# S = S0 * np.exp(X)

# --- 3. Plotting: Paths ---
plt.figure(figsize=(10, 6))

# Theoretical Expected Path
# The MATLAB code uses (muS + lambda*muJ)*t as expectation for the plot
expected_path = (muS + lam * muJ) * t

# Plot
plt.plot(t, expected_path, 'k', linewidth=2, label='Expected path')
plt.plot(t, np.mean(X, axis=1), ':k', linewidth=2, label='Mean path')
# Plot a subset of paths (every 1000th path) to avoid clutter
plt.plot(t, X[:, ::1000], alpha=0.5)
# Re-plot expected/mean on top for visibility
plt.plot(t, expected_path, 'k', linewidth=2)
plt.plot(t, np.mean(X, axis=1), ':k', linewidth=2)

plt.legend(['Expected path', 'Mean path'])
plt.xlabel('t')
plt.ylabel('X')
plt.ylim([-1, 1.2])
plt.title(r'Paths of a Merton jump-diffusion process $X(t) = \mu t + \sigma W(t) + \Sigma_{i=1}^{N(t)} Z_i$')
plt.show()

# --- 4. Plotting: Probability Density Functions ---
plt.figure(figsize=(8, 10))

# Time steps to plot: 40 (0.2s), 100 (0.5s), 200 (1.0s)
# Note: Python uses 0-based indexing, but since we prepended a zero row,
# index 40 corresponds to the 40th step (time 0.2) just like in the MATLAB logic where X(1,:) is time 0.
time_indices = [40, 100, 200]
time_labels = [0.2, 0.5, 1.0]

for i, (idx, time_val) in enumerate(zip(time_indices, time_labels)):
    plt.subplot(3, 1, i+1)

    # Extract data slice
    data_slice = X[idx, :]

    # Create histogram
    # density=True normalizes it to form a PDF
    count, bins, ignored = plt.hist(data_slice, bins=100, density=True, alpha=0.7, color='C0', edgecolor='white')

    plt.xlim([-1, 1.2])
    plt.ylim([0, 3])
    plt.ylabel(f'$f_X(x, {time_val})$')

    if i == 0:
        plt.title('Probability density function of a Merton jump-diffusion process at different times')
    if i == 2:
        plt.xlabel('x')

plt.tight_layout()
plt.show()

