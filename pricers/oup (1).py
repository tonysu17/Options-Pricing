import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- 1. Define Parameters and Time Grid ---
npaths = 20000        # number of paths
T = 1.0               # time horizon
nsteps = 200          # number of time steps
dt = T / nsteps       # time step
t = np.linspace(0, T, nsteps + 1) # observation times (0 to T)

# Model parameters (Note: alpha is mean reversion speed, mu is mean level)
alpha = 5.0
mu = 0.07
sigma = 0.07
X0 = 0.03             # initial value

# --- 2. Monte Carlo Simulation ---

# Allocate paths (Shape: [npaths, nsteps+1])
# We prepend X0 to every path
X = np.zeros((npaths, nsteps + 1))
X[:, 0] = X0

# Sample standard Gaussian random numbers [npaths, nsteps]
N = np.random.randn(npaths, nsteps)

# Compute the standard deviation for a time step
# Choice 1: Standard Euler
# sdev = sigma * np.sqrt(dt)

# Choice 2: Euler with Analytic Moments (Exact Variance)
# Var(X_t) = sigma^2 / (2*alpha) * (1 - exp(-2*alpha*dt))
sdev = sigma * np.sqrt((1 - np.exp(-2 * alpha * dt)) / (2 * alpha))

# Pre-calculate mean reversion factor for efficiency
exp_alpha_dt = np.exp(-alpha * dt)

# Compute and accumulate increments
# Note: Iterating is necessary because X[i] depends on X[i-1] for the mean drift
# (unless we used a full vectorized exact solution formula, but this follows the MATLAB loop logic)
for i in range(nsteps):
    # X_next = Mean(X_curr) + Std * Z
    # Mean(X_curr) = mu + (X_curr - mu) * exp(-alpha*dt)
    X[:, i+1] = mu + (X[:, i] - mu) * exp_alpha_dt + sdev * N[:, i]

# --- 3. Plot 1: Paths ---
plt.figure(figsize=(10, 6))

EX = mu + (X0 - mu) * np.exp(-alpha * t) # Theoretical Expected path
MeanPath = np.mean(X, axis=0)

# Plot subset of paths
plt.plot(t, X[:50, :].T, alpha=0.3) # Plot first 20 paths
plt.plot(t, EX, 'k', linewidth=2, label='Expected path')
plt.plot(t, MeanPath, 'k:', linewidth=4, label='Mean path')
plt.axhline(mu, color='k', linestyle='--', label='Long-term average')

sdevinfty = sigma / np.sqrt(2 * alpha)
plt.ylim([mu - 4 * sdevinfty, mu + 4 * sdevinfty])
plt.xlabel('t')
plt.ylabel('X')
plt.title(r'Ornstein-Uhlenbeck process $dX = \alpha(\mu-X)dt + \sigma dW$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# --- 4. Plot 2: Variance ---
plt.figure(figsize=(10, 6))

# Theory Variance
Var_Theory = (sigma**2 / (2 * alpha)) * (1 - np.exp(-2 * alpha * t))
# Asymptotic Variance (sigma^2 / 2alpha)
Var_Infinite = (sigma**2 / (2 * alpha)) * np.ones_like(t)
# Sampled Variance
Var_Sampled1 = np.var(X, axis=0) # var(X)
Var_Sampled2 = np.mean((X - EX)**2, axis=0) # E[(X - E[X])^2]

plt.plot(t, Var_Theory, 'r', label='Theory', linewidth=2)
plt.plot(t, sigma**2 * t, 'g', label=r'$\sigma^2 t$ (BM approximation)')
plt.plot(t, Var_Infinite, 'b', label=r'$\sigma^2/(2\alpha)$ (Long term)')
plt.plot(t, Var_Sampled1, 'm', label='Sampled Var')
plt.plot(t, Var_Sampled2, 'c--', label='Sampled MSE')

plt.xlabel('t')
plt.ylabel(r'Var(X)')
plt.ylim([0, 0.0006])
plt.legend(loc='lower right')
plt.title('Ornstein-Uhlenbeck process: variance')
plt.grid(True, alpha=0.3)
plt.show()

# --- 5. Plot 3: Mean Absolute Deviation ---
plt.figure(figsize=(10, 6))

# Theory MAD: sigma * sqrt( (1-exp)/pi*alpha )
MAD_Theory = sigma * np.sqrt((1 - np.exp(-2 * alpha * t)) / (np.pi * alpha))
# Brownian Motion Limit: sigma * sqrt(2t/pi)
MAD_BM = sigma * np.sqrt(2 * t / np.pi)
# Long term average MAD: sigma / sqrt(pi*alpha)
MAD_Infinite = (sigma / np.sqrt(np.pi * alpha)) * np.ones_like(t)
# Sampled MAD
MAD_Sampled = np.mean(np.abs(X - EX), axis=0)

plt.plot(t, MAD_Theory, 'r', label='Theory')
plt.plot(t, MAD_BM, 'g', label=r'$\sigma\sqrt{2t/\pi}$')
plt.plot(t, MAD_Infinite, 'b', label='Long-term average')
plt.plot(t, MAD_Sampled, 'm', label='Sampled')

plt.xlabel('t')
plt.ylabel(r'$E(|X-E(X)|)$')
plt.ylim([0, 0.02])
plt.legend(loc='lower right')
plt.title('Ornstein-Uhlenbeck process: mean absolute deviation')
plt.grid(True, alpha=0.3)
plt.show()

# --- 6. Plot 4: PDF at different times ---
plt.figure(figsize=(10, 6))

x_grid = np.linspace(-0.02, mu + 4 * sdevinfty, 200)
t_snapshots = [0.05, 0.1, 0.2, 0.4, 1.0]

for i, t_val in enumerate(t_snapshots):
    # Find closest index
    idx = int(round(t_val / dt))

    # Analytical Parameters for this time
    ex_t = mu + (X0 - mu) * np.exp(-alpha * t_val)
    sd_t = sigma * np.sqrt((1 - np.exp(-2 * alpha * t_val)) / (2 * alpha))

    # Analytical PDF
    fa = norm.pdf(x_grid, ex_t, sd_t)

    # Sampled PDF (Histogram)
    counts, bins = np.histogram(X[:, idx], bins=50, density=True)
    centers = (bins[:-1] + bins[1:]) / 2

    color = f'C{i}'
    plt.plot(x_grid, fa, color=color, label=f'Theory t={t_val}')
    plt.plot(centers, counts, '.', color=color, alpha=0.5)

plt.xlabel('x')
plt.ylabel('f_X(x,t)')
plt.legend()
plt.title('Ornstein-Uhlenbeck process: PDF at different times')
plt.grid(True, alpha=0.3)
plt.show()

# --- 7. Plot 5: Autocovariance ---
# C(tau) = Cov(X(t), X(t+tau)) for large t
# We approximate using the time series of the paths
# Efficient computation using numpy correlation
# Note: MATLAB's xcorr is signal processing correlation.
# We will compute Autocovariance manually for "tau" lags from the data.

lags = np.arange(nsteps + 1)
C_sampled = np.zeros(len(lags))

# We want C(tau) roughly for the stationary part.
# But the script calculates it using the whole path X - EX
# Let's replicate the MATLAB loop logic efficiently
# MATLAB: xcorr(X(:,j)-EX, 'unbiased')
# This computes the auto-correlation of the fluctuations around the mean path

# Efficient vectorized Autocovariance
residuals = X - EX # Remove deterministic trend
# Variance at each lag averaged over paths
# To match MATLAB 'xcorr' exactly is tricky, but we can compute the
# covariance of the process at steady state logic.
# Let's compute Cov(X_t, X_{t+tau}) average over t?
# The MATLAB code `xcorr` on the residual vector effectively averages over time t.

# Simplified robust calculation for Python:
# Calculate autocovariance of the residuals averaged across all paths
# autocov[k] = mean( sum(res[i]*res[i+k]) )
# We use fftconvolve for speed or just loop since nsteps=200 is small.

for tau in range(nsteps):
    # Covariance between R[t] and R[t+tau]
    # We avail of the ergodicity or just average over the available pairs
    # MATLAB xcorr 'unbiased' scales by 1/(N-tau)

    # Slice 1: 0 to N-tau
    r1 = residuals[:, :nsteps+1-tau]
    # Slice 2: tau to N
    r2 = residuals[:, tau:]

    # Element-wise product, mean over paths, then mean over time t
    prod = r1 * r2
    # Mean over paths AND time
    C_sampled[tau] = np.mean(prod)

plt.figure(figsize=(10, 6))
# Theory: sigma^2/(2a) * exp(-alpha * tau)
C_theory = (sigma**2 / (2 * alpha)) * np.exp(-alpha * t)

plt.plot(t, C_theory, 'r', label='Theory (infinite t)')
plt.plot(t, C_sampled, 'g', label='Sampled')
plt.plot(0, sigma**2 / (2 * alpha), 'go', label='Var (Theory)')
plt.plot(0, np.mean(np.var(X, axis=0)), 'bo', label='Var (Sampled)')

plt.xlabel(r'$\tau$')
plt.ylabel(r'$C(\tau)$')
plt.legend()
plt.title('Ornstein-Uhlenbeck process: Autocovariance')
plt.grid(True, alpha=0.3)
plt.show()

# --- 8. Plot 6: Autocorrelation ---
plt.figure(figsize=(10, 6))

# Theory: exp(-alpha * tau)
Corr_Theory = np.exp(-alpha * t)
# Sampled: C(tau) / C(0)
Corr_Sampled = C_sampled / C_sampled[0]

plt.plot(t, Corr_Theory, 'r', label='Theory')
plt.plot(t, Corr_Sampled, 'g', label='Sampled')

plt.xlabel(r'$\tau$')
plt.ylabel(r'$c(\tau)$')
plt.legend()
plt.title('Ornstein-Uhlenbeck process: Autocorrelation')
plt.grid(True, alpha=0.3)
plt.show()

