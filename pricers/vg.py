import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define parameters and time grid ---
npaths = 20000        # number of paths
T = 1.0               # time horizon
nsteps = 200          # number of time steps
dt = T / nsteps       # time step
t = np.linspace(0, T, nsteps + 1) # observation times

# Model parameters
mu = 0.2              # drift of the arithmetic Brownian motion
sigma = 0.3           # volatility of the arithmetic Brownian motion
kappa = 0.05          # scale parameter of the gamma process (variance of time)

# --- 2. Monte Carlo Simulation ---

# Compute the increments of the gamma process
# MATLAB: gamrnd(shape, scale).
# Python: np.random.gamma(shape, scale)
# Shape = dt/kappa, Scale = kappa. Mean = dt, Variance = kappa*dt.
dG = np.random.gamma(shape=dt/kappa, scale=kappa, size=(nsteps, npaths))

# Compute the increments of the ABM on the gamma random clock
# dX = mu*dG + sigma * Z * sqrt(dG)
dX = mu * dG + sigma * np.random.randn(nsteps, npaths) * np.sqrt(dG)

# Accumulate the increments
# Prepend a row of zeros (X0 = 0)
X = np.vstack([np.zeros((1, npaths)), np.cumsum(dX, axis=0)])

# --- 3. Plot 1: Expected, mean and sample paths ---
plt.figure(figsize=(10, 6))

EX = mu * t # expected path
mean_path = np.mean(X, axis=1)

# Plot a subset of paths (every 1000th path) to match MATLAB style
plt.plot(t, X[:, ::1000], alpha=0.5)
plt.plot(t, EX, 'k', linewidth=2, label='Expected path')
plt.plot(t, mean_path, ':k', linewidth=2, label='Mean path')
# Re-plot expected/mean on top for visibility
plt.plot(t, EX, 'k', linewidth=2)
plt.plot(t, mean_path, ':k', linewidth=2)

plt.legend()
plt.xlabel('t')
plt.ylabel('X')
plt.ylim([-0.8, 1.2])
plt.title(r'Paths of a variance Gamma process $dX(t) = \mu dG(t) + \sigma dW(G(t))$')
plt.grid(True, alpha=0.3)
plt.show()

# --- 4. Plot 2: Probability density function at different times ---
plt.figure(figsize=(8, 10))

# Indices for t=0.2, t=0.5, t=1.0
# MATLAB indices: 40, 100, end. Python (0-based) needs equivalent logic.
# If 200 steps = 1.0s, then:
# t=0.2 -> step 40
# t=0.5 -> step 100
# t=1.0 -> step 200 (end)

indices = [40, 100, 200]
times = [0.2, 0.5, 1.0]
ylimits = [6, 3, 3]

for i, (idx, time_val) in enumerate(zip(indices, times)):
    plt.subplot(3, 1, i+1)

    # Extract data at this time step
    data_slice = X[idx, :]

    # Create histogram
    # density=True acts like the manual normalization in the MATLAB script
    count, bins, ignored = plt.hist(data_slice, bins=100, density=True,
                                    alpha=0.7, color='C0', edgecolor='white')

    plt.ylabel(f'$f_X(x, {time_val})$')
    plt.xlim([-0.8, 1.2])
    plt.ylim([0, ylimits[i]])

    if i == 0:
        plt.title('Probability density function of a variance Gamma process at different times')
    if i == 2:
        plt.xlabel('x')

plt.tight_layout()
plt.show()

