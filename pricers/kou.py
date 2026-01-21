import numpy as np
import matplotlib.pyplot as plt

# --- 1. Parameters ---
eta1 = 4.0
eta2 = 3.0
p = 0.4
xmax = 2.0        # truncation
deltax = 0.01     # grid step
binw = 0.1        # bin width
n = 10**6         # number of random samples

# --- 2. Compute the Theoretical PDF ---
# Grid creation (equivalent to -xmax:deltax:xmax)
# We add a tiny epsilon to the end to ensure the endpoint is included
x = np.arange(-xmax, xmax + deltax/100, deltax)

# Calculate PDF: fX
# The original logic uses element-wise multiplication with booleans (0 or 1)
# We can do the same in Python or use masking.
# fX = p*eta1*exp(-eta1*x).*(x>=0) + (1-p)*eta2*exp(eta2*x).*(x<0)

fX = np.zeros_like(x)
mask_pos = (x >= 0)
mask_neg = (x < 0)

fX[mask_pos] = p * eta1 * np.exp(-eta1 * x[mask_pos])
fX[mask_neg] = (1 - p) * eta2 * np.exp(eta2 * x[mask_neg])

# --- 3. Sample the Distribution (Inverse Transform Sampling) ---
U = np.random.rand(n)  # Standard uniform random variable [0, 1]
X = np.zeros(n)

# Logic for Inverse Transform Sampling:
# If U >= 1-p, we are in the positive exponential tail
# If U < 1-p,  we are in the negative exponential tail
mask_U_pos = (U >= 1 - p)
mask_U_neg = (U < 1 - p)

# Apply Inverse CDF formulas
# X = -1/eta1 * log((1-U)/p) for upper tail
# X =  1/eta2 * log(U/(1-p)) for lower tail
X[mask_U_pos] = -1 / eta1 * np.log((1 - U[mask_U_pos]) / p)
X[mask_U_neg] =  1 / eta2 * np.log(U[mask_U_neg] / (1 - p))

# --- 4. Plotting ---
plt.figure(figsize=(10, 6))

# Histogram of samples
# density=True normalizes the histogram to form a PDF (area sums to 1)
bins = np.arange(-xmax, xmax + binw, binw)
plt.hist(X, bins=bins, density=True, alpha=0.6, label='Sampled', edgecolor='white')

# Theoretical PDF Line
plt.plot(x, fX, 'r-', linewidth=2, label='Theory')

plt.xlabel('x')
plt.ylabel('$f_X$')
plt.title('Asymmetric double-sided distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([-xmax, xmax])
plt.show()

