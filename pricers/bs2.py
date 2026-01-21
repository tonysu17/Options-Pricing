import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import trapezoid

# ==============================================================================
# 1. Setup Parameters
# ==============================================================================

# Contract parameters
T = 1.0       # maturity
K = 1.1       # strike price

# Market parameters
S0 = 1.0      # initial stock price
r = 0.05      # risk-free interest rate
q = 0.02      # dividend rate

# Model parameter
sigma = 0.4   # volatility

# Fourier parameters
xwidth = 8.0  # width of the support in real space
ngrid = 2**6  # number of grid points (64)
alphac = -6.0 # damping parameter for call
alphap = 6.0  # damping parameter for put

# Monte Carlo parameters
nblocks = 10000 # number of blocks
npaths = 2000   # number of paths per block

# Controls
figures = True # Set to True to generate plots

# ==============================================================================
# 2. Analytical Solution
# ==============================================================================

# Black-Scholes, Own Implementation
t_start = time.perf_counter()

muABM = r - q - 0.5 * sigma**2 # drift coefficient of the arithmetic Brownian motion
d2 = (np.log(S0 / K) + muABM * T) / (sigma * np.sqrt(T))
d1 = d2 + sigma * np.sqrt(T)

Vcao = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
Vpao = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)

cputime_ao = time.perf_counter() - t_start

# Print results
print(f"{'':<20}{'Call':>15}{'Put':>15}{'CPU_time/s':>15}")
# Note: Python's scipy.stats is equivalent to Matlab's Financial Toolbox
print(f"{'Black-Scholes Own':<20}{Vcao:15.10f}{Vpao:15.10f}{cputime_ao:15.10f}")

# --- Plotting (if figures=True) ---
if figures:
    # Meshgrid setup
    S_range = np.arange(0, 2.05, 0.05)
    t_range = np.arange(0, T + 0.025, 0.025)
    St, t_mesh = np.meshgrid(S_range, t_range)

    # Recalculate d1, d2 on grid
    # Avoid division by zero at T-t = 0 (maturity)
    # We add a tiny epsilon or handle T=t separately
    T_minus_t = np.maximum(T - t_mesh, 1e-9)

    d2_mesh = (np.log(St / K) + muABM * T_minus_t) / (sigma * np.sqrt(T_minus_t))
    d1_mesh = d2_mesh + sigma * np.sqrt(T_minus_t)

    # Figure 1: Call Surface (Spot)
    Vc = St * np.exp(-q * T_minus_t) * norm.cdf(d1_mesh) - \
         K * np.exp(-r * T_minus_t) * norm.cdf(d2_mesh)
    # Fix maturity payoff explicitly
    Vc[-1, :] = np.maximum(St[-1, :] - K, 0)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(St, t_mesh, Vc, cmap='viridis')
    ax.set_xlabel('S'); ax.set_ylabel('t'); ax.set_zlabel('V')
    ax.set_title('Call Option Price Surface')
    plt.show()

    # Figure 2: Put Surface (Spot)
    Vp = K * np.exp(-r * T_minus_t) * norm.cdf(-d2_mesh) - \
         St * np.exp(-q * T_minus_t) * norm.cdf(-d1_mesh)
    Vp[-1, :] = np.maximum(K - St[-1, :], 0)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(St, t_mesh, Vp, cmap='viridis')
    ax.set_xlabel('S'); ax.set_ylabel('t'); ax.set_zlabel('V')
    ax.set_title('Put Option Price Surface')
    plt.show()

# ==============================================================================
# 3. Fourier Transform Method
# ==============================================================================
t_start = time.perf_counter()

# Grids
N = ngrid // 2
dx = xwidth / ngrid
x = dx * np.arange(-N, N)   # grid of the log price
b = xwidth / 2              # upper bound
dxi = np.pi / b             # Nyquist relation
xi = dxi * np.arange(-N, N) # grid in Fourier space

# --- Payoff Transform Logic ---
# Function to calculate G (Fourier Transform of Damped Payoff)
# Based on standard analytical result for Call/Put with damping alpha
def analytical_payoff_transform(xi, alpha, K, S0, option_type='call'):
    # Transform of (S - K)+ * s^(-alpha) where s = log(S/S0)
    # G(v) = K^(1 + alpha + i*v) / ((alpha + i*v)*(alpha + 1 + i*v)) ??
    # Let's use the explicit integral form derived from Plancherel logic
    # similar to the MATLAB 'payoff' or Carr-Madan standard.

    fac = alpha + 1j * xi

    if option_type == 'call':
        # Int_k^inf (S0 e^x - K) e^(alpha x) e^(i xi x) dx
        # Assumes alpha < -1 for convergence at +infinity
        # Limit at infinity is 0. Limit at k = log(K/S0)
        lower = np.log(K/S0)
        term_k = S0 * np.exp((1+fac)*lower)/(1+fac) - K * np.exp(fac*lower)/fac
        return -term_k # Upper - Lower = 0 - Term_k

    elif option_type == 'put':
        # Int_-inf^k (K - S0 e^x) e^(alpha x) e^(i xi x) dx
        # Assumes alpha > 0 for convergence at -infinity
        upper = np.log(K/S0)
        term_k = K * np.exp(fac*upper)/fac - S0 * np.exp((1+fac)*upper)/(1+fac)
        return term_k # Upper - Lower = Term_k - 0

# Calculate Gc and Gp
Gc = analytical_payoff_transform(xi, alphac, K, S0, 'call')
Gp = analytical_payoff_transform(xi, alphap, K, S0, 'put')

# --- Characteristic Function ---
# Psi(xi) for Arithmetic Brownian Motion: i*mu*xi - 0.5*sigma^2*xi^2
def get_psi_abm(u):
    return 1j * muABM * u - 0.5 * (sigma * u)**2

# Shifted Characteristic Functions
# Psic = exp(psi(xi + i*alphac) * T)
psi_c_val = get_psi_abm(xi + 1j * alphac)
Psic = np.exp(psi_c_val * T)

# Psip = exp(psi(xi + i*alphap) * T)
psi_p_val = get_psi_abm(xi + 1j * alphap)
Psip = np.exp(psi_p_val * T)

# --- Integration (Plancherel / Parseval) ---
# We integrate Real(G * conj(Psi)) from 0 to infinity (indices N to end)
# MATLAB: trapz(real(Gc(N+1:end).*conj(Psic(N+1:end))))*dxi
# Python indices: N to end
idx_start = N # Corresponds to xi=0 (or close to it depending on even/odd N)

integrand_c = np.real(Gc[idx_start:] * np.conjugate(Psic[idx_start:]))
VcF = np.exp(-r * T) / np.pi * trapezoid(integrand_c, dx=dxi)

integrand_p = np.real(Gp[idx_start:] * np.conjugate(Psip[idx_start:]))
VpF = np.exp(-r * T) / np.pi * trapezoid(integrand_p, dx=dxi)

cputime_F = time.perf_counter() - t_start
print(f"{'Fourier':<20}{VcF:15.10f}{VpF:15.10f}{cputime_F:15.10f}")

# ==============================================================================
# 4. Monte Carlo Simulation
# ==============================================================================
t_start = time.perf_counter()

VcMCb = np.zeros(nblocks)
VpMCb = np.zeros(nblocks)

for i in range(nblocks):
    # 1. Arithmetic Brownian Motion X(T)
    # Size: (1, npaths)
    # X = muABM*T + sigma*randn*sqrt(T)
    X = muABM * T + sigma * np.random.randn(1, npaths) * np.sqrt(T)

    # 2. Transform to Geometric Brownian Motion S(T)
    # S = S0 * exp(X)
    S_T = S0 * np.exp(X)

    # 3. Discounted Payoff
    VcMCb[i] = np.exp(-r * T) * np.mean(np.maximum(S_T - K, 0))
    VpMCb[i] = np.exp(-r * T) * np.mean(np.maximum(K - S_T, 0))

VcMC = np.mean(VcMCb)
VpMC = np.mean(VpMCb)
scMC = np.sqrt(np.var(VcMCb, ddof=1) / nblocks) # Standard Error
spMC = np.sqrt(np.var(VpMCb, ddof=1) / nblocks)

cputime_MC = time.perf_counter() - t_start

print(f"{'Monte Carlo':<20}{VcMC:15.10f}{VpMC:15.10f}{cputime_MC:15.10f}")
print(f"{'Monte Carlo stdev':<20}{scMC:15.10f}{spMC:15.10f}")