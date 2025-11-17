# models/characteristic_funcs.py
import numpy as np

def heston_char_func(u, T, r, q, v0, kappa, theta, sigma, rho):
    """
    Heston (1993) characteristic function for the log-asset price.
    """
    i = 1j
    u = np.array(u, dtype=np.complex128)
    
    # Risk-neutral drift
    mu = r - q
    
    # Heston trap correction terms can be added here
    d = np.sqrt((rho * sigma * i * u - kappa)**2 - sigma**2 * (2 * i * u - u**2))
    g = (kappa - rho * sigma * i * u - d) / (kappa - rho * sigma * i * u + d)
    
    C = mu * i * u * T + (kappa * theta / sigma**2) * (
        (kappa - rho * sigma * i * u - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))
    )
    
    D = ((kappa - rho * sigma * i * u - d) / sigma**2) * (
        (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
    )
    
    return np.exp(C + D * v0)

def vg_char_func(u, T, r, q, sigma_vg, nu, theta_vg):
    """
    Variance Gamma (VG) characteristic function for the log-asset price.
    """
    i = 1j
    u = np.array(u, dtype=np.complex128)
    
    # Martingale correction for risk-neutral measure
    omega = (1 / nu) * np.log(1 - theta_vg * nu - 0.5 * sigma_vg**2 * nu)
    
    # Risk-neutral drift
    mu = r - q + omega
    
    # Characteristic function of the VG process
    phi_vg = (1 - i * u * theta_vg * nu + 0.5 * sigma_vg**2 * nu * u**2)**(-T / nu)
    
    return np.exp(i * u * mu * T) * phi_vg