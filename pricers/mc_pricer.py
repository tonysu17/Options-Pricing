# pricers/mc_pricer.py
import numpy as np

def heston_mc_pricer(S0, K, r, q, T, v0, kappa, theta, sigma, rho, n_paths=10000, n_steps=252):
    """
    Prices a European call option using Monte Carlo simulation for the Heston model.
    """
    dt = T / n_steps
    
    # Generate correlated random numbers
    z1 = np.random.standard_normal((n_paths, n_steps))
    z2 = np.random.standard_normal((n_paths, n_steps))
    dw1 = np.sqrt(dt) * z1
    dw2 = np.sqrt(dt) * (rho * z1 + np.sqrt(1 - rho**2) * z2)
    
    # Initialize paths
    S = np.full(n_paths, S0)
    v = np.full(n_paths, v0)
    
    # Simulate paths
    for _ in range(n_steps):
        # Full truncation scheme for variance
        v_pos = np.maximum(v, 0)
        S = S * np.exp((r - q - 0.5 * v_pos) * dt + np.sqrt(v_pos) * dw1[:, _])
        v = v + kappa * (theta - v) * dt + sigma * np.sqrt(v_pos) * dw2[:, _]
        
    # Calculate discounted payoff
    payoff = np.maximum(S - K, 0)
    price = np.mean(payoff) * np.exp(-r * T)
    
    return price

def vg_mc_pricer(S0, K, r, q, T, sigma_vg, nu, theta_vg, n_paths=10000, n_steps=1):
    """
    Prices a European call option using Monte Carlo simulation for the VG model.
    Since VG has independent increments, we can simulate in a single step.
    """
    dt = T / n_steps
    
    # Martingale correction
    omega = (1 / nu) * np.log(1 - theta_vg * nu - 0.5 * sigma_vg**2 * nu)
    mu = r - q + omega
    
    # Initialize terminal prices
    ST = np.zeros(n_paths)
    
    for i in range(n_paths):
        log_S = np.log(S0)
        for _ in range(n_steps):
            # Simulate Gamma process increment
            g = np.random.gamma(shape=dt/nu, scale=nu)
            # Simulate standard normal
            z = np.random.standard_normal()
            # VG process increment
            delta_X = theta_vg * g + sigma_vg * np.sqrt(g) * z
            # Update log price
            log_S += mu * dt + delta_X
        ST[i] = np.exp(log_S)
        
    # Calculate discounted payoff
    payoff = np.maximum(ST - K, 0)
    price = np.mean(payoff) * np.exp(-r * T)
    
    return price