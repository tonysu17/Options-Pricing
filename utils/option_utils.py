# utils/option_utils.py
from scipy.stats import norm
from scipy.optimize import brentq
import numpy as np

def bs_call_price(S, K, T, r, q, sigma):
    """ Black-Scholes-Merton formula for European call price. """
    if T <= 0 or sigma <= 0:
        return np.maximum(S * np.exp(-q * T) - K * np.exp(-r * T), 0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_vol_call(price, S, K, T, r, q):
    """ Computes implied volatility for a call option using Brent's method. """
    if price <= np.maximum(S * np.exp(-q * T) - K * np.exp(-r * T), 0) * 0.999:
        return 0.0 # Price is below intrinsic value
    
    def objective_func(sigma):
        return bs_call_price(S, K, T, r, q, sigma) - price
    
    try:
        iv = brentq(objective_func, 1e-6, 5.0) # Search between 0.01% and 500% vol
    except ValueError:
        iv = np.nan # No solution found in the interval
    return iv