# pricers/fft_pricer.py
import numpy as np

def price_options_fft(S0, r, q, T, char_func, model_params, N=4096, eta=0.1, alpha=1.5):
    """
    Prices European call options for a range of strikes using the Carr-Madan FFT method.
    
    Args:
        S0 (float): Initial asset price.
        r (float): Risk-free interest rate.
        q (float): Dividend yield.
        T (float): Time to maturity.
        char_func (function): The characteristic function of the log-asset price model.
        model_params (dict): Dictionary of parameters for the characteristic function.
        N (int): Number of FFT points (must be a power of 2).
        eta (float): Spacing of the frequency grid.
        alpha (float): Damping parameter.

    Returns:
        tuple: A tuple containing arrays of strikes and corresponding call prices.
    """
    i = 1j
    
    # Grid spacing
    delta_k = 2 * np.pi / (N * eta)
    
    # Frequency grid
    u = np.arange(N) * eta
    
    # Log-strike grid centered around log(S0)
    k0 = np.log(S0) - (N / 2) * delta_k
    k = k0 + np.arange(N) * delta_k
    strikes = np.exp(k)
    
    # Simpson's rule weights for higher accuracy
    simpson_weights = np.array( +  * (N // 2 - 1) + ) * (eta / 3)
    
    # Calculate integrand for FFT
    phi = char_func(u - i * (alpha + 1), T, r, q, **model_params)
    integrand = (np.exp(-r * T) * phi) / ((alpha + i * u) * (alpha + 1 + i * u))
    
    # Perform FFT
    fft_input = np.exp(-i * k0 * u) * integrand * simpson_weights
    fft_output = np.fft.fft(fft_input)
    
    # Recover call prices
    call_prices = (np.exp(-alpha * k) / np.pi) * np.real(fft_output)
    
    return strikes, call_prices