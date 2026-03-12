import numpy as np
from scipy.stats import norm

def bsm_price(S, K, T, R, sigma):
    """Calcule le prix d'un call européen avec Black-Scholes."""
    d1 = (np.log(S / K) + (R + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-R * T) * norm.cdf(d2)
    return price
#Price d'un put 

def bsm_price2(S, K, T, R, sigma):
    """Calcule le prix d'un put européen avec Black-Scholes."""
    d1 = (np.log(S / K) + (R + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price =  K * np.exp(-R * T) * norm.cdf(-d2)-S * norm.cdf(-d1)
    return price
def bsm_vega(S, K, T, R, sigma):
    """Calcule le Vega de l'option."""
    d1 = (np.log(S / K) + (R + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    return vega

def implied_vol_newton_call(S, K, T, R, market_price, sigma_guess=0.2, n_iter=100, tol=1e-6):
    sigma = sigma_guess
    for i in range(n_iter):
        price = bsm_price(S, K, T, R, sigma)
        diff = price - market_price
        
        if abs(diff) < tol:
            return sigma, i
        
        vega = bsm_vega(S, K, T, R, sigma)
        
        # Mise à jour Newton-Raphson
        sigma = sigma - diff / vega
        
    return sigma, n_iter
#Put
def implied_vol_newton_put(S, K, T, R, market_price, sigma_guess=0.2, n_iter=100, tol=1e-6):
    sigma = sigma_guess
    for i in range(n_iter):
        price = bsm_price2(S, K, T, R, sigma)
        diff = price - market_price
        
        if abs(diff) < tol:
            return sigma, i
        
        vega = bsm_vega(S, K, T, R, sigma)
        
        # Mise à jour Newton-Raphson
        sigma = sigma - diff / vega
        
    return sigma, n_iter
# Paramètres de test
S, K, T, R = 100, 100, 1, 0.03
market_price = 6

vol_imp, iterations = implied_vol_newton_call(S, K, T, R, market_price)
vol_imp2, iterations2 = implied_vol_newton_put(S, K, T, R, market_price)


print(f"Volatilité Implicite call : {vol_imp:.2%}")
print(f"Nombre d'itérations : {iterations}")
print(f"Volatilité Implicite : {vol_imp2:.2%}")
print(f"Nombre d'itérations : {iterations2}")
