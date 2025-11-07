import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from .black_scholes import bs_call_price, bs_put_price, d1


def implied_vol(price, S, K, T, r, option_type='call', method='brent'):
    if T <= 0:
        return np.nan

    if option_type == 'call':
        intrinsic = max(S - K * np.exp(-r * T), 0)
        max_price = S
        bs_func = bs_call_price
    else:
        intrinsic = max(K * np.exp(-r * T) - S, 0)
        max_price = K * np.exp(-r * T)
        bs_func = bs_put_price

    if price < intrinsic - 1e-10:
        return np.nan
    if price > max_price + 1e-10:
        return np.nan

    # price at intrinsic means zero vol
    if price <= intrinsic + 1e-10:
        return 0.0

    if method == 'newton':
        return _newton_iv(price, S, K, T, r, option_type)
    else:
        return _brent_iv(price, S, K, T, r, bs_func)


def _brent_iv(price, S, K, T, r, bs_func):
    def objective(sigma):
        return bs_func(S, K, T, r, sigma) - price

    try:
        return brentq(objective, 1e-6, 5.0, xtol=1e-8)
    except ValueError:
        return np.nan


def _newton_iv(price, S, K, T, r, option_type, sigma0=0.3, tol=1e-8, max_iter=50):
    sigma = sigma0

    if option_type == 'call':
        bs_func = bs_call_price
    else:
        bs_func = bs_put_price

    for _ in range(max_iter):
        bs_price = bs_func(S, K, T, r, sigma)
        diff = bs_price - price

        if abs(diff) < tol:
            return sigma

        # vega for Newton step
        d1_val = d1(S, K, T, r, sigma)
        vega = S * norm.pdf(d1_val) * np.sqrt(T)

        if vega < 1e-12:
            break

        sigma = sigma - diff / vega
        sigma = np.clip(sigma, 1e-6, 5.0)

    return sigma
