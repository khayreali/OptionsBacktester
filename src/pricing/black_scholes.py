import numpy as np
from scipy.stats import norm


def d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def bs_call_price(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0)
    if sigma <= 0:
        return max(S - K * np.exp(-r * T), 0)

    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    return S * norm.cdf(d1_val) - K * np.exp(-r * T) * norm.cdf(d2_val)


def bs_put_price(S, K, T, r, sigma):
    if T <= 0:
        return max(K - S, 0)
    if sigma <= 0:
        return max(K * np.exp(-r * T) - S, 0)

    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2_val) - S * norm.cdf(-d1_val)


def bs_greeks(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0:
        if option_type == 'call':
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0
        return {'delta': delta, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}

    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    sqrt_T = np.sqrt(T)
    disc = np.exp(-r * T)
    pdf_d1 = norm.pdf(d1_val)

    # delta
    if option_type == 'call':
        delta = norm.cdf(d1_val)
    else:
        delta = norm.cdf(d1_val) - 1

    # gamma
    gamma = pdf_d1 / (S * sigma * sqrt_T)

    # vega (per 1% vol move)
    vega = S * pdf_d1 * sqrt_T / 100

    # theta (per day)
    theta_common = -(S * pdf_d1 * sigma) / (2 * sqrt_T)
    if option_type == 'call':
        theta = theta_common - r * K * disc * norm.cdf(d2_val)
    else:
        theta = theta_common + r * K * disc * norm.cdf(-d2_val)
    theta = theta / 365

    # rho (per 1% rate move)
    if option_type == 'call':
        rho = K * T * disc * norm.cdf(d2_val) / 100
    else:
        rho = -K * T * disc * norm.cdf(-d2_val) / 100

    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }
