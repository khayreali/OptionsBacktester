import numpy as np
from scipy.optimize import minimize, least_squares


def svi_raw(k, a, b, rho, m, sigma):
    """SVI total variance: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))"""
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))


def svi_iv(k, a, b, rho, m, sigma, T):
    """Convert SVI total variance to implied vol."""
    w = svi_raw(k, a, b, rho, m, sigma)
    return np.sqrt(np.maximum(w, 0) / T)


def svi_fit(strikes, ivs, forward, T):
    """
    Fit SVI to a single expiry smile.

    strikes: array of strikes
    ivs: array of implied vols
    forward: forward price
    T: time to expiry

    Returns dict with SVI params and fit quality.
    """
    k = np.log(strikes / forward)
    w = ivs**2 * T  # total variance

    def objective(params):
        a, b, rho, m, sigma = params
        w_fit = svi_raw(k, a, b, rho, m, sigma)
        return np.sum((w - w_fit)**2)

    # butterfly constraint: a + b*sigma*sqrt(1-rho^2) >= 0
    def butterfly(params):
        a, b, rho, m, sigma = params
        return a + b * sigma * np.sqrt(1 - rho**2)

    x0 = [np.mean(w), 0.1, -0.3, 0.0, 0.1]

    bounds = [
        (1e-6, None),     # a > 0
        (1e-6, None),     # b > 0
        (-0.99, 0.99),    # |rho| < 1
        (-0.5, 0.5),      # m near ATM
        (1e-6, None),     # sigma > 0
    ]

    result = minimize(
        objective, x0,
        method='SLSQP',
        bounds=bounds,
        constraints={'type': 'ineq', 'fun': butterfly},
    )

    a, b, rho, m, sigma = result.x
    w_fit = svi_raw(k, a, b, rho, m, sigma)
    rmse = np.sqrt(np.mean((w - w_fit)**2))

    return {
        'a': a, 'b': b, 'rho': rho, 'm': m, 'sigma': sigma,
        'success': result.success,
        'rmse': rmse,
        'rmse_iv': np.sqrt(rmse / T),  # approx IV RMSE
    }


def poly_fit(strikes, ivs, forward, degree=2):
    """
    Simple polynomial fit to smile as fallback.
    Fits iv = c0 + c1*k + c2*k^2 + ... where k = log(K/F)
    """
    k = np.log(strikes / forward)

    coeffs = np.polyfit(k, ivs, degree)

    iv_fit = np.polyval(coeffs, k)
    rmse = np.sqrt(np.mean((ivs - iv_fit)**2))

    return {
        'coeffs': coeffs,
        'degree': degree,
        'rmse': rmse,
    }


def poly_iv(k, coeffs):
    """Evaluate polynomial smile."""
    return np.polyval(coeffs, k)


class SmileFitter:
    """Convenience class to fit and evaluate smile models."""

    def __init__(self, method='svi'):
        self.method = method
        self.params = None
        self.forward = None
        self.T = None

    def fit(self, strikes, ivs, forward, T):
        self.forward = forward
        self.T = T

        if self.method == 'svi':
            self.params = svi_fit(strikes, ivs, forward, T)
        else:
            self.params = poly_fit(strikes, ivs, forward)

        return self.params

    def __call__(self, K):
        """Get IV at strike K."""
        k = np.log(K / self.forward)

        if self.method == 'svi':
            p = self.params
            return svi_iv(k, p['a'], p['b'], p['rho'], p['m'], p['sigma'], self.T)
        else:
            return poly_iv(k, self.params['coeffs'])
