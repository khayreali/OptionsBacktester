import numpy as np


def moneyness(S: float, K: float) -> float:
    return S / K


def log_moneyness(S: float, K: float, r: float = 0.0, T: float = 0.0) -> float:
    # forward log-moneyness
    F = S * np.exp(r * T)
    return np.log(K / F)


def annualize_vol(daily_returns: np.ndarray) -> float:
    return np.std(daily_returns) * np.sqrt(252)


def realized_vol(prices: np.ndarray, window: int = 20) -> np.ndarray:
    log_returns = np.diff(np.log(prices))
    vol = np.full(len(prices), np.nan)

    for i in range(window, len(prices)):
        vol[i] = np.std(log_returns[i-window:i]) * np.sqrt(252)

    return vol


def days_to_expiry(expiry_date, as_of_date) -> int:
    return (expiry_date - as_of_date).days


def business_days_between(start_date, end_date) -> int:
    # rough estimate, doesn't account for holidays
    import pandas as pd
    return len(pd.bdate_range(start_date, end_date)) - 1
