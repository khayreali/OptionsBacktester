import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List


@dataclass
class PnLBreakdown:
    """Single period P&L attribution."""
    delta_pnl: float
    gamma_pnl: float
    theta_pnl: float
    vega_pnl: float
    residual: float
    total: float


class PnLAttribution:
    """
    Break down option P&L into greek components using Taylor expansion.

    For option price change dV:
    dV ≈ delta * dS + 0.5 * gamma * dS^2 + theta * dt + vega * d(sigma)

    The residual captures higher-order effects and model error.
    """

    def __init__(self, multiplier: int = 100):
        self.multiplier = multiplier
        self.history: List[PnLBreakdown] = []

    def attribute(
        self,
        dS: float,
        dt: float,
        d_sigma: float,
        delta: float,
        gamma: float,
        theta: float,
        vega: float,
        actual_option_pnl: float,
    ) -> PnLBreakdown:
        """
        Attribute option P&L to greeks for a single period.

        dS: spot change
        dt: time change in years (e.g., 1/365 for daily)
        d_sigma: IV change
        delta, gamma, theta, vega: greeks at start of period
        actual_option_pnl: realized option P&L (already multiplied by contract size)
        """
        # scale greeks by multiplier
        delta_pnl = delta * dS * self.multiplier
        gamma_pnl = 0.5 * gamma * dS**2 * self.multiplier
        theta_pnl = theta * dt * 365 * self.multiplier  # theta is daily, dt in years
        vega_pnl = vega * d_sigma * 100 * self.multiplier  # vega per 1% vol

        explained = delta_pnl + gamma_pnl + theta_pnl + vega_pnl
        residual = actual_option_pnl - explained

        breakdown = PnLBreakdown(
            delta_pnl=delta_pnl,
            gamma_pnl=gamma_pnl,
            theta_pnl=theta_pnl,
            vega_pnl=vega_pnl,
            residual=residual,
            total=actual_option_pnl,
        )
        self.history.append(breakdown)
        return breakdown

    def attribute_series(
        self,
        spots: np.ndarray,
        times: np.ndarray,  # T values (time to expiry)
        ivs: np.ndarray,
        deltas: np.ndarray,
        gammas: np.ndarray,
        thetas: np.ndarray,
        vegas: np.ndarray,
        option_prices: np.ndarray,
    ) -> pd.DataFrame:
        """
        Attribute P&L for a time series.
        All arrays should be same length. Uses values at start of each period.
        """
        n = len(spots)
        results = []

        for i in range(1, n):
            dS = spots[i] - spots[i-1]
            dt = times[i-1] - times[i]  # time passes, T decreases
            d_sigma = ivs[i] - ivs[i-1]
            actual_pnl = (option_prices[i] - option_prices[i-1]) * self.multiplier

            breakdown = self.attribute(
                dS=dS,
                dt=dt,
                d_sigma=d_sigma,
                delta=deltas[i-1],
                gamma=gammas[i-1],
                theta=thetas[i-1],
                vega=vegas[i-1],
                actual_option_pnl=actual_pnl,
            )
            results.append(vars(breakdown))

        return pd.DataFrame(results)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([vars(b) for b in self.history])

    def summary(self) -> dict:
        if not self.history:
            return {}

        df = self.to_dataframe()
        return {
            'total_pnl': df['total'].sum(),
            'delta_pnl': df['delta_pnl'].sum(),
            'gamma_pnl': df['gamma_pnl'].sum(),
            'theta_pnl': df['theta_pnl'].sum(),
            'vega_pnl': df['vega_pnl'].sum(),
            'residual': df['residual'].sum(),
            'pct_explained': 1 - abs(df['residual'].sum()) / abs(df['total'].sum()) if df['total'].sum() != 0 else 0,
        }


def attribute_hedged_pnl(hedge_df: pd.DataFrame, multiplier: int = 100) -> pd.DataFrame:
    """
    Convenience function to attribute P&L from a DeltaHedger result DataFrame.
    """
    attr = PnLAttribution(multiplier=multiplier)

    spots = hedge_df['spot'].values
    ivs = hedge_df['iv'].values
    deltas = hedge_df['delta'].values
    gammas = hedge_df['gamma'].values
    thetas = hedge_df['theta'].values
    vegas = hedge_df['vega'].values
    option_prices = hedge_df['option_price'].values

    # compute time to expiry changes (assume daily = 1/365)
    n = len(spots)
    times = np.arange(n)[::-1] / 365  # decreasing time

    result = attr.attribute_series(
        spots, times, ivs, deltas, gammas, thetas, vegas, option_prices
    )

    # add hedge P&L
    result['hedge_pnl'] = hedge_df['hedge_pnl'].values[1:]
    result['net_pnl'] = result['total'] + result['hedge_pnl']

    return result
