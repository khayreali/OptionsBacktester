import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, List

from src.models import Option
from src.pricing import bs_call_price, bs_put_price, bs_greeks


@dataclass
class HedgeState:
    date: date
    spot: float
    option_price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    iv: float
    hedge_shares: float
    shares_traded: float
    transaction_cost: float
    option_pnl: float
    hedge_pnl: float
    total_pnl: float
    cumulative_pnl: float


class DeltaHedger:
    """
    Simulates delta-hedged P&L for an options position.
    Steps through spot data, rehedges at specified frequency,
    tracks all P&L components.
    """

    def __init__(
        self,
        option: Option,
        spot_data: pd.DataFrame,
        vol_surface=None,
        constant_vol: float = 0.20,
        r: float = 0.05,
        rehedge_frequency: str = 'daily',
        transaction_cost: float = 0.0,  # $ per share
        position_size: int = 1,  # number of contracts (each = 100 shares)
    ):
        self.option = option
        self.spot_data = spot_data.copy()
        self.vol_surface = vol_surface
        self.constant_vol = constant_vol
        self.r = r
        self.rehedge_frequency = rehedge_frequency
        self.transaction_cost = transaction_cost
        self.position_size = position_size
        self.multiplier = position_size * 100

        self.history: List[HedgeState] = []
        self.hedge_shares = 0.0
        self.cumulative_pnl = 0.0
        self.total_transaction_costs = 0.0

    def _get_vol(self, K: float, T: float) -> float:
        if self.vol_surface is not None:
            try:
                return self.vol_surface.interpolate(K, T)
            except:
                pass
        return self.constant_vol

    def _price_option(self, S: float, T: float, sigma: float) -> float:
        if T <= 0:
            return self.option.intrinsic(S)
        if self.option.option_type == 'call':
            return bs_call_price(S, self.option.K, T, self.r, sigma)
        return bs_put_price(S, self.option.K, T, self.r, sigma)

    def _get_greeks(self, S: float, T: float, sigma: float) -> dict:
        if T <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        return bs_greeks(S, self.option.K, T, self.r, sigma, self.option.option_type)

    def _should_rehedge(self, idx: int) -> bool:
        if self.rehedge_frequency == 'daily':
            return True
        elif self.rehedge_frequency == 'weekly':
            return idx % 5 == 0
        elif self.rehedge_frequency == 'never':
            return idx == 0
        return True

    def run(self) -> pd.DataFrame:
        df = self.spot_data.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.date
        else:
            df['date'] = df.index

        df = df.sort_values('date').reset_index(drop=True)

        prev_spot = None
        prev_option_price = None

        for idx, row in df.iterrows():
            dt = row['date']
            spot = row['close']

            T = self.option.time_to_expiry(dt)
            if T < 0:
                break

            sigma = self._get_vol(self.option.K, T)
            option_price = self._price_option(spot, T, sigma)
            greeks = self._get_greeks(spot, T, sigma)

            # P&L calculation
            if prev_spot is None:
                option_pnl = 0.0
                hedge_pnl = 0.0
            else:
                option_pnl = (option_price - prev_option_price) * self.multiplier
                hedge_pnl = self.hedge_shares * (spot - prev_spot)

            total_pnl = option_pnl + hedge_pnl

            # rehedge if needed
            shares_traded = 0.0
            txn_cost = 0.0

            if self._should_rehedge(idx):
                target_hedge = -greeks['delta'] * self.multiplier
                shares_traded = target_hedge - self.hedge_shares
                txn_cost = abs(shares_traded) * self.transaction_cost
                self.hedge_shares = target_hedge
                self.total_transaction_costs += txn_cost

            total_pnl -= txn_cost
            self.cumulative_pnl += total_pnl

            state = HedgeState(
                date=dt,
                spot=spot,
                option_price=option_price,
                delta=greeks['delta'],
                gamma=greeks['gamma'],
                theta=greeks['theta'],
                vega=greeks['vega'],
                iv=sigma,
                hedge_shares=self.hedge_shares,
                shares_traded=shares_traded,
                transaction_cost=txn_cost,
                option_pnl=option_pnl,
                hedge_pnl=hedge_pnl,
                total_pnl=total_pnl,
                cumulative_pnl=self.cumulative_pnl,
            )
            self.history.append(state)

            prev_spot = spot
            prev_option_price = option_price

        return self.to_dataframe()

    def to_dataframe(self) -> pd.DataFrame:
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame([vars(s) for s in self.history])

    def summary(self) -> dict:
        if not self.history:
            return {}

        df = self.to_dataframe()
        daily_pnl = df['total_pnl']

        return {
            'total_pnl': self.cumulative_pnl,
            'total_transaction_costs': self.total_transaction_costs,
            'mean_daily_pnl': daily_pnl.mean(),
            'std_daily_pnl': daily_pnl.std(),
            'sharpe': daily_pnl.mean() / daily_pnl.std() * np.sqrt(252) if daily_pnl.std() > 0 else 0,
            'max_drawdown': (df['cumulative_pnl'] - df['cumulative_pnl'].cummax()).min(),
            'num_days': len(df),
        }
