import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import date
from typing import List, Dict, Optional

from src.models import Option
from src.pricing import bs_call_price, bs_put_price, bs_greeks


@dataclass
class PositionEntry:
    option: Option
    quantity: int  # positive = long, negative = short
    entry_price: float
    entry_date: date
    entry_spot: float
    tag: str = ''  # optional label


@dataclass
class PositionSnapshot:
    """Current state of a position."""
    option: Option
    quantity: int
    entry_price: float
    current_price: float
    spot: float
    T: float
    iv: float
    delta: float
    gamma: float
    theta: float
    vega: float
    unrealized_pnl: float
    market_value: float


class PositionManager:
    """
    Track multiple option positions and aggregate greeks.
    Simple implementation - no fancy optimization.
    """

    def __init__(self, r: float = 0.05):
        self.r = r
        self.positions: List[PositionEntry] = []

    def add_position(
        self,
        option: Option,
        quantity: int,
        entry_price: float,
        entry_date: date,
        entry_spot: float,
        tag: str = '',
    ):
        pos = PositionEntry(
            option=option,
            quantity=quantity,
            entry_price=entry_price,
            entry_date=entry_date,
            entry_spot=entry_spot,
            tag=tag,
        )
        self.positions.append(pos)

    def remove_position(self, idx: int):
        if 0 <= idx < len(self.positions):
            self.positions.pop(idx)

    def get_snapshot(
        self,
        pos: PositionEntry,
        spot: float,
        as_of: date,
        vol_surface=None,
        constant_vol: float = 0.20,
    ) -> PositionSnapshot:
        """Get current state of a single position."""
        T = pos.option.time_to_expiry(as_of)

        if vol_surface is not None:
            try:
                iv = vol_surface.interpolate(pos.option.K, T)
            except:
                iv = constant_vol
        else:
            iv = constant_vol

        if T <= 0:
            current_price = pos.option.intrinsic(spot)
            greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
        else:
            if pos.option.option_type == 'call':
                current_price = bs_call_price(spot, pos.option.K, T, self.r, iv)
            else:
                current_price = bs_put_price(spot, pos.option.K, T, self.r, iv)
            greeks = bs_greeks(spot, pos.option.K, T, self.r, iv, pos.option.option_type)

        multiplier = pos.quantity * 100
        unrealized_pnl = (current_price - pos.entry_price) * multiplier
        market_value = current_price * multiplier

        return PositionSnapshot(
            option=pos.option,
            quantity=pos.quantity,
            entry_price=pos.entry_price,
            current_price=current_price,
            spot=spot,
            T=T,
            iv=iv,
            delta=greeks['delta'] * multiplier,
            gamma=greeks['gamma'] * multiplier,
            theta=greeks['theta'] * multiplier,
            vega=greeks['vega'] * multiplier,
            unrealized_pnl=unrealized_pnl,
            market_value=market_value,
        )

    def get_portfolio_greeks(
        self,
        spot: float,
        as_of: date,
        vol_surface=None,
        constant_vol: float = 0.20,
    ) -> Dict:
        """Aggregate greeks across all positions."""
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        total_pnl = 0.0
        total_value = 0.0

        for pos in self.positions:
            snap = self.get_snapshot(pos, spot, as_of, vol_surface, constant_vol)
            total_delta += snap.delta
            total_gamma += snap.gamma
            total_theta += snap.theta
            total_vega += snap.vega
            total_pnl += snap.unrealized_pnl
            total_value += snap.market_value

        return {
            'delta': total_delta,
            'gamma': total_gamma,
            'theta': total_theta,
            'vega': total_vega,
            'unrealized_pnl': total_pnl,
            'market_value': total_value,
            'num_positions': len(self.positions),
        }

    def get_all_snapshots(
        self,
        spot: float,
        as_of: date,
        vol_surface=None,
        constant_vol: float = 0.20,
    ) -> List[PositionSnapshot]:
        """Get snapshots for all positions."""
        return [
            self.get_snapshot(pos, spot, as_of, vol_surface, constant_vol)
            for pos in self.positions
        ]

    def to_dataframe(
        self,
        spot: float,
        as_of: date,
        vol_surface=None,
        constant_vol: float = 0.20,
    ) -> pd.DataFrame:
        """Export all positions to DataFrame."""
        rows = []
        for i, pos in enumerate(self.positions):
            snap = self.get_snapshot(pos, spot, as_of, vol_surface, constant_vol)
            rows.append({
                'idx': i,
                'underlying': pos.option.underlying,
                'strike': pos.option.K,
                'expiry': pos.option.expiry,
                'type': pos.option.option_type,
                'quantity': pos.quantity,
                'entry_price': pos.entry_price,
                'current_price': snap.current_price,
                'T': snap.T,
                'iv': snap.iv,
                'delta': snap.delta,
                'gamma': snap.gamma,
                'theta': snap.theta,
                'vega': snap.vega,
                'unrealized_pnl': snap.unrealized_pnl,
                'tag': pos.tag,
            })
        return pd.DataFrame(rows)

    def filter_by_expiry(self, as_of: date) -> 'PositionManager':
        """Return new manager with only non-expired positions."""
        new_mgr = PositionManager(self.r)
        for pos in self.positions:
            if not pos.option.is_expired(as_of):
                new_mgr.positions.append(pos)
        return new_mgr

    def hedge_shares_needed(
        self,
        spot: float,
        as_of: date,
        current_hedge: float = 0.0,
        vol_surface=None,
        constant_vol: float = 0.20,
    ) -> float:
        """Calculate shares needed to delta-hedge the portfolio."""
        greeks = self.get_portfolio_greeks(spot, as_of, vol_surface, constant_vol)
        target_hedge = -greeks['delta']
        return target_hedge - current_hedge
