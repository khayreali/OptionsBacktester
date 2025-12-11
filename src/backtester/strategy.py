from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional

from src.models import Option
from src.pricing import bs_call_price, bs_put_price


@dataclass
class StrategyLeg:
    option: Option
    quantity: int  # positive = long, negative = short
    entry_price: float


class SimpleStrategy:
    """
    Factory for common options strategies.
    Each method returns a list of StrategyLeg objects.
    """

    def __init__(
        self,
        underlying: str = 'SPY',
        r: float = 0.05,
        default_vol: float = 0.18,
    ):
        self.underlying = underlying
        self.r = r
        self.default_vol = default_vol

    def _make_option(self, K: float, expiry: date, opt_type: str) -> Option:
        return Option(
            strike=K,
            expiry=expiry,
            option_type=opt_type,
            underlying=self.underlying,
        )

    def _price(self, S: float, K: float, T: float, sigma: float, opt_type: str) -> float:
        if opt_type == 'call':
            return bs_call_price(S, K, T, self.r, sigma)
        return bs_put_price(S, K, T, self.r, sigma)

    def long_call(
        self,
        spot: float,
        K: float,
        expiry: date,
        entry_date: date,
        sigma: Optional[float] = None,
    ) -> List[StrategyLeg]:
        """Long a single call."""
        sigma = sigma or self.default_vol
        T = (expiry - entry_date).days / 365.0
        opt = self._make_option(K, expiry, 'call')
        price = self._price(spot, K, T, sigma, 'call')

        return [StrategyLeg(option=opt, quantity=1, entry_price=price)]

    def long_put(
        self,
        spot: float,
        K: float,
        expiry: date,
        entry_date: date,
        sigma: Optional[float] = None,
    ) -> List[StrategyLeg]:
        """Long a single put."""
        sigma = sigma or self.default_vol
        T = (expiry - entry_date).days / 365.0
        opt = self._make_option(K, expiry, 'put')
        price = self._price(spot, K, T, sigma, 'put')

        return [StrategyLeg(option=opt, quantity=1, entry_price=price)]

    def short_put(
        self,
        spot: float,
        K: float,
        expiry: date,
        entry_date: date,
        sigma: Optional[float] = None,
    ) -> List[StrategyLeg]:
        """Short a single put (cash-secured put)."""
        sigma = sigma or self.default_vol
        T = (expiry - entry_date).days / 365.0
        opt = self._make_option(K, expiry, 'put')
        price = self._price(spot, K, T, sigma, 'put')

        return [StrategyLeg(option=opt, quantity=-1, entry_price=price)]

    def short_call(
        self,
        spot: float,
        K: float,
        expiry: date,
        entry_date: date,
        sigma: Optional[float] = None,
    ) -> List[StrategyLeg]:
        """Short a single call (covered call without the stock)."""
        sigma = sigma or self.default_vol
        T = (expiry - entry_date).days / 365.0
        opt = self._make_option(K, expiry, 'call')
        price = self._price(spot, K, T, sigma, 'call')

        return [StrategyLeg(option=opt, quantity=-1, entry_price=price)]

    def long_straddle(
        self,
        spot: float,
        K: float,
        expiry: date,
        entry_date: date,
        sigma: Optional[float] = None,
    ) -> List[StrategyLeg]:
        """Long straddle: long call + long put at same strike."""
        sigma = sigma or self.default_vol
        T = (expiry - entry_date).days / 365.0

        call = self._make_option(K, expiry, 'call')
        put = self._make_option(K, expiry, 'put')

        call_price = self._price(spot, K, T, sigma, 'call')
        put_price = self._price(spot, K, T, sigma, 'put')

        return [
            StrategyLeg(option=call, quantity=1, entry_price=call_price),
            StrategyLeg(option=put, quantity=1, entry_price=put_price),
        ]

    def short_straddle(
        self,
        spot: float,
        K: float,
        expiry: date,
        entry_date: date,
        sigma: Optional[float] = None,
    ) -> List[StrategyLeg]:
        """Short straddle: short call + short put at same strike."""
        legs = self.long_straddle(spot, K, expiry, entry_date, sigma)
        for leg in legs:
            leg.quantity = -1
        return legs

    def long_strangle(
        self,
        spot: float,
        K_put: float,
        K_call: float,
        expiry: date,
        entry_date: date,
        sigma: Optional[float] = None,
    ) -> List[StrategyLeg]:
        """Long strangle: long OTM put + long OTM call."""
        sigma = sigma or self.default_vol
        T = (expiry - entry_date).days / 365.0

        call = self._make_option(K_call, expiry, 'call')
        put = self._make_option(K_put, expiry, 'put')

        call_price = self._price(spot, K_call, T, sigma, 'call')
        put_price = self._price(spot, K_put, T, sigma, 'put')

        return [
            StrategyLeg(option=call, quantity=1, entry_price=call_price),
            StrategyLeg(option=put, quantity=1, entry_price=put_price),
        ]

    def bull_call_spread(
        self,
        spot: float,
        K1: float,
        K2: float,
        expiry: date,
        entry_date: date,
        sigma: Optional[float] = None,
    ) -> List[StrategyLeg]:
        """Bull call spread: long lower strike call, short higher strike call."""
        if K1 >= K2:
            raise ValueError("K1 must be less than K2 for bull call spread")

        sigma = sigma or self.default_vol
        T = (expiry - entry_date).days / 365.0

        long_call = self._make_option(K1, expiry, 'call')
        short_call = self._make_option(K2, expiry, 'call')

        long_price = self._price(spot, K1, T, sigma, 'call')
        short_price = self._price(spot, K2, T, sigma, 'call')

        return [
            StrategyLeg(option=long_call, quantity=1, entry_price=long_price),
            StrategyLeg(option=short_call, quantity=-1, entry_price=short_price),
        ]

    def bear_put_spread(
        self,
        spot: float,
        K1: float,
        K2: float,
        expiry: date,
        entry_date: date,
        sigma: Optional[float] = None,
    ) -> List[StrategyLeg]:
        """Bear put spread: long higher strike put, short lower strike put."""
        if K1 >= K2:
            raise ValueError("K1 must be less than K2 for bear put spread")

        sigma = sigma or self.default_vol
        T = (expiry - entry_date).days / 365.0

        long_put = self._make_option(K2, expiry, 'put')
        short_put = self._make_option(K1, expiry, 'put')

        long_price = self._price(spot, K2, T, sigma, 'put')
        short_price = self._price(spot, K1, T, sigma, 'put')

        return [
            StrategyLeg(option=long_put, quantity=1, entry_price=long_price),
            StrategyLeg(option=short_put, quantity=-1, entry_price=short_price),
        ]

    def iron_condor(
        self,
        spot: float,
        K_put_long: float,
        K_put_short: float,
        K_call_short: float,
        K_call_long: float,
        expiry: date,
        entry_date: date,
        sigma: Optional[float] = None,
    ) -> List[StrategyLeg]:
        """
        Iron condor: bull put spread + bear call spread.
        Profit if spot stays between short strikes.
        """
        sigma = sigma or self.default_vol
        T = (expiry - entry_date).days / 365.0

        legs = []

        # bull put spread (credit)
        legs.append(StrategyLeg(
            option=self._make_option(K_put_long, expiry, 'put'),
            quantity=1,
            entry_price=self._price(spot, K_put_long, T, sigma, 'put'),
        ))
        legs.append(StrategyLeg(
            option=self._make_option(K_put_short, expiry, 'put'),
            quantity=-1,
            entry_price=self._price(spot, K_put_short, T, sigma, 'put'),
        ))

        # bear call spread (credit)
        legs.append(StrategyLeg(
            option=self._make_option(K_call_short, expiry, 'call'),
            quantity=-1,
            entry_price=self._price(spot, K_call_short, T, sigma, 'call'),
        ))
        legs.append(StrategyLeg(
            option=self._make_option(K_call_long, expiry, 'call'),
            quantity=1,
            entry_price=self._price(spot, K_call_long, T, sigma, 'call'),
        ))

        return legs


def strategy_to_positions(legs: List[StrategyLeg], entry_date: date, entry_spot: float):
    """Convert strategy legs to PositionManager entries."""
    from src.backtester import PositionManager

    mgr = PositionManager()
    for i, leg in enumerate(legs):
        mgr.add_position(
            option=leg.option,
            quantity=leg.quantity,
            entry_price=leg.entry_price,
            entry_date=entry_date,
            entry_spot=entry_spot,
            tag=f'leg_{i}',
        )
    return mgr
