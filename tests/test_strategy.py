import pytest
from datetime import date, timedelta

from src.backtester.strategy import SimpleStrategy, strategy_to_positions


class TestSimpleStrategy:
    def setup_method(self):
        self.strategy = SimpleStrategy(underlying='SPY')
        self.spot = 100
        self.entry_date = date(2024, 1, 1)
        self.expiry = date(2024, 2, 15)

    def test_long_call(self):
        legs = self.strategy.long_call(
            spot=self.spot, K=100, expiry=self.expiry, entry_date=self.entry_date
        )
        assert len(legs) == 1
        assert legs[0].quantity == 1
        assert legs[0].option.option_type == 'call'
        assert legs[0].entry_price > 0

    def test_short_put(self):
        legs = self.strategy.short_put(
            spot=self.spot, K=100, expiry=self.expiry, entry_date=self.entry_date
        )
        assert len(legs) == 1
        assert legs[0].quantity == -1
        assert legs[0].option.option_type == 'put'

    def test_long_straddle(self):
        legs = self.strategy.long_straddle(
            spot=self.spot, K=100, expiry=self.expiry, entry_date=self.entry_date
        )
        assert len(legs) == 2

        types = [leg.option.option_type for leg in legs]
        assert 'call' in types
        assert 'put' in types

        for leg in legs:
            assert leg.quantity == 1
            assert leg.option.strike == 100

    def test_short_straddle(self):
        legs = self.strategy.short_straddle(
            spot=self.spot, K=100, expiry=self.expiry, entry_date=self.entry_date
        )
        assert len(legs) == 2

        for leg in legs:
            assert leg.quantity == -1

    def test_bull_call_spread(self):
        legs = self.strategy.bull_call_spread(
            spot=self.spot, K1=95, K2=105, expiry=self.expiry, entry_date=self.entry_date
        )
        assert len(legs) == 2

        long_leg = [l for l in legs if l.quantity > 0][0]
        short_leg = [l for l in legs if l.quantity < 0][0]

        assert long_leg.option.strike == 95
        assert short_leg.option.strike == 105

    def test_bull_call_spread_invalid_strikes(self):
        with pytest.raises(ValueError):
            self.strategy.bull_call_spread(
                spot=self.spot, K1=105, K2=95, expiry=self.expiry, entry_date=self.entry_date
            )

    def test_bear_put_spread(self):
        legs = self.strategy.bear_put_spread(
            spot=self.spot, K1=95, K2=105, expiry=self.expiry, entry_date=self.entry_date
        )
        assert len(legs) == 2

        long_leg = [l for l in legs if l.quantity > 0][0]
        short_leg = [l for l in legs if l.quantity < 0][0]

        assert long_leg.option.strike == 105
        assert short_leg.option.strike == 95

    def test_iron_condor(self):
        legs = self.strategy.iron_condor(
            spot=self.spot,
            K_put_long=85,
            K_put_short=90,
            K_call_short=110,
            K_call_long=115,
            expiry=self.expiry,
            entry_date=self.entry_date,
        )
        assert len(legs) == 4

        long_legs = [l for l in legs if l.quantity > 0]
        short_legs = [l for l in legs if l.quantity < 0]

        assert len(long_legs) == 2
        assert len(short_legs) == 2

    def test_strategy_to_positions(self):
        legs = self.strategy.long_straddle(
            spot=self.spot, K=100, expiry=self.expiry, entry_date=self.entry_date
        )

        mgr = strategy_to_positions(legs, self.entry_date, self.spot)

        assert len(mgr.positions) == 2
        greeks = mgr.get_portfolio_greeks(self.spot, self.entry_date, constant_vol=0.20)

        # straddle should have near-zero delta at ATM
        assert abs(greeks['delta']) < 10
