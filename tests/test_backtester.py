import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta

from src.models import Option
from src.backtester import DeltaHedger, PnLAttribution, PositionManager


def make_spot_data(start_date, num_days, start_price=100, drift=0.0, vol=0.2):
    """Generate synthetic spot data."""
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    returns = np.random.normal(drift/252, vol/np.sqrt(252), num_days)
    prices = start_price * np.exp(np.cumsum(returns))
    prices[0] = start_price
    return pd.DataFrame({'date': dates, 'close': prices})


class TestDeltaHedger:
    def test_basic_run(self):
        np.random.seed(42)

        opt = Option(strike=100, expiry=date(2024, 3, 15), option_type='call')
        spot_data = make_spot_data(date(2024, 1, 1), 30)

        hedger = DeltaHedger(
            option=opt,
            spot_data=spot_data,
            constant_vol=0.20,
            rehedge_frequency='daily',
        )
        result = hedger.run()

        assert len(result) == 30
        assert 'delta' in result.columns
        assert 'cumulative_pnl' in result.columns

    def test_hedger_with_transaction_costs(self):
        np.random.seed(42)

        opt = Option(strike=100, expiry=date(2024, 3, 15), option_type='call')
        spot_data = make_spot_data(date(2024, 1, 1), 20)

        hedger = DeltaHedger(
            option=opt,
            spot_data=spot_data,
            constant_vol=0.20,
            transaction_cost=0.01,
        )
        hedger.run()

        assert hedger.total_transaction_costs > 0

    def test_weekly_rehedge(self):
        np.random.seed(42)

        opt = Option(strike=100, expiry=date(2024, 3, 15), option_type='call')
        spot_data = make_spot_data(date(2024, 1, 1), 20)

        hedger = DeltaHedger(
            option=opt,
            spot_data=spot_data,
            constant_vol=0.20,
            rehedge_frequency='weekly',
        )
        result = hedger.run()

        # should have fewer trades with weekly rehedging
        num_trades = (result['shares_traded'].abs() > 0).sum()
        assert num_trades < 20

    def test_summary(self):
        np.random.seed(42)

        opt = Option(strike=100, expiry=date(2024, 3, 15), option_type='call')
        spot_data = make_spot_data(date(2024, 1, 1), 30)

        hedger = DeltaHedger(option=opt, spot_data=spot_data, constant_vol=0.20)
        hedger.run()

        summary = hedger.summary()
        assert 'total_pnl' in summary
        assert 'sharpe' in summary
        assert 'max_drawdown' in summary


class TestPnLAttribution:
    def test_single_period(self):
        attr = PnLAttribution(multiplier=100)

        breakdown = attr.attribute(
            dS=1.0,
            dt=1/365,
            d_sigma=0.0,
            delta=0.5,
            gamma=0.02,
            theta=-0.05,
            vega=0.10,
            actual_option_pnl=50.0,
        )

        # delta P&L should be 0.5 * 1.0 * 100 = 50
        assert abs(breakdown.delta_pnl - 50.0) < 0.01

        # gamma P&L should be 0.5 * 0.02 * 1^2 * 100 = 1
        assert abs(breakdown.gamma_pnl - 1.0) < 0.01

    def test_vega_pnl(self):
        attr = PnLAttribution(multiplier=100)

        # 1% vol increase with vega of 0.10
        breakdown = attr.attribute(
            dS=0.0,
            dt=0.0,
            d_sigma=0.01,  # 1% increase
            delta=0.5,
            gamma=0.02,
            theta=-0.05,
            vega=0.10,
            actual_option_pnl=100.0,
        )

        # vega P&L = 0.10 * 0.01 * 100 * 100 = 10
        assert abs(breakdown.vega_pnl - 10.0) < 0.01

    def test_summary(self):
        attr = PnLAttribution(multiplier=100)

        for _ in range(5):
            attr.attribute(
                dS=0.5, dt=1/365, d_sigma=0.001,
                delta=0.5, gamma=0.02, theta=-0.05, vega=0.10,
                actual_option_pnl=30.0,
            )

        summary = attr.summary()
        assert 'total_pnl' in summary
        assert 'delta_pnl' in summary
        assert 'residual' in summary
        assert summary['total_pnl'] == 150.0  # 5 * 30


class TestPositionManager:
    def test_add_position(self):
        mgr = PositionManager()
        opt = Option(strike=100, expiry=date(2024, 6, 21), option_type='call')

        mgr.add_position(
            option=opt,
            quantity=1,
            entry_price=5.0,
            entry_date=date(2024, 1, 1),
            entry_spot=100,
        )

        assert len(mgr.positions) == 1

    def test_portfolio_greeks(self):
        mgr = PositionManager()

        # long call
        call = Option(strike=100, expiry=date(2024, 6, 21), option_type='call')
        mgr.add_position(call, quantity=1, entry_price=5.0, entry_date=date(2024, 1, 1), entry_spot=100)

        # short put at same strike
        put = Option(strike=100, expiry=date(2024, 6, 21), option_type='put')
        mgr.add_position(put, quantity=-1, entry_price=4.0, entry_date=date(2024, 1, 1), entry_spot=100)

        greeks = mgr.get_portfolio_greeks(spot=100, as_of=date(2024, 1, 1), constant_vol=0.20)

        # long call + short put = synthetic long, delta should be close to 100
        assert greeks['delta'] > 80

    def test_to_dataframe(self):
        mgr = PositionManager()
        opt = Option(strike=100, expiry=date(2024, 6, 21), option_type='call')
        mgr.add_position(opt, quantity=2, entry_price=5.0, entry_date=date(2024, 1, 1), entry_spot=100)

        df = mgr.to_dataframe(spot=105, as_of=date(2024, 1, 15), constant_vol=0.20)

        assert len(df) == 1
        assert df['quantity'].iloc[0] == 2
        assert df['unrealized_pnl'].iloc[0] > 0  # spot moved up, call should profit

    def test_hedge_shares_needed(self):
        mgr = PositionManager()
        opt = Option(strike=100, expiry=date(2024, 6, 21), option_type='call')
        mgr.add_position(opt, quantity=1, entry_price=5.0, entry_date=date(2024, 1, 1), entry_spot=100)

        shares = mgr.hedge_shares_needed(spot=100, as_of=date(2024, 1, 1), constant_vol=0.20)

        # long call has positive delta, need to short shares to hedge
        assert shares < 0

    def test_filter_expired(self):
        mgr = PositionManager()

        # expired option
        opt1 = Option(strike=100, expiry=date(2024, 1, 1), option_type='call')
        mgr.add_position(opt1, quantity=1, entry_price=5.0, entry_date=date(2023, 12, 1), entry_spot=100)

        # non-expired option
        opt2 = Option(strike=100, expiry=date(2024, 6, 21), option_type='call')
        mgr.add_position(opt2, quantity=1, entry_price=5.0, entry_date=date(2023, 12, 1), entry_spot=100)

        filtered = mgr.filter_by_expiry(date(2024, 3, 1))

        assert len(filtered.positions) == 1
        assert filtered.positions[0].option.expiry == date(2024, 6, 21)
