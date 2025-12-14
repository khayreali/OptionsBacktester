import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta

from src.models import Option
from src.backtester import DeltaHedger, PnLAttribution, attribute_hedged_pnl


def make_spot_data(start_date, num_days, start_price=100, drift=0.0, vol=0.2, seed=None):
    """Generate synthetic spot data."""
    if seed is not None:
        np.random.seed(seed)
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    returns = np.random.normal(drift/252, vol/np.sqrt(252), num_days)
    prices = start_price * np.exp(np.cumsum(returns))
    prices[0] = start_price
    return pd.DataFrame({'date': dates, 'close': prices})


class TestHedgedVsUnhedged:
    """Test that delta-hedged position has lower variance than unhedged."""

    def test_hedged_lower_variance_call(self):
        np.random.seed(42)

        opt = Option(strike=100, expiry=date(2024, 3, 15), option_type='call')
        spot_data = make_spot_data(date(2024, 1, 1), 45, start_price=100, vol=0.25)

        # run hedged
        hedger = DeltaHedger(
            option=opt,
            spot_data=spot_data,
            constant_vol=0.25,
            rehedge_frequency='daily',
            position_size=1,
        )
        hedged_result = hedger.run()
        hedged_std = hedged_result['total_pnl'].std()

        # run "unhedged" (never rehedge after initial)
        hedger_unhedged = DeltaHedger(
            option=opt,
            spot_data=spot_data,
            constant_vol=0.25,
            rehedge_frequency='never',
            position_size=1,
        )
        unhedged_result = hedger_unhedged.run()
        unhedged_std = unhedged_result['total_pnl'].std()

        # hedged should have lower daily P&L variance
        assert hedged_std < unhedged_std

    def test_hedged_lower_variance_put(self):
        np.random.seed(123)

        opt = Option(strike=100, expiry=date(2024, 3, 15), option_type='put')
        spot_data = make_spot_data(date(2024, 1, 1), 45, start_price=100, vol=0.25)

        hedger = DeltaHedger(
            option=opt,
            spot_data=spot_data,
            constant_vol=0.25,
            rehedge_frequency='daily',
            position_size=-1,  # short put
        )
        hedged_result = hedger.run()
        hedged_std = hedged_result['total_pnl'].std()

        hedger_unhedged = DeltaHedger(
            option=opt,
            spot_data=spot_data,
            constant_vol=0.25,
            rehedge_frequency='never',
            position_size=-1,
        )
        unhedged_result = hedger_unhedged.run()
        unhedged_std = unhedged_result['total_pnl'].std()

        assert hedged_std < unhedged_std

    def test_multiple_seeds(self):
        # run with different random seeds to ensure result is consistent
        variance_ratios = []

        for seed in [1, 2, 3, 4, 5]:
            opt = Option(strike=100, expiry=date(2024, 3, 15), option_type='call')
            spot_data = make_spot_data(date(2024, 1, 1), 40, vol=0.30, seed=seed)

            hedger = DeltaHedger(
                option=opt, spot_data=spot_data, constant_vol=0.30,
                rehedge_frequency='daily',
            )
            hedged = hedger.run()

            hedger_un = DeltaHedger(
                option=opt, spot_data=spot_data, constant_vol=0.30,
                rehedge_frequency='never',
            )
            unhedged = hedger_un.run()

            ratio = hedged['total_pnl'].std() / (unhedged['total_pnl'].std() + 1e-10)
            variance_ratios.append(ratio)

        # on average, hedged should have lower variance
        assert np.mean(variance_ratios) < 1.0


class TestPnLAttribution:
    """Test that P&L attribution components sum correctly."""

    def test_attribution_sums_to_total(self):
        attr = PnLAttribution(multiplier=100)

        # simulate a few periods
        for _ in range(10):
            dS = np.random.normal(0, 2)
            dt = 1/365
            d_sigma = np.random.normal(0, 0.005)
            delta = 0.5
            gamma = 0.02
            theta = -0.05
            vega = 0.10

            # actual pnl is arbitrary for this test
            actual = np.random.normal(0, 50)

            breakdown = attr.attribute(
                dS=dS, dt=dt, d_sigma=d_sigma,
                delta=delta, gamma=gamma, theta=theta, vega=vega,
                actual_option_pnl=actual,
            )

            # components + residual should equal total
            components_sum = (
                breakdown.delta_pnl +
                breakdown.gamma_pnl +
                breakdown.theta_pnl +
                breakdown.vega_pnl +
                breakdown.residual
            )
            assert abs(components_sum - breakdown.total) < 1e-10

    def test_attribution_from_hedger(self):
        np.random.seed(42)

        opt = Option(strike=100, expiry=date(2024, 2, 28), option_type='call')
        spot_data = make_spot_data(date(2024, 1, 1), 30, vol=0.20, seed=42)

        hedger = DeltaHedger(
            option=opt,
            spot_data=spot_data,
            constant_vol=0.20,
            rehedge_frequency='daily',
        )
        result = hedger.run()

        attr_df = attribute_hedged_pnl(result, multiplier=100)

        # each row should have components summing to total (minus residual equals explained)
        for _, row in attr_df.iterrows():
            explained = row['delta_pnl'] + row['gamma_pnl'] + row['theta_pnl'] + row['vega_pnl']
            assert abs(explained + row['residual'] - row['total']) < 1e-10

    def test_delta_pnl_direction(self):
        # if spot goes up and we're long delta, delta P&L should be positive
        attr = PnLAttribution(multiplier=100)

        breakdown = attr.attribute(
            dS=2.0,  # spot up $2
            dt=1/365,
            d_sigma=0,
            delta=0.5,  # long delta
            gamma=0.02,
            theta=-0.05,
            vega=0.10,
            actual_option_pnl=100,
        )

        # delta pnl = 0.5 * 2 * 100 = 100
        assert breakdown.delta_pnl == 100

    def test_gamma_pnl_always_positive(self):
        # gamma P&L is always positive for long gamma regardless of direction
        attr = PnLAttribution(multiplier=100)

        for dS in [-3.0, 3.0]:
            breakdown = attr.attribute(
                dS=dS,
                dt=1/365,
                d_sigma=0,
                delta=0.5,
                gamma=0.02,  # long gamma
                theta=-0.05,
                vega=0.10,
                actual_option_pnl=100,
            )
            # gamma pnl = 0.5 * 0.02 * 9 * 100 = 9
            assert breakdown.gamma_pnl > 0

    def test_theta_pnl_negative_for_long(self):
        attr = PnLAttribution(multiplier=100)

        breakdown = attr.attribute(
            dS=0,
            dt=1/365,
            d_sigma=0,
            delta=0.5,
            gamma=0.02,
            theta=-0.05,  # negative theta
            vega=0.10,
            actual_option_pnl=-5,
        )

        # theta pnl should be negative (time decay)
        assert breakdown.theta_pnl < 0


class TestHedgerMechanics:
    """Test hedger mechanics work correctly."""

    def test_hedge_shares_opposite_delta(self):
        np.random.seed(42)

        opt = Option(strike=100, expiry=date(2024, 3, 15), option_type='call')
        spot_data = make_spot_data(date(2024, 1, 1), 10, start_price=100)

        hedger = DeltaHedger(
            option=opt,
            spot_data=spot_data,
            constant_vol=0.20,
            rehedge_frequency='daily',
            position_size=1,  # long 1 call
        )
        result = hedger.run()

        # for long call (positive delta), hedge shares should be negative
        assert result['hedge_shares'].iloc[0] < 0

    def test_short_position_hedge_direction(self):
        np.random.seed(42)

        opt = Option(strike=100, expiry=date(2024, 3, 15), option_type='put')
        spot_data = make_spot_data(date(2024, 1, 1), 10, start_price=100)

        hedger = DeltaHedger(
            option=opt,
            spot_data=spot_data,
            constant_vol=0.20,
            rehedge_frequency='daily',
            position_size=-1,  # short 1 put
        )
        result = hedger.run()

        # short put has positive delta (from short's perspective), hedge should be negative
        # but wait - short put delta = -(-0.5) = +0.5, so hedge = -50
        assert result['hedge_shares'].iloc[0] < 0

    def test_transaction_costs_increase_with_rehedging(self):
        np.random.seed(42)

        opt = Option(strike=100, expiry=date(2024, 3, 15), option_type='call')
        spot_data = make_spot_data(date(2024, 1, 1), 30)

        hedger_daily = DeltaHedger(
            option=opt,
            spot_data=spot_data,
            constant_vol=0.20,
            rehedge_frequency='daily',
            transaction_cost=0.01,
        )
        hedger_daily.run()

        hedger_weekly = DeltaHedger(
            option=opt,
            spot_data=spot_data,
            constant_vol=0.20,
            rehedge_frequency='weekly',
            transaction_cost=0.01,
        )
        hedger_weekly.run()

        # daily rehedging should have higher transaction costs
        assert hedger_daily.total_transaction_costs > hedger_weekly.total_transaction_costs

    def test_zero_transaction_cost(self):
        np.random.seed(42)

        opt = Option(strike=100, expiry=date(2024, 3, 15), option_type='call')
        spot_data = make_spot_data(date(2024, 1, 1), 20)

        hedger = DeltaHedger(
            option=opt,
            spot_data=spot_data,
            constant_vol=0.20,
            transaction_cost=0.0,
        )
        hedger.run()

        assert hedger.total_transaction_costs == 0
