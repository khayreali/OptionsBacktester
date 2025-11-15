import pytest
from datetime import date

from src.models import Option, Greeks, Position


class TestOption:
    def test_time_to_expiry(self):
        opt = Option(strike=400, expiry=date(2024, 6, 21), option_type='call')
        T = opt.time_to_expiry(date(2024, 1, 1))
        assert 0.45 < T < 0.50

    def test_expired_returns_zero_tte(self):
        opt = Option(strike=400, expiry=date(2024, 1, 1), option_type='call')
        T = opt.time_to_expiry(date(2024, 6, 1))
        assert T == 0.0

    def test_intrinsic_call_itm(self):
        opt = Option(strike=400, expiry=date(2024, 6, 21), option_type='call')
        assert opt.intrinsic(420) == 20

    def test_intrinsic_call_otm(self):
        opt = Option(strike=400, expiry=date(2024, 6, 21), option_type='call')
        assert opt.intrinsic(380) == 0

    def test_intrinsic_put(self):
        opt = Option(strike=400, expiry=date(2024, 6, 21), option_type='put')
        assert opt.intrinsic(380) == 20
        assert opt.intrinsic(420) == 0

    def test_k_property(self):
        opt = Option(strike=100, expiry=date(2024, 6, 21), option_type='call')
        assert opt.K == 100


class TestGreeks:
    def test_from_dict(self):
        d = {'delta': 0.5, 'gamma': 0.02, 'theta': -0.05, 'vega': 0.15}
        g = Greeks.from_dict(d)
        assert g.delta == 0.5
        assert g.gamma == 0.02

    def test_scaled(self):
        g = Greeks(delta=0.5, gamma=0.02, theta=-0.05, vega=0.15)
        scaled = g.scaled(100)
        assert scaled.delta == 50
        assert scaled.gamma == 2.0


class TestPosition:
    def test_long_position(self):
        opt = Option(strike=400, expiry=date(2024, 6, 21), option_type='call')
        pos = Position(option=opt, quantity=1, entry_price=10, entry_date=date(2024, 1, 1))
        assert pos.is_long
        assert not pos.is_short

    def test_short_position(self):
        opt = Option(strike=400, expiry=date(2024, 6, 21), option_type='put')
        pos = Position(option=opt, quantity=-1, entry_price=8, entry_date=date(2024, 1, 1))
        assert pos.is_short
        assert not pos.is_long

    def test_unrealized_pnl(self):
        opt = Option(strike=400, expiry=date(2024, 6, 21), option_type='call')
        pos = Position(option=opt, quantity=1, entry_price=10, entry_date=date(2024, 1, 1))
        pnl = pos.unrealized_pnl(12)
        assert pnl == 200  # (12 - 10) * 100

    def test_update_greeks(self):
        opt = Option(strike=400, expiry=date(2024, 6, 21), option_type='call')
        pos = Position(option=opt, quantity=1, entry_price=10, entry_date=date(2024, 1, 1))
        pos.update_greeks({'delta': 0.5, 'gamma': 0.02, 'theta': -0.05, 'vega': 0.15})
        assert pos.greeks.delta == 50  # scaled by 100
        assert pos.net_delta() == 50
