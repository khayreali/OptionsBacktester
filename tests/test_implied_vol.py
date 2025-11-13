import pytest
import numpy as np

from src.pricing import bs_call_price, bs_put_price, implied_vol


class TestIVRecovery:
    """Test that IV solver recovers the known vol."""

    def test_roundtrip_call_atm(self):
        S, K, T, r, sigma = 100, 100, 0.5, 0.05, 0.25
        price = bs_call_price(S, K, T, r, sigma)
        iv = implied_vol(price, S, K, T, r, 'call')
        assert abs(iv - sigma) < 1e-6

    def test_roundtrip_put_atm(self):
        S, K, T, r, sigma = 100, 100, 0.5, 0.05, 0.25
        price = bs_put_price(S, K, T, r, sigma)
        iv = implied_vol(price, S, K, T, r, 'put')
        assert abs(iv - sigma) < 1e-6

    def test_roundtrip_call_otm(self):
        S, K, T, r, sigma = 100, 120, 0.5, 0.05, 0.35
        price = bs_call_price(S, K, T, r, sigma)
        iv = implied_vol(price, S, K, T, r, 'call')
        assert abs(iv - sigma) < 1e-5

    def test_roundtrip_put_otm(self):
        S, K, T, r, sigma = 100, 80, 0.5, 0.05, 0.35
        price = bs_put_price(S, K, T, r, sigma)
        iv = implied_vol(price, S, K, T, r, 'put')
        assert abs(iv - sigma) < 1e-5

    def test_roundtrip_call_itm(self):
        S, K, T, r, sigma = 100, 80, 0.5, 0.05, 0.25
        price = bs_call_price(S, K, T, r, sigma)
        iv = implied_vol(price, S, K, T, r, 'call')
        assert abs(iv - sigma) < 1e-5

    def test_roundtrip_various_vols(self):
        S, K, T, r = 100, 100, 0.5, 0.05
        for sigma in [0.10, 0.20, 0.30, 0.50, 0.80, 1.0]:
            price = bs_call_price(S, K, T, r, sigma)
            iv = implied_vol(price, S, K, T, r, 'call')
            assert abs(iv - sigma) < 1e-4, f"Failed for sigma={sigma}"

    def test_roundtrip_various_strikes(self):
        S, T, r, sigma = 100, 0.25, 0.05, 0.20
        for K in [80, 90, 100, 110, 120]:
            price = bs_call_price(S, K, T, r, sigma)
            iv = implied_vol(price, S, K, T, r, 'call')
            assert abs(iv - sigma) < 1e-5, f"Failed for K={K}"

    def test_roundtrip_various_expiries(self):
        S, K, r, sigma = 100, 100, 0.05, 0.25
        for T in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]:
            price = bs_call_price(S, K, T, r, sigma)
            iv = implied_vol(price, S, K, T, r, 'call')
            assert abs(iv - sigma) < 1e-5, f"Failed for T={T}"


class TestNewtonMethod:
    """Test Newton-Raphson solver."""

    def test_newton_basic(self):
        S, K, T, r, sigma = 100, 100, 0.5, 0.05, 0.25
        price = bs_call_price(S, K, T, r, sigma)
        iv = implied_vol(price, S, K, T, r, 'call', method='newton')
        assert abs(iv - sigma) < 1e-4

    def test_newton_otm(self):
        S, K, T, r, sigma = 100, 115, 0.25, 0.05, 0.30
        price = bs_call_price(S, K, T, r, sigma)
        iv = implied_vol(price, S, K, T, r, 'call', method='newton')
        assert abs(iv - sigma) < 1e-4

    def test_newton_put(self):
        S, K, T, r, sigma = 100, 95, 0.5, 0.05, 0.22
        price = bs_put_price(S, K, T, r, sigma)
        iv = implied_vol(price, S, K, T, r, 'put', method='newton')
        assert abs(iv - sigma) < 1e-4


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_time_returns_nan(self):
        iv = implied_vol(5, 100, 100, 0, 0.05, 'call')
        assert np.isnan(iv)

    def test_negative_time_returns_nan(self):
        iv = implied_vol(5, 100, 100, -0.1, 0.05, 'call')
        assert np.isnan(iv)

    def test_price_below_intrinsic_call(self):
        # call with S=110, K=100 has intrinsic ~10
        # price of 5 is below intrinsic
        iv = implied_vol(5, 110, 100, 0.5, 0.05, 'call')
        assert np.isnan(iv)

    def test_price_below_intrinsic_put(self):
        # put with S=90, K=100 has intrinsic ~10
        iv = implied_vol(5, 90, 100, 0.5, 0.05, 'put')
        assert np.isnan(iv)

    def test_price_above_spot_call(self):
        # call can't be worth more than spot
        iv = implied_vol(110, 100, 100, 0.5, 0.05, 'call')
        assert np.isnan(iv)

    def test_price_above_strike_put(self):
        # put can't be worth more than discounted strike
        iv = implied_vol(105, 100, 100, 0.5, 0.05, 'put')
        assert np.isnan(iv)

    def test_very_small_price(self):
        # deep OTM option with tiny price
        S, K, T, r, sigma = 100, 150, 0.1, 0.05, 0.20
        price = bs_call_price(S, K, T, r, sigma)
        if price > 0.001:
            iv = implied_vol(price, S, K, T, r, 'call')
            assert abs(iv - sigma) < 0.01 or np.isnan(iv)

    def test_price_at_intrinsic(self):
        # price exactly at intrinsic should give very low vol
        S, K, T, r = 110, 100, 0.5, 0.05
        intrinsic = S - K * np.exp(-r * T)
        iv = implied_vol(intrinsic + 0.01, S, K, T, r, 'call')
        # should be very low vol (under 10%) or zero
        assert iv < 0.10 or np.isnan(iv)

    def test_high_vol_option(self):
        # test that we can recover high vols
        S, K, T, r, sigma = 100, 100, 0.5, 0.05, 1.5
        price = bs_call_price(S, K, T, r, sigma)
        iv = implied_vol(price, S, K, T, r, 'call')
        assert abs(iv - sigma) < 0.01

    def test_short_expiry(self):
        S, K, T, r, sigma = 100, 100, 0.02, 0.05, 0.25  # ~1 week
        price = bs_call_price(S, K, T, r, sigma)
        iv = implied_vol(price, S, K, T, r, 'call')
        assert abs(iv - sigma) < 1e-4

    def test_long_expiry(self):
        S, K, T, r, sigma = 100, 100, 3.0, 0.05, 0.25  # 3 years
        price = bs_call_price(S, K, T, r, sigma)
        iv = implied_vol(price, S, K, T, r, 'call')
        assert abs(iv - sigma) < 1e-4


class TestConsistency:
    """Test consistency between methods."""

    def test_brent_newton_agree(self):
        S, K, T, r, sigma = 100, 100, 0.5, 0.05, 0.25
        price = bs_call_price(S, K, T, r, sigma)

        iv_brent = implied_vol(price, S, K, T, r, 'call', method='brent')
        iv_newton = implied_vol(price, S, K, T, r, 'call', method='newton')

        assert abs(iv_brent - iv_newton) < 1e-3
