import pytest
import numpy as np

from src.pricing import bs_call_price, bs_put_price, bs_greeks


class TestBlackScholesKnownValues:
    """Test against known/textbook values."""

    def test_call_price_hull_example(self):
        # Hull textbook example (roughly)
        # S=42, K=40, T=0.5, r=10%, sigma=20%
        S, K, T, r, sigma = 42, 40, 0.5, 0.10, 0.20
        price = bs_call_price(S, K, T, r, sigma)
        # should be around 4.76
        assert 4.5 < price < 5.0

    def test_put_price_hull_example(self):
        S, K, T, r, sigma = 42, 40, 0.5, 0.10, 0.20
        price = bs_put_price(S, K, T, r, sigma)
        # should be around 0.81
        assert 0.6 < price < 1.0

    def test_atm_call_approximation(self):
        # ATM call ~ 0.4 * S * sigma * sqrt(T) for small r
        S, K, T, sigma = 100, 100, 1.0, 0.20
        price = bs_call_price(S, K, T, 0.0, sigma)
        approx = 0.4 * S * sigma * np.sqrt(T)
        # should be close but not exact
        assert abs(price - approx) < 2

    def test_deep_itm_call_approaches_intrinsic(self):
        S, K, T, r, sigma = 150, 100, 0.1, 0.05, 0.20
        price = bs_call_price(S, K, T, r, sigma)
        intrinsic = S - K
        # should be very close to intrinsic
        assert price > intrinsic
        assert price < intrinsic + 2

    def test_deep_otm_call_near_zero(self):
        S, K, T, r, sigma = 50, 100, 0.1, 0.05, 0.20
        price = bs_call_price(S, K, T, r, sigma)
        assert price < 0.01


class TestPutCallParity:
    """Put-call parity: C - P = S - K*exp(-rT)"""

    def test_parity_atm(self):
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
        call = bs_call_price(S, K, T, r, sigma)
        put = bs_put_price(S, K, T, r, sigma)
        parity = call - put
        expected = S - K * np.exp(-r * T)
        assert abs(parity - expected) < 1e-10

    def test_parity_otm_call(self):
        S, K, T, r, sigma = 100, 120, 0.5, 0.05, 0.30
        call = bs_call_price(S, K, T, r, sigma)
        put = bs_put_price(S, K, T, r, sigma)
        parity = call - put
        expected = S - K * np.exp(-r * T)
        assert abs(parity - expected) < 1e-10

    def test_parity_itm_call(self):
        S, K, T, r, sigma = 100, 80, 0.25, 0.03, 0.25
        call = bs_call_price(S, K, T, r, sigma)
        put = bs_put_price(S, K, T, r, sigma)
        parity = call - put
        expected = S - K * np.exp(-r * T)
        assert abs(parity - expected) < 1e-10

    def test_parity_various_vols(self):
        S, K, T, r = 100, 100, 0.5, 0.05
        for sigma in [0.10, 0.20, 0.40, 0.80]:
            call = bs_call_price(S, K, T, r, sigma)
            put = bs_put_price(S, K, T, r, sigma)
            parity = call - put
            expected = S - K * np.exp(-r * T)
            assert abs(parity - expected) < 1e-10


class TestGreeks:
    """Test greek values at various spots."""

    def test_call_delta_bounds(self):
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20
        greeks = bs_greeks(S, K, T, r, sigma, 'call')
        assert 0 < greeks['delta'] < 1

    def test_put_delta_bounds(self):
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20
        greeks = bs_greeks(S, K, T, r, sigma, 'put')
        assert -1 < greeks['delta'] < 0

    def test_call_put_delta_relationship(self):
        # put delta = call delta - 1
        S, K, T, r, sigma = 100, 100, 0.5, 0.05, 0.25
        call_greeks = bs_greeks(S, K, T, r, sigma, 'call')
        put_greeks = bs_greeks(S, K, T, r, sigma, 'put')
        assert abs(put_greeks['delta'] - (call_greeks['delta'] - 1)) < 1e-10

    def test_gamma_same_for_call_put(self):
        S, K, T, r, sigma = 100, 100, 0.5, 0.05, 0.25
        call_greeks = bs_greeks(S, K, T, r, sigma, 'call')
        put_greeks = bs_greeks(S, K, T, r, sigma, 'put')
        assert abs(call_greeks['gamma'] - put_greeks['gamma']) < 1e-10

    def test_vega_same_for_call_put(self):
        S, K, T, r, sigma = 100, 100, 0.5, 0.05, 0.25
        call_greeks = bs_greeks(S, K, T, r, sigma, 'call')
        put_greeks = bs_greeks(S, K, T, r, sigma, 'put')
        assert abs(call_greeks['vega'] - put_greeks['vega']) < 1e-10

    def test_gamma_positive(self):
        for S in [80, 100, 120]:
            greeks = bs_greeks(S, 100, 0.25, 0.05, 0.20, 'call')
            assert greeks['gamma'] > 0

    def test_gamma_highest_atm(self):
        K, T, r, sigma = 100, 0.25, 0.05, 0.20
        gamma_atm = bs_greeks(100, K, T, r, sigma, 'call')['gamma']
        gamma_itm = bs_greeks(110, K, T, r, sigma, 'call')['gamma']
        gamma_otm = bs_greeks(90, K, T, r, sigma, 'call')['gamma']
        assert gamma_atm > gamma_itm
        assert gamma_atm > gamma_otm

    def test_theta_negative_long_option(self):
        greeks = bs_greeks(100, 100, 0.25, 0.05, 0.20, 'call')
        # theta is negative (time decay hurts long positions)
        assert greeks['theta'] < 0

    def test_vega_positive(self):
        greeks = bs_greeks(100, 100, 0.25, 0.05, 0.20, 'call')
        assert greeks['vega'] > 0

    def test_deep_itm_call_delta_near_one(self):
        greeks = bs_greeks(150, 100, 0.25, 0.05, 0.20, 'call')
        assert greeks['delta'] > 0.95

    def test_deep_otm_call_delta_near_zero(self):
        greeks = bs_greeks(50, 100, 0.25, 0.05, 0.20, 'call')
        assert greeks['delta'] < 0.05

    def test_delta_increases_with_spot(self):
        K, T, r, sigma = 100, 0.5, 0.05, 0.20
        delta_90 = bs_greeks(90, K, T, r, sigma, 'call')['delta']
        delta_100 = bs_greeks(100, K, T, r, sigma, 'call')['delta']
        delta_110 = bs_greeks(110, K, T, r, sigma, 'call')['delta']
        assert delta_90 < delta_100 < delta_110


class TestExpiredOptions:
    def test_expired_itm_call(self):
        price = bs_call_price(105, 100, 0, 0.05, 0.20)
        assert price == 5

    def test_expired_otm_call(self):
        price = bs_call_price(95, 100, 0, 0.05, 0.20)
        assert price == 0

    def test_expired_itm_put(self):
        price = bs_put_price(95, 100, 0, 0.05, 0.20)
        assert price == 5

    def test_expired_otm_put(self):
        price = bs_put_price(105, 100, 0, 0.05, 0.20)
        assert price == 0

    def test_expired_greeks_delta(self):
        # expired ITM call should have delta = 1
        greeks = bs_greeks(105, 100, 0, 0.05, 0.20, 'call')
        assert greeks['delta'] == 1.0

        # expired OTM call should have delta = 0
        greeks = bs_greeks(95, 100, 0, 0.05, 0.20, 'call')
        assert greeks['delta'] == 0.0
