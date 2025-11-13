import pytest
import numpy as np
import pandas as pd

from src.volatility import VolSurface, svi_fit, poly_fit, SmileFitter


class TestVolSurface:
    def test_fit_from_ivs(self):
        strikes = np.array([90, 95, 100, 105, 110])
        expiries = np.array([0.25, 0.5, 1.0])

        # create a simple smile that increases away from ATM
        grid = np.zeros((3, 5))
        for i, T in enumerate(expiries):
            for j, K in enumerate(strikes):
                moneyness = (K - 100) / 100
                grid[i, j] = 0.20 + 0.5 * moneyness**2 + 0.02 * T

        surf = VolSurface()
        surf.fit_from_ivs(strikes, expiries, grid)

        # check interpolation at grid points
        assert abs(surf.interpolate(100, 0.5) - grid[1, 2]) < 0.01

    def test_interpolate_between_points(self):
        strikes = np.array([90, 100, 110])
        expiries = np.array([0.25, 0.5])
        grid = np.array([
            [0.25, 0.20, 0.25],
            [0.26, 0.21, 0.26],
        ])

        surf = VolSurface()
        surf.fit_from_ivs(strikes, expiries, grid)

        # should interpolate smoothly
        iv = surf.interpolate(95, 0.375)
        assert 0.18 < iv < 0.28

    def test_get_smile(self):
        strikes = np.array([90, 95, 100, 105, 110])
        expiries = np.array([0.25, 0.5])
        grid = np.array([
            [0.25, 0.22, 0.20, 0.22, 0.25],
            [0.26, 0.23, 0.21, 0.23, 0.26],
        ])

        surf = VolSurface()
        surf.fit_from_ivs(strikes, expiries, grid)

        K, ivs = surf.get_smile(0.25)
        assert len(K) == 5
        assert len(ivs) == 5

    def test_to_dataframe(self):
        strikes = np.array([90, 100, 110])
        expiries = np.array([0.25, 0.5])
        grid = np.array([
            [0.25, 0.20, 0.25],
            [0.26, 0.21, 0.26],
        ])

        surf = VolSurface()
        surf.fit_from_ivs(strikes, expiries, grid)

        df = surf.to_dataframe()
        assert len(df) == 6
        assert 'T' in df.columns
        assert 'strike' in df.columns
        assert 'iv' in df.columns


class TestSVIFit:
    def test_svi_fit_basic(self):
        # generate synthetic smile data
        forward = 100
        T = 0.5
        strikes = np.array([85, 90, 95, 100, 105, 110, 115])

        # simple parabolic smile
        k = np.log(strikes / forward)
        ivs = 0.20 + 0.3 * k**2

        result = svi_fit(strikes, ivs, forward, T)

        assert result['success']
        assert result['rmse_iv'] < 0.02

    def test_svi_fit_with_skew(self):
        forward = 100
        T = 0.25
        strikes = np.array([90, 95, 100, 105, 110])

        k = np.log(strikes / forward)
        ivs = 0.22 - 0.1 * k + 0.4 * k**2  # skewed smile

        result = svi_fit(strikes, ivs, forward, T)

        assert result['success']


class TestPolyFit:
    def test_poly_fit_quadratic(self):
        forward = 100
        strikes = np.array([90, 95, 100, 105, 110])
        k = np.log(strikes / forward)
        ivs = 0.20 + 0.5 * k**2

        result = poly_fit(strikes, ivs, forward, degree=2)

        assert result['rmse'] < 0.01
        assert len(result['coeffs']) == 3


class TestSmileFitter:
    def test_svi_fitter(self):
        forward = 100
        T = 0.5
        strikes = np.array([90, 95, 100, 105, 110])
        k = np.log(strikes / forward)
        ivs = 0.20 + 0.4 * k**2

        fitter = SmileFitter(method='svi')
        fitter.fit(strikes, ivs, forward, T)

        # evaluate at ATM
        iv_atm = fitter(100)
        assert abs(iv_atm - 0.20) < 0.02

    def test_poly_fitter(self):
        forward = 100
        T = 0.5
        strikes = np.array([90, 95, 100, 105, 110])
        k = np.log(strikes / forward)
        ivs = 0.20 + 0.4 * k**2

        fitter = SmileFitter(method='poly')
        fitter.fit(strikes, ivs, forward, T)

        iv_atm = fitter(100)
        assert abs(iv_atm - 0.20) < 0.01
