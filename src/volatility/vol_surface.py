import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator, interp1d
from typing import Optional

from src.pricing import implied_vol


class VolSurface:
    def __init__(self):
        self.strikes = None
        self.expiries = None
        self.grid = None  # 2D array: [expiry_idx, strike_idx]
        self._interp = None
        self._interp_method = None
        self.spot = None

    def fit_surface(self, options_data: pd.DataFrame, spot: float, r: float = 0.05):
        """
        Fit IV surface from option prices.
        options_data needs: strike, T, option_type, price
        """
        self.spot = spot
        df = options_data.copy()

        ivs = []
        for _, row in df.iterrows():
            iv = implied_vol(
                row['price'], spot, row['strike'], row['T'],
                r, row['option_type']
            )
            ivs.append(iv)
        df['iv'] = ivs

        df = df[~np.isnan(df['iv'])]

        if len(df) < 4:
            raise ValueError("Need at least 4 valid IV points")

        self.strikes = np.sort(df['strike'].unique())
        self.expiries = np.sort(df['T'].unique())

        self.grid = np.full((len(self.expiries), len(self.strikes)), np.nan)

        for i, T in enumerate(self.expiries):
            for j, K in enumerate(self.strikes):
                mask = (df['T'] == T) & (df['strike'] == K)
                if mask.any():
                    self.grid[i, j] = df.loc[mask, 'iv'].values[0]

        self._fill_missing()
        self._build_interpolator()

    def fit_from_ivs(self, strikes: np.ndarray, expiries: np.ndarray, iv_grid: np.ndarray):
        """Fit from pre-computed IV grid."""
        self.strikes = np.array(strikes, dtype=float)
        self.expiries = np.array(expiries, dtype=float)
        self.grid = np.array(iv_grid, dtype=float)

        self._fill_missing()
        self._build_interpolator()

    def _fill_missing(self):
        """Fill NaN values by linear interpolation."""
        for i in range(len(self.expiries)):
            row = self.grid[i, :]
            valid = ~np.isnan(row)

            if valid.sum() == 0:
                continue
            elif valid.sum() == 1:
                self.grid[i, :] = row[valid][0]
            else:
                f = interp1d(self.strikes[valid], row[valid],
                            kind='linear', fill_value='extrapolate')
                self.grid[i, :] = f(self.strikes)

        for j in range(len(self.strikes)):
            col = self.grid[:, j]
            valid = ~np.isnan(col)
            if valid.sum() >= 2:
                f = interp1d(self.expiries[valid], col[valid],
                            kind='linear', fill_value='extrapolate')
                self.grid[:, j] = f(self.expiries)

    def _build_interpolator(self):
        """Build 2D interpolator, adapting method to grid size."""
        n_exp = len(self.expiries)
        n_strikes = len(self.strikes)

        # cubic spline needs at least 4 points in each dimension
        if n_exp >= 4 and n_strikes >= 4:
            self._interp = RectBivariateSpline(
                self.expiries, self.strikes, self.grid, kx=3, ky=3
            )
            self._interp_method = 'cubic'
        else:
            # fall back to linear
            self._interp = RegularGridInterpolator(
                (self.expiries, self.strikes), self.grid,
                method='linear', bounds_error=False, fill_value=None
            )
            self._interp_method = 'linear'

    def interpolate(self, K: float, T: float) -> float:
        """Get IV at strike K and expiry T."""
        if self._interp is None:
            raise ValueError("Surface not fitted")

        K = np.clip(K, self.strikes.min(), self.strikes.max())
        T = np.clip(T, self.expiries.min(), self.expiries.max())

        if self._interp_method == 'cubic':
            return float(self._interp(T, K)[0, 0])
        else:
            return float(self._interp([[T, K]])[0])

    def get_smile(self, T: float, strikes: Optional[np.ndarray] = None) -> tuple:
        """IV smile at a given expiry."""
        if strikes is None:
            strikes = self.strikes
        vols = np.array([self.interpolate(K, T) for K in strikes])
        return strikes, vols

    def get_term_structure(self, K: float, expiries: Optional[np.ndarray] = None) -> tuple:
        """IV term structure at a given strike."""
        if expiries is None:
            expiries = self.expiries
        vols = np.array([self.interpolate(K, T) for T in expiries])
        return expiries, vols

    def to_dataframe(self) -> pd.DataFrame:
        """Export as tidy DataFrame."""
        rows = []
        for i, T in enumerate(self.expiries):
            for j, K in enumerate(self.strikes):
                rows.append({'T': T, 'strike': K, 'iv': self.grid[i, j]})
        return pd.DataFrame(rows)
