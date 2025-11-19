import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Optional, List

from src.pricing import bs_call_price, bs_put_price


def load_price_data(filepath: str) -> pd.DataFrame:
    """Load spot data from CSV."""
    df = pd.read_csv(filepath)

    date_cols = [c for c in df.columns if 'date' in c.lower()]
    if date_cols:
        df['date'] = pd.to_datetime(df[date_cols[0]])
    else:
        df['date'] = pd.to_datetime(df.iloc[:, 0])

    close_cols = [c for c in df.columns if 'close' in c.lower() or 'adj' in c.lower()]
    if close_cols:
        df['close'] = df[close_cols[0]]
    else:
        df['close'] = df.iloc[:, 4]

    return df[['date', 'close']].dropna()


def load_spot_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Load spot data from yfinance."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start, end=end)

        if hist.empty:
            raise ValueError(f"No data for {ticker}")

        df = hist.reset_index()
        df['date'] = pd.to_datetime(df['Date'])
        df['close'] = df['Close']
        df['high'] = df['High']
        df['low'] = df['Low']
        df['volume'] = df['Volume']

        return df[['date', 'close', 'high', 'low', 'volume']]

    except Exception as e:
        print(f"yfinance failed: {e}")
        print("Generating synthetic spot data...")
        return generate_synthetic_spot(ticker, start, end)


def generate_synthetic_spot(
    ticker: str,
    start: str,
    end: str,
    initial_price: float = 470.0,
    annual_vol: float = 0.18,
    annual_drift: float = 0.08,
) -> pd.DataFrame:
    """Generate synthetic spot data when yfinance fails."""
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    dates = pd.bdate_range(start_dt, end_dt)
    n = len(dates)

    daily_vol = annual_vol / np.sqrt(252)
    daily_drift = annual_drift / 252

    returns = np.random.normal(daily_drift, daily_vol, n)
    prices = initial_price * np.exp(np.cumsum(returns))

    # add some realistic intraday range
    high = prices * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.005, n)))
    volume = np.random.randint(50_000_000, 150_000_000, n)

    return pd.DataFrame({
        'date': dates,
        'close': prices,
        'high': high,
        'low': low,
        'volume': volume,
    })


def load_options_chain(ticker: str, num_expiries: int = 5) -> pd.DataFrame:
    """
    Load options chain from yfinance.
    Falls back to synthetic data if yfinance is flaky.
    """
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        expirations = stock.options

        if not expirations:
            raise ValueError("No options data available")

        all_chains = []
        spot = stock.history(period='1d')['Close'].iloc[-1]

        for exp in expirations[:num_expiries]:
            try:
                chain = stock.option_chain(exp)

                for opt_type, data in [('call', chain.calls), ('put', chain.puts)]:
                    df = data.copy()
                    df['option_type'] = opt_type
                    df['expiry'] = exp
                    df['spot'] = spot
                    all_chains.append(df)

            except Exception:
                continue

        if not all_chains:
            raise ValueError("Failed to fetch any options data")

        result = pd.concat(all_chains, ignore_index=True)
        result = result.rename(columns={
            'impliedVolatility': 'implied_vol',
            'lastPrice': 'last_price',
        })

        cols = ['strike', 'expiry', 'option_type', 'implied_vol', 'bid', 'ask', 'last_price', 'spot']
        cols = [c for c in cols if c in result.columns]

        return result[cols]

    except Exception as e:
        print(f"yfinance options failed: {e}")
        print("Generating synthetic options chain...")

        # need spot price for synthetic data
        try:
            import yfinance as yf
            spot = yf.Ticker(ticker).history(period='1d')['Close'].iloc[-1]
        except:
            spot = 470.0

        return generate_synthetic_options(spot)


def generate_synthetic_options(
    spot: float,
    r: float = 0.05,
    base_vol: float = 0.18,
    expiry_days: List[int] = None,
    num_strikes: int = 15,
) -> pd.DataFrame:
    """
    Generate realistic synthetic options data.
    Creates a vol smile with skew and term structure.
    """
    if expiry_days is None:
        expiry_days = [7, 14, 30, 45, 60, 90]

    today = date.today()
    rows = []

    for days in expiry_days:
        expiry = today + timedelta(days=days)
        T = days / 365.0

        # generate strikes around spot
        strike_range = spot * 0.15  # +/- 15%
        strikes = np.linspace(spot - strike_range, spot + strike_range, num_strikes)

        for K in strikes:
            moneyness = np.log(K / spot)

            # vol smile: higher vol OTM puts, slight smile for calls
            skew = -0.12  # negative skew
            smile = 0.08  # smile curvature
            term_adj = 0.02 * np.sqrt(T)  # vol increases with time slightly

            iv = base_vol + skew * moneyness + smile * moneyness**2 + term_adj
            iv = max(iv, 0.05)  # floor at 5%

            for opt_type in ['call', 'put']:
                if opt_type == 'call':
                    price = bs_call_price(spot, K, T, r, iv)
                else:
                    price = bs_put_price(spot, K, T, r, iv)

                # add bid-ask spread (wider for OTM options)
                spread_pct = 0.02 + 0.03 * abs(moneyness)
                spread = price * spread_pct
                bid = max(price - spread / 2, 0.01)
                ask = price + spread / 2

                rows.append({
                    'strike': round(K, 2),
                    'expiry': expiry,
                    'option_type': opt_type,
                    'implied_vol': round(iv, 4),
                    'bid': round(bid, 2),
                    'ask': round(ask, 2),
                    'last_price': round(price, 2),
                    'spot': spot,
                    'T': T,
                })

    return pd.DataFrame(rows)


def options_chain_to_surface_format(
    chain_df: pd.DataFrame,
    spot: float,
    as_of_date: date = None,
    max_T: float = 2.0,
) -> pd.DataFrame:
    """
    Convert options chain to format expected by VolSurface.fit_surface().
    Returns DataFrame with: strike, T, option_type, price
    """
    df = chain_df.copy()

    if as_of_date is None:
        as_of_date = date.today()

    # compute T
    df['expiry'] = pd.to_datetime(df['expiry'])
    df['T'] = (df['expiry'] - pd.Timestamp(as_of_date)).dt.days / 365.0

    # filter out unreasonable T values
    df = df[(df['T'] > 0.01) & (df['T'] < max_T)]

    # use mid price
    if 'price' not in df.columns:
        if 'bid' in df.columns and 'ask' in df.columns:
            df['price'] = (df['bid'] + df['ask']) / 2
        elif 'last_price' in df.columns:
            df['price'] = df['last_price']

    # filter out zero/negative prices
    df = df[df['price'] > 0.01]

    return df[['strike', 'T', 'option_type', 'price']]
