# OptionsBacktester

A Python tool for backtesting delta-hedged options strategies. Fits an implied volatility surface from market data (or synthetic data), simulates delta hedging through time, and breaks down P&L into greek components.

Built this to better understand how delta hedging actually works in practice and where the P&L comes from.

![Demo](options_demo.gif)

## What it does

- **IV Surface Fitting**: Takes options chain data and fits an implied vol surface using cubic interpolation. Also has SVI fitting if you want to get fancy.
- **Delta Hedging Simulation**: Steps through historical spot data day-by-day, rebalances the hedge to stay delta-neutral, tracks transaction costs.
- **P&L Attribution**: Breaks down your P&L into delta, gamma, theta, vega components using Taylor expansion. Shows you what's actually driving your returns.
- **Strategy Helpers**: Quick functions to set up common strategies (straddles, spreads, iron condors, etc.)

## Quick Start

```bash
# install dependencies
pip install -e .

# run the example backtest
python run_backtest.py
```

This will:
1. Load SPY data for 2024
2. Generate a synthetic options chain (yfinance historical options are spotty)
3. Fit a vol surface
4. Simulate a short ATM put with daily delta hedging
5. Print greeks over time and P&L breakdown
6. Save some plots

## Example Output

```
[5] Running delta hedge simulation...
    Days simulated: 33
    Total P&L: $234.93
    Transaction costs: $1.38
    Sharpe ratio: 7.96

[6] Greeks exposure over time:

    Date         Spot      Delta    Gamma    Theta    Vega
    -------------------------------------------------------
    2024-01-02  $ 461.25   -0.433   0.0130  -0.1039   0.6371
    2024-01-19  $ 470.79   -0.291   0.0141  -0.1290   0.4470
    2024-02-06  $ 482.06   -0.055   0.0076  -0.0783   0.0890

[7] P&L Attribution:
    Delta P&L:  $ -775.92
    Gamma P&L:  $  203.63
    Theta P&L:  $ -305.52
    Vega P&L:   $  -14.19
    Residual:   $ -120.23
```

## Project Structure

```
OptionsBacktester/
├── src/
│   ├── models/       # Option, Greeks, Position dataclasses
│   ├── pricing/      # Black-Scholes, implied vol solvers
│   ├── volatility/   # Vol surface fitting, SVI
│   ├── backtester/   # Delta hedger, P&L attribution, strategies
│   └── utils/        # Data loading, helpers
├── tests/            # pytest tests
├── data/             # Sample data files
├── notebooks/        # Jupyter notebooks for analysis
└── run_backtest.py   # Main example script
```

## Running Tests

```bash
pytest tests/ -v
```

## Tech Stack

- Python 3.9+
- NumPy, SciPy (pricing math)
- pandas (data handling)
- matplotlib (plots)
- yfinance (market data, when it works)

## Would be nice to add

- **Stochastic vol models**: Right now it's just Black-Scholes. Would be cool to add Heston or SABR to see how vol dynamics affect hedging P&L.
- **Transaction cost optimization**: Currently rehedges daily to zero delta. Could add a band around zero to reduce turnover.
- **More realistic vol surface dynamics**: The vol surface is static right now. In reality it moves around, which is a big part of options P&L.
- **Greeks hedging**: Only does delta hedging. Could add gamma scalping or vega hedging.

## Notes

- The synthetic options generator creates realistic-ish vol smiles with skew. Good enough for testing but don't use for actual trading decisions.
- yfinance options data can be flaky, so the backtest uses synthetic data by default for reproducibility.
- P&L attribution residual can be large when spot moves a lot - that's the higher-order terms we're ignoring.
