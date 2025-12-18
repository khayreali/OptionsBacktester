#!/usr/bin/env python3
"""
Options backtester example script.
Loads SPY data, fits vol surface, runs a delta-hedged strategy, and prints results.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta

# add src to path
sys.path.insert(0, '.')

from src.models import Option
from src.pricing import bs_call_price, bs_put_price, bs_greeks
from src.volatility import VolSurface, plot_surface, plot_smiles_multi
from src.backtester import DeltaHedger, PnLAttribution, PositionManager, attribute_hedged_pnl
from src.backtester.strategy import SimpleStrategy, strategy_to_positions
from src.utils.data_loader import (
    load_spot_data, load_options_chain, generate_synthetic_options,
    options_chain_to_surface_format
)


def main():
    print("=" * 60)
    print("OPTIONS BACKTESTER")
    print("=" * 60)
    print()

    # --- Load spot data ---
    print("[1] Loading SPY spot data for 2024...")
    spot_data = load_spot_data('SPY', '2024-01-01', '2024-06-30')
    print(f"    Loaded {len(spot_data)} days of data")
    print(f"    Date range: {spot_data['date'].min()} to {spot_data['date'].max()}")
    print(f"    Price range: ${spot_data['close'].min():.2f} - ${spot_data['close'].max():.2f}")
    print()

    # --- Load/generate options chain ---
    print("[2] Generating options chain...")
    initial_spot = spot_data['close'].iloc[0]
    print(f"    Initial spot: ${initial_spot:.2f}")

    # get entry date for generating synthetic options
    entry_date_for_chain = spot_data['date'].iloc[0]
    if isinstance(entry_date_for_chain, pd.Timestamp):
        entry_date_for_chain = entry_date_for_chain.date()

    # yfinance gives current options (2026), we need historical
    # so always use synthetic for backtesting past dates
    chain = generate_synthetic_options(initial_spot, expiry_days=[14, 30, 45, 60, 90])
    print(f"    Generated {len(chain)} synthetic options")
    print(f"    Expiries: {sorted(chain['expiry'].unique())[:5]}")
    print()

    # --- Fit vol surface ---
    print("[3] Fitting vol surface...")
    vol_surface = VolSurface()

    # convert chain to surface format
    surface_data = options_chain_to_surface_format(chain, initial_spot)
    surface_data = surface_data[surface_data['T'] > 0.01]  # drop very short dated

    try:
        vol_surface.fit_surface(surface_data, initial_spot)
        print(f"    Fitted surface with {len(vol_surface.strikes)} strikes x {len(vol_surface.expiries)} expiries")
        atm_vol = vol_surface.interpolate(initial_spot, 0.1)
        print(f"    ATM 30-day vol: {atm_vol*100:.1f}%")
    except Exception as e:
        print(f"    Surface fitting failed ({e}), using constant vol")
        vol_surface = None

    print()

    # --- Set up strategy ---
    print("[4] Setting up strategy: Short ATM Put (delta hedged)")
    entry_date = spot_data['date'].iloc[0]
    if isinstance(entry_date, pd.Timestamp):
        entry_date = entry_date.date()

    # 45 DTE put
    expiry = entry_date + timedelta(days=45)
    strike = round(initial_spot / 5) * 5  # round to nearest $5

    strategy = SimpleStrategy(underlying='SPY')
    legs = strategy.short_put(
        spot=initial_spot,
        K=strike,
        expiry=expiry,
        entry_date=entry_date,
        sigma=0.18,
    )

    print(f"    Strike: ${strike}")
    print(f"    Expiry: {expiry}")
    print(f"    Premium collected: ${legs[0].entry_price:.2f}")
    print()

    # --- Run delta hedge simulation ---
    print("[5] Running delta hedge simulation...")
    option = legs[0].option

    # filter spot data to option lifetime
    # handle timezone-aware dates from yfinance
    spot_data['date'] = pd.to_datetime(spot_data['date']).dt.tz_localize(None)
    expiry_ts = pd.Timestamp(expiry)
    spot_subset = spot_data[spot_data['date'] <= expiry_ts].copy()

    hedger = DeltaHedger(
        option=option,
        spot_data=spot_subset,
        vol_surface=vol_surface,
        constant_vol=0.18,
        r=0.05,
        rehedge_frequency='daily',
        transaction_cost=0.01,
        position_size=-1,  # short position
    )

    result = hedger.run()
    summary = hedger.summary()

    print(f"    Days simulated: {summary['num_days']}")
    print(f"    Total P&L: ${summary['total_pnl']:.2f}")
    print(f"    Transaction costs: ${summary['total_transaction_costs']:.2f}")
    print(f"    Sharpe ratio: {summary['sharpe']:.2f}")
    print(f"    Max drawdown: ${summary['max_drawdown']:.2f}")
    print()

    # --- Greeks over time ---
    print("[6] Greeks exposure over time:")
    print()
    print("    Date         Spot      Delta    Gamma    Theta    Vega")
    print("    " + "-" * 55)

    # sample every 5 days
    for i in range(0, len(result), max(1, len(result) // 8)):
        row = result.iloc[i]
        print(f"    {row['date']}  ${row['spot']:7.2f}  {row['delta']:7.3f}  {row['gamma']:7.4f}  {row['theta']:7.4f}  {row['vega']:7.4f}")

    print()

    # --- P&L Attribution ---
    print("[7] P&L Attribution:")
    try:
        attr_df = attribute_hedged_pnl(result, multiplier=100)

        print(f"    Delta P&L:  ${attr_df['delta_pnl'].sum():8.2f}")
        print(f"    Gamma P&L:  ${attr_df['gamma_pnl'].sum():8.2f}")
        print(f"    Theta P&L:  ${attr_df['theta_pnl'].sum():8.2f}")
        print(f"    Vega P&L:   ${attr_df['vega_pnl'].sum():8.2f}")
        print(f"    Residual:   ${attr_df['residual'].sum():8.2f}")
        print(f"    ---------------------")
        print(f"    Total:      ${attr_df['total'].sum():8.2f}")
    except Exception as e:
        print(f"    Attribution failed: {e}")

    print()

    # --- Generate plots ---
    print("[8] Generating plots...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Spot price
    axes[0, 0].plot(result['date'], result['spot'], 'b-', linewidth=1)
    axes[0, 0].axhline(strike, color='r', linestyle='--', alpha=0.5, label=f'Strike ${strike}')
    axes[0, 0].set_title('SPY Spot Price')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Delta
    axes[0, 1].plot(result['date'], result['delta'], 'g-', linewidth=1)
    axes[0, 1].axhline(0, color='gray', linestyle='-', alpha=0.3)
    axes[0, 1].set_title('Option Delta')
    axes[0, 1].set_ylabel('Delta')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Hedge position
    axes[0, 2].plot(result['date'], result['hedge_shares'], 'purple', linewidth=1)
    axes[0, 2].axhline(0, color='gray', linestyle='-', alpha=0.3)
    axes[0, 2].set_title('Hedge Position (Shares)')
    axes[0, 2].set_ylabel('Shares')
    axes[0, 2].tick_params(axis='x', rotation=45)

    # Cumulative P&L
    axes[1, 0].plot(result['date'], result['cumulative_pnl'], 'b-', linewidth=1)
    axes[1, 0].axhline(0, color='gray', linestyle='-', alpha=0.3)
    axes[1, 0].fill_between(result['date'], 0, result['cumulative_pnl'],
                            where=result['cumulative_pnl'] >= 0, alpha=0.3, color='green')
    axes[1, 0].fill_between(result['date'], 0, result['cumulative_pnl'],
                            where=result['cumulative_pnl'] < 0, alpha=0.3, color='red')
    axes[1, 0].set_title('Cumulative P&L')
    axes[1, 0].set_ylabel('P&L ($)')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Daily P&L
    axes[1, 1].bar(result['date'], result['total_pnl'], color='steelblue', alpha=0.7)
    axes[1, 1].axhline(0, color='gray', linestyle='-', alpha=0.3)
    axes[1, 1].set_title('Daily P&L')
    axes[1, 1].set_ylabel('P&L ($)')
    axes[1, 1].tick_params(axis='x', rotation=45)

    # Implied vol
    axes[1, 2].plot(result['date'], result['iv'] * 100, 'orange', linewidth=1)
    axes[1, 2].set_title('Implied Volatility')
    axes[1, 2].set_ylabel('IV (%)')
    axes[1, 2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('output_backtest_results.png', dpi=150, bbox_inches='tight')
    print("    Saved: output_backtest_results.png")

    # Vol surface plot (if fitted)
    if vol_surface is not None:
        try:
            fig2, ax2 = plot_surface(vol_surface, title='Implied Volatility Surface')
            plt.savefig('output_vol_surface.png', dpi=150, bbox_inches='tight')
            print("    Saved: output_vol_surface.png")
        except Exception as e:
            print(f"    Vol surface plot failed: {e}")

    print()
    print("=" * 60)
    print("BACKTEST COMPLETE")
    print("=" * 60)

    # show plots if running interactively
    try:
        plt.show()
    except:
        pass

    return result, summary


if __name__ == '__main__':
    result, summary = main()
