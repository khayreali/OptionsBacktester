import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_surface(vol_surface, title='IV Surface', figsize=(10, 7)):
    """3D surface plot of implied volatility."""
    if vol_surface.grid is None:
        raise ValueError("Surface not fitted")

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    K, T = np.meshgrid(vol_surface.strikes, vol_surface.expiries)

    surf = ax.plot_surface(
        K, T, vol_surface.grid * 100,  # convert to percentage
        cmap='viridis',
        edgecolor='none',
        alpha=0.8
    )

    ax.set_xlabel('Strike')
    ax.set_ylabel('Time to Expiry (years)')
    ax.set_zlabel('IV (%)')
    ax.set_title(title)

    fig.colorbar(surf, shrink=0.5, aspect=10, label='IV (%)')

    return fig, ax


def plot_smile(vol_surface, T, strikes=None, ax=None, label=None):
    """Plot IV smile for a single expiry."""
    if strikes is None:
        strikes = vol_surface.strikes

    K, ivs = vol_surface.get_smile(T, strikes)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(K, ivs * 100, 'o-', label=label or f'T={T:.2f}')
    ax.set_xlabel('Strike')
    ax.set_ylabel('IV (%)')
    ax.set_title(f'Volatility Smile (T={T:.3f})')
    ax.grid(True, alpha=0.3)

    if label:
        ax.legend()

    return ax


def plot_term_structure(vol_surface, K, expiries=None, ax=None):
    """Plot IV term structure for a single strike."""
    if expiries is None:
        expiries = vol_surface.expiries

    T, ivs = vol_surface.get_term_structure(K, expiries)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(T, ivs * 100, 's-')
    ax.set_xlabel('Time to Expiry (years)')
    ax.set_ylabel('IV (%)')
    ax.set_title(f'Term Structure (K={K:.0f})')
    ax.grid(True, alpha=0.3)

    return ax


def plot_smiles_multi(vol_surface, expiries=None, figsize=(10, 6)):
    """Plot multiple smiles on one chart."""
    if expiries is None:
        expiries = vol_surface.expiries

    fig, ax = plt.subplots(figsize=figsize)

    for T in expiries:
        K, ivs = vol_surface.get_smile(T)
        ax.plot(K, ivs * 100, 'o-', label=f'T={T:.2f}y', markersize=4)

    ax.set_xlabel('Strike')
    ax.set_ylabel('IV (%)')
    ax.set_title('Volatility Smiles by Expiry')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax
