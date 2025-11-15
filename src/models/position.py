from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from .option import Option
from .greeks import Greeks


@dataclass
class Position:
    option: Option
    quantity: int
    entry_price: float
    entry_date: date
    greeks: Greeks = field(default_factory=Greeks)

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def notional(self) -> float:
        return abs(self.quantity) * 100 * self.entry_price

    def market_value(self, current_price: float) -> float:
        return self.quantity * 100 * current_price

    def unrealized_pnl(self, current_price: float) -> float:
        entry_value = self.quantity * 100 * self.entry_price
        current_value = self.quantity * 100 * current_price
        return current_value - entry_value

    def update_greeks(self, greeks_dict: dict):
        raw = Greeks.from_dict(greeks_dict)
        self.greeks = raw.scaled(self.quantity * 100)

    def net_delta(self) -> float:
        return self.greeks.delta


@dataclass
class Portfolio:
    positions: list = field(default_factory=list)
    cash: float = 0.0

    def add_position(self, pos: Position):
        self.positions.append(pos)
        self.cash -= pos.quantity * 100 * pos.entry_price

    def total_delta(self) -> float:
        return sum(p.net_delta() for p in self.positions)

    def total_gamma(self) -> float:
        return sum(p.greeks.gamma for p in self.positions)

    def total_theta(self) -> float:
        return sum(p.greeks.theta for p in self.positions)

    def total_vega(self) -> float:
        return sum(p.greeks.vega for p in self.positions)
