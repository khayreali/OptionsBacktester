from dataclasses import dataclass
from datetime import date
from typing import Literal


@dataclass
class Option:
    strike: float
    expiry: date
    option_type: Literal['call', 'put']
    underlying: str = 'SPY'

    @property
    def K(self):
        return self.strike

    def time_to_expiry(self, as_of: date) -> float:
        days = (self.expiry - as_of).days
        return max(days / 365.0, 0.0)

    def is_expired(self, as_of: date) -> bool:
        return as_of >= self.expiry

    def intrinsic(self, S: float) -> float:
        if self.option_type == 'call':
            return max(S - self.strike, 0)
        return max(self.strike - S, 0)
