from dataclasses import dataclass


@dataclass
class Greeks:
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            delta=d.get('delta', 0.0),
            gamma=d.get('gamma', 0.0),
            theta=d.get('theta', 0.0),
            vega=d.get('vega', 0.0),
            rho=d.get('rho', 0.0),
        )

    def scaled(self, multiplier: float):
        return Greeks(
            delta=self.delta * multiplier,
            gamma=self.gamma * multiplier,
            theta=self.theta * multiplier,
            vega=self.vega * multiplier,
            rho=self.rho * multiplier,
        )
