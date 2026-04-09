"""
Electricity price simulation using an Ornstein-Uhlenbeck process with a
deterministic daily cycle and occasional jump component to simulate
balancing price spikes.

The diurnal shape reflects typical UK day-ahead prices (episode starts at 4pm):
- Evening peak    (periods 0-8,   ~4pm-8pm)
- Evening decline (periods 8-16,  ~8pm-midnight)
- Overnight trough(periods 16-28, ~midnight-6am)
- Morning ramp    (periods 28-36, ~6am-10am)
- Midday plateau  (periods 36-48, ~10am-4pm)

Prices are in £/MWh over 48 half-hour periods (one day).
"""

from dataclasses import dataclass, field

import numpy as np

from ev_dispatch import FloatArray


@dataclass
class PriceProcessConfig:
    """Parameters governing the synthetic price process."""

    mean_price: float = 80.0  # Long-run mean price (£/MWh)
    mean_reversion_speed: float = 0.3  # How fast price reverts to mean
    volatility: float = 10.0  # Noise magnitude
    jump_probability: float = 0.05  # Probability of a spike per timestep
    jump_magnitude: float = 150.0  # Size of spike above current price
    periods_per_day: int = 48  # Half-hourly

    # Daily shape: additive offsets (£/MWh) applied per period.
    # Default produces a realistic UK-style evening and morning peak.
    daily_cycle_offsets: list[float] = field(
        default_factory=lambda: _default_daily_cycle_offsets()
    )


def _default_daily_cycle_offsets() -> list[float]:
    """
    Build a 48-period diurnal price shape broadly reflecting UK day-ahead prices.

    Episode starts at 4pm (period 0) to keep overnight charge cycle in sequence:
      periods 0-8:    evening peak (~4pm-8pm), +80
      periods 8-16:   evening decline (~8pm-midnight), tapering +80 to -20
      periods 16-28:  overnight trough (~midnight-6am), -20
      periods 28-36:  morning ramp (~6am-10am), tapering -20 to +40
      periods 36-48:  midday plateau (~10am-4pm), 0
    """
    offsets = np.zeros(48)
    offsets[0:8] = 80.0  # 4pm-8pm evening peak
    offsets[8:16] = np.linspace(80.0, -20.0, 8)  # 8pm-midnight decline
    offsets[16:28] = -20.0  # midnight-6am trough
    offsets[28:36] = np.linspace(-20.0, 40.0, 8)  # 6am-10am morning ramp
    offsets[36:48] = 0.0  # 10am-4pm midday plateau
    return offsets.tolist()


class PriceProcess:
    """
    Simulates day-ahead and intraday electricity prices with a daily cycle
    plus stochastic noise.

    Generates independent price scenario realisations for use in
    ADP training and policy evaluation.
    """

    def __init__(self, config: PriceProcessConfig, seed: int | None = None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self._daily_cycle_offsets = np.array(config.daily_cycle_offsets)

    def sample_scenario(self) -> FloatArray:
        """
        Generate one price scenario over a full day.

        Returns
        -------
        FloatArray
            Array of shape (periods_per_day,) with prices in £/MWh.
            Prices are clipped to a minimum of 0.
        """
        cfg = self.config
        prices = np.zeros(cfg.periods_per_day)
        current_price = cfg.mean_price

        for t in range(cfg.periods_per_day):
            reversion = cfg.mean_reversion_speed * (cfg.mean_price - current_price)
            noise = cfg.volatility * self.rng.standard_normal()
            jump = self._sample_jump()
            current_price = (
                current_price + self._daily_cycle_offsets[t] + reversion + noise + jump
            )
            prices[t] = max(current_price, 0.0)

        return prices

    def sample_scenarios(self, n: int) -> FloatArray:
        """
        Generate multiple independent price scenarios.

        Returns
        -------
        FloatArray
            Array of shape (n, periods_per_day).
        """
        return np.stack([self.sample_scenario() for _ in range(n)])

    def _sample_jump(self) -> float:
        if self.rng.random() < self.config.jump_probability:
            return self.config.jump_magnitude * self.rng.exponential()
        return 0.0
