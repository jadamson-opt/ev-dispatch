"""
Piecewise-linear concave value function approximation (VFA) for a representative EV asset.

The VFA approximates the expected future revenue achievable from a given SOC
at a given time period, under the best attainable policy from that point forward.

Represented as a piecewise-linear concave function over SOC in [0, 1], with one
set of slopes per timestep. Concavity is maintained after each update, reflecting
the diminishing marginal returns of additional stored energy.

The slope at a given SOC is the shadow price. This is the marginal value of one additional
kWh stored at that point, which is broadcast to assets as their dispatch signal.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class VFAConfig:
    """Parameters controlling the VFA structure and learning."""

    n_segments: int = 10          # Number of piecewise-linear segments over [0, 1]
    # learning_rate: float = 0.05   # Step size for slope updates
    learning_rate: float = 1.0    # Step size for slope updates — intended to be normalised
                                  # by n_available in the trainer, giving effective per-asset
                                  # rate of learning_rate / n_available per period
    discount_factor: float = 0.99
    mean_price: float = 80.0  # Long-run mean price (£/MWh)


class AssetValueFunction:
    """
    Piecewise-linear concave VFA for a single representative EV asset.

    Stores one set of slopes per timestep. The value at a given SOC is obtained
    by integrating slopes from 0 up to that SOC. Concavity is enforced after
    each update by projecting slopes onto the non-increasing cone.

    Slopes are zeroed out for periods beyond the expected availability horizon,
    preventing the TD bootstrap from pulling current-period values toward
    stale initialised estimates in periods where no assets are ever present.

    All SOC values are fractions in [0, 1]. Slopes are in £/MWh equivalent,
    representing the marginal value of moving one SOC unit (i.e. full battery
    capacity) of stored energy.
    """

    def __init__(
            self,
            config: VFAConfig,
            periods_per_day: int,
            last_available_period: int,
            max_discharge_per_period: float
    ):
        self.config = config
        self.periods_per_day = periods_per_day

        breakpoints = np.linspace(0.0, 1.0, config.n_segments + 1)
        self.breakpoints = breakpoints
        self.segment_width = breakpoints[1] - breakpoints[0]

       #  soc_starts = np.array([k / config.n_segments for k in range(config.n_segments)])
       #  initial_slopes = config.mean_price * (2.0 - soc_starts)  # exactly 2x mean at k=0, ~1x at k=n-1
       #  self.slopes = np.tile(initial_slopes, (periods_per_day, 1))

        # Initialise slopes with a concave shape: high marginal value at low SOC,
        # decreasing toward mean_price at full SOC. This prevents the policy from
        # discharging to zero in early episodes before the VFA has learned, by
        # making the last units of stored energy appear expensive to lose.
        # The slopes also taper towards zero near expected departure, since revenue
        # is limited by how much energy can be discharged before then.
        soc_starts = np.array([k / config.n_segments for k in range(config.n_segments)])

        self.slopes = np.zeros((periods_per_day, config.n_segments))
        for t in range(periods_per_day):
            periods_remaining = max(last_available_period - t, 0)
            max_dischargeable_soc = min(periods_remaining * max_discharge_per_period, 1.0)
            max_value = max_dischargeable_soc * config.mean_price
            # Concave shape across SOC segments, scaled by physically reachable revenue
            self.slopes[t] = max_value * (2.0 - soc_starts)  # exactly 2x mean at k=0, ~1x at k=n-1

    def value(self, soc: float, period: int) -> float:
        """
        Estimate the future value of holding a given SOC at a given period.

        Computed by integrating slopes from 0 to soc to get the accumulated
        value up to that specific SOC point.
        """
        soc = float(np.clip(soc, 0.0, 1.0))
        segment_values = self.slopes[period] * self.segment_width
        cumulative_values = np.concatenate([[0.0], np.cumsum(segment_values)])

        segment_index = self._get_segment_index(soc)
        gap_from_segment_start = soc - self.breakpoints[segment_index]
        return float(
            cumulative_values[segment_index] + self.slopes[period, segment_index] * gap_from_segment_start
        )

    def shadow_price(self, soc: float, period: int) -> float:
        """
        Return the marginal value of stored energy at this SOC and period,
        (the value associated with one additional unit of SOC).

        This is the slope of the value function at the given SOC. It is the signal
        broadcast to assets to drive their charge/discharge decisions.
        """
        soc = float(np.clip(soc, 0.0, 1.0))
        seg = self._get_segment_index(soc)
        return float(self.slopes[period, seg])

    def update(self, soc: float, period: int, target_value: float, learning_rate: float) -> None:
        """
        Update the VFA slope at the given SOC and period toward a target value.

        The target is the temporal difference (TD) estimate: immediate reward
        + discounted next-period value.
        After updating the slope, concavity is enforced across all segments.

        Parameters
        ----------
        soc:
            SOC at which the update is made from.
        period:
            Timestep being updated.
        target_value:
            TD target: r_t + gamma * V_{t+1}(s_{t+1}).
        learning_rate:
            Override the config learning rate — used by the trainer to pass
            a fleet-size-normalised effective rate.
        """
        if soc < self.segment_width * 0.5:
            return  # too close to zero, update unreliable
        current_value = self.value(soc, period)
        value_error = target_value - current_value

        # Translate value error into an implied slope correction
        seg = self._get_segment_index(soc)
        # slope_correction = value_error / (self.segment_width + 1e-9)
        self.slopes[period, seg] = np.maximum(0, self.slopes[period, seg] + learning_rate * value_error)

        self._enforce_concavity(period)

    def slopes_at_period(self, period: int) -> np.ndarray:
        """Return the full slope array for a given period (for inspection/plotting)."""
        return self.slopes[period].copy()

    def _get_segment_index(self, soc: float) -> int:
        """Return the segment index containing this SOC value."""
        segment_index = int(soc / self.segment_width)
        return min(segment_index, self.config.n_segments - 1)

    def _enforce_concavity(self, period: int) -> None:
        """
        Force the slopes to be concave in SOC, meaning they must non-increasing.

        Concavity of the value function requires slopes to be non-increasing
        in SOC, since higher stored energy has weakly lower marginal value.
        In other words, there are diminishing marginal returns from charging to
        higher SOC (e.g. less value when you're nearly at full capacity where there
        is less room left to exploit price spikes).

        Enforced by forward-scanning and clipping each slope to the previous.
        """
        for k in range(1, self.config.n_segments):
            if self.slopes[period, k] > self.slopes[period, k - 1]:
                self.slopes[period, k] = self.slopes[period, k - 1]