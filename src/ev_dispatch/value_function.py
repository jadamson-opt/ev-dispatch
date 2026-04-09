"""
Piecewise-linear concave value function approximation (VFA) for a representative EV asset.

Maintains one VFA per asset type (commuter, WFH). Each VFA approximates
the expected future arbitrage revenue achievable from a given SOC at a given
time period, under the best attainable policy from that point forward.

Commuter VFA: slopes taper toward zero near the expected last departure,
reflecting that stored energy has no value after assets have left.

WFH VFA: slopes are uniform across time periods since WFH assets are
available throughout the day. Only the concave SOC shape is initialised.

Te VFA is represented as a piecewise-linear concave function over SOC in [0, 1], with one
set of slopes per timestep. Concavity is maintained after each update, reflecting
the diminishing marginal returns of additional stored energy.

The slope at a given SOC is the shadow price. This is the marginal value of one additional
unit of SOC stored at that point, which is broadcast to assets as their dispatch signal.
"""

from dataclasses import dataclass

import numpy as np

from ev_dispatch import FloatArray
from ev_dispatch.ev_asset import AssetType


@dataclass
class VFAConfig:
    """Parameters controlling the VFA structure and learning."""

    n_segments: int = 10  # Number of piecewise-linear segments over [0, 1]
    learning_rate: float = (
        1.0  # Step size for slope updates — normalised by n_available in the trainer
    )
    discount_factor: float = 0.99
    mean_price: float = 80.0  # Long-run mean price (£/MWh)


class AssetValueFunction:
    """
    Piecewise-linear concave VFA for a single representative EV asset.

    Stores one set of slopes per timestep. The value at a given SOC is obtained
    by integrating slopes from 0 up to that SOC. Concavity is enforced after
    each update along with a zero lower bound.

    All SOC values are fractions in [0, 1]. Slopes are in £/SOC,
    representing the marginal value of moving one SOC unit (i.e. full battery
    capacity) of stored energy.

    Slopes are initialised with a simple heuristic dependent on asset type.
    """

    def __init__(
        self,
        config: VFAConfig,
        asset_type: AssetType,
        periods_per_day: int,
        expected_last_plugged_in_period: int,
        max_discharge_per_period: float,
        battery_capacity_kwh: float,
    ):
        self.config = config
        self.asset_type = asset_type
        self.periods_per_day = periods_per_day

        breakpoints = np.linspace(0.0, 1.0, config.n_segments + 1)
        self.breakpoints = breakpoints
        self.segment_width = breakpoints[1] - breakpoints[0]

        self.slopes = self._initialise_slopes(
            asset_type,
            expected_last_plugged_in_period,
            max_discharge_per_period,
            battery_capacity_kwh,
        )

    def value(self, soc: float, period: int) -> float:
        """
        Estimate the future value of holding a given SOC at a given period, in £.

        Computed by integrating slopes from 0 to soc to get the accumulated
        value up to that specific SOC point.
        """
        soc = min(max(soc, 0.0), 1.0)
        segment_values = self.slopes[period] * self.segment_width
        cumulative_values = np.concatenate([[0.0], np.cumsum(segment_values)])

        segment_index = self._get_segment_index(soc)
        gap_from_segment_start = soc - self.breakpoints[segment_index]
        return float(
            cumulative_values[segment_index]
            + self.slopes[period, segment_index] * gap_from_segment_start
        )

    def shadow_price(self, soc: float, period: int) -> float:
        """
        Return the marginal value of stored energy at this SOC and period,
        (the value associated with one additional unit of SOC). The unit is
        £/SOC.

        This is the slope of the value function at the given SOC. It is the signal
        broadcast to assets to drive their charge/discharge decisions.
        """
        soc = min(max(soc, 0.0), 1.0)
        seg = self._get_segment_index(soc)
        return float(self.slopes[period, seg])

    def update(
        self, soc: float, period: int, target_value: float, learning_rate: float
    ) -> None:
        """
        Update the VFA slope at the given SOC and period toward a target value.

        The target is the temporal difference (TD) estimate: immediate reward
        + discounted next-period value.
        After updating the slope, concavity is enforced across all segments.

        Skips update when SOC is too close to zero since current_value is
        always zero there, producing unreliable large positive errors.

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
        self.slopes[period, seg] = np.maximum(
            0, self.slopes[period, seg] + learning_rate * value_error
        )

        self._enforce_concavity(period, seg)

    def slopes_at_period(self, period: int) -> np.ndarray:
        """Return the full slope array for a given period (for inspection/plotting)."""
        return self.slopes[period].copy()

    def _initialise_slopes(
        self,
        asset_type: AssetType,
        last_available_period: int,
        max_discharge_per_period: float,
        battery_capacity_kwh: float,
    ) -> FloatArray:
        """
        Initialise slopes using an expectation of future revenue.

        For commuters: slopes scale with how much energy can physically be
        discharged before last departure, times the expected price.
        This produces a natural taper toward zero near the departure horizon.

        For WFH: slopes are uniform across periods since assets are available
        throughout the day. Only the concave SOC shape is applied.
        """
        cfg = self.config
        soc_starts = np.array([k / cfg.n_segments for k in range(cfg.n_segments)])
        slopes = np.zeros((self.periods_per_day, cfg.n_segments))
        mean_price_per_soc = cfg.mean_price * (
            battery_capacity_kwh / 1000
        )  # £ per unit SOC

        for t in range(self.periods_per_day):
            if asset_type == AssetType.COMMUTER:
                periods_remaining = max(last_available_period - t, 0)
                max_dischargeable_soc = min(
                    periods_remaining * max_discharge_per_period, 1.0
                )
                expected_value = mean_price_per_soc * max_dischargeable_soc
            else:
                # WFH: uniform time value, only SOC shape varies
                expected_value = mean_price_per_soc

            slopes[t] = expected_value * (
                2.0 - soc_starts
            )  # exactly 2x mean at k=0, ~1x at k=n-1

        return slopes

    def _get_segment_index(self, soc: float) -> int:
        """Return the segment index containing this SOC value."""
        segment_index = int(soc / self.segment_width)
        return min(segment_index, self.config.n_segments - 1)

    def _enforce_concavity(self, period: int, updated_from_seg: int) -> None:
        """
        Force the slopes to be concave in SOC, meaning they must non-increasing.

        Concavity of the value function requires slopes to be non-increasing
        in SOC, since higher stored energy has weakly lower marginal value.
        In other words, there are diminishing marginal returns from charging to
        higher SOC (e.g. less value when you're nearly at full capacity where there
        is less room left to exploit price spikes).

        Enforced by forward-scanning and clipping each slope to the previous.
        """
        for k in range(updated_from_seg + 1, self.config.n_segments):
            if self.slopes[period, k] > self.slopes[period, updated_from_seg]:
                self.slopes[period, k] = self.slopes[period, updated_from_seg]
            else:
                break  # already satisfies concavity, no need to continue


def make_vfa_registry(
    config: VFAConfig,
    periods_per_day: int,
    last_commuter_departure: int,
    max_discharge_per_period: float,
    battery_capacity_kwh: float,
) -> dict[AssetType, AssetValueFunction]:
    """
    Construct one VFA per asset type and return as a dict.

    Parameters
    ----------
    config:
        The VFA config.
    periods_per_day:
        Number of time periods within a day.
    last_commuter_departure:
        Latest expected departure period across commuter assets.
        Used to initialise the commuter VFA slope taper.
    max_discharge_per_period:
        Maximum SOC dischargeable in one period.
    battery_capacity_kwh:
        Maximum battery capacity in kWh.
    """
    return {
        AssetType.COMMUTER: AssetValueFunction(
            config=config,
            asset_type=AssetType.COMMUTER,
            periods_per_day=periods_per_day,
            expected_last_plugged_in_period=last_commuter_departure,
            max_discharge_per_period=max_discharge_per_period,
            battery_capacity_kwh=battery_capacity_kwh,
        ),
        AssetType.WFH: AssetValueFunction(
            config=config,
            asset_type=AssetType.WFH,
            periods_per_day=periods_per_day,
            expected_last_plugged_in_period=periods_per_day - 1,
            max_discharge_per_period=max_discharge_per_period,
            battery_capacity_kwh=battery_capacity_kwh,
        ),
    }
