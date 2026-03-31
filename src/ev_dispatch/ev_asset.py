"""
Individual EV asset model.

Handles SOC dynamics, availability windows, departure deadlines,
SOC floor (buffer against unplanned trips), and deadline penalties.

Availability is modelled as a list of windows rather than a single
plugin/departure pair, allowing for mid-day unplanned trips where
the EV leaves, returns, and plugs back in before the main departure.

SoC = State of charge (remaining energy in a battery as a fraction of total capacity)
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class PluginWindow:
    """A continuous period during which the EV is plugged in."""

    start_period: int
    end_period: int  # Asset departs at this period (exclusive)
    required_soc: float  # SOC required by end_period


@dataclass
class UserProfile:
    """
    Statistical description of a user's charging behaviour.
    Here we treat this as input. In practice, it would be learned /predicted
    from real user data.

    All times are period indices (0–47 for a half-hourly day).
    """

    plugin_period_mean: float = 4.0  # ~6pm if episode starts at 4pm
    plugin_period_std: float = 4.0
    departure_period_mean: float = 30.0  # ~7am next day
    departure_period_std: float = 2.0
    required_soc: float = 0.8
    return_soc_mean: float = 0.3  # SOC when they plug in
    return_soc_std: float = 0.1
    unplanned_departure_probability: float = 0.05  # Captures unexpected usage
    unplanned_trip_duration_periods: int = 4  # ~2 hours away
    unplanned_required_soc: float = 0.15  # Minimum needed for the unplanned trip


@dataclass
class EVConfig:
    """Physical and operational parameters for an EV."""

    battery_capacity_kwh: float = 60.0
    max_charge_rate_kw: float = 7.4  # Standard home charger
    max_discharge_rate_kw: float = 7.4
    charge_efficiency: float = 0.92
    discharge_efficiency: float = 0.92
    soc_floor: float = 0.2  # Minimum dispatchable SOC
    period_duration_hours: float = 0.5  # Half-hourly


class EVAsset:
    """
    Simulates a single EV over one day episode.

    At episode start, plugin windows are sampled from the user profile.
    An unplanned trip inserts a mid-day gap in availability. The asset
    departs early, returns after a fixed duration, then continues charging
    until the main planned departure.

    The asset tracks SOC and enforces all feasibility constraints at each timestep.
    """

    def __init__(
        self,
        asset_id: str,
        config: EVConfig,
        profile: UserProfile,
        rng: np.random.Generator,
    ):
        self.asset_id = asset_id
        self.config = config
        self.profile = profile
        self._rng = rng

        self._starting_soc: float = 0.0
        self.soc: float = 0.0
        self.plugin_windows: list[PluginWindow] = []

    def reset(self) -> None:
        """
        Sample a new episode from the user profile.

        Constructs one or two plugin windows depending on whether
        an unplanned departure occurs today.
        """

        self._starting_soc = float(
            np.clip(
                self._rng.normal(
                    self.profile.return_soc_mean, self.profile.return_soc_std
                ),
                0.0,
                1.0,
            )
        )
        self.soc = self._starting_soc

        plugin_period = int(
            np.clip(
                self._rng.normal(
                    self.profile.plugin_period_mean, self.profile.plugin_period_std
                ),
                0,
                47,
            )
        )

        planned_departure_period = int(
            np.clip(
                self._rng.normal(
                    self.profile.departure_period_mean,
                    self.profile.departure_period_std,
                ),
                0,
                47,
            )
        )

        unplanned_rand_sample = self._rng.random()
        has_unplanned_departure = (
            unplanned_rand_sample < self.profile.unplanned_departure_probability
        )

        if (
            has_unplanned_departure
            and plugin_period
            < planned_departure_period
            - self.profile.unplanned_trip_duration_periods
            - 1
        ):
            unplanned_departure = int(plugin_period + self._rng.integers(1, 4))  # leaves within 2 hours of getting home
            return_period = (
                unplanned_departure + self.profile.unplanned_trip_duration_periods
            )

            self.plugin_windows = [
                PluginWindow(
                    start_period=plugin_period,
                    end_period=unplanned_departure,
                    required_soc=self.profile.unplanned_required_soc,
                ),
                PluginWindow(
                    start_period=return_period,
                    end_period=planned_departure_period,
                    required_soc=self.profile.required_soc,
                ),
            ]
        else:
            self.plugin_windows = [
                PluginWindow(
                    start_period=plugin_period,
                    end_period=planned_departure_period,
                    required_soc=self.profile.required_soc,
                )
            ]

    def is_plugged_in(self, period: int) -> bool:
        """Return True if the EV is plugged in during this period."""
        return any(w.start_period <= period < w.end_period for w in self.plugin_windows)

    def apply_action(self, action_kw: float, period: int) -> tuple[float, float]:
        """
        Apply a charge (positive) or discharge (negative) action.

        The action is clipped to feasible bounds before application.

        Parameters
        ----------
        action_kw:
            Requested charge/discharge rate in kW.
        period:
            Current timestep (used to check availability).

        Returns
        -------
        tuple[float, float]
            (actual_action_kw, energy_transferred_kwh)
            Energy is positive for charging, negative for discharging.
        """
        if not self.is_plugged_in(period):
            return 0.0, 0.0

        feasible_action = self._clip_to_feasible(action_kw)
        energy_kwh = self._update_soc(feasible_action)
        return feasible_action, energy_kwh

    def deadline_penalty(self, period: int, penalty_per_kwh: float = 500.0) -> float:
        """
        Return a penalty if a departure deadline is missed at this period.

        Checks all windows whose end_period matches the current period.

        Parameters
        ----------
        period:
            Current timestep.
        penalty_per_kwh:
            Cost per kWh of shortfall below required SOC at departure.
        """
        total_penalty = 0.0
        for window in self.plugin_windows:
            if window.end_period == period and self.soc < window.required_soc:
                shortfall_kwh = (
                    window.required_soc - self.soc
                ) * self.config.battery_capacity_kwh
                total_penalty += penalty_per_kwh * shortfall_kwh
        return total_penalty

    @property
    def max_charge_per_period(self) -> float:
        """Maximum SOC increase achievable in one period at full charge rate."""
        cfg = self.config
        return (
            cfg.max_charge_rate_kw
            * cfg.period_duration_hours
            * cfg.charge_efficiency
            / cfg.battery_capacity_kwh
        )

    @property
    def max_discharge_per_period(self) -> float:
        """Maximum SOC decrease from one period of full discharge."""
        cfg = self.config
        return (
            cfg.max_discharge_rate_kw
            * cfg.period_duration_hours
            / cfg.discharge_efficiency
            / cfg.battery_capacity_kwh
        )

    def _clip_to_feasible(self, action_kw: float) -> float:
        """Clip action to charge/discharge rate limits and SOC bounds."""
        cfg = self.config
        period_hours = cfg.period_duration_hours

        if action_kw > 0:  # Charging
            max_charge_kwh = (
                (1.0 - self.soc) * cfg.battery_capacity_kwh / cfg.charge_efficiency
            )
            max_action = min(cfg.max_charge_rate_kw, max_charge_kwh / period_hours)
            return min(action_kw, max_action)
        elif action_kw < 0:  # Discharging
            # SOC floor limits how much can be discharged
            dispatchable_soc = max(self.soc - cfg.soc_floor, 0.0)
            max_discharge_kwh = (
                dispatchable_soc * cfg.battery_capacity_kwh * cfg.discharge_efficiency
            )
            max_action = min(
                cfg.max_discharge_rate_kw, max_discharge_kwh / period_hours
            )
            return -min(-action_kw, max_action)
        return 0.0

    def _update_soc(self, action_kw: float) -> float:
        """
        Apply feasible action to SOC, return energy transferred in kWh.
        action_kw represents energy going to/from the grid.
        """
        cfg = self.config
        period_hours = cfg.period_duration_hours

        if action_kw > 0:
            energy_kwh = action_kw * period_hours * cfg.charge_efficiency
        else:
            energy_kwh = action_kw * period_hours / cfg.discharge_efficiency

        self.soc = float(
            np.clip(self.soc + energy_kwh / cfg.battery_capacity_kwh, 0.0, 1.0)
        )
        return energy_kwh
