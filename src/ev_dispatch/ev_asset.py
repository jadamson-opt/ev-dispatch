"""
Individual EV asset model.

Supports two asset types:
- Commuter: plugs in evening, departs morning, single overnight availability window
- WFH: available most of the day with occasional short random trips

Availability is modelled as a list of PluginWindows rather than a single
plugin/departure pair, allowing for gaps in availability from trips.

SoC = State of charge (remaining energy in a battery as a fraction of total capacity)
"""

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np


class AssetType(StrEnum):
    COMMUTER = "commuter"
    WFH = "wfh"


@dataclass
class PluginWindow:
    """A continuous period during which the EV is plugged in."""

    start_period: int
    end_period: int  # Asset departs at this period (exclusive)
    required_soc: float  # SOC required by end_period


@dataclass
class UserProfile:
    """
    Base profile with fields common to all asset types to describe a user's
    charging behaviour.

    Here we treat this as input. In practice, it would be learned /predicted
    from real user data.

    All times are period indices (0–47 for a half-hourly day, starting 4pm).
    """

    asset_type: AssetType = field(init=False)
    return_soc_mean: float = 0.5
    return_soc_std: float = 0.1
    unplanned_departure_probability: float = 0.05
    unplanned_trip_duration_periods: int = 4
    unplanned_required_soc: float = 0.15


@dataclass
class CommuterProfile(UserProfile):
    """Profile for a commuter asset with a planned overnight departure."""

    asset_type: AssetType = field(default=AssetType.COMMUTER, init=False)
    plugin_period_mean: float = 4.0  # ~6pm if episode starts at 4pm
    plugin_period_std: float = 1.0
    departure_period_mean: float = 30.0  # ~7am next day
    departure_period_std: float = 1.0
    required_soc: float = 0.8


@dataclass
class WFHProfile(UserProfile):
    """Profile for a WFH asset with no planned departure."""

    asset_type: AssetType = field(default=AssetType.WFH, init=False)
    max_trips: int = 2
    trip_soc_consumption_mean: float = 0.1
    trip_soc_consumption_std: float = 0.03

    def sample_trip_soc_consumption(self, rng: np.random.Generator) -> float:
        """Sample SOC consumed by a single short trip."""
        return float(
            np.clip(
                rng.normal(
                    self.trip_soc_consumption_mean, self.trip_soc_consumption_std
                ),
                0.0,
                1.0,
            )
        )


@dataclass
class EVConfig:
    """Physical and operational parameters for an EV."""

    battery_capacity_kwh: float = 60.0
    max_charge_rate_kw: float = 7.4  # Standard home charger
    max_discharge_rate_kw: float = 7.4
    charge_efficiency: float = 0.92
    discharge_efficiency: float = 0.92
    period_duration_hours: float = 0.5  # Half-hourly

    @property
    def max_charge_per_period(self) -> float:
        """Maximum SOC increase achievable in one period at full charge rate."""
        return (
            self.max_charge_rate_kw
            * self.period_duration_hours
            * self.charge_efficiency
            / self.battery_capacity_kwh
        )

    @property
    def max_discharge_per_period(self) -> float:
        """Maximum SOC decrease from one period of full discharge."""
        return (
            self.max_discharge_rate_kw
            * self.period_duration_hours
            / self.discharge_efficiency
            / self.battery_capacity_kwh
        )


class EVAsset:
    """
    Simulates a single EV over one day episode.

    At episode start, availability windows are sampled from the user profile.
    Commuters have a single overnight window with a possible unplanned evening
    trip. WFH users are available most of the day with occasional short trips.

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
        Sample a new episode from the user profile and constructs PluginWindows.
        """

        self._starting_soc = float(
            min(
                max(
                    self._rng.normal(
                        self.profile.return_soc_mean, self.profile.return_soc_std
                    ),
                    0.0,
                ),
                1.0,
            )
        )
        self.soc = self._starting_soc

        if isinstance(self.profile, CommuterProfile):
            self.plugin_windows = self._sample_commuter_windows(self.profile)
        elif isinstance(self.profile, WFHProfile):
            self.plugin_windows = self._sample_wfh_windows(self.profile)
        else:
            raise NotImplementedError(
                f"Unsupported user profile: {self.profile.asset_type}"
            )

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
        return self.config.max_charge_per_period

    @property
    def max_discharge_per_period(self) -> float:
        """Maximum SOC decrease from one period of full discharge."""
        return self.config.max_discharge_per_period

    def _sample_commuter_windows(self, profile: CommuterProfile) -> list[PluginWindow]:
        """
        Sample plugin windows for a commuter asset.

        Produces one overnight window, with a possible short unplanned
        evening trip creating a gap near the start of the window.
        """
        plugin_period = round(
            np.clip(
                self._rng.normal(profile.plugin_period_mean, profile.plugin_period_std),
                0,
                47,
            )
        )

        planned_departure_period = round(
            np.clip(
                self._rng.normal(
                    profile.departure_period_mean,
                    profile.departure_period_std,
                ),
                0,
                47,
            )
        )

        unplanned_rand_sample = self._rng.random()
        has_unplanned_departure = (
            unplanned_rand_sample < self.profile.unplanned_departure_probability
            and plugin_period
            < planned_departure_period - profile.unplanned_trip_duration_periods - 1
        )

        if has_unplanned_departure:
            unplanned_departure = round(
                plugin_period + self._rng.integers(1, 4)
            )  # leaves within 2 hours of getting home
            return_period = (
                unplanned_departure + profile.unplanned_trip_duration_periods
            )

            return [
                PluginWindow(
                    start_period=plugin_period,
                    end_period=unplanned_departure,
                    required_soc=profile.unplanned_required_soc,
                ),
                PluginWindow(
                    start_period=return_period,
                    end_period=planned_departure_period,
                    required_soc=profile.required_soc,
                ),
            ]

        return [
            PluginWindow(
                start_period=plugin_period,
                end_period=planned_departure_period,
                required_soc=profile.required_soc,
            )
        ]

    def _sample_wfh_windows(self, profile: WFHProfile) -> list[PluginWindow]:
        """
        Sample plugin windows for a WFH asset.

        WFH users are plugged in for most of the day with occasional short
        trips during daytime hours (periods 30–46, roughly 7am to 3pm).
        Each trip creates a gap in availability.
        """
        n_trips = int(self._rng.integers(0, profile.max_trips + 1))

        windows = []
        current_start = 0
        current_period = 0

        trip_periods = (
            sorted(self._rng.integers(30, 46, size=n_trips).tolist())
            if n_trips > 0
            else []
        )

        for trip_start in trip_periods:
            if trip_start <= current_period:
                continue
            trip_end = min(trip_start + profile.unplanned_trip_duration_periods, 47)
            trip_soc = profile.sample_trip_soc_consumption(self._rng)
            windows.append(
                PluginWindow(
                    start_period=current_start,
                    end_period=trip_start,
                    required_soc=trip_soc,
                )
            )
            current_start = trip_end
            current_period = trip_end

        if current_start < 47:
            windows.append(
                PluginWindow(
                    start_period=current_start,
                    end_period=47,
                    required_soc=0.0,
                )
            )

        return windows

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
            max_discharge_kwh = (
                self.soc * cfg.battery_capacity_kwh * cfg.discharge_efficiency
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
            min(max(self.soc + energy_kwh / cfg.battery_capacity_kwh, 0.0), 1.0)
        )
        return energy_kwh
