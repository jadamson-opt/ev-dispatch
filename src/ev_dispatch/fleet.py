"""
Fleet aggregator.

Manages a heterogeneous collection of EV assets (commuters and WFH users),
enforces the portfolio capacity buffer, and coordinates dispatch decisions
across assets at each timestep.

Profiles are generated per-asset at construction time, with commuter departure
means sampled from U(28, 32) and std devs from U(1, 2), giving some
heterogeneity across the fleet.
"""

from dataclasses import dataclass

import numpy as np

from ev_dispatch.ev_asset import CommuterProfile, EVAsset, EVConfig, WFHProfile


@dataclass
class FleetConfig:
    """Operational parameters for the fleet aggregator."""

    n_assets: int = 500
    wfh_fraction: float = 0.4
    portfolio_buffer_fraction: float = (
        0.2  # Reserve this fraction of available capacity
    )
    periods_per_day: int = 48
    period_duration_hours: float = 0.5


class Fleet:
    """
    Aggregates EV assets and manages portfolio-level dispatch.

    Commuter and WFH assets are constructed with individually sampled profiles,
    giving somewhat realistic variation in departure times and usage patterns across
    the fleet.

    The portfolio buffer ensures a fraction of available capacity is always
    held in reserve against unplanned departures or grid commitment shortfalls.
    """

    def __init__(
        self,
        config: FleetConfig,
        asset_config: EVConfig,
        seed: int | None = None,
    ):
        self.config = config
        self.ev_config = asset_config
        self.rng = np.random.default_rng(seed)

        # Spawn independent child generators from master seed so all asset
        # behaviour is fully reproducible from a single top-level seed
        child_rngs = self.rng.spawn(config.n_assets)
        self.assets = [
            EVAsset(
                asset_id=f"ev_{i}",
                config=asset_config,
                profile=self._sample_profile(i, config),
                rng=child_rngs[i],
            )
            for i in range(config.n_assets)
        ]

    def reset(self) -> None:
        """Reset all assets to a new sampled episode."""
        for asset in self.assets:
            asset.reset()

    def step(
        self, requested_actions_kw: np.ndarray, price_per_mwh: float, period: int
    ) -> dict:
        """
        Apply dispatch actions across the fleet for one timestep.

        Actions are scaled down proportionally if the portfolio buffer
        would be breached. Returns a summary of fleet outcomes.

        Parameters
        ----------
        requested_actions_kw:
            Array of shape (n_assets,) with charge (+) / discharge (-) in kW.
        price_per_mwh:
            Current electricity price used to compute revenue.
        period:
            Current timestep index.

        Returns
        -------
        dict with keys:
            total_energy_kwh: Net energy dispatched (positive = charged)
            penalised_revenue: £ revenue this period (negative = cost of charging) - penalty
            total_penalty: Deadline penalties incurred
            mean_soc: Fleet mean SOC after actions
            n_available: Number of plugged-in assets
            asset_energies_kwh: Array of shape (n_assets,) with asset charge/discharge in kwh
        """
        plugged_in_mask = np.array([a.is_plugged_in(period) for a in self.assets])
        scaled_actions = self._apply_portfolio_buffer(requested_actions_kw, period)

        total_energy_kwh = 0.0
        total_penalty = 0.0
        asset_energies_kwh = np.zeros(self.config.n_assets)

        for i, asset in enumerate(self.assets):
            _, energy_kwh = asset.apply_action(scaled_actions[i], period)
            asset_energies_kwh[i] = energy_kwh
            total_energy_kwh += energy_kwh
            total_penalty += asset.deadline_penalty(period)

        revenue = self._compute_revenue(total_energy_kwh, price_per_mwh)
        mean_soc = float(np.mean([a.soc for a in self.assets]))

        return {
            "total_energy_kwh": total_energy_kwh,
            "penalised_revenue": revenue - total_penalty,
            "total_penalty": total_penalty,
            "mean_soc": mean_soc,
            "n_plugged_in": int(plugged_in_mask.sum()),
            "asset_energies_kwh": asset_energies_kwh,
        }

    def soc_array(self) -> np.ndarray:
        """Return current SOC for all assets as an array of shape (n_assets,)."""
        return np.array([a.soc for a in self.assets])

    def available_discharge_capacity_kw(self, period: int) -> float:
        """Total dispatchable discharge capacity across available assets (kW)."""
        return sum(
            a.soc * a.config.battery_capacity_kwh / a.config.period_duration_hours
            for a in self.assets
            if a.is_plugged_in(period) and a.soc > 0
        )

    def _sample_profile(
        self, asset_index: int, config: FleetConfig
    ) -> CommuterProfile | WFHProfile:
        """
        Sample an individual user profile for one asset.

        WFH fraction determines asset type. Commuter departure means are
        sampled from U(28, 32) and std devs from U(1, 2) to give realistic
        heterogeneity in departure patterns across the fleet.
        """
        is_wfh = asset_index < int(config.n_assets * config.wfh_fraction)

        if is_wfh:
            return WFHProfile()

        return CommuterProfile(
            departure_period_mean=float(self.rng.triangular(28, 30, 32)),
            departure_period_std=1.0,
        )

    def _apply_portfolio_buffer(
        self, requested_actions_kw: np.ndarray, period: int
    ) -> np.ndarray:
        """
        Scale discharge actions down to respect the portfolio buffer.

        Net discharge is computed as total requested discharge minus total
        requested charge. Charging actions increase future fleet capacity,
        so they reduce the effective draw on the portfolio. This netting is
        valid when the buffer protects near-future capacity rather than
        instant physical cover.

        Only discharge actions are scaled down if the net exceeds the
        buffered limit — charging actions are left unchanged.
        """
        total_available_discharge = self.available_discharge_capacity_kw(period)
        buffer_kw = self.config.portfolio_buffer_fraction * total_available_discharge
        max_net_discharge_kw = total_available_discharge - buffer_kw

        total_discharge_kw = -np.sum(np.minimum(requested_actions_kw, 0))
        net_discharge_kw = -np.sum(requested_actions_kw)

        if net_discharge_kw > max_net_discharge_kw:
            scale = max(
                0.0, max_net_discharge_kw + (total_discharge_kw - net_discharge_kw)
            ) / (total_discharge_kw + 1e-9)
            scaled = requested_actions_kw.copy()
            scaled[requested_actions_kw < 0] *= scale
            return scaled

        return requested_actions_kw.copy()

    @staticmethod
    def _compute_revenue(total_energy_kwh: float, price_per_mwh: float) -> float:
        """
        Revenue from energy dispatch.

        Discharging to grid earns revenue; charging costs money.
        Energy is in kWh, price in £/MWh, so divide by 1000.
        """
        return -total_energy_kwh * price_per_mwh / 1000.0
