import numpy as np

from ev_dispatch.ev_asset import EVConfig
from ev_dispatch.fleet import Fleet, FleetConfig


def make_fleet(n_assets: int = 10, seed: int = 42) -> Fleet:
    fleet = Fleet(FleetConfig(n_assets=n_assets), EVConfig(), seed=seed)
    fleet.reset()
    return fleet


def plugged_in_period(fleet: Fleet) -> int:
    """Return a period where at least one asset is plugged in."""
    for period in range(fleet.config.periods_per_day):
        if any(a.is_plugged_in(period) for a in fleet.assets):
            return period
    raise ValueError("No assets plugged in during any period")


class TestStep:
    def test_charging_returns_negative_revenue(self):
        fleet = make_fleet()
        period = plugged_in_period(fleet)
        actions = np.array([a.config.max_charge_rate_kw for a in fleet.assets])
        result = fleet.step(actions, price_per_mwh=100.0, period=period)
        assert result["penalised_revenue"] < 0

    def test_discharging_returns_positive_revenue(self):
        fleet = make_fleet()
        period = plugged_in_period(fleet)
        for asset in fleet.assets:
            asset.soc = 0.8
        actions = np.array([-a.config.max_discharge_rate_kw for a in fleet.assets])
        result = fleet.step(actions, price_per_mwh=100.0, period=period)
        assert result["penalised_revenue"] + result["total_penalty"] > 0

    def test_portfolio_buffer_limits_discharge(self):
        fleet = make_fleet()
        period = plugged_in_period(fleet)
        for asset in fleet.assets:
            asset.soc = 0.5
        actions = np.array([-a.config.max_discharge_rate_kw for a in fleet.assets])
        result = fleet.step(actions, price_per_mwh=100.0, period=period)
        max_discharge = fleet.available_discharge_capacity_kw(period)
        actual_discharge = (
            -result["total_energy_kwh"] / fleet.config.period_duration_hours
        )
        assert (
            actual_discharge
            <= max_discharge * (1 - fleet.config.portfolio_buffer_fraction) + 1e-6
        )
