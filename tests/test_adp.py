import numpy as np

from ev_dispatch.adp import ADPPolicy
from ev_dispatch.ev_asset import EVConfig
from ev_dispatch.fleet import Fleet, FleetConfig
from ev_dispatch.value_function import VFAConfig, make_vfa_registry


def make_policy(
    shadow_price_mwh: float = 80.0, forcing_buffer: float = 0.0
) -> tuple[ADPPolicy, Fleet]:
    """
    Construct a policy and fleet with known shadow prices.

    Shadow prices are set by initialising a VFA with uniform slopes that
    correspond to the given £/MWh equivalent shadow price.
    """
    fleet = Fleet(FleetConfig(n_assets=5, wfh_fraction=0.0), EVConfig(), seed=42)
    fleet.reset()

    max_discharge = fleet.assets[0].max_discharge_per_period
    battery_kwh = fleet.ev_config.battery_capacity_kwh
    registry = make_vfa_registry(
        VFAConfig(mean_price=shadow_price_mwh),
        periods_per_day=48,
        last_commuter_departure=32,
        max_discharge_per_period=max_discharge,
        battery_capacity_kwh=battery_kwh,
    )

    policy = ADPPolicy(
        registry, epsilon=0.0, forcing_buffer=forcing_buffer, soc_floor=0.0
    )
    return policy, fleet


def plugged_in_period(fleet: Fleet) -> int:
    for period in range(20, 30):  # overnight window, all commuters should be available
        if all(a.is_plugged_in(period) for a in fleet.assets):
            return period
    raise ValueError("No period found where all assets are plugged in")


class TestSelectActions:
    def test_charges_when_price_below_shadow_price(self):
        policy, fleet = make_policy(shadow_price_mwh=100.0)
        period = plugged_in_period(fleet)
        prices = np.full(48, 50.0)  # price well below shadow price
        for asset in fleet.assets:
            asset.soc = 0.5
        actions = policy.select_actions(fleet, prices, period)
        assert all(a > 0 for a in actions if a != 0.0)

    def test_discharges_when_price_above_shadow_price(self):
        policy, fleet = make_policy(shadow_price_mwh=50.0)
        period = plugged_in_period(fleet)
        prices = np.full(48, 150.0)  # price well above shadow price
        for asset in fleet.assets:
            asset.soc = 0.5
        actions = policy.select_actions(fleet, prices, period)
        assert all(a <= 0 for a in actions)

    def test_unavailable_asset_gets_zero_action(self):
        policy, fleet = make_policy()
        # Find a period before any asset plugs in
        period = 0
        prices = np.full(48, 150.0)
        actions = policy.select_actions(fleet, prices, period)
        for asset, action in zip(fleet.assets, actions):
            if not asset.is_plugged_in(period):
                assert action == 0.0

    def test_forcing_overrides_shadow_price(self):
        policy, fleet = make_policy(shadow_price_mwh=50.0, forcing_buffer=3.0)
        prices = np.full(48, 150.0)  # high price would normally trigger discharge
        for asset in fleet.assets:
            asset.soc = 0.1  # very low SOC — forcing should kick in near departure
        expected_departure = int(fleet.assets[0].profile.departure_period_mean)
        period = expected_departure - 2  # close to deadline
        actions = policy.select_actions(fleet, prices, period)
        plugged_in = [a.is_plugged_in(period) for a in fleet.assets]
        for action, available in zip(actions, plugged_in):
            if available:
                assert action >= 0  # should charge or idle, not discharge
