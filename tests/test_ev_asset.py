import numpy as np

from ev_dispatch.ev_asset import CommuterProfile, EVAsset, EVConfig, WFHProfile


def make_commuter(soc: float = 0.5) -> EVAsset:
    asset = EVAsset("test", EVConfig(), CommuterProfile(), np.random.default_rng(42))
    asset.reset()
    asset.soc = soc
    return asset


def make_wfh(soc: float = 0.5) -> EVAsset:
    asset = EVAsset("test", EVConfig(), WFHProfile(), np.random.default_rng(42))
    asset.reset()
    asset.soc = soc
    return asset


class TestReset:
    def test_commuter_has_at_least_one_window(self):
        asset = make_commuter()
        assert len(asset.plugin_windows) >= 1

    def test_commuter_window_end_after_start(self):
        asset = make_commuter()
        for window in asset.plugin_windows:
            assert window.end_period > window.start_period

    def test_wfh_has_at_least_one_window(self):
        asset = make_wfh()
        assert len(asset.plugin_windows) >= 1


class TestApplyAction:
    def test_charge_increases_soc(self):
        asset = make_commuter(soc=0.5)
        period = asset.plugin_windows[0].start_period
        _, energy = asset.apply_action(asset.config.max_charge_rate_kw, period)
        assert energy > 0
        assert asset.soc > 0.5

    def test_discharge_decreases_soc(self):
        asset = make_commuter(soc=0.5)
        period = asset.plugin_windows[0].start_period
        _, energy = asset.apply_action(-asset.config.max_discharge_rate_kw, period)
        assert energy < 0
        assert asset.soc < 0.5

    def test_unavailable_asset_returns_zero(self):
        asset = make_commuter(soc=0.5)
        unavailable_period = asset.plugin_windows[-1].end_period + 1
        action, energy = asset.apply_action(
            asset.config.max_charge_rate_kw, unavailable_period
        )
        assert action == 0.0
        assert energy == 0.0
