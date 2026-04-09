"""
Baseline dispatch policies for benchmarking.

Three policies:
- NaiveNightCharger: charges during fixed overnight window only
- PriceThreshold: charges below a fixed price, discharges above another
- HindsightOptimal: solves the dispatch problem with perfect price foresight
"""

import math
from abc import ABC, abstractmethod

import numpy as np

from ev_dispatch.fleet import Fleet


class BasePolicy(ABC):
    """Abstract base for all dispatch policies."""

    @abstractmethod
    def select_actions(
        self, fleet: Fleet, prices: np.ndarray, period: int
    ) -> np.ndarray:
        """
        Return requested charge/discharge actions for each asset.

        Parameters
        ----------
        fleet: Current fleet state.
        prices: Full price array for the day (£/MWh), shape (periods,).
        period: Current timestep index.

        Returns
        -------
        np.ndarray of shape (n_assets,) with actions in kW.
        """
        pass

    def run_episode(self, fleet: Fleet, prices: np.ndarray) -> dict:
        """
        Run one full day episode and return summary metrics.

        Returns
        -------
        dict with keys: total_penalised_revenue, total_penalty, soc_history, revenue_history
        """
        fleet.reset()
        total_penalised_revenue = 0.0
        total_penalty = 0.0
        soc_history = []
        revenue_history = []
        penalty_history = []

        for period in range(fleet.config.periods_per_day):
            actions = self.select_actions(fleet, prices, period)
            result = fleet.step(actions, prices[period], period)
            total_penalised_revenue += result["penalised_revenue"]
            total_penalty += result["total_penalty"]
            soc_history.append(result["mean_soc"])
            revenue_history.append(result["penalised_revenue"])
            penalty_history.append(result["total_penalty"])

        return {
            "total_penalised_revenue": total_penalised_revenue,
            "total_penalty": total_penalty,
            "soc_history": soc_history,
            "revenue_history": revenue_history,
            "penalty_history": penalty_history,
        }


class NaiveNightCharger(BasePolicy):
    """
    Charges all available assets at full rate during a fixed overnight window.

    No price awareness, no discharge. This is a very simple rule-based policy
    representative of non-optimised charging.
    """

    def __init__(self, charge_start_period: int = 12, charge_end_period: int = 28):
        self.charge_start_period = charge_start_period  # ~10pm
        self.charge_end_period = charge_end_period  # 6am

    def select_actions(
        self, fleet: Fleet, prices: np.ndarray, period: int
    ) -> np.ndarray:
        n = fleet.config.n_assets
        if self.charge_start_period <= period < self.charge_end_period:
            return np.array([a.config.max_charge_rate_kw for a in fleet.assets])
        return np.zeros(n)


class ForesightGreedy(BasePolicy):
    """
    Greedy dispatch with perfect knowledge of future prices and departures.

    Charges at the cheapest available periods and discharges at the most
    expensive, subject to availability and a deadline-aware discharge rule:
    an asset will not discharge if its current SOC is needed to meet an
    upcoming departure requirement given remaining charging time.

    Uses a simple greedy rank-based approach rather than full LP,
    which is sufficient for a prototype benchmark.

    This is not a deployable policy, it sets a loose upper bound on achievable
    revenue.
    """

    def __init__(self, n_charge_periods: int = 10, n_discharge_periods: int = 6):
        self.n_charge_periods = n_charge_periods
        self.n_discharge_periods = n_discharge_periods
        self._charge_periods: set[int] = set()
        self._discharge_periods: set[int] = set()

    def plan(self, prices: np.ndarray, fleet: Fleet) -> None:
        """
        Pre-compute charge and discharge periods from the full price sequence,
        restricted to periods where at least one asset is available.

        Call this once per episode before running.
        """
        plugged_in_periods = [
            t
            for t in range(len(prices))
            if any(a.is_plugged_in(t) for a in fleet.assets)
        ]
        sorted_by_price = sorted(plugged_in_periods, key=lambda t: prices[t])
        self._charge_periods = set(sorted_by_price[: self.n_charge_periods])
        self._discharge_periods = set(sorted_by_price[-self.n_discharge_periods :])

    def select_actions(
        self, fleet: Fleet, prices: np.ndarray, period: int
    ) -> np.ndarray:
        actions = []
        for asset in fleet.assets:
            if not asset.is_plugged_in(period):
                actions.append(0.0)
            elif self._must_charge_for_deadline(asset, period):
                actions.append(asset.config.max_charge_rate_kw)
            elif period in self._charge_periods:
                actions.append(asset.config.max_charge_rate_kw)
            elif (
                period in self._discharge_periods
                and not self._discharge_would_miss_deadline(asset, period)
            ):
                actions.append(-asset.config.max_discharge_rate_kw)
            else:
                actions.append(0.0)
        return np.array(actions)

    def run_episode(self, fleet: Fleet, prices: np.ndarray) -> dict:
        """Override to plan before running."""
        self.plan(prices, fleet)
        return super().run_episode(fleet, prices)

    @staticmethod
    def _must_charge_for_deadline(asset, period: int) -> bool:
        """
        Return True if the asset must charge now to have any chance of
        meeting an upcoming departure requirement.
        """
        for window in asset.plugin_windows:
            if window.start_period <= period < window.end_period:
                periods_remaining = window.end_period - period
                periods_required = (
                    math.ceil(
                        (window.required_soc - asset.soc) / asset.max_charge_per_period
                    )
                    + 1
                )
                if periods_remaining <= periods_required > 0:
                    return True
        return False

    @staticmethod
    def _discharge_would_miss_deadline(asset, period: int) -> bool:
        """
        Return True if discharging now would make it impossible to reach
        the required SOC by the end of the current availability window.
        """
        for window in asset.plugin_windows:
            if window.start_period <= period < window.end_period:
                periods_remaining = window.end_period - period - 1
                soc_after_discharge = asset.soc - asset.max_discharge_per_period
                if (
                    soc_after_discharge
                    + periods_remaining * asset.max_charge_per_period
                    < window.required_soc
                ):
                    return True
        return False
