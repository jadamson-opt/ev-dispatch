"""
ADP training loop and learned dispatch policy.

The aggregator trains a shared AssetValueFunction over many simulated episodes.
At each timestep, each asset receives a shadow price derived from the VFA slope
at its current SOC. Assets charge if the current electricity price is below their
shadow price, and discharge if above, subject to availability, SOC floor, and
expected deadline constraints.

The VFA learns to encode departure patterns through the deadline penalty signal:
shadow prices rise as an asset approaches its expected departure with insufficient
SOC, naturally overriding the arbitrage signal and forcing charging.
"""

import math

import numpy as np
from dataclasses import dataclass

from ev_dispatch import FloatArray
from ev_dispatch.fleet import Fleet
from ev_dispatch.price_process import PriceProcess
from ev_dispatch.value_function import AssetValueFunction, VFAConfig
from ev_dispatch.baseline import BasePolicy


@dataclass
class ADPConfig:
    """Parameters controlling the ADP training procedure."""

    n_training_episodes: int = 2000
    # epsilon: float = 0.1  # Exploration rate for epsilon-greedy action selection
    epsilon: float = 0.02  # Exploration rate for epsilon-greedy action selection
    epsilon_decay: float = 0.995  # Multiplicative decay applied each episode
    initial_forcing_buffer: float = (
        2.0  # Initial forcing buffer in units of departure std dev
    )
    forcing_buffer_decay: float = (
        0.995  # Multiplicative decay of forcing buffer each episode
    )
    rolling_window: int = 50


class ADPPolicy(BasePolicy):
    """
    Dispatch policy that selects actions greedily against a trained VFA.

    Each asset compares the current electricity price to its shadow price.
    The shadow price is the marginal value of stored energy from the VFA,
    adjusted upward when the asset is behind on its charging deadline.

    This policy can be used both during training (with exploration) and
    for evaluation (epsilon=0).
    """

    def __init__(
        self, vfa: AssetValueFunction, epsilon: float = 0.0, forcing_buffer: float = 0
    ):
        self.vfa = vfa
        self.epsilon = epsilon
        self.forcing_buffer = forcing_buffer

    def select_actions(
        self, fleet: Fleet, prices: FloatArray, period: int
    ) -> FloatArray:
        """
        Select charge/discharge actions for each asset based on shadow prices.

        Deadline awareness uses the user's _expected_ departure time from their
        profile rather than the actual sampled window (the policy operates
        under uncertainty just as a real deployed system would). The VFA learns
        to hedge against residual uncertainty through the penalty signal.

        No SOC floor is enforced here. The VFA shadow prices
        should learn when discharge below the naive floor is genuinely worth it.
        """
        current_price = prices[period]
        actions = []

        for asset in fleet.assets:
            if not asset.is_plugged_in(period):
                actions.append(0.0)
                continue

            if self._must_charge_for_deadline(asset, period, self.forcing_buffer):
                actions.append(asset.config.max_charge_rate_kw)
                continue

            discharge_would_miss_deadline = self._discharge_would_miss_deadline(
                asset, period, self.forcing_buffer
            )
            if self.epsilon > 0 and np.random.random() < self.epsilon:
                if discharge_would_miss_deadline:
                    action_choices = [asset.config.max_charge_rate_kw, 0.0]
                else:
                    action_choices = [
                        0.0,
                        -asset.config.max_discharge_rate_kw,
                        asset.config.max_charge_rate_kw,
                    ]
                actions.append(float(np.random.choice(action_choices)))
                continue

            shadow_price = self.vfa.shadow_price(asset.soc, period)

            if current_price < shadow_price:
                actions.append(asset.config.max_charge_rate_kw)
            elif current_price > shadow_price and not discharge_would_miss_deadline:
                actions.append(-asset.config.max_discharge_rate_kw)
            else:
                actions.append(0.0)

        return np.array(actions)

    @staticmethod
    def _must_charge_for_deadline(asset, period: int, forcing_buffer: float) -> bool:
        """
        Return True if the asset must charge every remaining period to meet
        its expected departure requirement.

        Uses profile-level expectations rather than actual sampled windows,
        reflecting that a deployed policy cannot observe tonight's true departure.

        forcing_buffer is a multiplier on departure_period_std, adding extra
        period conservatism that scales with departure time uncertainty.
        """
        expected_departure = int(asset.profile.departure_period_mean)
        expected_required_soc = asset.profile.required_soc
        if asset.soc >= expected_required_soc:
            return False

        buffer_periods = int(forcing_buffer * asset.profile.departure_period_std)
        periods_remaining = expected_departure - period
        periods_required = (
            math.ceil((expected_required_soc - asset.soc) / asset.max_charge_per_period)
            + 1
            + buffer_periods
        )
        return periods_remaining <= periods_required > 0

    @staticmethod
    def _discharge_would_miss_deadline(
        asset, period: int, forcing_buffer: float
    ) -> bool:
        """
        Return True if discharging now would make the expected departure
        requirement unreachable.

        Applies the same std-dev-scaled buffer as _must_charge_for_deadline
        so the two rules are consistent in their conservatism. The buffer is
        expressed as additional required SOC in the reachability check.

        Uses profile-level expectations for the same reason as above.
        """
        expected_departure = int(asset.profile.departure_period_mean)
        expected_required_soc = asset.profile.required_soc
        buffer_periods = int(forcing_buffer * asset.profile.departure_period_std)

        buffered_periods_remaining = max(
            0, expected_departure - period - 1 - buffer_periods
        )
        soc_after_discharge = asset.soc - asset.max_discharge_per_period
        max_reachable_soc = (
            soc_after_discharge
            + buffered_periods_remaining * asset.max_charge_per_period
        )
        return max_reachable_soc < expected_required_soc


class ADPTrainer:
    """
    Trains an AssetValueFunction via forward ADP simulation.

    Each episode simulates a full day, stepping the fleet through 48 periods.
    After each transition, the VFA is updated using a temporal difference (TD)
    target composed of the immediate reward and the discounted next-period value estimate.

    The shared VFA represents a fleet-average asset. Updating from all assets
    at each timestep gives many training samples per episode, accelerating
    convergence.

    An epsilon-greedy exploration strategy is used. Early in training, more
    random exploration is encouraged. As time progress, exploration reduces
    in favour of more exploitation.
    """

    def __init__(
        self,
        fleet: Fleet,
        price_process: PriceProcess,
        vfa_config: VFAConfig,
        adp_config: ADPConfig,
    ):
        self.fleet = fleet
        self.price_process = price_process
        self.adp_config = adp_config

        # Estimate the last period where any asset is likely available.
        # Slopes beyond this are zeroed to prevent TD bootstrap from using
        # stale initialised values in unvisited periods.
        profile = fleet.assets[0].profile
        last_expected_available_period = min(
            int(profile.departure_period_mean), fleet.config.periods_per_day - 1
        )
        self.vfa = AssetValueFunction(
            vfa_config,
            fleet.config.periods_per_day,
            last_expected_available_period,
            fleet.assets[0].max_discharge_per_period,
        )
        self.episode_revenues: list[float] = []
        self.episode_arbitrage_revenues: list[float] = []

    def train(self) -> AssetValueFunction:
        """
        Run the full ADP training loop.

        Returns the trained AssetValueFunction.
        """
        cfg = self.adp_config
        epsilon = cfg.epsilon
        forcing_buffer = cfg.initial_forcing_buffer

        for episode in range(cfg.n_training_episodes):
            prices = self.price_process.sample_scenario()
            policy = ADPPolicy(self.vfa, epsilon=epsilon, forcing_buffer=forcing_buffer)
            episode_revenue, episode_arbitrage = self._run_training_episode(
                prices, policy
            )

            self.episode_revenues.append(episode_revenue)
            self.episode_arbitrage_revenues.append(episode_arbitrage)
            epsilon *= cfg.epsilon_decay

            if (episode + 1) % 100 == 0:
                mean_revenue = np.mean(self.episode_revenues[-100:])
                print(
                    f"Episode {episode + 1}: "
                    f"mean revenue (last 100) = £{mean_revenue:.2f}  "
                )

        return self.vfa

    def _run_training_episode(
        self, prices: FloatArray, policy: ADPPolicy
    ) -> tuple[float, float]:
        """
        Run one episode, updating the VFA after each transition.

        Returns:
            (total_revenue, total_arbitrage_revenue) where total_revenue
            includes penalties and total_arbitrage_revenue does not

        """
        self.fleet.reset()
        total_revenue = 0.0
        total_arbitrage_revenue = 0.0
        periods = self.fleet.config.periods_per_day

        for period in range(periods):
            socs_before = self.fleet.soc_array()
            actions = policy.select_actions(self.fleet, prices, period)
            result = self.fleet.step(actions, prices[period], period)
            socs_after = self.fleet.soc_array()

            total_revenue += result["revenue"]
            total_arbitrage_revenue += result["revenue"] + result["total_penalty"]
            self._update_vfa(socs_before, socs_after, result, period)

        return total_revenue, total_arbitrage_revenue

    def _update_vfa(
        self,
        socs_before: np.ndarray,
        socs_after: np.ndarray,
        step_result: dict,
        period: int,
    ) -> None:
        """
        Update VFA slopes for each asset using a TD target.

        The TD target at period t is:
            immediate_reward_per_asset + gamma * V_{t+1}(soc_{t+1})

        Immediate reward is approximated as the fleet revenue divided equally
        across available assets, a simplification consistent with using a
        shared representative VFA.
        """
        n_plugged_in = max(step_result["n_plugged_in"], 1)
        arbitrage_revenue = step_result["revenue"] + step_result["total_penalty"]
        reward_per_asset = arbitrage_revenue / n_plugged_in
        is_terminal = period == self.fleet.config.periods_per_day - 1

        # Normalise learning rate by n_available so the effective per-period
        # update is independent of fleet size — each asset contributes equally
        # regardless of how many are plugged in at this timestep
        effective_learning_rate = self.vfa.config.learning_rate / n_plugged_in

        for asset, soc_before, soc_after in zip(
            self.fleet.assets, socs_before, socs_after
        ):
            if not asset.is_plugged_in(period):
                continue

            if is_terminal:
                next_value = 0.0
            else:
                next_value = self.vfa.value(soc_after, period + 1)

            td_target = reward_per_asset + self.vfa.config.discount_factor * next_value
            self.vfa.update(soc_before, period, td_target, effective_learning_rate)
