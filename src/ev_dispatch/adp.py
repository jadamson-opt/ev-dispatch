"""
ADP training loop and learned dispatch policy.

The aggregator trains per-type AssetValueFunctions (one for commuters, one for
WFH users) over many simulated episodes.
At each timestep, each asset receives a shadow price derived from the VFA slope
at its current SOC. Assets charge if the current electricity price is below their
shadow price, and discharge if above, subject to availability, SOC floor, and
expected deadline constraints.

Deadline forcing is applied only to commuter assets, using their individual
profile's expected departure. WFH assets have no planned departure and rely
entirely on shadow prices for dispatch decisions.

Penalty costs are excluded from VFA updates. The VFA learns pure arbitrage
value. Deadline safety is handled entirely by the forcing rule.
"""

import math

import numpy as np
from dataclasses import dataclass

from ev_dispatch import FloatArray
from ev_dispatch.ev_asset import AssetType, CommuterProfile
from ev_dispatch.fleet import Fleet
from ev_dispatch.price_process import PriceProcess
from ev_dispatch.value_function import VFAConfig, make_vfa_registry, AssetValueFunction
from ev_dispatch.baseline import BasePolicy


@dataclass
class ADPConfig:
    """Parameters controlling the ADP training procedure."""

    n_training_episodes: int = 2000
    # epsilon: float = 0.1  # Exploration rate for epsilon-greedy action selection
    epsilon: float = 0.02  # Exploration rate for epsilon-greedy action selection
    epsilon_decay: float = 0.995  # Multiplicative decay applied each episode
    forcing_buffer: float = 2.0  # Multiplier on departure_period_std for when to force charging ahead of planned departure
    soc_floor: float = 0.15  # Minimum SOC for discharge to cover unplanned trips


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
        self,
        vfa_registry: dict[AssetType, AssetValueFunction],
        epsilon: float,
        forcing_buffer: float,
        soc_floor: float,
    ):
        self.vfa_registry = vfa_registry
        self.epsilon = epsilon
        self.forcing_buffer = forcing_buffer
        self.soc_floor = soc_floor

    def select_actions(
        self, fleet: Fleet, prices: FloatArray, period: int
    ) -> FloatArray:
        """
        Select charge/discharge actions for each asset based on shadow prices.

        Commuter deadline awareness uses profile-level expectations rather than
        actual sampled windows, reflecting that a deployed policy cannot observe
        tonight's true departure time.

        A SOC floor is enforced to protect against unplanned usage.
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

            should_prevent_discharge = self._should_prevent_discharge(asset, period)
            if self.epsilon > 0 and np.random.random() < self.epsilon:
                if should_prevent_discharge:
                    action_choices = [asset.config.max_charge_rate_kw, 0.0]
                else:
                    action_choices = [
                        0.0,
                        -asset.config.max_discharge_rate_kw,
                        asset.config.max_charge_rate_kw,
                    ]
                actions.append(float(np.random.choice(action_choices)))
                continue

            vfa = self.vfa_registry[asset.profile.asset_type]
            shadow_price = vfa.shadow_price(asset.soc, period)

            if current_price < shadow_price:
                actions.append(asset.config.max_charge_rate_kw)
            elif current_price > shadow_price and not should_prevent_discharge:
                actions.append(-asset.config.max_discharge_rate_kw)
            else:
                actions.append(0.0)

        return np.array(actions)

    @staticmethod
    def _must_charge_for_deadline(asset, period: int, forcing_buffer: float) -> bool:
        """
        Return True if a commuter asset must charge every remaining period to meet
        its expected departure requirement.

        Uses profile-level expectations rather than actual sampled windows,
        reflecting that a deployed policy cannot observe tonight's true departure.
        forcing_buffer is a multiplier on departure_period_std, adding extra
        period conservatism that scales with departure time uncertainty.

        Always returns False for WFH assets since no planned departure exists.
        """
        if not isinstance(asset.profile, CommuterProfile):
            return False

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

    def _should_prevent_discharge(self, asset, period: int) -> bool:
        """
        Return True if discharging now would make the expected departure
        requirement unreachable or if it would violate the soc floor.

        Applies the same std-dev-scaled buffer as _must_charge_for_deadline
        so the two rules are consistent in their conservatism. Uses profile-level
        expectations for the same reason as above.

        For WFH assets, there is no deadline based restriction.
        """
        soc_after_discharge = max(0, asset.soc - asset.max_discharge_per_period)
        if soc_after_discharge < self.soc_floor:
            return True

        if not isinstance(asset.profile, CommuterProfile):
            return False

        expected_departure = int(asset.profile.departure_period_mean)
        expected_required_soc = asset.profile.required_soc
        buffer_periods = int(self.forcing_buffer * asset.profile.departure_period_std)

        buffered_periods_remaining = max(
            0, expected_departure - period - 1 - buffer_periods
        )
        max_reachable_soc = (
            soc_after_discharge
            + buffered_periods_remaining * asset.max_charge_per_period
        )
        return max_reachable_soc < expected_required_soc


class ADPTrainer:
    """
    Trains per-type AssetValueFunctions via forward ADP simulation.

    Each episode simulates a full day, stepping the fleet through 48 periods.
    After each transition, the appropriate type VFA is updated using a temporal
    difference (TD) target composed of the immediate reward and the discounted
    next-period value estimate. Penalty costs are excluded from VFA updates.

    Separate per-type revenue tracking allows convergence of commuter and WFH
    arbitrage to be monitored independently. Within those profiles, the shared
    VFAs represent a profile-average asset. Updating from all assets at each
    timestep gives many training samples per episode, accelerating convergence.

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

        last_commuter_expected_departure = max(
            (a.profile.departure_period_mean for a in fleet.assets if isinstance(a.profile, CommuterProfile)),
            default=fleet.config.periods_per_day,
        )
        self.vfa_registry = make_vfa_registry(
            config=vfa_config,
            periods_per_day=fleet.config.periods_per_day,
            last_commuter_departure=int(last_commuter_expected_departure),
            max_discharge_per_period=fleet.assets[0].max_discharge_per_period,
        )
        self.episode_revenues: list[float] = []
        self.episode_arbitrage_revenues: list[float] = []

    # @profile
    def train(self) -> dict[AssetType, AssetValueFunction]:
        """
        Run the full ADP training loop.

        Returns the trained AssetValueFunction.
        """
        cfg = self.adp_config
        epsilon = cfg.epsilon

        for episode in range(cfg.n_training_episodes):
            prices = self.price_process.sample_scenario()
            policy = ADPPolicy(
                self.vfa_registry, epsilon=epsilon, forcing_buffer=cfg.forcing_buffer, soc_floor=cfg.soc_floor
            )
            episode_revenue, episode_arbitrage = self._run_training_episode(
                prices, policy
            )

            self.episode_revenues.append(episode_revenue)
            self.episode_arbitrage_revenues.append(episode_arbitrage)
            epsilon *= cfg.epsilon_decay

            if (episode + 1) % 100 == 0:
                mean_revenue = np.mean(self.episode_revenues[-100:])
                mean_arbitrage = np.mean(self.episode_arbitrage_revenues[-100:])
                print(
                    f"Episode {episode + 1}: "
                    f"mean revenue (last 100) = £{mean_revenue:.2f}  "
                    f"mean arbitrage (last 100) = £{mean_arbitrage:.2f}  "
                )

        return self.vfa_registry

    def _run_training_episode(
        self, prices: FloatArray, policy: ADPPolicy
    ) -> tuple[float, float]:
        """
        Run one episode, updating each type VFA after each transition.

        Returns:
            (total_revenue, total_arbitrage_revenue) where total_revenue
            includes penalties and total_arbitrage_revenue does not

        """
        self.fleet.reset()
        total_revenue = 0.0
        total_arbitrage_revenue = 0.0

        for period in range(self.fleet.config.periods_per_day):
            socs_before = self.fleet.soc_array()
            actions = policy.select_actions(self.fleet, prices, period)
            result = self.fleet.step(actions, prices[period], period)
            socs_after = self.fleet.soc_array()

            total_revenue += result["revenue"]
            total_arbitrage_revenue += result["revenue"] + result["total_penalty"]
            self._update_vfas(socs_before, socs_after, result, period)

        return total_revenue, total_arbitrage_revenue

    def _update_vfas(
        self,
        socs_before: np.ndarray,
        socs_after: np.ndarray,
        step_result: dict,
        period: int,
    ) -> None:
        """
        Update VFA slopes for each asset using a TD target based on arbitrage revenue only.

        The TD target at period t is:
            immediate_reward_per_asset + gamma * V_{t+1}(soc_{t+1})

        Revenue is split equally across all available assets regardless of type.
        Per-type attribution would require per-asset energy tracking in fleet.step
        and is left as a future improvement.

        Learning rate is normalised by n_available for stability.
        """
        n_plugged_in = max(step_result["n_plugged_in"], 1)
        arbitrage_revenue = step_result["revenue"] + step_result["total_penalty"]
        reward_per_asset = arbitrage_revenue / n_plugged_in
        is_terminal = period == self.fleet.config.periods_per_day - 1

        for asset, soc_before, soc_after in zip(
            self.fleet.assets, socs_before, socs_after
        ):
            if not asset.is_plugged_in(period):
                continue

            vfa = self.vfa_registry[asset.profile.asset_type]
            effective_learning_rate = vfa.config.learning_rate / n_plugged_in
            next_value = 0.0 if is_terminal else vfa.value(soc_after, period + 1)
            td_target = reward_per_asset + vfa.config.discount_factor * next_value
            vfa.update(soc_before, period, td_target, effective_learning_rate)
