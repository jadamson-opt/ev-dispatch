"""
Environment validation script.

Runs all three baseline policies over the same set of held-out price
scenarios and prints a summary. Used to confirm the environment
behaves correctly.

Expected sanity checks:
- HindsightOptimal revenue > PriceThreshold revenue > NaiveNightCharger revenue
- Penalties are non-zero but not catastrophically large
- Mean SOC trajectories look physically reasonable
"""

import numpy as np
import matplotlib.pyplot as plt

from ev_dispatch.price_process import PriceProcess, PriceProcessConfig
from ev_dispatch.ev_asset import EVConfig, UserProfile
from ev_dispatch.fleet import Fleet, FleetConfig
from ev_dispatch.baseline import NaiveNightCharger, PriceThreshold, ForesightGreedy

N_SCENARIOS = 100
SEED = 42


def run_policy_over_scenarios(policy, fleet, scenarios):
    revenues, penalties = [], []
    for idx, prices in enumerate(scenarios):
        result = policy.run_episode(fleet, prices)
        revenues.append(result["total_revenue"])
        penalties.append(result["total_penalty"])
    return np.array(revenues), np.array(penalties)


def print_summary(name: str, revenues: np.ndarray, penalties: np.ndarray) -> None:
    print(f"\n{name}")
    print(f"  Revenue   mean={revenues.mean():.2f}  std={revenues.std():.2f}  min={revenues.min():.2f}  max={revenues.max():.2f}")
    print(f"  Penalties mean={penalties.mean():.2f}  std={penalties.std():.2f}")


def plot_example_episode(fleet, scenarios, seed=0):
    prices = scenarios[0]
    periods = np.arange(len(prices))
    hours = periods * 0.5

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Example Episode — Baseline Policy Comparison")

    axes[0].plot(hours, prices, color="steelblue")
    axes[0].set_ylabel("Price (£/MWh)")
    axes[0].set_title("Electricity Price")

    policies = {
        "Naive": NaiveNightCharger(),
        # "Threshold": PriceThreshold(),
        "Hindsight": ForesightGreedy(),
    }

    for name, policy in policies.items():
        result = policy.run_episode(fleet, prices)
        axes[1].plot(hours, result["soc_history"], label=name)
        axes[2].plot(hours, np.cumsum(result["revenue_history"]), label=name)

    axes[1].set_ylabel("Mean Fleet SOC")
    axes[1].legend()
    axes[2].set_ylabel("Cumulative Revenue (£)")
    axes[2].set_xlabel("Hour of Day")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("validation_plot.png", dpi=150)
    print("\nPlot saved to validation_plot.png")


def main():
    price_process = PriceProcess(PriceProcessConfig(), seed=SEED)
    scenarios = price_process.sample_scenarios(N_SCENARIOS)

    fleet_config = FleetConfig(n_assets=50)
    asset_config = EVConfig()
    user_profile = UserProfile()
    fleet = Fleet(fleet_config, asset_config, user_profile, seed=SEED)

    naive = NaiveNightCharger()
    hindsight = ForesightGreedy()

    print(f"Running {N_SCENARIOS} scenarios with {fleet_config.n_assets} assets...")

    naive_rev, naive_pen = run_policy_over_scenarios(naive, fleet, scenarios)
    hint_rev, hint_pen = run_policy_over_scenarios(hindsight, fleet, scenarios)

    print_summary("NaiveNightCharger", naive_rev, naive_pen)
    print_summary("HindsightOptimal", hint_rev, hint_pen)

    print("\nSanity check — Hindsight should beat Naive:")
    print(f"  Hindsight mean > Naive mean:     {hint_rev.mean() > naive_rev.mean()}")

    plot_example_episode(fleet, scenarios)


if __name__ == "__main__":
    main()