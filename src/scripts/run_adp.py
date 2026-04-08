"""
Train the ADP policy and evaluate against baselines.

Usage:
    uv run scripts/run_adp.py
"""

import numpy as np

from ev_dispatch.price_process import PriceProcess, PriceProcessConfig
from ev_dispatch.ev_asset import EVConfig, UserProfile
from ev_dispatch.fleet import Fleet, FleetConfig
from ev_dispatch.baseline import NaiveNightCharger, PriceThreshold, ForesightGreedy
from ev_dispatch.adp import ADPTrainer, ADPConfig, ADPPolicy
from ev_dispatch.value_function import VFAConfig
from ev_dispatch.evaluation import (
    evaluate_policies,
    plot_revenue_distributions,
    plot_example_episode,
    plot_shadow_prices,
    plot_training_convergence,
    print_summary,
)

TRAIN_SEED = 42
EVAL_SEED = 99
N_EVAL_SCENARIOS = 200


def main() -> None:
    price_config = PriceProcessConfig()
    fleet_config = FleetConfig(n_assets=200)
    asset_config = EVConfig()
    user_profile = UserProfile()

    # --- Training ---
    train_price_process = PriceProcess(price_config, seed=TRAIN_SEED)
    train_fleet = Fleet(fleet_config, asset_config, user_profile, seed=TRAIN_SEED)

    trainer = ADPTrainer(
        fleet=train_fleet,
        price_process=train_price_process,
        vfa_config=VFAConfig(),
        adp_config=ADPConfig(n_training_episodes=2000),
    )

    print("Training ADP policy...")
    vfa = trainer.train()

    # --- Evaluation ---
    eval_price_process = PriceProcess(price_config, seed=EVAL_SEED)
    eval_scenarios = eval_price_process.sample_scenarios(N_EVAL_SCENARIOS)
    eval_fleet = Fleet(fleet_config, asset_config, user_profile, seed=EVAL_SEED)

    policies = {
        "Naive": NaiveNightCharger(),
        # "PriceThreshold": PriceThreshold(),
        "Foresight": ForesightGreedy(),
        "ADP": ADPPolicy(vfa, epsilon=0.0, forcing_buffer=2.0),
    }

    print(f"\nEvaluating over {N_EVAL_SCENARIOS} held-out scenarios...")
    results = evaluate_policies(policies, eval_fleet, eval_scenarios)
    print_summary(results)

    # --- Plots ---
    example_prices = eval_scenarios[0]

    plot_revenue_distributions(results).show()
    plot_example_episode(policies, eval_fleet, example_prices).show()
    plot_shadow_prices(vfa).show()
    plot_training_convergence(trainer.episode_revenues, trainer.episode_arbitrage_revenues).show()


if __name__ == "__main__":
    main()