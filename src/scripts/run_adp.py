"""
Train the ADP policy and evaluate against baselines.

Usage:
    uv run scripts/run_adp.py
"""

from ev_dispatch.adp import ADPConfig, ADPPolicy, ADPTrainer
from ev_dispatch.baseline import ForesightGreedy, NaiveNightCharger
from ev_dispatch.ev_asset import EVConfig
from ev_dispatch.evaluation import (
    evaluate_policies,
    plot_example_episode,
    plot_revenue_distributions,
    plot_shadow_prices,
    plot_training_convergence,
    print_summary,
)
from ev_dispatch.fleet import Fleet, FleetConfig
from ev_dispatch.price_process import PriceProcess, PriceProcessConfig
from ev_dispatch.value_function import VFAConfig

TRAIN_SEED = 42
EVAL_SEED = 99
N_EVAL_SCENARIOS = 200


def main() -> None:
    price_config = PriceProcessConfig()
    fleet_config = FleetConfig(n_assets=500)
    asset_config = EVConfig()

    # --- Training ---
    train_price_process = PriceProcess(price_config, seed=TRAIN_SEED)
    fleet = Fleet(fleet_config, asset_config, seed=TRAIN_SEED)

    trainer = ADPTrainer(
        fleet=fleet,
        price_process=train_price_process,
        vfa_config=VFAConfig(),
        adp_config=ADPConfig(n_training_episodes=2000),
    )

    print("Training ADP policy...")
    vfa_registry = trainer.train()

    # --- Evaluation ---
    eval_price_process = PriceProcess(price_config, seed=EVAL_SEED)
    eval_scenarios = eval_price_process.sample_scenarios(N_EVAL_SCENARIOS)

    policies = {
        "Naive": NaiveNightCharger(),
        "Foresight": ForesightGreedy(),
        "ADP": ADPPolicy(
            vfa_registry,
            epsilon=0.0,
            forcing_buffer=ADPConfig.forcing_buffer,
            soc_floor=ADPConfig.soc_floor,
        ),
    }

    print(f"\nEvaluating over {N_EVAL_SCENARIOS} held-out scenarios...")
    results = evaluate_policies(policies, fleet, eval_scenarios)
    print_summary(results)

    # --- Plots ---
    example_prices = eval_scenarios[0]

    plot_revenue_distributions(results).show()
    plot_example_episode(policies, fleet, example_prices).show()
    plot_shadow_prices(vfa_registry).show()
    plot_training_convergence(
        trainer.episode_penalised_revenues, trainer.episode_arbitrage_revenues
    ).show()


if __name__ == "__main__":
    main()
