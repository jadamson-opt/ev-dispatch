# EV Fleet Dispatch: ADP Prototype

A prototype implementation of Approximate Dynamic Programming (ADP) for optimal dispatch of a distributed fleet of electric vehicles (EVs). The system learns when to charge and discharge EVs to maximise arbitrage revenue from electricity price variation, while ensuring vehicles are sufficiently charged for their users' journeys.

## Contents

- [Problem Overview](#problem-overview)
- [Why ADP?](#why-adp)
- [Approach](#approach)
- [Prerequisites and Installation](#prerequisites-and-installation)
- [Running](#running)
- [Outputs](#outputs)
- [Code Structure](#code-structure)
- [Known Limitations](#known-limitations)
- [Future Work](#future-work)

## Problem Overview

Aggregators managing large fleets of residential EVs face a sequential decision problem under uncertainty: when should each vehicle charge (buying electricity) or discharge back to the grid (selling electricity), given:

- **Electricity prices** that vary significantly across the day, with morning and evening peaks and overnight troughs
- **User availability constraints**: vehicles must reach a required state of charge (SOC) before departure
- **Uncertain demand**: departure times and unplanned trips introduce risk

The opportunity is to buy cheap overnight electricity and sell during peak periods (V2G), while guaranteeing vehicles are ready when users need them.

### Scope of this Prototype

- **Two user types**: commuters (overnight charging, morning departure) and work-from-home (WFH) users (available most of the day, occasional short trips)
- **Synthetic price process**: Ornstein-Uhlenbeck mean-reverting process with a deterministic diurnal shape reflecting UK day-ahead price patterns and occasional jump spikes
- **Single asset class**: residential EVs with 60kWh batteries and 7.4kW home chargers
- **Binary dispatch**: assets charge or discharge at full rate, or remain idle
- **No battery degradation**, network constraints, or ancillary service markets
- **No prior commitment**: No day-ahead or intraday commitments to respect

## Why ADP?

**vs Rule-based systems**: fixed charge/discharge thresholds based on price or time of day are pragmatic and simple to implement and audit, but could leave significant value on the table and don't adapt to changing market conditions or individual user patterns.

**vs Model Predictive Control (MPC)**: a well-established approach in battery dispatch literature. MPC solves a deterministic optimisation over a rolling horizon at each timestep, re-solving as new information arrives. It handles constraints cleanly but relies on point forecasts. The optimal solution under expected prices is not the same as the expected optimal solution. Stochastic MPC extends this to handle uncertainty explicitly but at greater computational cost. ADP and MPC sit on the spectrum between explicit lookahead and learned value approximation. MPC rolls out the future explicitly, while ADP approximates that future value offline through the learned value function, trading solution accuracy for inference speed. In practice, hybrid approaches could be effective — MPC or stochastic optimisation for day-ahead scheduling where re-solving is tractable, and ADP for real-time dispatch where inference speed is critical.

**vs Stochastic Optimisation**: approaches such as scenario-based stochastic programming and stochastic metaheuristics (e.g. using estimation-based local search) handle uncertainty explicitly and can produce high-quality solutions. Computational cost varies significantly — metaheuristic approaches can scale better than strict scenario-tree formulations — but all require executing a solve at each decision point, and cost grows with fleet size, scenario count, and decision frequency. ADP learns a value function offline and evaluates it in microseconds at runtime, making it well-suited to the high-frequency sequential nature of real-time dispatch.

**vs Classical ML**: predictive models are well-suited to forecasting prices and availability, and are used here for exactly that. But prediction alone does not optimise decisions. A model that accurately forecasts tomorrow's prices cannot tell you what to do today without reasoning about the future value of the energy you store now. ADP explicitly represents that future value through the learned value function.

**vs Neural RL**: general-purpose RL learns everything from data, requiring large amounts of simulation and producing opaque policies. ADP exploits domain structure (concavity of the value function, separability across assets, physical bounds on energy flow) to learn faster, enforce constraints reliably, and produce interpretable shadow prices that operators can audit and reason about.

**Scalability**: the separable VFA decomposes the fleet problem into independent per-asset decisions driven by shadow prices. Inference complexity is constant per asset regardless of fleet size.

**Interpretability**: the shadow price has a direct economic meaning — the marginal value of stored energy in £/MWh. Dispatch decisions can be explained and audited, which matters in regulated energy markets.

## Approach

The dispatch problem is framed as a sequential decision problem under uncertainty following Warren Powell's RLSO framework [[1]](#references), using a two-level hierarchical decomposition.

### Fleet Level: Value Function Approximation (VFA)

A piecewise-linear concave value function $\bar{V}_t(s)$ is learned for each asset type, approximating the expected future arbitrage revenue achievable from a given SOC $s$ at time $t$. The slope of this function at a given SOC is the **shadow price**, representing the marginal value of stored energy in £/MWh equivalent.

The VFA is trained via forward ADP simulation over many daily episodes. At each timestep, a Temporal Difference (TD) update adjusts the slope toward a bootstrap target:

$$\text{TD target} = r_t + \gamma \cdot \bar{V}_{t+1}(s_{t+1})$$

Where $r_t$ is the immediate arbitrage revenue from the asset's action (discharge revenue or charge cost in £).

### Asset Level: Shadow Price Policy

Each asset receives a shadow price signal from its type-specific VFA and makes a threshold decision:

- **Charge** if current electricity price < shadow price
- **Discharge** if current electricity price > shadow price
- **Idle** otherwise

### Deadline Safety: Forcing Rule

Deadline management is handled by a separate deterministic rule rather than the VFA, based on the principle that hard constraints under uncertainty are better enforced by robust rules than learned policies. The forcing rule determines whether an asset can still reach its required SOC by its expected departure time at maximum charge rate, and overrides the shadow price signal to force charging if not. A configurable buffer scaled by departure time standard deviation adds conservatism proportional to uncertainty in the user's actual departure.

### Key Design Decisions

**Separable VFA**: the fleet value function is assumed to decompose additively across assets, enabling independent per-asset updates and shadow price computation. This is the key approximation that makes the problem tractable at scale.

**Piecewise-linear representation**: slopes are stored per SOC segment and time period, enforcing concavity (diminishing marginal returns of stored energy) throughout training. Physically derived initialisation based on maximum dischargeable energy at expected prices provides a sensible starting policy from episode one.

**Per-type VFAs**: commuter and WFH assets have structurally different value profiles. Commuters can only access overnight prices; WFH users can arbitrage during midday and peak periods. Separate VFAs allow each type to learn its own temporal shadow price structure.

**Penalty exclusion from VFA updates**: deadline penalties are excluded from VFA training. The VFA learns pure arbitrage value; deadline safety is entirely the forcing rule's responsibility. This prevents large penalty signals from corrupting the arbitrage value function.

## Prerequisites and Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/jadamson-opt/ev-dispatch.git
cd ev-dispatch
uv sync
```

## Running

### Validate the environment

```bash
uv run scripts/validate_environment.py
```

Runs naive and greedy-foresight policies over 100 scenarios and produces a validation plot. Use this to confirm the simulation environment behaves correctly before training.

### Train and evaluate the ADP policy

```bash
uv run scripts/run_adp.py
```

Trains the ADP policy over 2000 episodes then evaluates all policies over 200 held-out scenarios, producing four interactive Plotly figures.

## Outputs

**Revenue Distribution**: box plot comparing episode revenue across policies. ADP should outperform the naive baseline; greedy foresight sets an upper bound no deployable policy can reach.

**Example Episode**: price, mean fleet SOC, cumulative revenue, and cumulative arbitrage revenue for a single scenario.

**VFA Shadow Prices**: shadow price curves across SOC at six times of day, shown separately for commuter and WFH VFAs. The commuter VFA should show urgency near departure; the WFH VFA should reflect access to both morning and evening peaks.

**Training Convergence**: rolling mean of total and arbitrage revenue across episodes. The gap between the two lines is mean penalty cost. A stable arbitrage line with low penalty gap indicates healthy training.

## Code Structure

```
src/ev_dispatch/
├── price_process.py     # Synthetic electricity price simulation
├── ev_asset.py          # Individual EV asset, plugin windows, SOC dynamics
├── fleet.py             # Fleet aggregator, portfolio buffer, dispatch coordination
├── value_function.py    # Piecewise-linear VFA, shadow prices, TD updates
├── adp.py               # ADP training loop, ADPPolicy, deadline forcing
├── baselines.py         # Naive and greedy-foresight baseline policies
└── evaluation.py        # Metrics and Plotly visualisations

scripts/
├── validate_environment.py   # Baseline-only sanity check
└── run_adp.py                # Full training and evaluation pipeline
```

**`price_process.py`**: generates synthetic price scenarios using an OU process with diurnal offsets. Separate instances for training and evaluation ensure the VFA is assessed on unseen price realisations.

**`ev_asset.py`**: models a single EV with `PluginWindow` availability windows, SOC dynamics with charge/discharge efficiency, and deadline penalties. Supports `CommuterProfile` and `WFHProfile` subtypes with distinct availability sampling logic.

**`fleet.py`**: manages the asset collection, applies the portfolio capacity buffer, and aggregates step results.

**`value_function.py`**: stores slopes as a `(periods, segments)` array per asset type. `shadow_price()` returns the slope at a given SOC and period converted to £/MWh; `update()` applies a TD correction and enforces concavity. `make_vfa_registry()` constructs one VFA per asset type with appropriate initialisation.

**`adp.py`**: `ADPTrainer` runs the forward simulation training loop, routing VFA updates to the correct type. `ADPPolicy` implements the shadow price threshold decision with deadline forcing.

**`baselines.py`**: naive and greedy-foresight policies sharing a common `BasePolicy` interface. `ForesightGreedy` plans charge/discharge periods from the full price sequence with deadline awareness, providing a loose upper bound.

## Known Limitations

- **Shared VFA per asset type**: all commuters share one value function regardless of individual departure time. An asset departing at 06:30 has a different value profile to one departing at 08:00 but receives the same shadow prices
- **Synthetic prices**: the OU process does not capture real price regime changes, seasonal variation, or day-to-day correlation
- **No battery degradation**: frequent deep cycling accelerates degradation; ignoring this overstates long-run profitability
- **Consuming SOC**: Unplanned EV journeys do not currently use up battery charge, they just require a certain starting charge
- **Single-day episodes**: the VFA has no memory across days; a user's SOC history and upcoming schedule are not exploited

## Future Work

**Day-ahead / intraday two-stage structure**: in a production aggregator context, day-ahead and intraday market commitments may create obligations that constrain real-time dispatch. The fleet must honour volumes already sold in forward markets, with imbalance penalties for shortfalls. The current prototype optimises freely against spot prices without respecting prior commitments. An extension would be to add an outer decision layer where the aggregator commits capacity to day-ahead markets based on price forecasts, with the ADP policy dispatching within those commitments at real-time. This reflects how a production system would likely combine approaches — stochastic optimisation or MPC for the day-ahead layer where re-solving is tractable, ADP for real-time dispatch, and deterministic pre-committed rules for frequency response.

**Bayesian updating**: In the real-world, we could update the probability distributions as new data becomes available.

**Price-conditioned VFA**: extend the state to include current price level or regime, allowing shadow prices to adapt on high-price vs low-price days.

**Battery degradation cost**: add a cycle cost term to the objective, causing the VFA to learn more conservative discharge thresholds.

**Ancillary services**: reserve fleet capacity for frequency response contracts modelled on a separate timescale from the arbitrage problem.

**Real price data**: replace the synthetic process with historical UK half-hourly prices to validate policy performance against real market conditions.

## References

[1] Powell, W.B. (2022). *Reinforcement Learning and Stochastic Optimization: A Unified Framework for Sequential Decisions*. John Wiley & Sons, Hoboken, NJ.