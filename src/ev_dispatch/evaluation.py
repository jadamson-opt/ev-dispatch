"""
Evaluation and visualisation of dispatch policies.

Runs all policies over a shared held-out scenario set and produces
Plotly figures comparing performance across:
- Revenue distributions
- Example SOC trajectories
- Example dispatch and price schedules
- VFA shadow price curves across the day
- ADP training convergence
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ev_dispatch import FloatArray
from ev_dispatch.fleet import Fleet
from ev_dispatch.price_process import PriceProcess
from ev_dispatch.baseline import BasePolicy
from ev_dispatch.value_function import AssetValueFunction
from ev_dispatch.adp import ADPPolicy


def evaluate_policies(
    policies: dict[str, BasePolicy],
    fleet: Fleet,
    price_scenarios: FloatArray,
) -> dict[str, dict]:
    """
    Run each policy over all scenarios and collect metrics.

    Parameters
    ----------
    policies:
        Mapping of policy name to policy instance.
    fleet:
        Fleet to simulate (reset at each episode).
    price_scenarios:
        Array of shape (n_scenarios, periods_per_day) with held-out prices.

    Returns
    -------
    dict mapping policy name to metrics dict with keys:
        revenues, penalties, soc_histories, revenue_histories
    """
    results = {}
    for name, policy in policies.items():
        revenues, penalties = [], []
        soc_histories, revenue_histories = [], []

        for prices in price_scenarios:
            episode = policy.run_episode(fleet, prices)
            revenues.append(episode["total_revenue"])
            penalties.append(episode["total_penalty"])
            soc_histories.append(episode["soc_history"])
            revenue_histories.append(episode["revenue_history"])

        results[name] = {
            "revenues": np.array(revenues),
            "penalties": np.array(penalties),
            "soc_histories": np.array(soc_histories),
            "revenue_histories": np.array(revenue_histories),
        }

    return results


def plot_revenue_distributions(results: dict[str, dict]) -> go.Figure:
    """
    Box plot comparing revenue distributions across policies.

    Useful for assessing both mean performance and tail risk.
    """
    fig = go.Figure()
    for name, metrics in results.items():
        fig.add_trace(go.Box(
            y=metrics["revenues"],
            name=name,
            boxmean=True,
        ))

    fig.update_layout(
        title="Revenue Distribution by Policy",
        yaxis_title="Episode Revenue (£)",
        xaxis_title="Policy",
        showlegend=False,
    )
    return fig


def plot_example_episode(
    policies: dict[str, BasePolicy],
    fleet: Fleet,
    prices: np.ndarray,
    periods_per_day: int = 48,
) -> go.Figure:
    """
    Plot SOC trajectories, cumulative revenue, and price for one episode.

    Runs each policy on the same price scenario for a direct comparison.
    """
    hours = np.arange(periods_per_day) * 0.5
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=("Electricity Price", "Mean Fleet SOC", "Cumulative Revenue (£)"),
        vertical_spacing=0.08,
    )

    fig.add_trace(go.Scatter(x=hours, y=prices, name="Price", line=dict(color="steelblue")), row=1, col=1)

    for name, policy in policies.items():
        episode = policy.run_episode(fleet, prices)
        fig.add_trace(go.Scatter(x=hours, y=episode["soc_history"], name=name), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=hours,
            y=np.cumsum(episode["revenue_history"]),
            name=name,
            showlegend=False,
        ), row=3, col=1)

    fig.update_xaxes(title_text="Hours from 4pm", row=3, col=1)
    fig.update_yaxes(title_text="£/MWh", row=1, col=1)
    fig.update_yaxes(title_text="SOC", row=2, col=1)
    fig.update_yaxes(title_text="£", row=3, col=1)
    fig.update_layout(title="Example Episode — Policy Comparison", height=700)
    return fig


def plot_shadow_prices(vfa: AssetValueFunction, periods_to_plot: list[int] | None = None) -> go.Figure:
    """
    Plot VFA shadow price curves (slopes) across SOC for selected periods.

    Shows how the marginal value of stored energy varies with SOC and time.
    Shadow prices should be higher before morning departure peaks and lower
    during the overnight trough — reflecting the opportunity cost of storage.

    Parameters
    ----------
    periods_to_plot:
        List of period indices to show. Defaults to six evenly spaced periods.
    """
    if periods_to_plot is None:
        periods_to_plot = [0, 8, 16, 24, 32, 40]

    soc_values = vfa.breakpoints[:-1] + vfa.segment_width / 2  # midpoints
    hours_from_4pm = [p * 0.5 for p in periods_to_plot]

    fig = go.Figure()
    for period, hour in zip(periods_to_plot, hours_from_4pm):
        label = f"{_period_to_clock(period)}"
        fig.add_trace(go.Scatter(
            x=soc_values,
            y=vfa.slopes_at_period(period),
            name=label,
            mode="lines+markers",
        ))

    fig.update_layout(
        title="VFA Shadow Prices by SOC and Time of Day",
        xaxis_title="State of Charge",
        yaxis_title="Shadow Price (£/MWh equivalent)",
        legend_title="Time",
    )
    return fig


def plot_training_convergence(
        episode_revenues: list[float],
        episode_arbitrage_revenues: list[float],
        window: int = 50
) -> go.Figure:
    """
    Plot ADP training convergence as rolling means of total and arbitrage revenue.

    A converging curve indicates the VFA is stabilising. Continued high
    variance suggests more training episodes or a lower learning rate is needed.

    Comparing the two series helps diagnose whether poor total revenue stems
    from the VFA not learning arbitrage, or from the forcing rule failing to
    prevent deadline penalties.
    """
    total = np.array(episode_revenues)
    arbitrage = np.array(episode_arbitrage_revenues)
    episodes = np.arange(len(total))

    rolling_total = np.convolve(total, np.ones(window) / window, mode="valid")
    rolling_arbitrage = np.convolve(arbitrage, np.ones(window) / window, mode="valid")
    rolling_episodes = np.arange(window, len(total) + 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=episodes, y=total,
        name="Total Revenue", line=dict(color="lightcoral"), opacity=0.3,
    ))
    fig.add_trace(go.Scatter(
        x=episodes, y=arbitrage,
        name="Arbitrage Revenue", line=dict(color="lightblue"), opacity=0.3,
    ))
    fig.add_trace(go.Scatter(
        x=rolling_episodes, y=rolling_total,
        name=f"Total Rolling Mean (n={window})", line=dict(color="crimson", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=rolling_episodes, y=rolling_arbitrage,
        name=f"Arbitrage Rolling Mean (n={window})", line=dict(color="steelblue", width=2),
    ))

    fig.update_layout(
        title="ADP Training Convergence",
        xaxis_title="Episode",
        yaxis_title="Episode Revenue (£)",
    )
    return fig


def print_summary(results: dict[str, dict]) -> None:
    """Print a tabular revenue and penalty summary across policies."""
    print(f"\n{'Policy':<20} {'Mean Rev':>10} {'Std Rev':>10} {'Min Rev':>10} {'Max Rev':>10} {'Mean Pen':>10}")
    print("-" * 72)
    for name, metrics in results.items():
        rev = metrics["revenues"]
        pen = metrics["penalties"]
        print(f"{name:<20} {rev.mean():>10.2f} {rev.std():>10.2f} {rev.min():>10.2f} {rev.max():>10.2f} {pen.mean():>10.2f}")


def _period_to_clock(period: int) -> str:
    """Convert a period index (episode starts 4pm) to a clock time string."""
    total_minutes = 16 * 60 + period * 30
    hour = (total_minutes // 60) % 24
    minute = total_minutes % 60
    return f"{hour:02d}:{minute:02d}"