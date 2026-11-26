"""Visualize an exploratory event-short series and a market buy-and-hold series.

Important: these two series have different exposure and compounding rules.
They are NOT comparable as live trading portfolios or verified alpha.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data/processed/backtest_input.csv"
OUTPUT_DIR = ROOT / "result/generated"
TIME_COLUMN = "px_time_et"
EVENT_HORIZON_MINUTES = 60


def load_input(path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load and sort the original event/backtest input."""
    frame = pd.read_csv(path)
    frame[TIME_COLUMN] = pd.to_datetime(
        frame[TIME_COLUMN], errors="coerce", utc=True
    ).dt.tz_convert("America/New_York")
    return (
        frame.dropna(subset=[TIME_COLUMN])
        .sort_values(TIME_COLUMN)
        .reset_index(drop=True)
    )


def market_buy_and_hold(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute a compounded minute-price return series."""
    market = frame.dropna(subset=["open"]).copy()
    market["bh_ret"] = market["open"].pct_change()
    market["bh_cum"] = (1 + market["bh_ret"]).cumprod() - 1
    return market


def exploratory_event_short(
    frame: pd.DataFrame, horizon_minutes: int = EVENT_HORIZON_MINUTES
) -> pd.DataFrame:
    """Sum signed forward returns across events (not a portfolio equity curve)."""
    return_column = f"ret_{horizon_minutes}m"
    if return_column not in frame:
        raise ValueError(f"Missing required column: {return_column}")

    events = frame.dropna(subset=[return_column]).copy()
    events["short_ret"] = -events[return_column]
    events["cum_short"] = events["short_ret"].cumsum()
    return events


def plot_comparison(
    market: pd.DataFrame,
    events: pd.DataFrame,
    output: Path,
    start_date: str | None = None,
) -> None:
    """Plot two descriptive series after rebasing at a common start."""
    if start_date is None:
        first = max(market[TIME_COLUMN].iloc[0], events[TIME_COLUMN].iloc[0])
    else:
        first = pd.Timestamp(start_date, tz="America/New_York")

    market_subset = market.loc[market[TIME_COLUMN] >= first].copy()
    event_subset = events.loc[events[TIME_COLUMN] >= first].copy()
    if market_subset.empty or event_subset.empty:
        raise ValueError(f"No market/events available after {first}")

    market_subset["rebased"] = market_subset["bh_cum"] - market_subset["bh_cum"].iloc[0]
    if start_date is None:
        event_subset["rebased"] = (
            event_subset["cum_short"] - event_subset["cum_short"].iloc[0]
        )
    else:
        event_subset["rebased"] = event_subset["short_ret"].cumsum()

    fig, axis = plt.subplots(figsize=(11, 5))
    axis.plot(
        market_subset[TIME_COLUMN],
        market_subset["rebased"],
        label="S&P 500 Buy & Hold (compounded)",
        linewidth=2,
    )
    axis.plot(
        event_subset[TIME_COLUMN],
        event_subset["rebased"],
        label="Event short returns (summed; non-comparable exposure)",
        linewidth=2,
    )
    axis.axhline(0, color="gray", linestyle="--", linewidth=1)
    axis.set(
        title="Exploratory return series — not a trading performance comparison",
        xlabel="Time",
        ylabel="Rebased return series",
    )
    axis.legend()
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    frame = load_input()
    market = market_buy_and_hold(frame)
    events = exploratory_event_short(frame)

    plot_comparison(market, events, OUTPUT_DIR / "series_first_available.png")
    plot_comparison(
        market,
        events,
        OUTPUT_DIR / "series_2018_09.png",
        start_date="2018-09-01",
    )
    print(f"Saved descriptive plots in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
