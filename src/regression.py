"""Exploratory robust regression of topic scores on forward market returns.

This script does not establish causality or adjust for multiple testing.
It requires data/processed/reg_input_price_returns.csv (not committed).
"""
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[1]
TOPIC_PATH = ROOT / "data/processed/reg_input_topic0_minute.csv"
RETURNS_PATH = ROOT / "data/processed/reg_input_price_returns.csv"
OUTPUT_PATH = ROOT / "result/generated/regression_trade_war_topic0.csv"

HORIZONS = (1, 10, 20, 30, 45, 60, 120)
TOPIC_COLUMN = "topic0"
TIME_COLUMN = "datetime"
MIN_EVENT_GAP_MINUTES = 10


def remove_nearby_events(
    events: pd.DataFrame, minimum_gap_minutes: int = MIN_EVENT_GAP_MINUTES
) -> pd.DataFrame:
    """Keep the first event in each minimum-gap window."""
    events = events.sort_values(TIME_COLUMN).reset_index(drop=True)
    min_gap = pd.Timedelta(minutes=minimum_gap_minutes)
    keep_positions = []
    last_time = None

    for position, timestamp in enumerate(events[TIME_COLUMN]):
        if last_time is None or timestamp - last_time >= min_gap:
            keep_positions.append(position)
            last_time = timestamp

    return events.iloc[keep_positions].reset_index(drop=True)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read topic scores and aligned forward-return data."""
    if not RETURNS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {RETURNS_PATH}. Generate this processed input from "
            "the original minute-price data before running regression."
        )

    topic = pd.read_csv(TOPIC_PATH)
    returns = pd.read_csv(RETURNS_PATH)

    for frame in (topic, returns):
        frame[TIME_COLUMN] = pd.to_datetime(frame[TIME_COLUMN], errors="coerce")

    topic = topic.dropna(subset=[TIME_COLUMN, TOPIC_COLUMN])
    returns = returns.dropna(subset=[TIME_COLUMN])
    return topic, returns


def fit_regressions(events: pd.DataFrame) -> pd.DataFrame:
    """Fit one HuberT robust linear model for each available horizon."""
    records = []
    for horizon in HORIZONS:
        return_column = f"ret_{horizon}m"
        if return_column not in events:
            continue

        sample = events[[TOPIC_COLUMN, return_column]].dropna()
        if sample.empty:
            continue

        predictor = sm.add_constant(sample[[TOPIC_COLUMN]])
        fit = sm.RLM(
            sample[return_column],
            predictor,
            M=sm.robust.norms.HuberT(),
        ).fit()

        records.append(
            {
                "horizon_min": horizon,
                "events_used": int(fit.nobs),
                "beta_topic0": float(fit.params[TOPIC_COLUMN]),
                "p_value": float(fit.pvalues[TOPIC_COLUMN]),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    topic, returns = load_inputs()
    threshold = topic[TOPIC_COLUMN].median()
    events = topic.loc[topic[TOPIC_COLUMN] >= threshold].sort_values(TIME_COLUMN)

    # Keep the original forward-asof matching behavior for comparability.
    # This matching needs timestamp-tolerance and timezone validation.
    matched = pd.merge_asof(
        events,
        returns.sort_values(TIME_COLUMN),
        on=TIME_COLUMN,
        direction="forward",
    ).dropna()

    selected = remove_nearby_events(matched)
    results = fit_regressions(selected)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(results.to_string(index=False))
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
