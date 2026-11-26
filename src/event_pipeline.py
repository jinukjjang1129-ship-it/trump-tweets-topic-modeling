"""Conditional event study with explicit clocks and executable portfolio rules.

The vendor clock is a required assumption, not inferred from price patterns.
All joins use UTC; trading windows use America/New_York with DST conversion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (1, 10, 20, 30, 45, 60, 120)
SPLIT = pd.Timestamp("2018-09-01", tz="UTC")
END = pd.Timestamp("2019-09-23", tz="UTC")


def load_prices(raw: Path, timezone: str) -> tuple[pd.DataFrame, dict]:
    """Read vendor OHLC and reject conflicting/invalid observations."""
    paths = sorted(raw.glob("DAT_ASCII_SPXUSD_M1_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No minute OHLC files in {raw}")
    frame = pd.concat(
        [
            pd.read_csv(
                p,
                sep=";",
                header=None,
                names=["time", "open", "high", "low", "close", "volume"],
            )
            for p in paths
        ],
        ignore_index=True,
    )
    naive = pd.to_datetime(frame.pop("time"), format="%Y%m%d %H%M%S")
    frame["time"] = naive.dt.tz_localize(
        timezone, ambiguous="NaT", nonexistent="NaT"
    ).dt.tz_convert("UTC")
    invalid = (
        frame.time.isna()
        | frame[["open", "high", "low", "close"]].isna().any(axis=1)
        | (frame[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (frame.high < frame[["open", "close", "low"]].max(axis=1))
        | (frame.low > frame[["open", "close", "high"]].min(axis=1))
    )
    audit = {
        "rows": len(frame),
        "invalid_rows": int(invalid.sum()),
        "duplicate_timestamps": int(frame.time.duplicated().sum()),
    }
    if invalid.any():
        raise ValueError(f"Invalid OHLC/clock rows: {audit}")
    if (
        frame.loc[frame.time.duplicated(False)]
        .groupby("time")
        .open.nunique()
        .gt(1)
        .any()
    ):
        raise ValueError("Conflicting opens at the same timestamp")
    frame = frame.drop_duplicates("time").sort_values("time")
    return frame.set_index("time"), audit


def load_tweets(raw: Path) -> pd.DataFrame:
    """Preserve tweet IDs and sub-minute posting time; deduplicate by ID."""
    frame = pd.read_csv(next(raw.glob("trump*.csv")), dtype={"id": str})
    frame["time"] = pd.to_datetime(frame.datetime_utc, utc=True, errors="raise")
    return frame.dropna(subset=["text"]).drop_duplicates("id").sort_values("time")


def regular_session(frame: pd.DataFrame) -> pd.DataFrame:
    """Restrict to weekday 09:30–16:00 NY; no exchange holiday certification."""
    local = frame.index.tz_convert("America/New_York")
    minutes = local.hour * 60 + local.minute
    return frame.loc[(local.day_of_week < 5) & (minutes >= 570) & (minutes <= 960)]


def forward_returns(prices: pd.DataFrame, horizon: int) -> pd.Series:
    """Exact elapsed-minute return, requiring every minute and the same session."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    times = prices.index
    future = times + pd.Timedelta(minutes=horizon)
    exit_open = prices.open.reindex(future).to_numpy()
    # Endpoint presence alone is insufficient when the feed has missing bars.
    positions = pd.Series(np.arange(len(times)), index=times)
    contiguous = positions.reindex(future).to_numpy() - positions.to_numpy() == horizon
    same_day = (
        times.tz_convert("America/New_York").date
        == future.tz_convert("America/New_York").date
    )
    result = pd.Series(exit_open / prices.open.to_numpy() - 1, index=times)
    return result.where(contiguous & same_day)


def align_events(events: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Enter strictly after posting: ceiling minute plus one-minute latency."""
    events = events.sort_values("time").copy()
    events["entry"] = events.time.dt.ceil("min") + pd.Timedelta(minutes=1)
    events = events.drop_duplicates("entry")
    # An exact join prevents overnight/weekend roll-forward.
    events = events.merge(
        prices.reset_index().rename(columns={"time": "entry"}),
        on="entry",
        how="inner",
        validate="one_to_one",
    )
    return events.sort_values("entry").reset_index(drop=True)


def remove_overlaps(events: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """First eligible event wins; no concurrent positions or future-return filter."""
    positions = []
    free_at = None
    for i, time in enumerate(events.entry):
        if free_at is None or time >= free_at:
            positions.append(i)
            free_at = time + pd.Timedelta(minutes=horizon)
    return events.iloc[positions].copy()


def portfolio(events: pd.DataFrame, side: int, cost_bps: float) -> pd.DataFrame:
    """One unit of entry equity, 1x notional; costs are round-trip basis points."""
    if side not in (-1, 1) or cost_bps < 0:
        raise ValueError("side must be +/-1 and costs nonnegative")
    result = events.copy()
    result["net_return"] = side * result["ret_60m"] - cost_bps / 10000
    if (result.net_return <= -1).any():
        raise ValueError("Portfolio insolvency")
    result["equity"] = (1 + result.net_return).cumprod()
    return result


def switch_portfolio(
    prices: pd.DataFrame, events: pd.DataFrame, cost_bps: float
) -> pd.DataFrame:
    """Switch a long position to short for 60 minutes, with self-financing shares.

    Both portfolios start with equity=1 and target absolute notional=equity at
    each trade. Between trades shares stay fixed. No financing/borrow is modeled.
    A switch trades twice the notional, unlike a single entry or liquidation.
    """
    if cost_bps < 0 or prices.empty:
        raise ValueError("Nonnegative costs and nonempty prices are required")
    desired = pd.Series(1, index=prices.index)
    for entry in events.entry:
        exit_time = entry + pd.Timedelta(minutes=60)
        desired.loc[(desired.index >= entry) & (desired.index < exit_time)] = -1
    curves = {}
    fee = cost_bps / 20000
    for name, targets in (
        ("long", np.ones(len(prices), dtype=int)),
        ("hedged", desired.to_numpy()),
    ):
        cash, shares, previous = 1.0, 0.0, 0
        curve = []
        for price, side in zip(prices.open.to_numpy(), targets, strict=True):
            equity = cash + shares * price
            if equity <= 0:
                raise ValueError("Portfolio insolvency")
            if side != previous:
                # Solve cash+shares*price after fees = abs(shares*price).
                direction = 1 if side == 1 else -1
                target_notional = (equity + fee * direction * shares * price) / (
                    1 + fee
                )
                new_shares = side * target_notional / price
                turnover = abs(new_shares - shares) * price
                cash -= (new_shares - shares) * price + turnover * fee
                shares, previous = new_shares, side
            curve.append(cash + shares * price)
        curve[-1] -= abs(shares) * prices.open.iloc[-1] * fee
        curves[name] = curve
    return pd.DataFrame(curves, index=prices.index)


def regress(events: pd.DataFrame) -> pd.DataFrame:
    """OLS with day-clustered inference; purge each horizon and split boundary."""
    rows = []
    for period in ("development", "holdout"):
        for horizon in HORIZONS:
            subset = events.loc[
                (events.entry < SPLIT)
                if period == "development"
                else (events.entry >= SPLIT)
            ].copy()
            if period == "development":
                subset = subset.loc[
                    subset.entry + pd.Timedelta(minutes=horizon) < SPLIT
                ]
            subset = remove_overlaps(subset, horizon).dropna(
                subset=[f"ret_{horizon}m", "topic0"]
            )
            days = subset.entry.dt.strftime("%Y-%m-%d")
            if len(subset) < 10 or days.nunique() < 3 or subset.topic0.nunique() < 2:
                continue
            fit = sm.OLS(
                subset[f"ret_{horizon}m"], sm.add_constant(subset[["topic0"]])
            ).fit(cov_type="cluster", cov_kwds={"groups": days})
            rows.append(
                {
                    "period": period,
                    "horizon": horizon,
                    "n": len(subset),
                    "days": days.nunique(),
                    "beta": fit.params["topic0"],
                    "p_value": fit.pvalues["topic0"],
                }
            )
    results = pd.DataFrame(rows)
    if not results.empty:
        results["q_bh"] = multipletests(results.p_value, method="fdr_bh")[1]
    return results


def keyword_regressions(prices: pd.DataFrame, tweets: pd.DataFrame) -> pd.DataFrame:
    """Compare word-event minutes with others; cluster dependent returns by day.

    This is a sensitivity study, not a replication of the old Huber p-values.
    The correction family includes all words, horizons and both periods here.
    """
    rows = []
    for word in ("tariff", "china", "deal"):
        pattern = rf"\b{word}(?:s|ed|ing)?\b"
        times = tweets.loc[tweets.text.str.contains(pattern, case=False), "time"]
        entries = (times.dt.ceil("min") + pd.Timedelta(minutes=1)).drop_duplicates()
        indicator = pd.Series(
            prices.index.isin(entries).astype(int), index=prices.index
        )
        for period in ("development", "holdout"):
            for horizon in HORIZONS:
                sample = prices[[f"ret_{horizon}m"]].assign(keyword=indicator).dropna()
                mask = (
                    sample.index < SPLIT
                    if period == "development"
                    else sample.index >= SPLIT
                )
                sample = sample.loc[mask]
                if period == "development":
                    sample = sample.loc[
                        sample.index + pd.Timedelta(minutes=horizon) < SPLIT
                    ]
                if sample.keyword.sum() < 3:
                    continue
                fit = sm.OLS(
                    sample.iloc[:, 0], sm.add_constant(sample[["keyword"]])
                ).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": sample.index.strftime("%Y-%m-%d")},
                )
                rows.append(
                    {
                        "word": word,
                        "period": period,
                        "horizon": horizon,
                        "n": len(sample),
                        "event_minutes": int(sample.keyword.sum()),
                        "beta": fit.params["keyword"],
                        "p_value": fit.pvalues["keyword"],
                    }
                )
    results = pd.DataFrame(rows)
    if not results.empty:
        results["q_bh"] = multipletests(results.p_value, method="fdr_bh")[1]
    return results


def run(
    timezone: str, output: Path, cost_bps: float, topic_path: Path, signal: str
) -> None:
    """Write conditional results without overwriting historical research inputs."""
    output.mkdir(parents=True, exist_ok=True)
    prices, audit = load_prices(ROOT / "data/raw", timezone)
    prices = regular_session(prices)
    prices = prices.loc[
        (prices.index >= pd.Timestamp("2016-11-08", tz="UTC")) & (prices.index < END)
    ].copy()
    for horizon in HORIZONS:
        prices[f"ret_{horizon}m"] = forward_returns(prices, horizon)
    topics = pd.read_csv(topic_path)
    topics["time"] = pd.to_datetime(topics.datetime, utc=True)
    # Existing topic0 was fitted on the full corpus: conditional diagnostics only.
    threshold = float(topics.loc[topics.time < SPLIT, "topic0"].median())
    topics = topics.loc[topics.topic0 >= threshold, ["time", "topic0"]]
    aligned = align_events(topics, prices)
    regression = regress(aligned)
    regression.to_csv(output / "topic_regression.csv", index=False)
    tweets = load_tweets(ROOT / "data/raw")
    keyword_regressions(prices, tweets).to_csv(
        output / "keyword_regression.csv", index=False
    )
    tariff = tweets.loc[
        tweets.text.str.contains(r"\btariff(?:s|ed|ing)?\b", case=False, regex=True),
        ["time", "id"],
    ]
    tariff = tariff.loc[
        (tariff.time >= pd.Timestamp("2016-11-08", tz="UTC")) & (tariff.time < END)
    ]
    if signal == "topic-and-tariff":
        tariff["minute"] = tariff.time.dt.floor("min")
        tariff = tariff.merge(
            topics.rename(columns={"time": "minute"}), on="minute", how="inner"
        )
    matched = align_events(tariff, prices)
    trades = remove_overlaps(matched, 60).dropna(subset=["ret_60m"])
    trades.to_csv(output / "tariff_trades.csv", index=False)
    switch_rows = []
    switch_fig, switch_axis = plt.subplots(figsize=(10, 4))
    for label, start in (("full", prices.index.min()), ("holdout", SPLIT)):
        selected_prices = prices.loc[prices.index >= start]
        selected_trades = trades.loc[trades.entry >= start]
        equity = switch_portfolio(selected_prices, selected_trades, cost_bps)
        equity.to_csv(output / f"switch_equity_{label}.csv")
        switch_rows.append(
            {
                "period": label,
                "long_return": equity.long.iloc[-1] - 1,
                "hedged_return": equity.hedged.iloc[-1] - 1,
                "difference_pp": 100 * (equity.hedged.iloc[-1] - equity.long.iloc[-1]),
            }
        )
        if label == "holdout":
            switch_axis.plot(equity.index, equity.long - 1, label="Long 1x")
            switch_axis.plot(equity.index, equity.hedged - 1, label="Long/short switch")
    switch_axis.set(
        title=f"Conditional switch simulation; clock={timezone}",
        ylabel="Net return",
        xlabel="UTC",
    )
    switch_axis.legend()
    switch_fig.tight_layout()
    switch_fig.savefig(output / "switch_comparison.png", dpi=160)
    plt.close(switch_fig)
    summaries = []
    fig, axis = plt.subplots(figsize=(10, 4))
    for period in ("development", "holdout"):
        selected = trades.loc[
            (trades.entry < SPLIT)
            if period == "development"
            else (trades.entry >= SPLIT)
        ]
        for side in (-1, 1):
            result = portfolio(selected, side, cost_bps)
            summaries.append(
                {
                    "period": period,
                    "side": side,
                    "trades": len(result),
                    "roundtrip_cost_bps": cost_bps,
                    "return": float(result.equity.iloc[-1] - 1)
                    if len(result)
                    else None,
                }
            )
            if len(result):
                axis.plot(
                    result.entry,
                    result.equity - 1,
                    label=f"{period} {'short' if side == -1 else 'long'}",
                )
    axis.set(
        title=f"Tariff events: matched 1x exposure; clock={timezone}",
        ylabel="Net return",
        xlabel="Entry (UTC)",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "matched_exposure.png", dpi=160)
    plt.close(fig)
    legacy = pd.read_csv(ROOT / "data/processed/backtest_input.csv")
    legacy["time"] = pd.to_datetime(legacy.px_time_et, utc=True)
    legacy = legacy.loc[legacy.time >= SPLIT]
    legacy_bh = float(legacy.open.iloc[-1] / legacy.open.iloc[0] - 1)
    legacy_short = float(-legacy.ret_60m.sum())
    report = {
        "price_timezone_assumption": timezone,
        "quality": audit,
        "rth_rows": len(prices),
        "topic_threshold_development": threshold,
        "tariff_tweets": len(tariff),
        "matched_tariff_entries": len(matched),
        "nonoverlapping_complete_trades": len(trades),
        "portfolio": summaries,
        "long_short_switch": switch_rows,
        "topic_input": topic_path.name,
        "signal": signal,
        "legacy_from_2018_09": {
            "summed_short": legacy_short,
            "buy_hold_rebased_ratio": legacy_bh,
            "difference_pp": 100 * (legacy_short - legacy_bh),
        },
        "limitations": [
            "Vendor clock provenance requires confirmation",
            "SPXUSD feed is not certified as tradable S&P500 cash",
            "Weekday/session filter is not an exchange calendar",
            (
                "Existing topic0 model uses full-period vocabulary/fit"
                if topic_path.name == "reg_input_topic0_minute.csv"
                else "Custom topic input: check its training provenance"
            ),
            "Historical holdout has already been explored",
            "BH family covers this run only, not every past hypothesis",
            "Rules are a reconstruction; original switching ledger is missing",
            "Costs exclude borrow/financing and are a scenario",
        ],
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--price-timezone",
        required=True,
        choices=["Etc/GMT+5", "UTC", "America/New_York"],
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=2.0,
        help="Round trip fee+slippage scenario, default 2 bps",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--topic-input",
        type=Path,
        default=ROOT / "data/processed/reg_input_topic0_minute.csv",
    )
    parser.add_argument(
        "--signal", choices=["tariff", "topic-and-tariff"], default="topic-and-tariff"
    )
    args = parser.parse_args()
    run(args.price_timezone, args.output, args.cost_bps, args.topic_input, args.signal)
