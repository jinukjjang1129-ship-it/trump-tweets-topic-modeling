"""Regression tests for clock, leakage, execution and portfolio invariants."""

import pandas as pd
import pytest

from backtest import load_input
from event_pipeline import (
    align_events,
    forward_returns,
    portfolio,
    regular_session,
    remove_overlaps,
    switch_portfolio,
)


def prices(times: list[str], values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"open": values}, index=pd.to_datetime(times, utc=True).rename("time")
    )


def test_elapsed_minutes_rejects_missing_bars():
    p = prices(
        ["2019-01-02 15:00", "2019-01-02 15:01", "2019-01-02 15:03"], [100, 101, 103]
    )
    assert forward_returns(p, 1).iloc[0] == pytest.approx(0.01)
    assert pd.isna(forward_returns(p, 3).iloc[0])


def test_session_boundary_not_next_trading_row():
    p = prices(["2019-01-02 21:00", "2019-01-03 14:30"], [100, 102])
    assert forward_returns(p, 1).isna().all()


def test_dst_regular_session():
    p = prices(
        ["2019-01-02 14:30", "2019-07-02 13:30", "2019-07-02 12:30"], [100, 100, 100]
    )
    assert len(regular_session(p)) == 2


def test_entry_after_post_and_no_weekend_rollforward():
    p = prices(
        ["2019-01-02 15:00", "2019-01-02 15:01", "2019-01-07 14:30"], [100, 101, 102]
    )
    e = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2019-01-02 15:00:00", "2019-01-05 15:00:00"], utc=True
            )
        }
    )
    matched = align_events(e, p)
    assert len(matched) == 1
    assert matched.entry.iloc[0] > matched.time.iloc[0]
    assert matched.open.iloc[0] == 101


def test_horizon_specific_nonoverlap():
    e = pd.DataFrame(
        {
            "entry": pd.to_datetime(
                ["2019-01-02 15:00", "2019-01-02 15:10", "2019-01-02 16:00"], utc=True
            )
        }
    )
    assert len(remove_overlaps(e, 60)) == 2
    assert len(remove_overlaps(e, 120)) == 1


def test_matched_exposure_and_costs():
    e = pd.DataFrame({"ret_60m": [0.01, -0.02]})
    long = portfolio(e, 1, 2)
    short = portfolio(e, -1, 2)
    assert (long.net_return + short.net_return).tolist() == pytest.approx([-0.0004] * 2)
    assert long.equity.iloc[-1] == pytest.approx(1.0098 * 0.9798)


def test_mixed_dst_offsets_legacy(tmp_path):
    p = tmp_path / "input.csv"
    p.write_text(
        "px_time_et,open,ret_60m\n2019-01-02 10:00:00-05:00,1,0\n"
        "2019-07-02 10:00:00-04:00,2,0\n"
    )
    assert str(load_input(p).px_time_et.dt.tz) == "America/New_York"


def test_switch_equals_long_without_events():
    p = prices(["2019-01-02 15:00", "2019-01-02 15:01"], [100, 110])
    equity = switch_portfolio(p, pd.DataFrame({"entry": []}), 2)
    assert equity.long.tolist() == equity.hedged.tolist()
    assert equity.long.iloc[-1] == pytest.approx(1.1 * (1 - 0.0001) / 1.0001)


def test_switch_short_profits_when_price_falls():
    p = prices(["2019-01-02 15:00", "2019-01-02 15:01"], [100, 90])
    e = pd.DataFrame({"entry": [p.index[0]]})
    equity = switch_portfolio(p, e, 0)
    assert equity.hedged.iloc[-1] == pytest.approx(1.1)
    assert equity.long.iloc[-1] == pytest.approx(0.9)
