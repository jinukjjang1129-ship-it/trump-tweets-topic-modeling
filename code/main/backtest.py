# code/main/backtest.py
import os
import pandas as pd
import matplotlib.pyplot as plt


def load_backtest_input(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 시간 파싱 (CSV 저장 시 tz가 문자열로 남아있을 수 있어서 안전하게 처리)
    df["px_time_et"] = pd.to_datetime(df["px_time_et"], errors="coerce")
    df = df.dropna(subset=["px_time_et"]).sort_values("px_time_et").reset_index(drop=True)
    return df


def make_buyhold(df: pd.DataFrame) -> pd.DataFrame:
    bh = df.dropna(subset=["open"]).copy()
    bh = bh.sort_values("px_time_et").reset_index(drop=True)
    bh["bh_ret"] = bh["open"].pct_change()
    bh["bh_cum"] = (1 + bh["bh_ret"]).cumprod() - 1
    return bh


def make_event_short(df: pd.DataFrame, horizon_min: int = 60) -> pd.DataFrame:
    col = f"ret_{horizon_min}m"
    if col not in df.columns:
        raise ValueError(f"'{col}' 컬럼이 없습니다. backtest_input.csv에 {col} 포함되어야 합니다.")

    tw = df.dropna(subset=[col]).copy()
    tw = tw.sort_values("px_time_et").reset_index(drop=True)
    tw["short_ret"] = -tw[col]
    tw["cum_short"] = tw["short_ret"].cumsum()
    return tw


def plot_rebased_from_first_available(
    bh: pd.DataFrame,
    tw: pd.DataFrame,
    out_path: str
):
    # 둘 다 존재하는 시점부터 시작
    plot_start = max(bh["px_time_et"].iloc[0], tw["px_time_et"].iloc[0])

    bh_plot = bh[bh["px_time_et"] >= plot_start].copy()
    tw_plot = tw[tw["px_time_et"] >= plot_start].copy()

    # 각자 시작값으로 리베이스
    bh0 = bh_plot["bh_cum"].iloc[0]
    bh_plot["bh_rebased"] = bh_plot["bh_cum"] - bh0

    tw0 = tw_plot["cum_short"].iloc[0]
    tw_plot["tw_rebased"] = tw_plot["cum_short"] - tw0

    plt.figure(figsize=(11, 5))
    plt.plot(bh_plot["px_time_et"], bh_plot["bh_rebased"],
             label="S&P 500 Buy & Hold (rebased @ first available)", linewidth=2)
    plt.plot(tw_plot["px_time_et"], tw_plot["tw_rebased"],
             label="Trade War Topic: 1h Short (Event Only, rebased)", linewidth=2)

    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlim(left=plot_start)
    plt.title("Rebased Comparison (Start from First Available Data)")
    plt.ylabel("Cumulative Return (Rebased)")
    plt.xlabel("Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_rebased_from_fixed_date(
    bh: pd.DataFrame,
    tw: pd.DataFrame,
    start_date: str,
    out_path: str
):
    # start_date는 문자열로 받아서 안전하게 파싱
    start_dt = pd.to_datetime(start_date, errors="coerce")
    if pd.isna(start_dt):
        raise ValueError("start_date 파싱 실패. 예: '2018-09-01' 또는 '2018-09-01 00:00:00'")

    bh_plot = bh[bh["px_time_et"] >= start_dt].copy()
    if len(bh_plot) == 0:
        raise ValueError("start_date 이후 Buy&Hold 데이터가 없습니다.")

    # BH 리베이스
    bh0 = bh_plot["bh_cum"].iloc[0]
    bh_plot["bh_rebased"] = bh_plot["bh_cum"] - bh0

    # 이벤트 숏은 start_date 이후 이벤트만 누적되게
    tw_plot = tw[tw["px_time_et"] >= start_dt].copy()
    if len(tw_plot) == 0:
        raise ValueError("start_date 이후 이벤트(short) 데이터가 없습니다.")
    tw_plot["tw_rebased"] = tw_plot["short_ret"].cumsum()

    plt.figure(figsize=(11, 5))
    plt.plot(bh_plot["px_time_et"], bh_plot["bh_rebased"],
             label=f"S&P 500 Buy & Hold (rebased @ {start_dt.date()})", linewidth=2)
    plt.plot(tw_plot["px_time_et"], tw_plot["tw_rebased"],
             label=f"Trade War Topic: 1h Short (Event Only, rebased @ {start_dt.date()})", linewidth=2)

    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlim(left=start_dt)
    plt.title(f"Rebased Comparison from {start_dt.date()} (Both Start at 0)")
    plt.ylabel("Cumulative Return (Rebased)")
    plt.xlabel("Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    # 경로 (repo 기준 상대경로)
    input_path = "data/processed/backtest_input.csv"

    # 결과 저장 폴더
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    df = load_backtest_input(input_path)
    bh = make_buyhold(df)
    tw = make_event_short(df, horizon_min=60)

    # 그래프 1) 둘 다 존재하는 첫 시점부터 리베이스
    plot_rebased_from_first_available(
        bh, tw,
        out_path=os.path.join(out_dir, "equity_curve_rebased_first_available.png")
    )

    # 그래프 2) 2018-09-01부터 리베이스
    plot_rebased_from_fixed_date(
        bh, tw,
        start_date="2018-09-01",
        out_path=os.path.join(out_dir, "equity_curve_rebased_2018-09.png")
    )

    print("✅ 완료: results/ 폴더에 그래프 2개 저장됨")


if __name__ == "__main__":
    main()
