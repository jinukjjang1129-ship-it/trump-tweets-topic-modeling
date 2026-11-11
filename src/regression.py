# code/main/regression.py
import os
import pandas as pd
import statsmodels.api as sm


# -----------------------
# 설정
# -----------------------
HORIZONS = [1, 10, 20, 30, 45, 60, 120]
MIN_GAP_MINUTES = 10

TOPIC_COL = "topic0"  # Trade War topic
TIME_COL = "datetime"


# -----------------------
# 유틸: overlap 제거
# -----------------------
def drop_overlaps(df, time_col, min_gap_minutes=10):
    df = df.sort_values(time_col).reset_index(drop=True).copy()
    keep = []
    last_time = None

    for t in df[time_col]:
        if last_time is None or (t - last_time) >= pd.Timedelta(minutes=min_gap_minutes):
            keep.append(True)
            last_time = t
        else:
            keep.append(False)

    return df.loc[keep].reset_index(drop=True)


# -----------------------
# main
# -----------------------
def main():

    # 입력 경로
    topic_path = "data/processed/reg_input_topic0_minute.csv"
    price_path = "data/processed/reg_input_price_returns.csv"

    # 결과 저장 폴더
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    # -----------------------
    # 데이터 로드
    # -----------------------
    topic = pd.read_csv(topic_path)
    price = pd.read_csv(price_path)

    topic[TIME_COL] = pd.to_datetime(topic[TIME_COL], errors="coerce")
    price[TIME_COL] = pd.to_datetime(price[TIME_COL], errors="coerce")

    topic = topic.dropna(subset=[TIME_COL, TOPIC_COL]).copy()
    price = price.dropna(subset=[TIME_COL]).copy()

    # -----------------------
    # 이벤트 정의 (median 기준)
    # -----------------------
    q = topic[TOPIC_COL].median()
    events = topic[topic[TOPIC_COL] >= q].copy()

    # -----------------------
    # 가격 데이터와 결합
    # -----------------------
    df = pd.merge_asof(
        events.sort_values(TIME_COL),
        price.sort_values(TIME_COL),
        on=TIME_COL,
        direction="forward"
    )

    df = df.dropna().reset_index(drop=True)

    # overlap 제거
    df = drop_overlaps(df, TIME_COL, MIN_GAP_MINUTES)

    # -----------------------
    # 회귀 실행
    # -----------------------
    rows = []

    for h in HORIZONS:
        ret_col = f"ret_{h}m"

        if ret_col not in df.columns:
            continue

        y = df[ret_col]
        X = sm.add_constant(df[[TOPIC_COL]])

        mask = y.notna()
        res = sm.RLM(
            y[mask],
            X[mask],
            M=sm.robust.norms.HuberT()
        ).fit()

        rows.append({
            "horizon_min": h,
            "events_used": int(res.nobs),
            "beta_topic0": float(res.params[TOPIC_COL]),
            "p_value": float(res.pvalues[TOPIC_COL])
        })

    results = pd.DataFrame(rows)

    # -----------------------
    # 저장
    # -----------------------
    out_path = os.path.join(out_dir, "regression_trade_war_topic0.csv")
    results.to_csv(out_path, index=False, encoding="utf-8")

    print("✅ 회귀 완료")
    print(f"결과 저장: {out_path}")
    print(results)


if __name__ == "__main__":
    main()
