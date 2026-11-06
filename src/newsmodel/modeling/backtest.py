"""
Simulate trading the model's predictions on the TEST split only (data the
model has never seen, chronologically after training/validation). Last
step of Job 3.

Answers "if I'd been running this live, would it have made money after
fees and slippage?" -- a different, harder question than "is test
accuracy above the naive baseline?"

Run (normally via job3-train):
    python -m newsmodel.modeling.backtest
"""

import pandas as pd
import joblib

from newsmodel import config
from newsmodel.modeling.dataset import DATASET_PATH
from newsmodel.modeling.train import MODEL_PATH, build_features, chronological_split, load_articles, prepare_rows

TRADES_PATH = config.PROCESSED_DIR / "backtest_trades.csv"


def simulate(test_df: pd.DataFrame, predictions, confidence):
    """
    On each "up" prediction, buy $1 notional at entry_price and sell at
    exit_price; on each "down", short $1 the same way. "flat" predictions,
    and predictions less sure than MIN_CONFIDENCE, are skipped (no trade).
    With LABEL_VS_MARKET, company trades are hedged: $1 the opposite way in
    MARKET_TICKER, so they earn the stock's move minus the market's, and pay
    costs on both legs.

    Ignores position sizing, concentration, and finite capital -- it tells
    you whether the SIGNAL has edge, not what a portfolio would return.

    Entry for news while the market is closed is the next official opening
    price, which you could get with an order placed before the open. For
    market_hours news it's the last price before the news, up to
    INTRADAY_BAR_MINUTES old -- slightly optimistic, since you'd trade a
    little after reading it. Compare them in the per-timing breakdown.
    """
    trades = []
    for (_, row), pred, sure in zip(test_df.iterrows(), predictions, confidence):
        if pred == "flat" or sure < config.MIN_CONFIDENCE:
            continue

        direction = 1 if pred == "up" else -1
        gross_return_pct = direction * row["trade_move"]

        # Slippage on both entry and exit, plus commission, per leg. Pessimistic on purpose.
        cost_pct = (config.SLIPPAGE_PCT * 2 + config.COMMISSION_PER_TRADE_PCT) * (2 if row["hedged"] else 1)
        net_return_pct = gross_return_pct - cost_pct

        trades.append({
            "ticker": row["ticker"],
            "news_timestamp": row["news_timestamp"],
            "news_session": row["news_session"],
            "entry": row["entry"],
            "role": row["role"],
            "prediction": pred,
            "actual_label": row["label"],
            "gross_return_pct": gross_return_pct,
            "net_return_pct": net_return_pct,
            "correct_direction": (pred == row["label"]),
        })

    return pd.DataFrame(trades)


def summarize(trades: pd.DataFrame):
    if trades.empty:
        print("No trades were made (model predicted 'flat' for everything in the test set,")
        print("or the test set is empty).")
        return

    n = len(trades)
    win_rate = (trades["net_return_pct"] > 0).mean()
    avg_net_return = trades["net_return_pct"].mean()
    total_net_return = trades["net_return_pct"].sum()  # naive sum, not compounded
    direction_accuracy = trades["correct_direction"].mean()

    print(f"\n=== Backtest summary (TEST split, {n} trades) ===")
    print(f"Directional accuracy (excl. 'flat' calls): {direction_accuracy:.3f}")
    print(f"Win rate after frictions:                   {win_rate:.3f}")
    print(f"Average net return per trade:               {avg_net_return:.3f}%")
    print(f"Naive summed net return across all trades:  {total_net_return:.2f}%")
    print("\n(The summed return is NOT a realistic portfolio return -- it assumes every")
    print(" trade gets fresh capital. It's a signal check, not a P&L projection.)")

    print("\nBreakdown by predicted direction:")
    print(trades.groupby("prediction")["net_return_pct"].agg(["count", "mean", "sum"]))

    print("\nBreakdown by when the news came out (market_hours entries are up to")
    print(f"{config.INTRADAY_BAR_MINUTES} minutes before the news, so slightly optimistic):")
    print(trades.groupby("news_session")["net_return_pct"].agg(["count", "mean", "sum"]))

    print("\nBreakdown by when the trade happened (at_news = right away, next_open = at the next open):")
    print(trades.groupby("entry")["net_return_pct"].agg(["count", "mean", "sum"]))

    print("\nBreakdown by how the news reaches the ticker (index = an INDEX_TICKERS row):")
    print(trades.groupby("role")["net_return_pct"].agg(["count", "mean", "sum"]))

    index_trades = trades[trades["ticker"].isin(config.INDEX_TICKERS)]
    if not index_trades.empty:
        print("\nBreakdown per market/sector fund:")
        print(index_trades.groupby("ticker")["net_return_pct"].agg(["count", "mean", "sum"]))


def main():
    if not MODEL_PATH.exists():
        raise SystemExit("No trained model yet. Run python -m newsmodel.modeling.train (or job3-train) first.")

    saved = joblib.load(MODEL_PATH)
    model, featurizer = saved["model"], saved["featurizer"]

    df = prepare_rows(pd.read_parquet(DATASET_PATH))
    _, _, test = chronological_split(df)

    proba = model.predict_proba(build_features(test, featurizer, fit=False, texts=load_articles()))
    predictions = model.classes_[proba.argmax(axis=1)]
    print(f"Trading predictions at least {config.MIN_CONFIDENCE:.0%} sure (MIN_CONFIDENCE)")

    trades = simulate(test, predictions, proba.max(axis=1))
    summarize(trades)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    trades.to_csv(TRADES_PATH, index=False)
    print(f"\nSaved per-trade detail to {TRADES_PATH} for inspection.")


if __name__ == "__main__":
    main()
