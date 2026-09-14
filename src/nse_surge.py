"""Leak-free NSE next-session 5% surge research pipeline.

Signals are created after the close of session t.  The target is true when the
high of the *next NSE session* is at least 5% above that session's opening
price.  Isolation Forest is fitted only on observations available before each
walk-forward test block.  It never receives the target labels.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

LOG = logging.getLogger("nse_surge")
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
NSE_EQUITY_LIST = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
FEATURES = [
    "ret_1d", "ret_5d", "ret_20d", "gap", "intraday_return", "range",
    "volume_ratio_20", "turnover_ratio_20", "volatility_20", "rsi_14",
    "close_vs_sma_20", "close_vs_high_20", "range_position_20",
]


def nse_symbols(limit: int = 0) -> list[str]:
    """Return active NSE EQ-series symbols from NSE's published master list."""
    response = requests.get(NSE_EQUITY_LIST, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    master = pd.read_csv(pd.io.common.StringIO(response.text))
    master.columns = master.columns.astype(str).str.strip()
    symbols = (
        master.loc[master["SERIES"].eq("EQ"), "SYMBOL"]
        .dropna().astype(str).str.strip().str.upper().drop_duplicates().tolist()
    )
    if not symbols:
        raise RuntimeError("NSE EQ master list returned no symbols")
    return symbols[:limit] if limit else symbols


def _ticker_frame(download: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if download.empty or not isinstance(download.columns, pd.MultiIndex):
        return pd.DataFrame()
    for level in range(download.columns.nlevels):
        if ticker in download.columns.get_level_values(level):
            frame = download.xs(ticker, level=level, axis=1, drop_level=True).copy()
            frame.columns = [str(c).lower() for c in frame.columns]
            needed = {"open", "high", "low", "close", "volume"}
            if needed.issubset(frame.columns):
                frame = frame.loc[:, ["open", "high", "low", "close", "volume"]].dropna(how="all")
                frame.index = pd.to_datetime(frame.index).tz_localize(None)
                return frame
    return pd.DataFrame()


def download_history(symbols: Iterable[str], start: str, end: str | None, batch_size: int) -> pd.DataFrame:
    """Download raw, unadjusted daily OHLCV; cached symbol files make retries safe."""
    RAW.mkdir(parents=True, exist_ok=True)
    requested = [f"{symbol}.NS" for symbol in symbols]
    saved: list[dict[str, object]] = []
    for offset in range(0, len(requested), batch_size):
        batch = requested[offset : offset + batch_size]
        missing = [t for t in batch if not (RAW / f"{t}.parquet").exists()]
        if not missing:
            continue
        LOG.info("Downloading %s–%s of %s", offset + 1, offset + len(batch), len(requested))
        try:
            prices = yf.download(missing, start=start, end=end, group_by="ticker", auto_adjust=False,
                                 actions=False, threads=True, progress=False, timeout=30)
        except Exception as exc:  # retain completed batches and continue
            LOG.warning("Batch failed: %s", exc)
            continue
        for ticker in missing:
            frame = _ticker_frame(prices, ticker)
            if len(frame) >= 60:
                frame.to_parquet(RAW / f"{ticker}.parquet")
                saved.append({"ticker": ticker, "rows": len(frame)})
        time.sleep(0.25)
    manifest = pd.DataFrame(saved)
    manifest.to_csv(REPORTS / "download_manifest.csv", index=False)
    return manifest


def _features_for_symbol(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.sort_index().copy()
    x.index.name = "date"
    x["symbol"] = symbol.removesuffix(".NS")
    previous_close = x["close"].shift(1)
    x["ret_1d"] = x["close"].pct_change()
    x["ret_5d"] = x["close"].pct_change(5)
    x["ret_20d"] = x["close"].pct_change(20)
    x["gap"] = x["open"].div(previous_close).sub(1)
    x["intraday_return"] = x["close"].div(x["open"]).sub(1)
    x["range"] = x["high"].div(x["low"]).sub(1)
    x["turnover"] = x["close"] * x["volume"]
    x["volume_ratio_20"] = x["volume"].div(x["volume"].rolling(20, min_periods=20).median())
    x["turnover_ratio_20"] = x["turnover"].div(x["turnover"].rolling(20, min_periods=20).median())
    x["volatility_20"] = x["ret_1d"].rolling(20, min_periods=20).std()
    delta = x["close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    rs = up.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().div(
        down.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().replace(0, np.nan)
    )
    x["rsi_14"] = 100 - 100 / (1 + rs)
    sma20 = x["close"].rolling(20, min_periods=20).mean()
    high20 = x["high"].rolling(20, min_periods=20).max()
    low20 = x["low"].rolling(20, min_periods=20).min()
    x["close_vs_sma_20"] = x["close"].div(sma20).sub(1)
    x["close_vs_high_20"] = x["close"].div(high20).sub(1)
    x["range_position_20"] = x["close"].sub(low20).div(high20.sub(low20).replace(0, np.nan))
    x["median_turnover_20"] = x["turnover"].rolling(20, min_periods=20).median()
    return x.reset_index()


def build_features() -> pd.DataFrame:
    files = sorted(RAW.glob("*.parquet"))
    if not files:
        raise RuntimeError("No raw histories found. Run download first.")
    panel = pd.concat([_features_for_symbol(f.stem, pd.read_parquet(f)) for f in files], ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    sessions = pd.Series(sorted(panel["date"].unique()))
    next_session = dict(zip(sessions.iloc[:-1], sessions.iloc[1:]))
    panel["next_session"] = panel["date"].map(next_session)
    future = panel[["symbol", "date", "open", "high"]].rename(
        columns={"date": "next_session", "open": "next_open", "high": "next_high"}
    )
    panel = panel.merge(future, on=["symbol", "next_session"], how="left", validate="many_to_one")
    panel["next_day_return_from_open"] = panel["next_high"].div(panel["next_open"]).sub(1)
    panel["target_5pct"] = np.where(panel["next_high"].notna(), (panel["next_day_return_from_open"] >= .05).astype(int), np.nan)
    # A practical entry-universe constraint, applied using only session-t information.
    panel["eligible"] = (panel["close"] >= 10) & (panel["median_turnover_20"] >= 5_000_000)
    panel.loc[:, FEATURES] = panel[FEATURES].replace([np.inf, -np.inf], np.nan)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PROCESSED / "features.parquet", index=False)
    return panel


def model() -> object:
    return make_pipeline(
        SimpleImputer(strategy="median"), RobustScaler(),
        IsolationForest(n_estimators=250, max_samples=50_000, contamination="auto", n_jobs=-1, random_state=7),
    )


def add_daily_signals(scores: pd.DataFrame, top_per_day: int) -> pd.DataFrame:
    out = scores.copy()
    out["daily_rank"] = out.groupby("date")["anomaly_score"].rank(method="first", ascending=False)
    out["signal"] = out["eligible"] & out["daily_rank"].le(top_per_day)
    return out


def walk_forward(panel: pd.DataFrame, train_sessions: int, test_sessions: int, top_per_day: int) -> pd.DataFrame:
    data = panel.dropna(subset=FEATURES).copy()
    dates = np.array(sorted(data["date"].unique()))
    if len(dates) <= train_sessions + test_sessions:
        raise RuntimeError("Insufficient sessions for requested walk-forward windows")
    scored: list[pd.DataFrame] = []
    for begin in range(train_sessions, len(dates), test_sessions):
        test_dates = dates[begin : begin + test_sessions]
        if len(test_dates) == 0:
            break
        train = data[data["date"] < test_dates[0]]
        test = data[data["date"].isin(test_dates)]
        # The detector is trained without target_5pct; this is deliberately unsupervised.
        fitted = model().fit(train[FEATURES])
        block = test[["date", "symbol", "eligible", "target_5pct", "next_session", "next_day_return_from_open"]].copy()
        block["anomaly_score"] = -fitted.decision_function(test[FEATURES])
        scored.append(block)
        LOG.info("Scored %s through %s", pd.Timestamp(test_dates[0]).date(), pd.Timestamp(test_dates[-1]).date())
    result = add_daily_signals(pd.concat(scored, ignore_index=True), top_per_day)
    return result[result["target_5pct"].notna()].copy()


def metric_report(scored: pd.DataFrame) -> dict[str, object]:
    actual = scored["target_5pct"].astype(bool)
    predicted = scored["signal"].astype(bool)
    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    tn = int((~predicted & ~actual).sum())
    fn = int((~predicted & actual).sum())
    return {
        "definition": "next-session High / next-session Open - 1 >= 5%",
        "rows_evaluated": int(len(scored)), "signals": int(predicted.sum()),
        "true_positives": tp, "false_positives": fp, "true_negatives": tn, "false_negatives": fn,
        "precision": None if tp + fp == 0 else tp / (tp + fp),
        "recall": None if tp + fn == 0 else tp / (tp + fn),
        "specificity": None if tn + fp == 0 else tn / (tn + fp),
        "false_positive_rate": None if tn + fp == 0 else fp / (tn + fp),
    }


def latest_signals(panel: pd.DataFrame, top_per_day: int) -> pd.DataFrame:
    data = panel.dropna(subset=FEATURES).copy()
    latest_date = data["date"].max()
    fitted = model().fit(data[data["date"] < latest_date][FEATURES])
    latest = data[data["date"].eq(latest_date)].copy()
    latest["anomaly_score"] = -fitted.decision_function(latest[FEATURES])
    latest = add_daily_signals(latest, top_per_day)
    return latest.loc[latest["signal"], ["date", "symbol", "close", "anomaly_score", "daily_rank", "next_session"]].sort_values("daily_rank")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["download", "run"])
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--limit", type=int, default=0, help="0 means all active NSE EQ symbols")
    parser.add_argument("--batch-size", type=int, default=75)
    parser.add_argument("--train-sessions", type=int, default=504)
    parser.add_argument("--test-sessions", type=int, default=21)
    parser.add_argument("--top-per-day", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    REPORTS.mkdir(parents=True, exist_ok=True)
    # Avoid relying on a user-profile cache location; this is also CI-safe.
    yf.set_tz_cache_location(str(ROOT / "data" / "yfinance_cache"))
    symbols = nse_symbols(args.limit)
    LOG.info("NSE EQ universe: %s symbols", len(symbols))
    download_history(symbols, args.start, args.end, args.batch_size)
    if args.command == "download":
        return
    panel = build_features()
    scored = walk_forward(panel, args.train_sessions, args.test_sessions, args.top_per_day)
    scored.to_parquet(REPORTS / "walk_forward_scores.parquet", index=False)
    report = metric_report(scored)
    (REPORTS / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    latest_signals(panel, args.top_per_day).to_csv(REPORTS / "latest_candidates.csv", index=False)
    LOG.info("Metrics: %s", json.dumps(report))


if __name__ == "__main__":
    main()

