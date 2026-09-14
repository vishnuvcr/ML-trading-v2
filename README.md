# NSE next-session surge research

This project tests an **unsupervised** anomaly-detection screen for NSE cash equities. A signal is generated after the close of session *t* and is intended for a next-session-open entry. The event is:

`(High[t+1] / Open[t+1]) - 1 >= 5%`

## What it does

- Fetches the active NSE `EQ` universe from NSE's published equity master list.
- Downloads unadjusted daily OHLCV from Yahoo Finance using `.NS` symbols.
- Excludes names below Rs. 10 or a trailing 20-session median turnover below Rs. 50 lakh.
- Fits an Isolation Forest only on feature rows available **before** every walk-forward test block. Target labels are never passed to the model.
- Uses each completed session's cross-sectional top five anomaly scores as candidates for the next session.
- Reports out-of-sample TP, FP, TN, FN, precision, recall, specificity, and false-positive rate, plus the latest candidates.

## Integrity constraints

No model can honestly guarantee 100% precision or 100% specificity for unknown future prices. This pipeline therefore does not claim either. If no threshold reaches the requested standard on previously untouched data, the correct result is that the standard was not met—not a retrofitted rule.

Features use OHLCV from session *t* or earlier. The target uses only session *t+1*, and it is unavailable to the signal-generation code. The `next_session` join deliberately requires the following market-wide session, so suspended/absent symbols are not silently treated as a later-date prediction.

## Run locally

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/nse_surge.py run --start 2016-01-01 --limit 0
```

`--limit 0` is the full active NSE EQ universe. Start with e.g. `--limit 100` to test connectivity. Results are written to `reports/`; raw Yahoo data is intentionally ignored by Git.

## GitHub Actions

The scheduled workflow runs around 15:45 IST on weekdays and can be launched manually from **Actions → NSE next-session surge research**. It uploads the reports as an artifact. A schedule only runs after the workflow is present on the default branch.

This is research software, not investment advice or an order-execution system. It does not connect to Paytm Money and never places trades.

