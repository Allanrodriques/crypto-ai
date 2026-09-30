# crypto-ml-predictor

A reproducible, leakage-safe research pipeline that predicts whether **BTCUSDT
will be up or down six hours from now** on `1h` candles, trains five models,
and reports what they are actually worth.

> **Research only.** This is not trading software. There is no broker
> integration, no order routing, no live execution, and no position sizing.
> `probability_up` is a model score, not a guaranteed probability of profit.

---

## Headline result: the models barely beat a coin flip, and the strategy loses money

From the run recorded in `reports/metrics/summary.md` (BTCUSDT 1h,
2022-01-09 → 2026-09-28, 41,342 rows, test period untouched until the end):

| Model | ROC-AUC | PR-AUC | Test accuracy | Backtest return |
| --- | ---: | ---: | ---: | ---: |
| majority baseline | n/a | 0.2457 | 0.7543 | — |
| momentum baseline | 0.4784 | 0.2382 | 0.4844 | — |
| **logistic regression (selected)** | **0.5911** | **0.3259** | 0.6332 | **−17.50%** |
| random forest | 0.5987 | 0.3145 | 0.6358 | — |
| xgboost | 0.5948 | 0.3112 | 0.7536 | — |

The selected strategy returned **−17.50%** against **−8.21%** for simply
holding BTC over the same window, with a 30.20% max drawdown and 2,744.94 in
paid costs. Cross-validated ROC-AUC across five expanding-window folds is
0.570 for the selected model, 0.563 for random forest, 0.565 for xgboost and
0.485 for momentum.

**Read that honestly:** these technical indicators carry a very weak signal
for this target. A ROC-AUC of ~0.59 is far from the 0.5 coin flip, but nowhere
near tradable, and after a 10 bps round trip the edge is entirely consumed.
The `majority_baseline` "wins" on accuracy only by predicting the majority
class 75% of the time — which is exactly why accuracy is a misleading metric
here and ROC-AUC is reported beside it. No model, threshold or feature set was
tuned on the test period, and the negative result stands as the finding.

The pipeline is the deliverable here, not the return. The value is that the
number is trustworthy: no lookahead, no purging mistakes, no threshold picked
on test data, and costs charged on both sides of every trade.

---

## Setup

Requires Python 3.11+ (developed and tested on 3.12).

```bash
cd crypto-ml-predictor
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # optional; nothing here is required
```

On macOS, XGBoost needs OpenMP:

```bash
brew install libomp
```

No Binance API key is needed. Market data is public.

---

## Quick start

```bash
# everything: download -> validate -> features -> dataset -> train -> evaluate -> predict
python scripts/run_pipeline.py

# fast sanity run on ~6 months (own config, own output dirs, no network re-fetch)
python scripts/run_pipeline.py --config config/smoke.yaml

# stage-by-stage
python scripts/download_data.py     --config config/config.yaml
python scripts/build_dataset.py     --config config/config.yaml
python scripts/train_model.py       --config config/config.yaml
python scripts/evaluate_model.py    --config config/config.yaml

# useful flags for run_pipeline.py
--models random_forest xgboost   # subset of models
--no-cv                          # skip cross-validation (much faster)
--no-download                    # reuse data/raw, go straight to features
--no-plots --no-predict          # trim the tail of the pipeline
```

Everything is driven by `config/config.yaml` — symbol, interval, dates,
features, splits, model hyper-parameters and backtest rules. No period,
threshold, ratio or seed is hard-coded in `src/`.

---

## Methodology

### 1. Data

`src/data/` pulls public klines from `data-api.binance.vision` (falling back
to `api.binance.com` and `api-gcp.binance.com`) with retries, rate limiting,
resumable checkpoints and atomic Parquet writes. `BTCUSDT_1h` for 2022→now is
~41.5k candles, fetched in ~45 requests.

`src/data/validator.py` then reports on the data and **never silently repairs
it**. Checks cover duplicates, ordering, interval alignment, nulls,
non-positive prices, OHLC consistency, volume, future-dated candles, gaps,
extreme moves and volume-less runs.

Two findings from the real dataset, kept as findings rather than patched over:

- **One candle is genuinely missing**: `2023-03-24 13:00 UTC`. Verified against
  the Binance API directly — the candle does not exist at the source, so it is
  a permanent upstream hole, not a downloader bug. Reported as an
  error-severity failure, which is why the run prints `Validation: FAIL`.
- **The `price_outliers` check was recalibrated after a false positive.** On
  the first full run it flagged 3 candles as corrupt because of a robust
  z-score above 20. Hourly crypto returns are fat-tailed and mostly tiny, so
  the MAD-derived scale is ~0.1% and a perfectly ordinary **7.25% hourly move
  looks like ~70 sigma**. A data-quality check that fires on real market moves
  gets ignored, which is worse than not having one. It now requires a candle
  to break *both* a robust z-score limit *and* an absolute
  `validation.price_move_max_log_return` floor (default 0.35 ≈ +42%/−30%).
  Retune that floor per interval and asset. `tests/test_binance_client.py`
  pins both behaviours.

### 2. Features — 39, all causal

`src/features/feature_engineering.py` builds 39 features in eight groups:

| Group | n | Features |
| --- | ---: | --- |
| returns | 5 | `return_1h` … `return_24h` |
| moving_averages | 7 | `sma_10`…`sma_200`, `ema_12`, `ema_26` |
| ma_relationships | 5 | `price_vs_sma20/50/200`, `sma20_vs_sma50`, `sma50_vs_sma200` |
| momentum | 5 | `rsi_14`, `macd`, `macd_signal`, `macd_histogram`, `roc_12` |
| volatility | 4 | `atr_14`, `atr_percent`, `rolling_volatility_20/50` |
| bollinger | 5 | `bb_middle`, `bb_upper`, `bb_lower`, `bb_width`, `bb_position` |
| volume | 3 | `volume_sma_20`, `volume_ratio`, `volume_change` |
| candle_structure | 5 | `candle_body`, `candle_range`, `upper_wick`, `lower_wick`, `body_to_range` |

Feature names and documentation are **derived from the same registry that
computes them**, so a period changed in `config/config.yaml` renames
`sma_50` → `sma_100` in the code, the schema and the docs together.

Causality is enforced and tested, not assumed. A feature at bar `t` may read
candle `t` and earlier, never `t+1`. `tests/test_features.py` verifies this by
copying the frame, poisoning future rows, and asserting the output is
bit-identical; a deliberate one-candle shift in the engine was caught this way
during development.

### 3. Target and splits

`target[t] = 1 if close[t+6]/close[t] - 1 >= 0.005 else 0`

`target` is the only place in the codebase permitted to look forward. The last
`horizon` rows have no resolvable label and are dropped.

Splits are chronological 70/15/15 and never shuffled. Non-final splits are
purged by exactly `horizon` rows so a training label can never resolve inside
the validation or test period:

| Period | Rows | Start | End | Share UP |
| --- | ---: | --- | --- | ---: |
| train | 28,933 | 2022-01-09 | 2025-04-28 | 26.7% |
| validation | 6,195 | 2025-04-29 | 2026-01-12 | 23.7% |
| test | 6,202 | 2026-01-12 | 2026-09-28 | 24.6% |

Model selection uses the validation period only. The selected model is then
refit on train+validation, that refit is what gets saved, and **the test
period is touched once, at the end, for reporting.**

### 4. Models

`majority_baseline` and `momentum_baseline` exist as honest reference points,
not decoration — a majority baseline that scores 0.7543 accuracy is what makes
the real models' 0.63 look weak instead of impressive. `logistic_regression`
(with a `StandardScaler` pipeline), `random_forest` and `xgboost` follow, with
purged expanding-window cross-validation.

### 5. Backtest

Simulation rules, deliberately conservative about what they claim:

- signal computed on candle `t`, fill at **`open[t+1]`** — never at `close[t]`
- long only, no leverage, one position at a time, no overlapping trades
- exit after the horizon, or earlier if a configured probability is hit
- 10 bps charged **per side**, so every trade pays entry and exit
- equity marked to market each bar; position units reset after a trade

That last point was a real bug. Sold `units` were never cleared, so equity
double-counted the position after every exit and inflated every derived
metric. `tests/test_backtest.py` now covers the fill timing, the cost model,
non-overlap, the equity curve and the reset.

### 6. Threshold honesty

The `probability_threshold` is a decision, not a discovery, so the pipeline
does not quietly pick one for you. Every backtest includes a
`threshold_sensitivity_on_validation` table computed from **train-only
validation probabilities captured before the refit** — never from a model
retrained on validation. From the recorded full run:

| Threshold | Trades | Return on validation |
| ---: | ---: | ---: |
| 0.50 | 626 | −76.27% |
| 0.55 | 415 | −59.04% |
| 0.60 | 253 | −52.39% |
| 0.65 | 151 | −26.14% |
| 0.70 | 82 | −28.26% |

Every threshold loses money, which is the most direct evidence that there is
no edge to tune into existence here. Raising the threshold reduces the number
of trades and the losses but never turns them positive. The configured 0.60
is not a recommendation; it is the configured default, and the table exists so
you can see the trade-off rather than take it on faith.

---

## Outputs

```
data/raw/          BTCUSDT_1h.parquet + .meta.json + .validation.json
data/processed/    dataset.parquet + dataset metadata
models/            <model>.joblib (train+validation refit), _cv.json, model_metadata.json
reports/metrics/   per-model metrics.json, calibration.csv, OOF predictions, summary.md
reports/backtests/ backtest.json, _trades.csv, _equity.csv
reports/plots/     10 figures
data/predictions/  scored history + latest prediction
```

The saved artifact for the selected model records `fitted_on:
train+validation`; the other models record `fitted_on: train`. This is
asserted after every run — an earlier version of the code trained the winner on
train+validation and then saved the *pre-refit* model, so the deployed artifact
disagreed with the reported one.

---

## Tests

```bash
python -m pytest -q -p no:logging
```

**125 tests, ~3.5s, no network.** The data layer runs against a `FakeSession`,
so the suite is deterministic and offline.

| File | Tests | Covers |
| --- | ---: | --- |
| `tests/test_binance_client.py` | 37 | pagination, retries, failover, rate limits, resume, validation, outlier calibration |
| `tests/test_features.py` | 30 | indicator math, schema, NaN hygiene, causality, truncation |
| `tests/test_dataset.py` | 28 | targets, purged splits, leakage assertions, persistence |
| `tests/test_backtest.py` | 30 | fill timing, costs, non-overlap, equity, early exit, threshold table |

Two bugs in this project were found by tests, not by inspection: the XGBoost
early-stopping split (it validated on the full `X` while fitting on a carved
subset) and the backtest `units` reset.

---

## Limitations

- **No edge.** See the headline. These features do not predict this horizon
  well enough to trade.
- Backtest is a simulation: flat costs, no slippage, no funding, no market
  impact, no partial fills, no live execution.
- One upstream candle is permanently missing from the dataset.
- Feature importance shows model reliance, not causality.
- Calibration is measured against the observed base rate; `probability_up` is
  a score, not a probability of profit.
- Costs, seeds and splits are configurable, so re-running with different
  settings will produce different numbers. The defaults and
  `random_state: 42` are what make the run above reproducible.

---

## V3: multi-horizon forward returns

V3 is a separate, self-contained research track under `src/v3/`, configured by
`config/v3.yaml`. It does not modify V1/V2 behaviour. Instead of classifying
"up or down in 6 hours", it regresses the **forward return** at eight horizons
(`1d, 3d, 7d, 14d, 30d, 60d, 90d, 180d`) and evaluates them with purged,
embargoed expanding-window walk-forward splits.

Run it with:

```bash
# BTCUSDT, excluding the derivatives bucket -- see the data note below
python scripts/run_v3.py --symbol BTCUSDT --exclude-buckets derivatives --n-splits 3
```

Full report, tables and figures: [`reports/experiments/v3/summary.md`](reports/experiments/v3/summary.md).

### V3 result: the naive zero baseline wins on all 8 horizons

This is a negative result, and it is the current state of the research. From
`reports/experiments/v3/` (BTCUSDT 1h, 2022-01-31 → 2026-09-28, 40,833 rows,
84 features, test folds untouched until scoring):

- **The naive `zero` baseline has the lowest RMSE at all 8 horizons.**
- **Every learned model has negative R²** at every horizon.
- Ridge does show **ranking** signal at longer horizons — Spearman IC ≈ **0.33
  at 30d** and ≈ **0.26 at 60d** — while still losing badly on RMSE and R².

The ranking signal is **not** evidence of profitable predictive performance. IC
measures ordering, not magnitude, and an ordering signal is routinely swamped
by unconditional volatility at these horizons. No cost, slippage or execution
model is applied to it, and no long/short P&L is claimed anywhere. The
`summary.md` carries a "naive-baseline check" section that states this in full
and is deliberately left intact.

### Known V3 issue: `min_train_fraction` is configured but not plumbed through

`config/v3.yaml` sets `v3.validation.min_train_fraction: 0.3`, but
`src/v3/walkforward.py` never passes that value to
`PurgedWalkForwardSplitter`, so **every run silently uses the splitter default
of `0.2`**. This is a known, unfixed bug — recorded here rather than patched, so
the reported results match the code that produced them. Anyone reproducing the
V3 numbers should either set the config value to `0.2` or pass
`min_train_fraction` explicitly; changing it alters fold geometry and therefore
the results, so it is a change worth making deliberately.

### V3 data note: BTC derivatives are excluded

BTC's cached derivatives data is truncated — `binance_futures_BTCUSDT.parquet`
holds 1,500 rows (ending 2022-03-04) and funding 500 rows (ending 2022-06-16),
while other symbols run to 2026-09-28. With `complete_rows_only: true`, keeping
the `derivatives` bucket leaves BTC with only 2,577 usable rows, which makes
30d/60d/90d/180d untrainable while the run still reports success. The recorded
run therefore excludes that bucket: 40,833 rows, 84 features, all 8 horizons
labelled. Raw data is unchanged and can be repaired by re-downloading.

### V3 artefacts in this repository

`reports/experiments/v3/` tracks the research output (CSV tables, PNG figures,
`summary.md`, `manifest.json`) — roughly 1 MB. The 48 fitted estimators the run
produced are **not** committed: eight `random_forest_*.joblib` files are ~1 GB
*each*, and they are fully regenerable by re-running the pipeline above.

---

## Layout

```
config/config.yaml        all tunables        config/smoke.yaml   6-month fast run
config/v3.yaml            V3 multi-horizon configuration
src/data/                 client, downloader, validator
src/features/             causal feature registry
src/dataset/              targets, purged splits
src/models/               baselines, logistic, random forest, xgboost
src/evaluation/           metrics, time-series CV, backtest, plots
src/pipeline/             train, evaluate, predict, run
src/v3/                   V3: horizons, targets, splits, dataset, models,
                          metrics, uncertainty, walkforward, analysis,
                          strategy, report, multiasset, pipeline, CLI
scripts/                  seven CLI entry points
tests/                    652 offline tests
reports/experiments/v3/   V3 research report, tables and figures
```
