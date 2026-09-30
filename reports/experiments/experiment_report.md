# V2 multi-factor experiment report

Every number below is a **measurement** on a held-out chronological test block. None of it establishes causation, and none of it was tuned on the test block. Where a feature group is compared with the baseline, the comparison is a paired moving-block bootstrap on the common rows rather than a subtraction of two independently estimated AUCs.

## Data availability

Coverage is measured on the hourly feature grid after as-of alignment with each source's publication lag. A source that is absent is reported as absent rather than imputed.

| source | rows | covered | coverage pct | first covered | last covered |
| --- | --- | --- | --- | --- | --- |
| binance_spot | 41550 | 41550 | 100.00 | 2022-01-01 00:00:00+00:00 | 2026-09-28 06:00:00+00:00 |
| binance_funding | 41550 | 41550 | 100.00 | 2022-01-01 00:00:00+00:00 | 2026-09-28 06:00:00+00:00 |
| binance_futures | 41550 | 41550 | 100.00 | 2022-01-01 00:00:00+00:00 | 2026-09-28 06:00:00+00:00 |
| fear_greed | 41550 | 41550 | 100.00 | 2022-01-01 00:00:00+00:00 | 2026-09-28 06:00:00+00:00 |

## Headline results

| experiment | n features | rows | period | test roc auc | test pr auc | wf roc auc mean | wf roc auc min | threshold | n trades | total return | excess vs buy hold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-00-OHLCV-TECHNICAL | 39 | 41342 | 2026-01-12 15:00:00 UTC -> 2026-09-28 00:00:00 UTC | 0.6012 | 0.3190 | 0.5919 | 0.5288 | 0.4400 | 14 | -0.0874 | -0.0035 |
| EXP-01-DERIVATIVES | 62 | 40102 | 2026-01-20 09:00:00 UTC -> 2026-09-28 00:00:00 UTC | 0.5967 | 0.3176 | 0.5831 | 0.4994 | 0.4200 | 14 | 0.0051 | 0.0827 |
| EXP-02-SENTIMENT | 51 | 40822 | 2026-01-15 21:00:00 UTC -> 2026-09-28 00:00:00 UTC | 0.5951 | 0.3072 | 0.5744 | 0.5484 | 0.4200 | 22 | -0.0153 | 0.1062 |
| EXP-03-MICROSTRUCTURE | 61 | 41061 | 2026-01-14 09:00:00 UTC -> 2026-09-28 00:00:00 UTC | 0.5996 | 0.3156 | 0.5806 | 0.5454 | 0.4600 | 0 | 0.0000 | 0.1169 |
| EXP-04-COMBINED | 107 | 40101 | 2026-01-20 09:00:00 UTC -> 2026-09-28 00:00:00 UTC | 0.5996 | 0.3212 | 0.5795 | 0.5074 | 0.4200 | 81 | -0.1264 | -0.0488 |

## Hypotheses under test

- **EXP-00-OHLCV-TECHNICAL** - Immutable reference point. Feature set is pinned by tests/test_features.py.
- **EXP-01-DERIVATIVES** - Open interest, long/short ratio and taker-ratio history are capped at ~30 days by Binance and are excluded rather than reconstructed or forward-filled.
- **EXP-02-SENTIMENT** - A reading for day D is treated as knowable from D+1 00:00 UTC.
- **EXP-03-MICROSTRUCTURE** - Order-book spread and depth have no historical endpoint on Binance. Book-shape features are therefore absent by design rather than reconstructed from snapshots.
- **EXP-04-COMBINED** - Higher-timeframe context uses only completed 4h/1d candles, shifted by one full bucket so no in-progress candle is ever read.

## Comparability

- EXP-01-DERIVATIVES has a unique feature manifest `09037eeb664eca5a`; compare it to the baseline only through the paired bootstrap, not by subtracting headline AUC.
- EXP-02-SENTIMENT has a unique feature manifest `1241497905abf074`; compare it to the baseline only through the paired bootstrap, not by subtracting headline AUC.
- EXP-00-OHLCV-TECHNICAL has a unique feature manifest `abbbc511924e4d10`; compare it to the baseline only through the paired bootstrap, not by subtracting headline AUC.
- EXP-04-COMBINED has a unique feature manifest `c5efc44af58a67e3`; compare it to the baseline only through the paired bootstrap, not by subtracting headline AUC.
- EXP-03-MICROSTRUCTURE has a unique feature manifest `d41026cc92661e22`; compare it to the baseline only through the paired bootstrap, not by subtracting headline AUC.
- Experiments cover different date ranges (2022-01-09 08:00:00 UTC, 2022-01-21 00:00:00 UTC, 2022-01-31 00:00:00 UTC, 2022-03-02 00:00:00 UTC); raw AUC differences across different periods are not comparable and the ablation table restricts to the common rows.

## Threshold selection

The probability cut is chosen on the **validation** block and then frozen; the test block never informs it. Probabilities used for that choice come from the train-only fit, so the cut is not tuned in-sample.

### EXP-00-OHLCV-TECHNICAL

- selected threshold: `0.44` (objective: excess_vs_buy_hold)
- validation trades at that cut: 30
- validation objective value: -0.0328
- first / second validation half: `0.38` / `0.46`  (disagree - treat the cut as unstable)
- test trades after freezing: 14

### EXP-01-DERIVATIVES

- selected threshold: `0.42` (objective: excess_vs_buy_hold)
- validation trades at that cut: 36
- validation objective value: 0.0137
- first / second validation half: `0.38` / `0.46`  (disagree - treat the cut as unstable)
- test trades after freezing: 14

### EXP-02-SENTIMENT

- selected threshold: `0.42` (objective: excess_vs_buy_hold)
- validation trades at that cut: 43
- validation objective value: -0.1433
- first / second validation half: `0.40` / `0.44`
- test trades after freezing: 22

### EXP-03-MICROSTRUCTURE

- selected threshold: `0.46` (objective: excess_vs_buy_hold)
- validation trades at that cut: 40
- validation objective value: -0.0018
- first / second validation half: `0.46` / `0.52`  (disagree - treat the cut as unstable)
- test trades after freezing: 0

### EXP-04-COMBINED

- selected threshold: `0.42` (objective: excess_vs_buy_hold)
- validation trades at that cut: 43
- validation objective value: -0.0490
- first / second validation half: `0.36` / `0.46`  (disagree - treat the cut as unstable)
- test trades after freezing: 81

## Backtest on the test block

Fees and slippage are charged on every leg. `excess` is the strategy return minus buy-and-hold over the same rows, which is the only comparison that nets out the market's own direction.

| experiment | n trades | total return | buy hold return | excess vs buy hold | win rate | profit factor | max drawdown | sharpe ratio | sortino ratio | calmar ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-00-OHLCV-TECHNICAL | 14 | -0.0874 | -0.0821 | -0.0035 | 0.2857 | 0.5372 | 0.1601 | -1.0185 | -0.1341 | -0.7569 |
| EXP-01-DERIVATIVES | 14 | 0.0051 | -0.0758 | 0.0827 | 0.7143 | 1.0398 | 0.1360 | 0.1221 | 0.0168 | 0.0546 |
| EXP-02-SENTIMENT | 22 | -0.0153 | -0.1197 | 0.1062 | 0.5455 | 0.9204 | 0.1389 | -0.0943 | -0.0162 | -0.1567 |
| EXP-03-MICROSTRUCTURE | 0 | 0.0000 | -0.1151 | 0.1169 | - | - | 0.0000 | - | - | - |
| EXP-04-COMBINED | 81 | -0.1264 | -0.0758 | -0.0488 | 0.5185 | 0.7596 | 0.2395 | -0.8639 | -0.2808 | -0.7461 |

## Performance by market regime

Regimes are defined by an explicit trailing trend and volatility rule, not by hindsight. A single period is not evidence of a stable edge; these splits are reported so the concentration of any apparent effect is visible.

### EXP-00-OHLCV-TECHNICAL

**trend regime**


| regime | n | share | roc auc | pr auc |
| --- | --- | --- | --- | --- |
| bear | 4492 | 72.4% | 0.5991 | 0.3294 |
| bull | 797 | 12.9% | 0.5574 | 0.2789 |
| sideways | 913 | 14.7% | 0.6149 | 0.2959 |
| unknown | - | - | - | - |

**vol regime**


| regime | n | share | roc auc | pr auc |
| --- | --- | --- | --- | --- |
| high_vol | 3367 | 54.3% | 0.5829 | 0.3469 |
| low_vol | 2835 | 45.7% | 0.5898 | 0.2606 |
| unknown | - | - | - | - |

### EXP-01-DERIVATIVES

**trend regime**


| regime | n | share | roc auc | pr auc |
| --- | --- | --- | --- | --- |
| bear | 4306 | 71.6% | 0.5960 | 0.3310 |
| bull | 797 | 13.2% | 0.5550 | 0.2768 |
| sideways | 913 | 15.2% | 0.5905 | 0.2636 |
| unknown | - | - | - | - |

**vol regime**


| regime | n | share | roc auc | pr auc |
| --- | --- | --- | --- | --- |
| high_vol | 3367 | 56.0% | 0.5803 | 0.3439 |
| low_vol | 2649 | 44.0% | 0.5809 | 0.2590 |
| unknown | - | - | - | - |

### EXP-02-SENTIMENT

**trend regime**


| regime | n | share | roc auc | pr auc |
| --- | --- | --- | --- | --- |
| bear | 4414 | 72.1% | 0.5901 | 0.3167 |
| bull | 797 | 13.0% | 0.5591 | 0.2832 |
| sideways | 913 | 14.9% | 0.6155 | 0.2708 |
| unknown | - | - | - | - |

**vol regime**


| regime | n | share | roc auc | pr auc |
| --- | --- | --- | --- | --- |
| high_vol | 3367 | 55.0% | 0.5587 | 0.3285 |
| low_vol | 2757 | 45.0% | 0.6014 | 0.2594 |
| unknown | - | - | - | - |

### EXP-03-MICROSTRUCTURE

**trend regime**


| regime | n | share | roc auc | pr auc |
| --- | --- | --- | --- | --- |
| bear | 4450 | 72.2% | 0.5963 | 0.3248 |
| bull | 797 | 12.9% | 0.5674 | 0.2966 |
| sideways | 913 | 14.8% | 0.6098 | 0.2930 |
| unknown | - | - | - | - |

**vol regime**


| regime | n | share | roc auc | pr auc |
| --- | --- | --- | --- | --- |
| high_vol | 3367 | 54.7% | 0.5723 | 0.3390 |
| low_vol | 2793 | 45.3% | 0.6003 | 0.2664 |
| unknown | - | - | - | - |

### EXP-04-COMBINED

**trend regime**


| regime | n | share | roc auc | pr auc |
| --- | --- | --- | --- | --- |
| bear | 4306 | 71.6% | 0.5955 | 0.3340 |
| bull | 797 | 13.2% | 0.5807 | 0.2935 |
| sideways | 913 | 15.2% | 0.6066 | 0.2715 |
| unknown | - | - | - | - |

**vol regime**


| regime | n | share | roc auc | pr auc |
| --- | --- | --- | --- | --- |
| high_vol | 3367 | 56.0% | 0.5750 | 0.3449 |
| low_vol | 2649 | 44.0% | 0.6006 | 0.2720 |
| unknown | - | - | - | - |

## Secondary studies

### Feature-group ablation

Each arm is trained under the same protocol and compared with the baseline on the **common** test rows only, using a paired moving-block bootstrap. A confidence interval spanning zero means the measured difference is not distinguishable from sampling noise; it does not mean the features are useless, only that this data cannot show it.

| variant | n features | n rows | test roc auc | delta roc auc | ci low | ci high | verdict | n trades | excess vs buy hold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 39 | 41342 | 0.6005 | 0.0000 | 0.0000 | 0.0000 | reference | 14 | -0.0035 |
| add_derivatives | 62 | 40102 | 0.5967 | -0.0039 | -0.0107 | 0.0017 | indistinguishable | 14 | 0.0827 |
| add_sentiment | 51 | 40822 | 0.5943 | -0.0062 | -0.0178 | 0.0057 | indistinguishable | 22 | 0.1062 |
| add_microstructure | 61 | 41061 | 0.5962 | -0.0043 | -0.0103 | 0.0019 | indistinguishable | 0 | 0.1169 |
| add_context | 50 | 41342 | 0.6022 | 0.0017 | -0.0037 | 0.0066 | indistinguishable | 23 | 0.0509 |
| drop_derivatives | 84 | 40821 | 0.5970 | -0.0035 | -0.0116 | 0.0050 | indistinguishable | 14 | 0.0944 |
| drop_sentiment | 95 | 40101 | 0.5971 | -0.0034 | -0.0121 | 0.0047 | indistinguishable | 22 | -0.1133 |
| drop_microstructure | 85 | 40102 | 0.5995 | -0.0010 | -0.0099 | 0.0072 | indistinguishable | 26 | 0.0485 |
| drop_context | 96 | 40101 | 0.6000 | -0.0005 | -0.0089 | 0.0067 | indistinguishable | 16 | 0.0777 |
| full_union | 107 | 40101 | 0.5996 | -0.0009 | -0.0135 | 0.0102 | indistinguishable | 81 | -0.0488 |

### Target-definition comparison

Different label definitions answer different questions, so their scores are not comparable to one another; the point is to show how much the headline number depends on a choice that was made before looking at any result.

| name | mode | horizon candles | threshold | majority share | test roc auc | test pr auc | test macro f1 | test balanced accuracy | delta roc auc vs binary | ci low | ci high | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary_h6_t50bps | binary | 6 | 0.0050 | 0.7543 | 0.5948 | 0.3112 | 0.0280 | 0.5044 | - | - | - | reference |
| binary_h6_t25bps | binary | 6 | 0.0025 | 0.6467 | 0.5657 | 0.4079 | 0.1209 | 0.5117 | 0.0016 | -0.0097 | 0.0120 | indistinguishable |
| binary_h6_t100bps | binary | 6 | 0.0100 | 0.8750 | 0.6608 | 0.2017 | 0.0051 | 0.5006 | 0.0151 | -0.0040 | 0.0329 | indistinguishable |
| binary_h12_t50bps | binary | 12 | 0.0050 | 0.6871 | 0.5498 | 0.3508 | 0.0666 | 0.5041 | -0.0126 | -0.0329 | 0.0089 | indistinguishable |
| binary_h24_t50bps | binary | 24 | 0.0050 | 0.6305 | 0.4667 | 0.3494 | 0.0474 | 0.5001 | -0.0640 | -0.1136 | -0.0155 | worse |

### Per-asset vs global model

A global model pooled across symbols is compared with each per-asset model on that asset's own test rows, again paired. Sentiment is excluded for non-BTC symbols, so the arms are not exchangeable across rows and the comparison is within-symbol only.

| symbol | strategy | n test | test roc auc | delta roc auc vs per asset | ci low | ci high | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BTCUSDT | per_asset | 6202 | 0.5948 | - | - | - | reference |
| ETHUSDT | per_asset | 6202 | 0.5678 | - | - | - | reference |
| SOLUSDT | per_asset | 6202 | 0.5445 | - | - | - | reference |
| BNBUSDT | per_asset | 6202 | 0.5501 | - | - | - | reference |
| XRPUSDT | per_asset | 6202 | 0.5742 | - | - | - | reference |
| BTCUSDT | global | 6202 | 0.6023 | 0.0075 | -0.0068 | 0.0213 | indistinguishable |
| ETHUSDT | global | 6202 | 0.5698 | 0.0019 | -0.0092 | 0.0116 | indistinguishable |
| SOLUSDT | global | 6202 | 0.5613 | 0.0167 | 0.0008 | 0.0332 | better |
| BNBUSDT | global | 6202 | 0.5833 | 0.0333 | 0.0081 | 0.0579 | better |
| XRPUSDT | global | 6202 | 0.5890 | 0.0148 | 0.0031 | 0.0283 | better |

## Warnings

- **EXP-00-OHLCV-TECHNICAL**: The two validation halves chose different thresholds, so this choice is period-dependent and should be treated as fragile.
- **EXP-00-OHLCV-TECHNICAL**: validation halves chose thresholds 0.38 and 0.46; the choice is period-dependent
- **EXP-01-DERIVATIVES**: The two validation halves chose different thresholds, so this choice is period-dependent and should be treated as fragile.
- **EXP-01-DERIVATIVES**: validation halves chose thresholds 0.38 and 0.46; the choice is period-dependent
- **EXP-03-MICROSTRUCTURE**: The two validation halves chose different thresholds, so this choice is period-dependent and should be treated as fragile.
- **EXP-03-MICROSTRUCTURE**: validation halves chose thresholds 0.46 and 0.52; the choice is period-dependent
- **EXP-04-COMBINED**: The two validation halves chose different thresholds, so this choice is period-dependent and should be treated as fragile.
- **EXP-04-COMBINED**: validation halves chose thresholds 0.36 and 0.46; the choice is period-dependent
## Interpretation limits

- One symbol, one period. A positive test-block result is evidence worth extending, not a finding.
- Features are strongly autocorrelated; a tree model's importance ranking is a dividend-split artefact as much as a signal ranking.
- Every source is fully covered over the feature grid, so the shorter group datasets come from trailing-indicator warm-up rather than missing history: a 60-day basis z-score has no value for its first 60 days. Group row counts run from 40101 to 41342 rows. Compare group results through the paired bootstrap on common rows rather than by subtracting two AUCs measured on different samples.
- Transaction costs are modelled as a flat rate, not as market impact, which flatters any high-turnover strategy.
- Sentiment is a daily reading carried across the day; it cannot react within a day and its features are therefore slow-moving relative to a 6h horizon.
