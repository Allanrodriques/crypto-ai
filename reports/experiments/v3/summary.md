# V3 walk-forward report: BTCUSDT

Every number below is computed from pooled out-of-sample test predictions. No row that a model was fitted on contributes to any metric in this document.

## 1. Run metadata

| field | value |
| --- | --- |
| symbol | BTCUSDT |
| horizons | 1d, 3d, 7d, 14d, 30d, 60d, 90d, 180d |
| features | 84 |
| models | zero, mean, trailing_mean, ridge, random_forest, xgboost |
| seed | 42 |
| n_splits | 3 |
| test fraction | 0.0800 |
| validation fraction | 0.0800 |
| nominal interval coverage | 0.9000 |
| total runtime (s) | 2,857.4 |
| analysis module | analyse: available, calibration_report: available, prediction_interval_report: available, decile_report: available, return_distribution_report: available, evaluate_by_regime: available |

**Purge and embargo geometry**

| horizon | days | purge | embargo | splits | labelled rows | folds | prediction rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1d | 1 | 1 days 00:00:00 | 1 days 00:00:00 | 3 | 40,809 | 3 | 9,795 |
| 3d | 3 | 3 days 00:00:00 | 3 days 00:00:00 | 3 | 40,761 | 3 | 9,784 |
| 7d | 7 | 7 days 00:00:00 | 7 days 00:00:00 | 3 | 40,665 | 3 | 9,761 |
| 14d | 14 | 14 days 00:00:00 | 14 days 00:00:00 | 3 | 40,497 | 3 | 9,721 |
| 30d | 30 | 30 days 00:00:00 | 30 days 00:00:00 | 3 | 40,113 | 3 | 9,628 |
| 60d | 60 | 60 days 00:00:00 | 60 days 00:00:00 | 3 | 39,393 | 3 | 9,456 |
| 90d | 90 | 90 days 00:00:00 | 90 days 00:00:00 | 3 | 38,673 | 3 | 9,283 |
| 180d | 180 | 180 days 00:00:00 | 180 days 00:00:00 | 3 | 36,513 | 3 | 8,764 |

### Caller-supplied notes

| field | value |
| --- | --- |
| plan | horizons=n/a; models=n/a; excluded_buckets=n/a; enabled_features=n/a |
| symbol | BTCUSDT |


## 2. Headline results

The lowest-RMSE model and the highest-IC model per horizon. They are frequently different models: RMSE rewards getting the magnitude right, IC rewards ordering the rows correctly, and a forecast can do one without the other.

| horizon | n | best by RMSE | rmse | mae | spearman ic | dir acc | best by IC | ic | rmse of IC model | models |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1d | 9,795 | zero | 0.0223 | 0.0160 | n/a | 0.0% | trailing_mean | 0.0236 | 0.0288 | 6 |
| 3d | 9,784 | zero | 0.0388 | 0.0289 | n/a | 0.0% | ridge | 0.0250 | 0.0406 | 6 |
| 7d | 9,761 | zero | 0.0609 | 0.0446 | n/a | 0.0% | ridge | 0.0446 | 0.0646 | 6 |
| 14d | 9,721 | zero | 0.0859 | 0.0638 | n/a | 0.0% | ridge | 0.1502 | 0.0931 | 6 |
| 30d | 9,628 | zero | 0.1272 | 0.0973 | n/a | 0.0% | ridge | 0.3288 | 0.1436 | 6 |
| 60d | 9,456 | zero | 0.1700 | 0.1452 | n/a | 0.0% | ridge | 0.2612 | 0.2202 | 6 |
| 90d | 9,283 | zero | 0.1868 | 0.1682 | n/a | 0.0% | ridge | 0.1301 | 0.3801 | 6 |
| 180d | 8,764 | zero | 0.2731 | 0.2463 | n/a | 0.0% | mean | 0.6351 | 0.4890 | 6 |

- **1d**: lowest RMSE is `zero` (RMSE 0.0223, MAE 0.0160, Spearman IC n/a, direction accuracy 0.0%); highest IC is `trailing_mean` (IC 0.0236).
- **3d**: lowest RMSE is `zero` (RMSE 0.0388, MAE 0.0289, Spearman IC n/a, direction accuracy 0.0%); highest IC is `ridge` (IC 0.0250).
- **7d**: lowest RMSE is `zero` (RMSE 0.0609, MAE 0.0446, Spearman IC n/a, direction accuracy 0.0%); highest IC is `ridge` (IC 0.0446).
- **14d**: lowest RMSE is `zero` (RMSE 0.0859, MAE 0.0638, Spearman IC n/a, direction accuracy 0.0%); highest IC is `ridge` (IC 0.1502).
- **30d**: lowest RMSE is `zero` (RMSE 0.1272, MAE 0.0973, Spearman IC n/a, direction accuracy 0.0%); highest IC is `ridge` (IC 0.3288).
- **60d**: lowest RMSE is `zero` (RMSE 0.1700, MAE 0.1452, Spearman IC n/a, direction accuracy 0.0%); highest IC is `ridge` (IC 0.2612).
- **90d**: lowest RMSE is `zero` (RMSE 0.1868, MAE 0.1682, Spearman IC n/a, direction accuracy 0.0%); highest IC is `ridge` (IC 0.1301).
- **180d**: lowest RMSE is `zero` (RMSE 0.2731, MAE 0.2463, Spearman IC n/a, direction accuracy 0.0%); highest IC is `mean` (IC 0.6351).

**Naive-baseline check**

A naive baseline wins on RMSE at every one of the 8 horizons (1d, 3d, 7d, 14d, 30d, 60d, 90d, 180d). Stated plainly because it is the expected result at long horizons: the learned models did not beat a constant or trailing-mean forecast out of sample, and no metric below changes that. The rank and direction statistics are reported because they are the only place any edge appears, not as a substitute for a win.

**Rank ladder (predicted decile against realised return)**

A prediction that carries information produces a monotonically rising ladder. A flat or falling one means the model is adding magnitude, not ordering.

| horizon | model | n | n deciles | top decile return | bottom decile return | long short spread | spearman ic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1d | zero | 9,795 | 10 | 0.00684 | -0.00147 | 0.00530 | n/a |
| 1d | mean | 9,795 | 10 | -0.00098 | -0.00662 | -0.00107 | -0.01984 |
| 1d | trailing_mean | 9,795 | 10 | 0.00481 | -0.00147 | 0.00390 | 0.02359 |
| 1d | ridge | 9,795 | 10 | -0.00189 | -0.00205 | 0.00182 | 0.02052 |
| 1d | random_forest | 9,795 | 10 | -0.00107 | 0.00262 | -0.00074 | -0.02123 |
| 1d | xgboost | 9,795 | 10 | -0.00080 | 0.00303 | -0.00160 | -0.01014 |
| 3d | zero | 9,784 | 10 | 0.02191 | -0.00452 | 0.01577 | n/a |
| 3d | mean | 9,784 | 10 | -0.00299 | -0.01742 | -0.00332 | -0.06377 |
| 3d | trailing_mean | 9,784 | 10 | -0.00299 | -0.01555 | -0.00413 | -0.07253 |
| 3d | ridge | 9,784 | 10 | -0.00446 | -0.00645 | 0.00301 | 0.02502 |
| 3d | random_forest | 9,784 | 10 | 0.00024 | 0.00682 | 0.00073 | 0.00392 |
| 3d | xgboost | 9,784 | 10 | 0.00263 | 0.00520 | 0.00127 | -0.00615 |
| 7d | zero | 9,761 | 10 | 0.05395 | -0.00950 | 0.03672 | n/a |
| 7d | mean | 9,761 | 10 | -0.01037 | -0.03955 | -0.00803 | -0.10944 |
| 7d | trailing_mean | 9,761 | 10 | -0.01037 | -0.03323 | -0.01131 | -0.12764 |
| 7d | ridge | 9,761 | 10 | -0.01271 | -0.01979 | 0.00369 | 0.04458 |
| 7d | random_forest | 9,761 | 10 | 0.00750 | 0.02961 | -0.01537 | -0.08964 |
| 7d | xgboost | 9,761 | 10 | 0.00519 | 0.02307 | -0.01033 | -0.03517 |
| 14d | zero | 9,721 | 10 | 0.09172 | -0.01091 | 0.06629 | n/a |
| 14d | mean | 9,721 | 10 | -0.03097 | -0.07374 | -0.01302 | -0.14250 |
| 14d | trailing_mean | 9,721 | 10 | 0.09172 | -0.04473 | 0.08527 | -0.03355 |
| 14d | ridge | 9,721 | 10 | -0.00747 | -0.01650 | 0.01912 | 0.15020 |
| 14d | random_forest | 9,721 | 10 | -0.03313 | -0.00409 | -0.01790 | -0.05684 |
| 14d | xgboost | 9,721 | 10 | -0.02690 | 0.00289 | -0.01818 | -0.01988 |
| 30d | zero | 9,628 | 10 | 0.17957 | -0.02090 | 0.12399 | n/a |
| 30d | mean | 9,628 | 10 | -0.08532 | -0.12684 | -0.02407 | -0.26109 |
| 30d | trailing_mean | 9,628 | 10 | -0.08532 | -0.05084 | -0.01248 | -0.21487 |
| 30d | ridge | 9,628 | 10 | 0.03168 | -0.05515 | 0.08329 | 0.32883 |
| 30d | random_forest | 9,628 | 10 | -0.11207 | 0.07124 | -0.08203 | -0.28940 |
| 30d | xgboost | 9,628 | 10 | -0.08568 | -0.07420 | -0.02254 | -0.00982 |
| 60d | zero | 9,456 | 10 | 0.25876 | -0.00499 | 0.11835 | n/a |
| 60d | mean | 9,456 | 10 | -0.19957 | -0.06133 | -0.04546 | -0.23413 |
| 60d | trailing_mean | 9,456 | 10 | -0.19957 | -0.06133 | -0.04546 | -0.23413 |
| 60d | ridge | 9,456 | 10 | 0.04980 | -0.07477 | 0.12022 | 0.26115 |
| 60d | random_forest | 9,456 | 10 | -0.01737 | -0.00274 | -0.03636 | -0.07178 |
| 60d | xgboost | 9,456 | 10 | 0.08754 | -0.01962 | -0.00565 | 0.03316 |
| 90d | zero | 9,283 | 10 | 0.21247 | 0.06381 | 0.04279 | n/a |
| 90d | mean | 9,283 | 10 | 0.00180 | 0.06381 | -0.08596 | -0.22520 |
| 90d | trailing_mean | 9,283 | 10 | -0.22403 | -0.03659 | -0.09511 | -0.21329 |
| 90d | ridge | 9,283 | 10 | 0.00659 | -0.14423 | 0.12668 | 0.13012 |
| 90d | random_forest | 9,283 | 10 | 0.00376 | -0.07279 | 0.01670 | -0.07415 |
| 90d | xgboost | 9,283 | 10 | -0.02177 | -0.10255 | 0.09062 | 0.04777 |
| 180d | zero | 8,764 | 10 | 0.15555 | 0.29925 | -0.08482 | n/a |
| 180d | mean | 8,764 | 10 | -0.20492 | -0.38027 | 0.20858 | 0.63510 |
| 180d | trailing_mean | 8,764 | 10 | -0.20492 | -0.28592 | 0.08071 | 0.18535 |
| 180d | ridge | 8,764 | 10 | 0.03289 | -0.30705 | 0.28645 | 0.50606 |
| 180d | random_forest | 8,764 | 10 | 0.06689 | -0.20393 | 0.18695 | 0.28913 |
| 180d | xgboost | 8,764 | 10 | -0.06537 | -0.12058 | -0.00594 | 0.08866 |

## 3. Uncertainty and interval coverage

Bands are split-conformal, calibrated on validation residuals only. That coverage claim is marginal and finite-sample: it holds only if the calibration residuals are exchangeable with the test residuals, and overlapping forward windows at these horizons break exactly that assumption. Coverage below nominal is therefore the expected direction of the error, not evidence that the construction is broken.

| horizon | model | empirical coverage | nominal | met nominal | mean width | calibration rows | source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1d | zero | 88.8% | n/a | n/a | n/a | n/a | pooled rows |
| 1d | mean | 88.9% | n/a | n/a | n/a | n/a | pooled rows |
| 1d | trailing_mean | 82.9% | n/a | n/a | n/a | n/a | pooled rows |
| 1d | ridge | 89.6% | n/a | n/a | n/a | n/a | pooled rows |
| 1d | random_forest | 91.4% | n/a | n/a | n/a | n/a | pooled rows |
| 1d | xgboost | 89.4% | n/a | n/a | n/a | n/a | pooled rows |
| 3d | zero | 89.2% | n/a | n/a | n/a | n/a | pooled rows |
| 3d | mean | 88.7% | n/a | n/a | n/a | n/a | pooled rows |
| 3d | trailing_mean | 88.5% | n/a | n/a | n/a | n/a | pooled rows |
| 3d | ridge | 90.3% | n/a | n/a | n/a | n/a | pooled rows |
| 3d | random_forest | 90.8% | n/a | n/a | n/a | n/a | pooled rows |
| 3d | xgboost | 89.2% | n/a | n/a | n/a | n/a | pooled rows |
| 7d | zero | 88.1% | n/a | n/a | n/a | n/a | pooled rows |
| 7d | mean | 86.5% | n/a | n/a | n/a | n/a | pooled rows |
| 7d | trailing_mean | 83.2% | n/a | n/a | n/a | n/a | pooled rows |
| 7d | ridge | 91.8% | n/a | n/a | n/a | n/a | pooled rows |
| 7d | random_forest | 89.0% | n/a | n/a | n/a | n/a | pooled rows |
| 7d | xgboost | 82.2% | n/a | n/a | n/a | n/a | pooled rows |
| 14d | zero | 86.6% | n/a | n/a | n/a | n/a | pooled rows |
| 14d | mean | 85.2% | n/a | n/a | n/a | n/a | pooled rows |
| 14d | trailing_mean | 94.9% | n/a | n/a | n/a | n/a | pooled rows |
| 14d | ridge | 89.3% | n/a | n/a | n/a | n/a | pooled rows |
| 14d | random_forest | 86.8% | n/a | n/a | n/a | n/a | pooled rows |
| 14d | xgboost | 82.5% | n/a | n/a | n/a | n/a | pooled rows |
| 30d | zero | 90.1% | n/a | n/a | n/a | n/a | pooled rows |
| 30d | mean | 88.4% | n/a | n/a | n/a | n/a | pooled rows |
| 30d | trailing_mean | 66.1% | n/a | n/a | n/a | n/a | pooled rows |
| 30d | ridge | 89.3% | n/a | n/a | n/a | n/a | pooled rows |
| 30d | random_forest | 78.7% | n/a | n/a | n/a | n/a | pooled rows |
| 30d | xgboost | 84.1% | n/a | n/a | n/a | n/a | pooled rows |
| 60d | zero | 73.5% | n/a | n/a | n/a | n/a | pooled rows |
| 60d | mean | 71.3% | n/a | n/a | n/a | n/a | pooled rows |
| 60d | trailing_mean | 57.8% | n/a | n/a | n/a | n/a | pooled rows |
| 60d | ridge | 94.4% | n/a | n/a | n/a | n/a | pooled rows |
| 60d | random_forest | 74.6% | n/a | n/a | n/a | n/a | pooled rows |
| 60d | xgboost | 65.8% | n/a | n/a | n/a | n/a | pooled rows |
| 90d | zero | 93.9% | n/a | n/a | n/a | n/a | pooled rows |
| 90d | mean | 62.7% | n/a | n/a | n/a | n/a | pooled rows |
| 90d | trailing_mean | 75.8% | n/a | n/a | n/a | n/a | pooled rows |
| 90d | ridge | 70.0% | n/a | n/a | n/a | n/a | pooled rows |
| 90d | random_forest | 65.6% | n/a | n/a | n/a | n/a | pooled rows |
| 90d | xgboost | 63.3% | n/a | n/a | n/a | n/a | pooled rows |
| 180d | zero | 81.1% | n/a | n/a | n/a | n/a | pooled rows |
| 180d | mean | 28.5% | n/a | n/a | n/a | n/a | pooled rows |
| 180d | trailing_mean | 51.1% | n/a | n/a | n/a | n/a | pooled rows |
| 180d | ridge | 88.6% | n/a | n/a | n/a | n/a | pooled rows |
| 180d | random_forest | 89.1% | n/a | n/a | n/a | n/a | pooled rows |
| 180d | xgboost | 75.4% | n/a | n/a | n/a | n/a | pooled rows |

Empirical coverage met the nominal target for 0 of 48 model/horizon pairs, against a tolerance of 2.0%.

**Pooled interval report**

| horizon | model | mean fold coverage | nominal coverage | coverage gap | mean interval width | median interval width | n predictions | coverage ci lower | coverage ci upper | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1d | zero | 0.88841 | 0.90000 | -0.01159 | 0.06879 | 0.06512 | 9,795 | 0.88202 | 0.89450 | prediction_interval_report |
| 1d | mean | 0.88882 | 0.90000 | -0.01118 | 0.06884 | 0.06560 | 9,795 | 0.88244 | 0.89489 | prediction_interval_report |
| 1d | trailing_mean | 0.82910 | 0.90000 | -0.07090 | 0.08119 | 0.07955 | 9,795 | 0.82151 | 0.83642 | prediction_interval_report |
| 1d | ridge | 0.89566 | 0.90000 | -0.00434 | 0.07190 | 0.06802 | 9,795 | 0.88945 | 0.90156 | prediction_interval_report |
| 1d | random_forest | 0.91414 | 0.90000 | 0.01414 | 0.08758 | 0.08864 | 9,795 | 0.90843 | 0.91953 | prediction_interval_report |
| 1d | xgboost | 0.89382 | 0.90000 | -0.00618 | 0.07557 | 0.07376 | 9,795 | 0.88757 | 0.89977 | prediction_interval_report |
| 3d | zero | 0.89156 | 0.90000 | -0.00844 | 0.11761 | 0.11592 | 9,784 | 0.88524 | 0.89757 | prediction_interval_report |
| 3d | mean | 0.88665 | 0.90000 | -0.01335 | 0.11791 | 0.11836 | 9,784 | 0.88022 | 0.89278 | prediction_interval_report |
| 3d | trailing_mean | 0.88481 | 0.90000 | -0.01519 | 0.12692 | 0.12299 | 9,784 | 0.87833 | 0.89099 | prediction_interval_report |
| 3d | ridge | 0.90290 | 0.90000 | 0.00290 | 0.13136 | 0.13163 | 9,784 | 0.89688 | 0.90861 | prediction_interval_report |
| 3d | random_forest | 0.90842 | 0.90000 | 0.00842 | 0.14640 | 0.15202 | 9,784 | 0.90255 | 0.91398 | prediction_interval_report |
| 3d | xgboost | 0.89206 | 0.90000 | -0.00794 | 0.14100 | 0.13964 | 9,784 | 0.88577 | 0.89806 | prediction_interval_report |
| 7d | zero | 0.88137 | 0.90000 | -0.01863 | 0.17919 | 0.17384 | 9,761 | 0.87480 | 0.88763 | prediction_interval_report |
| 7d | mean | 0.86539 | 0.90000 | -0.03461 | 0.17514 | 0.16532 | 9,761 | 0.85847 | 0.87201 | prediction_interval_report |
| 7d | trailing_mean | 0.83210 | 0.90000 | -0.06790 | 0.19352 | 0.18654 | 9,761 | 0.82454 | 0.83937 | prediction_interval_report |
| 7d | ridge | 0.91814 | 0.90000 | 0.01814 | 0.21828 | 0.21201 | 9,761 | 0.91254 | 0.92342 | prediction_interval_report |
| 7d | random_forest | 0.89027 | 0.90000 | -0.00973 | 0.25779 | 0.23232 | 9,761 | 0.88392 | 0.89633 | prediction_interval_report |
| 7d | xgboost | 0.82235 | 0.90000 | -0.07765 | 0.21168 | 0.21026 | 9,761 | 0.81465 | 0.82981 | prediction_interval_report |
| 14d | zero | 0.86568 | 0.90000 | -0.03432 | 0.25673 | 0.23810 | 9,721 | 0.85873 | 0.87229 | prediction_interval_report |
| 14d | mean | 0.85241 | 0.90000 | -0.04759 | 0.26560 | 0.27229 | 9,721 | 0.84519 | 0.85929 | prediction_interval_report |
| 14d | trailing_mean | 0.94868 | 0.90000 | 0.04868 | 0.39691 | 0.41467 | 9,721 | 0.94410 | 0.95288 | prediction_interval_report |
| 14d | ridge | 0.89273 | 0.90000 | -0.00727 | 0.32642 | 0.36334 | 9,721 | 0.88640 | 0.89870 | prediction_interval_report |
| 14d | random_forest | 0.86774 | 0.90000 | -0.03226 | 0.30878 | 0.30593 | 9,721 | 0.86083 | 0.87430 | prediction_interval_report |
| 14d | xgboost | 0.82516 | 0.90000 | -0.07484 | 0.28608 | 0.27280 | 9,721 | 0.81744 | 0.83254 | prediction_interval_report |
| 30d | zero | 0.90059 | 0.90000 | 0.00059 | 0.42741 | 0.43335 | 9,628 | 0.89447 | 0.90642 | prediction_interval_report |
| 30d | mean | 0.88408 | 0.90000 | -0.01592 | 0.45492 | 0.41558 | 9,628 | 0.87754 | 0.89033 | prediction_interval_report |
| 30d | trailing_mean | 0.66107 | 0.90000 | -0.23893 | 0.37770 | 0.38530 | 9,628 | 0.65158 | 0.67048 | prediction_interval_report |
| 30d | ridge | 0.89280 | 0.90000 | -0.00720 | 0.53738 | 0.54456 | 9,628 | 0.88648 | 0.89884 | prediction_interval_report |
| 30d | random_forest | 0.78718 | 0.90000 | -0.11282 | 0.43388 | 0.38346 | 9,628 | 0.77889 | 0.79524 | prediction_interval_report |
| 30d | xgboost | 0.84099 | 0.90000 | -0.05901 | 0.47663 | 0.54011 | 9,628 | 0.83354 | 0.84815 | prediction_interval_report |
| 60d | zero | 0.73519 | 0.90000 | -0.16481 | 0.44321 | 0.51380 | 9,456 | 0.72621 | 0.74399 | prediction_interval_report |
| 60d | mean | 0.71320 | 0.90000 | -0.18680 | 0.48127 | 0.51549 | 9,456 | 0.70400 | 0.72223 | prediction_interval_report |
| 60d | trailing_mean | 0.57794 | 0.90000 | -0.32206 | 0.63392 | 0.42667 | 9,456 | 0.56796 | 0.58786 | prediction_interval_report |
| 60d | ridge | 0.94427 | 0.90000 | 0.04427 | 0.99872 | 0.99976 | 9,456 | 0.93946 | 0.94871 | prediction_interval_report |
| 60d | random_forest | 0.74630 | 0.90000 | -0.15370 | 0.58561 | 0.53679 | 9,456 | 0.73743 | 0.75497 | prediction_interval_report |
| 60d | xgboost | 0.65778 | 0.90000 | -0.24222 | 0.62749 | 0.72073 | 9,456 | 0.64816 | 0.66728 | prediction_interval_report |
| 90d | zero | 0.93860 | 0.90000 | 0.03860 | 0.58709 | 0.54375 | 9,283 | 0.93353 | 0.94330 | prediction_interval_report |
| 90d | mean | 0.62745 | 0.90000 | -0.27255 | 0.59809 | 0.54119 | 9,283 | 0.61761 | 0.63727 | prediction_interval_report |
| 90d | trailing_mean | 0.75770 | 0.90000 | -0.14230 | 1.00504 | 1.03844 | 9,283 | 0.74891 | 0.76634 | prediction_interval_report |
| 90d | ridge | 0.69997 | 0.90000 | -0.20003 | 0.81961 | 0.74266 | 9,283 | 0.69059 | 0.70923 | prediction_interval_report |
| 90d | random_forest | 0.65580 | 0.90000 | -0.24420 | 0.59567 | 0.65930 | 9,283 | 0.64609 | 0.66542 | prediction_interval_report |
| 90d | xgboost | 0.63276 | 0.90000 | -0.26724 | 0.69896 | 0.69541 | 9,283 | 0.62291 | 0.64252 | prediction_interval_report |
| 180d | zero | 0.81125 | 0.90000 | -0.08875 | 0.94194 | 0.78415 | 8,764 | 0.80295 | 0.81933 | prediction_interval_report |
| 180d | mean | 0.28524 | 0.90000 | -0.61476 | 0.68209 | 0.80120 | 8,764 | 0.27590 | 0.29480 | prediction_interval_report |
| 180d | trailing_mean | 0.51113 | 0.90000 | -0.38887 | 1.12974 | 0.87518 | 8,764 | 0.50071 | 0.52164 | prediction_interval_report |
| 180d | ridge | 0.88615 | 0.90000 | -0.01385 | 1.35765 | 1.26544 | 8,764 | 0.87930 | 0.89261 | prediction_interval_report |
| 180d | random_forest | 0.89148 | 0.90000 | -0.00852 | 1.59666 | 1.38031 | 8,764 | 0.88480 | 0.89783 | prediction_interval_report |
| 180d | xgboost | 0.75351 | 0.90000 | -0.14649 | 1.43118 | 1.34290 | 8,764 | 0.74440 | 0.76245 | prediction_interval_report |

## 4. Probability and calibration

**A point forecast is not a probability.** The scores below are Brier scores of two genuinely different objects, and neither is a squashed regression output: the empirical score comes from each model's own out-of-sample residual CDF, and the calibrated score from a separately fitted direction classifier. Rescaling a return prediction into [0, 1] and scoring it here would measure nothing.

| horizon | model | brier (empirical) | brier (calibrated) | n |
| --- | --- | --- | --- | --- |
| 1d | zero | 0.2515 | 0.2505 | 9,795 |
| 1d | mean | 0.2519 | 0.2505 | 9,795 |
| 1d | trailing_mean | 0.3634 | 0.2505 | 9,795 |
| 1d | ridge | 0.2706 | 0.2505 | 9,795 |
| 1d | random_forest | 0.3239 | 0.2505 | 9,795 |
| 1d | xgboost | 0.2956 | 0.2505 | 9,795 |
| 3d | zero | 0.2593 | 0.2521 | 9,784 |
| 3d | mean | 0.2607 | 0.2521 | 9,784 |
| 3d | trailing_mean | 0.3567 | 0.2521 | 9,784 |
| 3d | ridge | 0.2910 | 0.2521 | 9,784 |
| 3d | random_forest | 0.3461 | 0.2521 | 9,784 |
| 3d | xgboost | 0.3120 | 0.2521 | 9,784 |
| 7d | zero | 0.2730 | 0.2536 | 9,761 |
| 7d | mean | 0.2766 | 0.2536 | 9,761 |
| 7d | trailing_mean | 0.3335 | 0.2536 | 9,761 |
| 7d | ridge | 0.3285 | 0.2536 | 9,761 |
| 7d | random_forest | 0.3543 | 0.2536 | 9,761 |
| 7d | xgboost | 0.3344 | 0.2536 | 9,761 |
| 14d | zero | 0.2930 | 0.2615 | 9,721 |
| 14d | mean | 0.3022 | 0.2615 | 9,721 |
| 14d | trailing_mean | 0.4818 | 0.2615 | 9,721 |
| 14d | ridge | 0.3429 | 0.2615 | 9,721 |
| 14d | random_forest | 0.3633 | 0.2615 | 9,721 |
| 14d | xgboost | 0.3017 | 0.2615 | 9,721 |
| 30d | zero | 0.3276 | 0.2410 | 9,628 |
| 30d | mean | 0.3424 | 0.2410 | 9,628 |
| 30d | trailing_mean | 0.4711 | 0.2410 | 9,628 |
| 30d | ridge | 0.3534 | 0.2410 | 9,628 |
| 30d | random_forest | 0.3024 | 0.2410 | 9,628 |
| 30d | xgboost | 0.2755 | 0.2410 | 9,628 |
| 60d | zero | 0.3260 | 0.2847 | 9,456 |
| 60d | mean | 0.3333 | 0.2847 | 9,456 |
| 60d | trailing_mean | 0.3410 | 0.2847 | 9,456 |
| 60d | ridge | 0.3250 | 0.2847 | 9,456 |
| 60d | random_forest | 0.4263 | 0.2847 | 9,456 |
| 60d | xgboost | 0.4154 | 0.2847 | 9,456 |
| 90d | zero | 0.4574 | 0.2143 | 9,283 |
| 90d | mean | 0.4564 | 0.2143 | 9,283 |
| 90d | trailing_mean | 0.5849 | 0.2143 | 9,283 |
| 90d | ridge | 0.4574 | 0.2143 | 9,283 |
| 90d | random_forest | 0.4779 | 0.2143 | 9,283 |
| 90d | xgboost | 0.3413 | 0.2143 | 9,283 |
| 180d | zero | 0.6982 | 0.3221 | 8,764 |
| 180d | mean | 0.6900 | 0.3221 | 8,764 |
| 180d | trailing_mean | 0.6651 | 0.3221 | 8,764 |
| 180d | ridge | 0.2886 | 0.3221 | 8,764 |
| 180d | random_forest | 0.5983 | 0.3221 | 8,764 |
| 180d | xgboost | 0.5655 | 0.3221 | 8,764 |

The direction model is fitted once per fold, independently of which regressor produced the point forecast, so its calibrated probability is identical across the models within a horizon. Repeated calibrated values down a column are expected, not a copy error.

The calibrated probability scored better than the empirical one for 47 of 48 comparable pairs. A Brier score on a single pooled sample is not a significance test, and the gap between two of them is well inside the noise of a few thousand overlapping rows. The best calibrated score here is 0.2143; a constant 50/50 forecast scores 0.25, so anything at or above that has not beaten the coin flip.

- `calibrated direction model`: 34 populated bin(s), largest gap between mean predicted probability and observed up-frequency 0.8181, mean absolute gap 0.2671.
- `zero (empirical)`: 22 populated bin(s), largest gap between mean predicted probability and observed up-frequency 0.8266, mean absolute gap 0.2673.

Source: equal-width reliability bins over the pooled probability columns; per-model calibration statistics are in prediction_calibration.csv. A large gap sitting in a sparsely populated bin is a sample-size artefact rather than miscalibration; the bin counts in `prediction_calibration.csv` are there so that distinction can be made.

## 5. Regime breakdown

| model | regime | n | rmse | mae | direction accuracy | mean actual return | mean predicted | hit rate ci lower | hit rate ci upper | std actual return |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ridge | vol_q1 | 2,448 | 0.01824 | 0.01304 | 0.56291 | 0.00143 | -0.00004 | 0.54318 | 0.58244 | 0.01858 |
| ridge | vol_q2 | 2,449 | 0.01991 | 0.01528 | 0.47530 | -0.00113 | 0.00072 | 0.45557 | 0.49510 | 0.01943 |
| ridge | vol_q3 | 2,449 | 0.02252 | 0.01710 | 0.52307 | -0.00057 | 0.00170 | 0.50327 | 0.54280 | 0.02199 |
| ridge | vol_q4 | 2,449 | 0.02929 | 0.02062 | 0.49898 | -0.00204 | 0.00394 | 0.47919 | 0.51877 | 0.02781 |
| ridge | trend_q1 | 2,448 | 0.02713 | 0.01923 | 0.49306 | -0.00166 | 0.00303 | 0.47328 | 0.51286 | 0.02554 |
| ridge | trend_q2 | 2,449 | 0.02152 | 0.01558 | 0.51531 | -0.00086 | -0.00002 | 0.49551 | 0.53507 | 0.02120 |
| ridge | trend_q3 | 2,449 | 0.02197 | 0.01618 | 0.53491 | 0.00119 | 0.00152 | 0.51512 | 0.55460 | 0.02206 |
| ridge | trend_q4 | 2,449 | 0.02031 | 0.01506 | 0.51695 | -0.00100 | 0.00180 | 0.49714 | 0.53670 | 0.01984 |
| ridge | vol_q1 | 2,446 | 0.03951 | 0.02638 | 0.57236 | 0.00413 | 0.00277 | 0.55266 | 0.59184 | 0.04059 |
| ridge | vol_q2 | 2,446 | 0.03594 | 0.02822 | 0.45707 | -0.00265 | 0.00365 | 0.43741 | 0.47687 | 0.03321 |
| ridge | vol_q3 | 2,446 | 0.03919 | 0.03047 | 0.52780 | -0.00460 | 0.00517 | 0.50799 | 0.54753 | 0.03678 |
| ridge | vol_q4 | 2,446 | 0.04701 | 0.03520 | 0.48651 | -0.00405 | 0.00848 | 0.46674 | 0.50632 | 0.04320 |
| ridge | trend_q1 | 2,446 | 0.04147 | 0.03094 | 0.56132 | -0.00042 | 0.00571 | 0.54158 | 0.58088 | 0.03919 |
| ridge | trend_q2 | 2,446 | 0.03488 | 0.02627 | 0.49428 | -0.00368 | 0.00264 | 0.47449 | 0.51408 | 0.03311 |
| ridge | trend_q3 | 2,446 | 0.04237 | 0.03014 | 0.53271 | -0.00052 | 0.00559 | 0.51290 | 0.55241 | 0.04098 |
| ridge | trend_q4 | 2,446 | 0.04321 | 0.03293 | 0.45544 | -0.00255 | 0.00612 | 0.43579 | 0.47523 | 0.04122 |
| ridge | vol_q1 | 2,440 | 0.06977 | 0.04612 | 0.56475 | 0.00868 | 0.00232 | 0.54500 | 0.58431 | 0.06970 |
| ridge | vol_q2 | 2,440 | 0.06021 | 0.04513 | 0.47418 | -0.00821 | 0.00585 | 0.45442 | 0.49402 | 0.05589 |
| ridge | vol_q3 | 2,440 | 0.06101 | 0.04672 | 0.58320 | -0.00853 | 0.00884 | 0.56352 | 0.60261 | 0.05751 |
| ridge | vol_q4 | 2,441 | 0.06677 | 0.05046 | 0.48587 | -0.00798 | 0.01287 | 0.46608 | 0.50570 | 0.05730 |
| ridge | trend_q1 | 2,440 | 0.07154 | 0.05337 | 0.52131 | -0.00948 | 0.01048 | 0.50147 | 0.54108 | 0.06384 |
| ridge | trend_q2 | 2,440 | 0.06556 | 0.04765 | 0.51639 | -0.00262 | 0.00604 | 0.49655 | 0.53618 | 0.06227 |
| ridge | trend_q3 | 2,440 | 0.06429 | 0.04565 | 0.54508 | -0.00297 | 0.00680 | 0.52527 | 0.56475 | 0.06230 |
| ridge | trend_q4 | 2,441 | 0.05591 | 0.04177 | 0.52519 | -0.00098 | 0.00657 | 0.50536 | 0.54495 | 0.05396 |
| ridge | vol_q1 | 2,430 | 0.10138 | 0.07515 | 0.56502 | 0.01589 | 0.00560 | 0.54522 | 0.58461 | 0.09982 |
| ridge | vol_q2 | 2,430 | 0.09735 | 0.07064 | 0.52305 | -0.02170 | 0.01280 | 0.50317 | 0.54285 | 0.08562 |
| ridge | vol_q3 | 2,430 | 0.08644 | 0.06445 | 0.58313 | -0.01896 | 0.02346 | 0.56341 | 0.60258 | 0.07705 |
| ridge | vol_q4 | 2,431 | 0.08647 | 0.06576 | 0.53023 | -0.01043 | 0.02928 | 0.51036 | 0.55001 | 0.07145 |
| ridge | trend_q1 | 2,430 | 0.09519 | 0.07163 | 0.55226 | -0.01309 | 0.01994 | 0.53242 | 0.57194 | 0.08384 |
| ridge | trend_q2 | 2,430 | 0.10113 | 0.07509 | 0.52099 | -0.00170 | 0.01704 | 0.50111 | 0.54080 | 0.09352 |
| ridge | trend_q3 | 2,430 | 0.09746 | 0.07157 | 0.54897 | -0.00741 | 0.01523 | 0.52912 | 0.56866 | 0.09220 |
| ridge | trend_q4 | 2,431 | 0.07694 | 0.05772 | 0.57919 | -0.01299 | 0.01893 | 0.55945 | 0.59867 | 0.06966 |
| ridge | vol_q1 | 2,407 | 0.14811 | 0.11687 | 0.64271 | -0.00411 | 0.00591 | 0.62335 | 0.66161 | 0.14451 |
| ridge | vol_q2 | 2,407 | 0.14810 | 0.11596 | 0.66099 | -0.03736 | 0.02329 | 0.64183 | 0.67963 | 0.13235 |
| ridge | vol_q3 | 2,407 | 0.14276 | 0.11004 | 0.66639 | -0.02592 | 0.05664 | 0.64730 | 0.68495 | 0.12159 |
| ridge | vol_q4 | 2,407 | 0.13489 | 0.10544 | 0.61903 | -0.01114 | 0.07158 | 0.59945 | 0.63822 | 0.09673 |
| ridge | trend_q1 | 2,407 | 0.14200 | 0.10975 | 0.67096 | -0.02237 | 0.04387 | 0.65193 | 0.68945 | 0.12075 |
| ridge | trend_q2 | 2,407 | 0.15727 | 0.12174 | 0.62568 | -0.01879 | 0.03639 | 0.60616 | 0.64479 | 0.13310 |
| ridge | trend_q3 | 2,407 | 0.14357 | 0.11447 | 0.66473 | -0.01514 | 0.02783 | 0.64562 | 0.68331 | 0.13607 |
| ridge | trend_q4 | 2,407 | 0.13014 | 0.10235 | 0.62775 | -0.02223 | 0.04931 | 0.60825 | 0.64685 | 0.11117 |
| ridge | vol_q1 | 2,364 | 0.21380 | 0.17733 | 0.61125 | -0.08031 | -0.04324 | 0.59144 | 0.63071 | 0.15503 |
| ridge | vol_q2 | 2,364 | 0.20932 | 0.17085 | 0.65228 | -0.06472 | 0.00051 | 0.63285 | 0.67122 | 0.16824 |
| ridge | vol_q3 | 2,364 | 0.22352 | 0.18458 | 0.63832 | -0.04561 | 0.07733 | 0.61875 | 0.65746 | 0.16391 |
| ridge | vol_q4 | 2,364 | 0.23338 | 0.18651 | 0.60068 | -0.01914 | 0.11864 | 0.58079 | 0.62024 | 0.15283 |
| ridge | trend_q1 | 2,364 | 0.22231 | 0.18206 | 0.59391 | -0.03218 | 0.06971 | 0.57397 | 0.61354 | 0.16180 |
| ridge | trend_q2 | 2,364 | 0.23894 | 0.19667 | 0.60406 | -0.06587 | 0.02485 | 0.58419 | 0.62359 | 0.16379 |
| ridge | trend_q3 | 2,364 | 0.21673 | 0.17693 | 0.64044 | -0.05935 | 0.00527 | 0.62088 | 0.65954 | 0.15868 |
| ridge | trend_q4 | 2,364 | 0.20117 | 0.16362 | 0.66413 | -0.05240 | 0.05341 | 0.64484 | 0.68289 | 0.16070 |
| random_forest | vol_q1 | 2,320 | 0.29335 | 0.25398 | 0.35000 | -0.11651 | 0.07006 | 0.33085 | 0.36964 | 0.14028 |
| random_forest | vol_q2 | 2,321 | 0.28573 | 0.24364 | 0.36536 | -0.10591 | 0.08635 | 0.34601 | 0.38516 | 0.14406 |
| random_forest | vol_q3 | 2,321 | 0.29452 | 0.24729 | 0.34468 | -0.08539 | 0.12805 | 0.32561 | 0.36426 | 0.16540 |
| random_forest | vol_q4 | 2,321 | 0.30390 | 0.25339 | 0.40327 | -0.05317 | 0.16586 | 0.38349 | 0.42338 | 0.19223 |
| random_forest | trend_q1 | 2,320 | 0.30484 | 0.26103 | 0.40302 | -0.05653 | 0.16432 | 0.38323 | 0.42312 | 0.18095 |
| random_forest | trend_q2 | 2,321 | 0.30159 | 0.25994 | 0.33735 | -0.09835 | 0.10434 | 0.31840 | 0.35684 | 0.15952 |
| random_forest | trend_q3 | 2,321 | 0.29019 | 0.24587 | 0.35933 | -0.10411 | 0.09110 | 0.34005 | 0.37907 | 0.15048 |
| random_forest | trend_q4 | 2,321 | 0.28055 | 0.23147 | 0.36364 | -0.10196 | 0.09062 | 0.34431 | 0.38342 | 0.15720 |
| ridge | vol_q1 | 2,191 | 0.34537 | 0.26977 | 0.63031 | -0.22505 | -0.15295 | 0.60988 | 0.65027 | 0.18493 |
| ridge | vol_q2 | 2,191 | 0.36415 | 0.28785 | 0.62894 | -0.18520 | -0.06271 | 0.60850 | 0.64892 | 0.21640 |
| ridge | vol_q3 | 2,191 | 0.42591 | 0.34648 | 0.60064 | -0.13482 | 0.04762 | 0.57997 | 0.62095 | 0.21856 |
| ridge | vol_q4 | 2,191 | 0.46397 | 0.38055 | 0.58193 | -0.06594 | 0.16655 | 0.56115 | 0.60242 | 0.24950 |
| ridge | trend_q1 | 2,191 | 0.43101 | 0.35188 | 0.58421 | -0.13891 | 0.05732 | 0.56344 | 0.60468 | 0.22749 |
| ridge | trend_q2 | 2,191 | 0.40247 | 0.31797 | 0.56413 | -0.18173 | 0.00158 | 0.54327 | 0.58476 | 0.21037 |
| ridge | trend_q3 | 2,191 | 0.37342 | 0.29792 | 0.65130 | -0.16080 | -0.06386 | 0.63110 | 0.67098 | 0.22224 |
| ridge | trend_q4 | 2,191 | 0.40170 | 0.31688 | 0.64217 | -0.12957 | 0.00346 | 0.62187 | 0.66198 | 0.24114 |

Regimes are defined by an explicit trailing rule rather than by hindsight, and each block is scored on one model rather than on all of them averaged together. One pass over one symbol is still not evidence that any difference between regimes is stable.

## 6. Data and caveats

| field | value |
| --- | --- |
| symbol | BTCUSDT |
| rows | 40,833 |
| features | 84 |
| targets | 8 |
| feature groups | technical, sentiment, microstructure, context |
| index name | timestamp |
| start | 2022-01-31 00:00:00+00:00 |
| end | 2026-09-28 12:00:00+00:00 |
| timezone | UTC |
| index unique | yes |
| index monotonic | yes |
| rows dropped | 723 |
| bucket registry available | yes |

**Drop report**

| dropped | rows |
| --- | --- |
| duplicate_timestamps | 0 |
| feature_warmup | 0 |
| max_rows_truncated | 0 |
| missing_feature | 723 |
| missing_price | 0 |
| non_finite_feature | 0 |

**Missing sources**

None recorded.

**Feature coverage**

Minimum 100.0%, median 100.0%, maximum 100.0%; 0 feature(s) below 50% coverage.

**Labelled rows per horizon**

| horizon | labelled | unlabelled tail | labelled share |
| --- | --- | --- | --- |
| 1d | 40,809 | 24 | 99.9% |
| 3d | 40,761 | 72 | 99.8% |
| 7d | 40,665 | 168 | 99.6% |
| 14d | 40,497 | 336 | 99.2% |
| 30d | 40,113 | 720 | 98.2% |
| 60d | 39,393 | 1,440 | 96.5% |
| 90d | 38,673 | 2,160 | 94.7% |
| 180d | 36,513 | 4,320 | 89.4% |

**Caveats**

- Forward-return windows at these horizons overlap almost completely, so the number of pooled rows overstates the independent evidence. Thousands of rows at a 30-day horizon are not thousands of independent observations - they are thousands of heavily correlated ones - and no significance statement in this report accounts for the dependence.
- One symbol, one pass, no nested cross-validation around the hyperparameter search. The search was selected on validation folds and the metrics are pooled from test folds, which is honest, but the selection itself is not re-validated.
- No transaction costs, slippage, funding or market impact are modelled anywhere in this report. The long-short spread is a pre-cost number and is not a PnL estimate.
- Geometry is shared by every horizon in the run, but a longer horizon consumes more of the sample in purging and embargo, so the long-horizon models are fitted on strictly less data. Fewer folds at the long end means the pooled metrics there rest on fewer independent decisions.

## 7. Files written

**Tables**

| file | contents |
| --- | --- |
| horizon_comparison.csv | best model per horizon with its pooled metrics |
| model_comparison.csv | every horizon/model pair with its pooled metrics |
| regime_analysis.csv | metrics by market regime, or a not-available row |
| prediction_calibration.csv | per-model calibration statistics for both probability routes |
| return_distributions.csv | return-distribution statistics |

**Figures**

| file | contents |
| --- | --- |
| horizon_rmse.png | pooled RMSE by horizon and model |
| horizon_ic.png | pooled Spearman IC by horizon and model |
| interval_coverage.png | conformal coverage against the nominal target |
| prediction_calibration.png | reliability diagram for the direction probability |
| return_distributions.png | realised vs predicted return distribution per horizon |
| model_comparison.png | per-horizon model ranking with direction accuracy |

Summary: `summary.md` (this document).

All twelve artefacts were written from the same pooled predictions by `src.v3.report`, so the numbers above and the numbers in the CSVs cannot disagree.
