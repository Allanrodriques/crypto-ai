"""crypto-ml-predictor — research ML pipeline for cryptocurrency direction prediction.

Layering (each layer only depends on the ones above it):

    data      Binance klines  -> validated OHLCV series
    features  causal technical indicators (no future data, ever)
    dataset   features + label -> leakage-guarded chronological splits
    models    baseline / logistic regression / random forest / XGBoost
    evaluation metrics, time-series CV, backtest
    pipeline  orchestration: train, predict, report
"""
