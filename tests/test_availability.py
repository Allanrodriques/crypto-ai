"""Availability-aware alignment: the guarantees the feature groups depend on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.alignment.availability import (
    AlignmentError,
    AvailabilityContract,
    align_asof,
    audit_no_lookahead,
    coverage_report,
    prediction_timestamps,
)


def _source(start: str, periods: int, freq: str, column: str = "value", **extra) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    frame = pd.DataFrame({column: np.arange(periods, dtype=float)}, index=index)
    for key, value in extra.items():
        frame[key] = value
    return frame


# --------------------------------------------------------------------------- no look-ahead

def test_alignment_never_uses_a_future_observation():
    """A value stamped at T may only come from an observation available at or before T."""
    source = _source("2024-01-01", 6, "1h")
    contract = AvailabilityContract(name="hourly", frequency="1h")
    target = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")

    aligned = align_asof(target, source, contract, value_columns=["value"])

    for stamp, value in zip(aligned.index, aligned["value"]):
        if pd.isna(value):
            continue
        assert stamp >= source.index[0]
        # The supplying observation's event time can never exceed the feature time.
        supplying = source.index[source.index <= stamp].max()
        assert value == float(source.index.get_loc(supplying))


def test_release_lag_shifts_every_observation_later():
    """A daily publication known only at T+1d must not appear at T."""
    source = _source("2024-01-01", 5, "1D")
    contract = AvailabilityContract(
        name="daily_news", frequency="1D", release_lag=pd.Timedelta(days=1)
    )
    target = pd.date_range("2024-01-01", periods=8, freq="1D", tz="UTC")

    aligned = align_asof(target, source, contract, value_columns=["value"])

    # Row 0 is the event itself but is not available until row 1.
    assert pd.isna(aligned["value"].iloc[0])
    assert aligned["value"].iloc[1] == 0.0
    assert aligned["value"].iloc[2] == 1.0


def test_intrabar_flag_is_carried_but_not_used_to_look_ahead():
    """`intrabar` documents granularity; it must not license a same-bar read."""
    source = _source("2024-01-01", 3, "1h")
    contract = AvailabilityContract(name="intrabar", frequency="1h", intrabar=True)
    target = source.index[:2]
    aligned = align_asof(target, source, contract, value_columns=["value"])
    assert aligned["value"].iloc[0] == 0.0  # exact match at its own stamp is fine
    assert aligned["value"].iloc[1] == 1.0


# --------------------------------------------------------------------------- max_age

def test_max_age_expires_a_stale_value():
    """A contract with max_age must stop carrying a value forward.

    This is the regression test for a real bug: `max_age` compared the target
    timestamp against the *joined* index, which is the target index, so the age
    was always zero and staleness never triggered.
    """
    source = _source("2024-01-01", 1, "1h")
    contract = AvailabilityContract(name="stale_source", frequency="1h", max_age=pd.Timedelta(hours=2))
    target = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")

    aligned = align_asof(target, source, contract, value_columns=["value"])

    assert aligned["value"].iloc[0] == 0.0
    assert aligned["value"].iloc[1] == 0.0
    assert aligned["value"].iloc[2] == 0.0
    # Three hours later the only observation is 3h old, past the 2h limit.
    assert pd.isna(aligned["value"].iloc[3])
    assert aligned["value"].isna().iloc[3:].all()


def test_max_age_none_carries_forward_indefinitely():
    source = _source("2024-01-01", 1, "1h")
    contract = AvailabilityContract(name="eternal", frequency="1h", max_age=None)
    target = pd.date_range("2024-01-01", periods=20, freq="1h", tz="UTC")
    aligned = align_asof(target, source, contract, value_columns=["value"])
    assert aligned["value"].notna().all()


# --------------------------------------------------------------------------- provenance

def test_provenance_records_the_supplying_availability_time():
    source = _source("2024-01-01", 3, "1h")
    contract = AvailabilityContract(name="prov", frequency="1h")
    target = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")

    aligned = align_asof(target, source, contract, value_columns=["value"])

    provenance = aligned.attrs["source_available_at"]
    assert provenance[0] == source.index[0]
    assert provenance[2] == source.index[2]
    # With no max_age the final observation legitimately carries forward, and the
    # provenance says so - it names the *source* time, not the feature time.
    assert provenance[3] == source.index[2]
    assert provenance[4] == source.index[2]


def test_provenance_is_missing_before_the_first_observation():
    """Before any source exists there is nothing to attribute a value to."""
    source = _source("2024-01-01 02:00", 2, "1h")
    contract = AvailabilityContract(name="late", frequency="1h")
    target = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")

    aligned = align_asof(target, source, contract, value_columns=["value"])

    provenance = aligned.attrs["source_available_at"]
    assert pd.isna(provenance[0])
    assert pd.isna(aligned["value"].iloc[0])
    assert provenance[2] == source.index[0]


def test_audit_passes_on_a_clean_alignment():
    source = _source("2024-01-01", 4, "1h")
    contract = AvailabilityContract(name="clean", frequency="1h")
    target = pd.date_range("2024-01-01", periods=8, freq="1h", tz="UTC")
    aligned = align_asof(target, source, contract, value_columns=["value"])
    audit_no_lookahead(aligned, contract, target, "value")  # must not raise


def test_audit_rejects_missing_provenance():
    """Without recorded provenance the audit must refuse to certify anything."""
    source = _source("2024-01-01", 3, "1h")
    contract = AvailabilityContract(name="noprov", frequency="1h")
    target = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
    aligned = align_asof(target, source, contract, value_columns=["value"])
    stripped = aligned.copy()
    stripped.attrs = {}
    with pytest.raises(AlignmentError, match="provenance"):
        audit_no_lookahead(stripped, contract, target, "value")


def test_audit_catches_a_planted_look_ahead():
    """A hand-corrupted provenance vector must be detected, not trusted."""
    source = _source("2024-01-01", 4, "1h")
    contract = AvailabilityContract(name="planted", frequency="1h")
    target = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
    aligned = align_asof(target, source, contract, value_columns=["value"])

    bad = pd.DatetimeIndex(aligned.attrs["source_available_at"]).to_series()
    bad.iloc[1] = target[1] + pd.Timedelta(days=1)  # published tomorrow
    aligned.attrs["source_available_at"] = pd.DatetimeIndex(bad.to_numpy())

    with pytest.raises(AlignmentError, match="look-ahead"):
        audit_no_lookahead(aligned, contract, target, "value")


def test_audit_catches_a_value_past_max_age():
    """A populated value whose provenance is older than max_age must be rejected.

    The frame is built by hand rather than through ``align_asof`` because
    ``align_asof`` correctly expires such a value to NaN itself; the audit is the
    independent second line of defence, so it has to be tested against a frame
    that bypassed that expiry.
    """
    target = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    contract = AvailabilityContract(name="old", frequency="1h", max_age=pd.Timedelta(minutes=30))
    stale_source_time = target - pd.Timedelta(hours=4)
    aligned = pd.DataFrame({"value": [0.0, 1.0, 2.0, 3.0, 4.0]}, index=target)
    aligned.attrs["source_available_at"] = stale_source_time

    with pytest.raises(AlignmentError, match="max_age"):
        audit_no_lookahead(aligned, contract, target, "value")


# --------------------------------------------------------------------------- misc

def test_explicit_available_at_column_wins_over_release_lag():
    """A source that states its own publication time needs no added lag."""
    source = _source("2024-01-01", 2, "1D")
    source["available_at"] = source.index + pd.Timedelta(hours=6)
    contract = AvailabilityContract(
        name="stated", frequency="1D", release_lag=pd.Timedelta(days=30)
    )
    target = pd.date_range("2024-01-01", periods=3, freq="6h", tz="UTC")

    aligned = align_asof(target, source, contract, value_columns=["value"])

    # The stated 6h publication decides, not the 30d contract lag: the value
    # appears at 06:00 and is still absent at 00:00.  The second observation is
    # not published until the following day, so 12:00 still sees the first.
    assert pd.isna(aligned["value"].iloc[0])
    assert aligned["value"].iloc[1] == 0.0
    assert aligned["value"].iloc[2] == 0.0


def test_alignment_rejects_a_non_datetime_source_index():
    source = pd.DataFrame({"value": [1.0]}, index=["not-a-time"])
    contract = AvailabilityContract(name="bad", frequency="1h")
    with pytest.raises(AlignmentError, match="DatetimeIndex"):
        align_asof(pd.date_range("2024-01-01", periods=1, freq="1h", tz="UTC"), source, contract)


def test_duplicate_availability_timestamps_keep_the_latest_state():
    index = pd.DatetimeIndex(["2024-01-01", "2024-01-01"], tz="UTC")
    source = pd.DataFrame({"value": [1.0, 2.0]}, index=index)
    contract = AvailabilityContract(name="dup", frequency="1h")
    target = pd.date_range("2024-01-01", periods=1, freq="1h", tz="UTC")
    aligned = align_asof(target, source, contract, value_columns=["value"])
    assert aligned["value"].iloc[0] == 2.0


def test_prediction_timestamps_are_one_bar_later():
    feature_times = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    predictions = prediction_timestamps(feature_times, "1h")
    assert (predictions - feature_times == pd.Timedelta(hours=1)).all()


def test_coverage_report_counts_populated_rows():
    """A max_age of zero means each observation is usable only at its own stamp."""
    source = _source("2024-01-01", 3, "1h")
    contract = AvailabilityContract(
        name="cov", frequency="1h", max_age=pd.Timedelta(0)
    )
    target = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
    aligned = align_asof(target, source, contract, value_columns=["value"])
    report = coverage_report(aligned, len(target))
    assert report["rows"] == 10
    assert report["covered"] == 3
    assert report["coverage_pct"] == 30.0
    assert report["first_covered"] == str(target[0])


def test_coverage_report_without_max_age_carries_forward():
    source = _source("2024-01-01", 1, "1h")
    contract = AvailabilityContract(name="hold", frequency="1h", max_age=None)
    target = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
    aligned = align_asof(target, source, contract, value_columns=["value"])
    assert coverage_report(aligned, len(target))["coverage_pct"] == 100.0
