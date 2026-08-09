from datetime import datetime, timezone, timedelta

from mvp.formatting import format_relative_time

_NOW = datetime(2026, 7, 23, 12, 0, 0, tzinfo=timezone.utc)


def _ago(**kwargs):
    return (_NOW - timedelta(**kwargs)).isoformat()


def test_just_now_for_recent_timestamps():
    assert format_relative_time(_ago(seconds=10), now=_NOW) == "just now"
    assert format_relative_time(_ago(seconds=59), now=_NOW) == "just now"


def test_minutes_ago():
    assert format_relative_time(_ago(minutes=1), now=_NOW) == "1 minute ago"
    assert format_relative_time(_ago(minutes=5), now=_NOW) == "5 minutes ago"


def test_hours_ago():
    assert format_relative_time(_ago(hours=1), now=_NOW) == "1 hour ago"
    assert format_relative_time(_ago(hours=3), now=_NOW) == "3 hours ago"


def test_days_ago_up_to_a_week():
    assert format_relative_time(_ago(days=1), now=_NOW) == "1 day ago"
    assert format_relative_time(_ago(days=6), now=_NOW) == "6 days ago"


def test_weeks_ago_after_a_week():
    """Regression test for a real bug found while building this: the first version used a
    30-day threshold before switching to weeks, so '21 days ago' never actually became
    '3 weeks ago' — caught by testing an actual 21-day case, not just the boundary values."""
    assert format_relative_time(_ago(days=10), now=_NOW) == "1 week ago"
    assert format_relative_time(_ago(days=21), now=_NOW) == "3 weeks ago"


def test_falls_back_to_a_plain_date_when_old_enough():
    old = _NOW - timedelta(days=400)
    result = format_relative_time(old.isoformat(), now=_NOW)
    assert result == old.strftime("%Y-%m-%d")


def test_handles_naive_iso_timestamps_without_timezone():
    """datetime.isoformat() without tzinfo (no +00:00 suffix) shouldn't crash — treated as
    UTC, matching what experiments.py actually stores."""
    naive = (_NOW - timedelta(hours=2)).replace(tzinfo=None).isoformat()
    assert format_relative_time(naive, now=_NOW) == "2 hours ago"


def test_future_timestamp_does_not_show_negative_duration():
    """Clock skew between processes shouldn't produce something like '-5 minutes ago'."""
    future = (_NOW + timedelta(minutes=5)).isoformat()
    result = format_relative_time(future, now=_NOW)
    assert "-" not in result
    assert result == "just now"


def test_real_experiment_timestamp_format():
    """Uses the actual isoformat() output shape experiments.create_experiment() produces."""
    from datetime import datetime as dt
    real_format = dt.now(timezone.utc).isoformat()
    result = format_relative_time(real_format)  # no `now=` override — real current time
    assert result == "just now"
