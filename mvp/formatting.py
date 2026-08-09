"""
Pure display-formatting helpers for the UI — no `import streamlit`, so they're testable
directly, same pattern as mvp/pipeline.py and mvp/experiments.py.
"""

from datetime import datetime, timezone


def format_relative_time(iso_timestamp, now=None):
    """
    '2026-07-23T08:00:44+00:00' -> 'just now' / '5 minutes ago' / '3 hours ago' /
    '2 days ago' / '3 weeks ago' / a plain date once it's old enough that "N units ago"
    stops being useful.

    now: inject a fixed "current time" for deterministic tests; defaults to the real current
    UTC time.
    """
    then = datetime.fromisoformat(iso_timestamp)
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)

    now = now or datetime.now(timezone.utc)
    delta_seconds = (now - then).total_seconds()

    if delta_seconds < 0:
        return "just now"  # clock skew between processes; don't show a negative duration
    if delta_seconds < 60:
        return "just now"
    if delta_seconds < 3600:
        minutes = int(delta_seconds // 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    if delta_seconds < 86400:
        hours = int(delta_seconds // 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    if delta_seconds < 86400 * 7:
        days = int(delta_seconds // 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"
    if delta_seconds < 86400 * 90:
        weeks = int(delta_seconds // (86400 * 7))
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"

    return then.strftime("%Y-%m-%d")
