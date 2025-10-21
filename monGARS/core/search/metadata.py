"""Helpers for extracting structured dates from HTML content."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Optional

from bs4 import BeautifulSoup

_DATE_PATTERN = re.compile(
    r"""
(?:
  (?P<y>\d{4})[-/.](?P<m>\d{1,2})[-/.](?P<d>\d{1,2})
|
  (?P<mon>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+(?P<d2>\d{1,2}),\s*(?P<y2>\d{4})
)
""",
    re.IGNORECASE | re.VERBOSE,
)

_MONTHS = {
    name.lower(): index
    for index, name in enumerate(
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
        start=1,
    )
}


def _make_dt(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


def _safe_parse(**groups: Optional[str]) -> Optional[datetime]:
    try:
        year = groups.get("y")
        month = groups.get("m")
        day = groups.get("d")
        if year and month and day:
            return _make_dt(int(year), int(month), int(day))
        month_name = groups.get("mon")
        d2 = groups.get("d2")
        y2 = groups.get("y2")
        if month_name and d2 and y2:
            return _make_dt(int(y2), _MONTHS[month_name.lower()[:3]], int(d2))
    except (KeyError, ValueError):
        return None
    return None


def parse_date_from_text(text: str | None) -> Optional[datetime]:
    if not text:
        return None
    match = _DATE_PATTERN.search(text)
    if not match:
        return None
    return _safe_parse(**match.groupdict())


def _iso_or_text(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    try:
        formatted = value[:-1] + "+00:00" if value.endswith("Z") else value
        dt = datetime.fromisoformat(formatted)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return parse_date_from_text(value)


def parse_schema_dates(html: str) -> tuple[Optional[datetime], Optional[datetime]]:
    """Return event and publication datetimes derived from structured HTML."""

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            payload = json.loads(tag.string or "{}")
        except json.JSONDecodeError:
            continue
        blocks = payload if isinstance(payload, list) else [payload]
        for block in blocks:
            if not isinstance(block, dict):
                continue
            pub = block.get("datePublished") or block.get("dateCreated")
            event = (
                block.get("startDate")
                or block.get("date")
                or block.get("temporalCoverage")
            )
            pub_dt = _iso_or_text(pub if isinstance(pub, str) else None)
            event_dt = _iso_or_text(event if isinstance(event, str) else None)
            if pub_dt or event_dt:
                return event_dt, pub_dt

    pub_meta = soup.find("meta", {"property": "article:published_time"})
    if pub_meta is None:
        pub_meta = soup.find("meta", {"name": "date"})
    event_meta = soup.find("meta", {"property": "event:start_time"})
    if event_meta is None:
        event_meta = soup.find("meta", {"name": "event_start"})
    pub_dt = (
        _iso_or_text(pub_meta["content"])
        if pub_meta and pub_meta.get("content")
        else None
    )
    event_dt = (
        _iso_or_text(event_meta["content"])
        if event_meta and event_meta.get("content")
        else None
    )
    return event_dt, pub_dt


__all__ = ["parse_schema_dates", "parse_date_from_text"]
