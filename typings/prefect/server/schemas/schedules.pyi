from __future__ import annotations

from datetime import datetime, timedelta

class IntervalSchedule:
    def __init__(
        self,
        *,
        interval: timedelta | int | float,
        anchor_date: datetime | None = ...,
        timezone: str | None = ...,
    ) -> None: ...

__all__ = ["IntervalSchedule"]
