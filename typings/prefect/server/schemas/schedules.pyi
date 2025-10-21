from __future__ import annotations

from datetime import datetime, timedelta

class IntervalSchedule:
    def __init__(
        self,
        *,
        interval: timedelta,
        anchor_date: datetime | None = ...,
    ) -> None: ...

__all__ = ["IntervalSchedule"]
