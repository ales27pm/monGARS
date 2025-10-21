"""Domain allow/deny policy and per-host rate budgeting for search hits."""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class HostBudget:
    """Token bucket budget for a single host."""

    capacity_per_min: int = 60
    tokens: float = 60.0
    last_ts: float = time.monotonic()


class DomainPolicy:
    """Apply allow/deny filters and simple per-host budgets."""

    def __init__(
        self,
        allow_patterns: list[str] | None = None,
        deny_patterns: list[str] | None = None,
        *,
        per_host_budget: int = 60,
    ) -> None:
        self.allow_patterns = [
            re.compile(pattern) for pattern in (allow_patterns or [])
        ]
        self.deny_patterns = [re.compile(pattern) for pattern in (deny_patterns or [])]
        self._budgets: dict[str, HostBudget] = {}
        self._lock = asyncio.Lock()
        self.per_host_budget = per_host_budget

    def is_allowed_domain(self, domain: str) -> bool:
        domain = domain.lower()
        if any(pattern.search(domain) for pattern in self.deny_patterns):
            return False
        if self.allow_patterns and not any(
            pattern.search(domain) for pattern in self.allow_patterns
        ):
            return False
        return True

    async def acquire_budget(self, domain: str) -> bool:
        """Consume a single token for ``domain`` if capacity is available."""

        async with self._lock:
            budget = self._budgets.get(domain)
            if budget is None:
                budget = HostBudget(
                    capacity_per_min=self.per_host_budget,
                    tokens=float(self.per_host_budget),
                    last_ts=time.monotonic(),
                )
                self._budgets[domain] = budget
            now = time.monotonic()
            elapsed = now - budget.last_ts
            if budget.capacity_per_min <= 0:
                budget.capacity_per_min = 1
            refill_rate = budget.capacity_per_min / 60.0
            budget.tokens = min(
                budget.capacity_per_min,
                budget.tokens + elapsed * refill_rate,
            )
            budget.last_ts = now
            if budget.tokens >= 1.0:
                budget.tokens -= 1.0
                return True
            return False


__all__ = ["DomainPolicy", "HostBudget"]
