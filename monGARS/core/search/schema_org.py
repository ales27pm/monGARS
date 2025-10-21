"""Parse schema.org metadata into a normalised structure."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Iterator, Mapping, Sequence

from bs4 import BeautifulSoup

from .metadata import parse_date_from_text, parse_schema_dates


@dataclass(slots=True)
class SchemaOrgMetadata:
    """Normalised subset of schema.org metadata relevant to Iris."""

    date_published: datetime | None = None
    date_modified: datetime | None = None
    event_start: datetime | None = None
    event_end: datetime | None = None
    authors: list[str] | None = None
    publisher: str | None = None
    organization: str | None = None
    location_name: str | None = None

    def is_empty(self) -> bool:
        return all(
            value in (None, [], "")
            for value in (
                self.date_published,
                self.date_modified,
                self.event_start,
                self.event_end,
                self.authors,
                self.publisher,
                self.organization,
                self.location_name,
            )
        )


def _ensure_timezone(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    candidates = text.split("/") if "/" in text else [text]
    for candidate in candidates:
        normalised = candidate.strip()
        if not normalised:
            continue
        try:
            iso_normalised = (
                normalised[:-1] + "+00:00" if normalised.endswith("Z") else normalised
            )
            parsed = datetime.fromisoformat(iso_normalised)
        except ValueError:
            parsed = parse_date_from_text(normalised)
        if parsed is not None:
            return _ensure_timezone(parsed)
    return None


def _flatten_payload(payload: object) -> Iterator[Mapping[str, object]]:
    if isinstance(payload, Mapping):
        graph = payload.get("@graph")
        if isinstance(graph, Sequence) and not isinstance(graph, (str, bytes)):
            for item in graph:
                yield from _flatten_payload(item)
        yield payload
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        for item in payload:
            yield from _flatten_payload(item)


def _extract_names(value: object) -> list[str]:
    names: list[str] = []
    queue: list[object] = [value]
    while queue:
        current = queue.pop(0)
        if current is None:
            continue
        if isinstance(current, str):
            cleaned = current.strip()
            if cleaned:
                names.append(cleaned)
            continue
        if isinstance(current, Mapping):
            name_value = current.get("name")
            if isinstance(name_value, str) and name_value.strip():
                names.append(name_value.strip())
            else:
                queue.extend(current.values())
            continue
        if isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            queue.extend(current)
    # Deduplicate while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        lowered = name.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        ordered.append(name)
    return ordered


def _coalesce(values: Iterable[str]) -> str | None:
    for value in values:
        cleaned = value.strip()
        if cleaned:
            return cleaned
    return None


def _extract_location(block: Mapping[str, object]) -> str | None:
    candidates = []
    for key in ("location", "contentLocation", "locationCreated", "spatialCoverage"):
        if key not in block:
            continue
        candidates.extend(_extract_names(block[key]))
    if "address" in block and isinstance(block["address"], Mapping):
        components = _extract_names(block["address"])
        if components:
            candidates.append(", ".join(components))
    return _coalesce(candidates)


def parse_schema_org(html: str) -> SchemaOrgMetadata | None:
    """Parse schema.org JSON-LD blocks from *html* into a structured object."""

    soup = BeautifulSoup(html, "html.parser")
    metadata = SchemaOrgMetadata()
    for tag in soup.find_all("script", {"type": "application/ld+json"}):
        text = tag.string or tag.text
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        for block in _flatten_payload(payload):
            if not isinstance(block, Mapping):
                continue
            published = block.get("datePublished") or block.get("dateCreated")
            modified = block.get("dateModified") or block.get("lastReviewed")
            start = (
                block.get("startDate")
                or block.get("startTime")
                or block.get("temporalCoverage")
            )
            end = block.get("endDate") or block.get("endTime")
            authors = block.get("author") or block.get("creator")
            publisher = block.get("publisher")
            organization = (
                block.get("sourceOrganization")
                or block.get("organization")
                or block.get("organizer")
                or block.get("provider")
            )
            location_name = _extract_location(block)

            metadata.date_published = metadata.date_published or _parse_datetime(
                published
            )
            metadata.date_modified = metadata.date_modified or _parse_datetime(modified)
            metadata.event_start = metadata.event_start or _parse_datetime(start)
            metadata.event_end = metadata.event_end or _parse_datetime(end)

            author_names = _extract_names(authors)
            if author_names:
                if metadata.authors is None:
                    metadata.authors = author_names
                else:
                    seen = {name.lower() for name in metadata.authors}
                    for name in author_names:
                        if name.lower() not in seen:
                            metadata.authors.append(name)
                            seen.add(name.lower())

            publisher_name = _coalesce(_extract_names(publisher))
            if publisher_name and metadata.publisher is None:
                metadata.publisher = publisher_name

            organisation_name = _coalesce(_extract_names(organization))
            if organisation_name and metadata.organization is None:
                metadata.organization = organisation_name

            if location_name and metadata.location_name is None:
                metadata.location_name = location_name

    if metadata.is_empty():
        # Fall back to basic meta tags if JSON-LD was absent or empty
        author_meta = [
            tag.get("content", "")
            for tag in soup.find_all("meta", attrs={"name": "author"})
        ]
        if author_meta:
            cleaned = [name.strip() for name in author_meta if name and name.strip()]
            if cleaned:
                metadata.authors = cleaned
        publisher_meta = soup.find("meta", attrs={"property": "og:site_name"})
        if publisher_meta and publisher_meta.get("content"):
            metadata.publisher = publisher_meta["content"].strip() or metadata.publisher
        location_meta = soup.find("meta", attrs={"property": "event:location"})
        if location_meta and location_meta.get("content"):
            metadata.location_name = (
                location_meta["content"].strip() or metadata.location_name
            )

    if metadata.date_published is None or metadata.event_start is None:
        fallback_event, fallback_pub = parse_schema_dates(html)
        fallback_event = _ensure_timezone(fallback_event)
        fallback_pub = _ensure_timezone(fallback_pub)
        if metadata.date_published is None:
            metadata.date_published = fallback_pub
        if metadata.event_start is None:
            metadata.event_start = fallback_event

    return None if metadata.is_empty() else metadata


__all__ = ["SchemaOrgMetadata", "parse_schema_org"]
