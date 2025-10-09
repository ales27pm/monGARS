# Iris Retrieval Service Overview

## Implementation Status

Iris is a fully implemented asynchronous web retrieval helper that ships with the core cognition
stack. The implementation at `monGARS/core/iris.py` provides typed document models, bounded
concurrency, retry-aware HTTP fetching via `httpx`, content-type validation, trafilatura-based
extraction, and BeautifulSoup fallbacks so that responses always degrade gracefully when the
primary extractor fails. Input validation guards against invalid URLs, oversized payloads, and
non-textual responses, ensuring the class can be used directly in production workflows without
additional wrappers. The behaviour is verified through dedicated unit tests in
`tests/test_iris.py`, which cover success paths, retry handling, invalid inputs, caching logic, and
snippet selection. The broader `docs/implementation_status.md` report also lists Iris scraping as
part of the completed Phase 2 functional expansion, confirming it is part of the supported feature
set.

## Integration Inside monGARS

Within the cognition pipeline, Iris is instantiated by the `CuriosityEngine` (`monGARS/core/cortex/
curiosity_engine.py`). During a conversation, the engine first attempts to satisfy information gaps
through the internal document retrieval service. When that service does not return data, the engine
falls back to `Iris.search`, which issues a DuckDuckGo lookup, resolves the first result, and calls
`Iris.fetch_document` to extract a concise snippet. The retrieved context is cached so repeated
questions reuse previous research, and OpenTelemetry counters are incremented to distinguish between
successful document service hits and Iris fallbacks. This workflow allows monGARS to enrich
conversations with up-to-date public information while still preferring curated internal knowledge.
