# Iris Retrieval Service Overview

## Implementation Status

Iris is a fully implemented asynchronous web retrieval helper that ships with the core cognition
stack. The implementation at `monGARS/core/iris.py` provides typed document models, bounded
concurrency, retry-aware HTTP fetching via a pooled `httpx.AsyncClient`, content-type validation,
trafilatura-based extraction, and BeautifulSoup fallbacks so responses degrade gracefully whenever
the primary extractor struggles. Input validation guards against invalid URLs, oversized payloads,
and non-textual responses, while cooperative throttling limits outbound request volume.

The latest refactor layers in two additional capabilities that align Iris with modern search
helpers:

- **Dual-tier caching** — Search snippets and fully extracted documents now use separate TTL-bound
  LRU caches so repeated requests avoid redundant network hops while still respecting freshness
  requirements. In-flight fetches are coalesced, ensuring concurrent callers share a single network
  request rather than stampeding an origin server.
- **Snippet intelligence** — Normalised summaries and sentence-aware snippet selection keep the
  returned context concise and relevant, trimming overly long extracts automatically.

The behaviour is verified through dedicated unit tests in `tests/test_iris.py`, which now cover
concurrency coalescing, cache reuse/expiry, success paths, retry handling, invalid inputs, and
snippet selection. The broader `docs/implementation_status.md` report also lists Iris scraping as
part of the completed Phase 2 functional expansion, confirming it is part of the supported feature
set.

## Integration Inside monGARS

Within the cognition pipeline, Iris is instantiated by the `CuriosityEngine` (`monGARS/core/cortex/
curiosity_engine.py`). During a conversation, the engine first attempts to satisfy information gaps
through the internal document retrieval service. When that service does not return data, the engine
falls back to `Iris.search`, which issues a DuckDuckGo lookup, resolves the first result, and calls
`Iris.fetch_document` to extract a concise snippet. The retrieved context flows through the new
document cache so repeated questions reuse previous research, and OpenTelemetry counters are
incremented to distinguish between successful document service hits and Iris fallbacks. This
workflow allows monGARS to enrich conversations with up-to-date public information while still
preferring curated internal knowledge.

## Runtime Optimisations

Iris exposes a handful of knobs so operators can tune throughput, freshness, and resource usage for
their workloads:

- `search_cache_ttl` / `search_cache_size`: control the snippet cache horizon and footprint.
- `document_cache_ttl` / `document_cache_size`: govern how long fully extracted pages remain
  reusable before expiring.
- `max_concurrency` and the internal semaphore: protect upstream services from bursts of parallel
  fetches while keeping enough concurrency to hide latency.
- Persistent HTTP client lifecycle (`aclose` / async context manager): amortises TLS handshakes and
  allows deterministic shutdown when Iris is embedded in longer-lived services.

These controls make it easy to tune Iris for aggressive live research (short TTLs with smaller
caches) or high-throughput assistants that revisit topics frequently (longer TTLs and larger cache
windows). Concurrency-sensitive deployments can further dial the semaphore and retry backoff to
respect rate limits from external search providers.

## Opportunities for Further Refinement

The latest refactor addresses the biggest performance hotspots, but a few enhancements could tighten
observability and accuracy even further:

- **Adaptive extraction heuristics** – Tracking extraction quality would allow Iris to automatically
  fall back to cached snippets when HTML is unusually noisy.
- **Structured telemetry** – Emitting counters for cache hits, inflight coalescing, and retry
  outcomes would help operators size TTLs and concurrency without manual tracing.
- **Domain-aware normalisation** – Adding site-specific cleaners (e.g., removing cookie banners or
  boilerplate) would improve snippet quality for news outlets and documentation portals.

These follow-on ideas are optional for correctness, but they would provide clearer observability and
slightly faster turnaround on hot queries.
