"""FastAPI routes exposing RAG context enrichment."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from monGARS.api.authentication import get_current_user
from monGARS.api.dependencies import get_rag_context_enricher
from monGARS.api.schemas import (
    RagContextRequest,
    RagContextResponse,
    RagReferenceSchema,
)
from monGARS.core.rag import (
    RagCodeReference,
    RagContextEnricher,
    RagDisabledError,
    RagEnrichmentResult,
    RagServiceError,
)

router = APIRouter(prefix="/api/v1/review", tags=["review"])
logger = logging.getLogger(__name__)


def _to_reference_schema(reference: RagCodeReference) -> RagReferenceSchema:
    return RagReferenceSchema(
        repository=reference.repository,
        file_path=reference.file_path,
        summary=reference.summary,
        score=reference.score,
        url=reference.url,
    )


def _to_response(result: RagEnrichmentResult) -> RagContextResponse:
    return RagContextResponse(
        enabled=True,
        focus_areas=result.focus_areas,
        references=[_to_reference_schema(ref) for ref in result.references],
    )


@router.post("/rag-context", response_model=RagContextResponse)
async def fetch_rag_context(
    request: RagContextRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
    enricher: Annotated[RagContextEnricher, Depends(get_rag_context_enricher)],
) -> RagContextResponse:
    """Return contextual code references relevant to the supplied query."""

    user_id = current_user.get("sub")
    logger.info(
        "api.review.rag_context.request",
        extra={
            "user_id": user_id,
            "repositories": request.repositories or "default",
            "query_length": len(request.query),
        },
    )
    try:
        result = await enricher.enrich(
            request.query,
            repositories=request.repositories,
            max_results=request.max_results,
        )
    except RagDisabledError as exc:
        logger.info(
            "api.review.rag_context.disabled",
            extra={"user_id": user_id},
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG disabled",
        ) from exc
    except RagServiceError as exc:
        logger.warning(
            "api.review.rag_context.service_unavailable",
            extra={"error": str(exc), "user_id": user_id},
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG context enrichment service is unavailable.",
        ) from exc
    except ValueError as exc:
        logger.debug(
            "api.review.rag_context.invalid_query",
            extra={"error": str(exc), "user_id": user_id},
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    return _to_response(result)
