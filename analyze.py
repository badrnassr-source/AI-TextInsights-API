from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_nlp_service
from app.schemas.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    SentimentRequest,
    SentimentResponse,
    SummaryRequest,
    SummaryResponse,
)
from app.services.nlp_service import NLPService, TextTooLongError

router = APIRouter(prefix="/analyze", tags=["analyze"])


@router.post("/sentiment", response_model=SentimentResponse)
async def sentiment(
    payload: SentimentRequest,
    nlp: NLPService = Depends(get_nlp_service),
) -> SentimentResponse:
    try:
        return await nlp.analyze_sentiment(payload.text, language_hint=payload.language_hint)
    except TextTooLongError as e:
        raise HTTPException(413, detail=str(e)) from e


@router.post("/summary", response_model=SummaryResponse)
async def summary(
    payload: SummaryRequest,
    nlp: NLPService = Depends(get_nlp_service),
) -> SummaryResponse:
    try:
        return await nlp.summarize(
            payload.text,
            max_sentences=payload.max_sentences,
            style=payload.style,
            language_hint=payload.language_hint,
        )
    except TextTooLongError as e:
        raise HTTPException(413, detail=str(e)) from e


@router.post("", response_model=AnalyzeResponse)
async def analyze_all(
    payload: AnalyzeRequest,
    nlp: NLPService = Depends(get_nlp_service),
) -> AnalyzeResponse:
    if not payload.include_sentiment and not payload.include_summary:
        raise HTTPException(422, detail="active au moins sentiment ou summary")
    try:
        return await nlp.analyze_full(
            payload.text,
            include_sentiment=payload.include_sentiment,
            include_summary=payload.include_summary,
            max_sentences=payload.max_sentences,
            language_hint=payload.language_hint,
        )
    except TextTooLongError as e:
        raise HTTPException(413, detail=str(e)) from e
