from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class ProviderInfo(BaseModel):
    # vader = local, llm = appel OpenAI (ou compatible)
    engine: Literal["llm", "vader"]
    model: Optional[str] = None


class TextInput(BaseModel):
    text: str = Field(..., min_length=1)

    @field_validator("text")
    @classmethod
    def no_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text vide")
        return v


class SentimentRequest(TextInput):
    language_hint: Optional[str] = None


class SentimentResponse(BaseModel):
    label: SentimentLabel
    score: float = Field(..., ge=-1.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    provider: ProviderInfo


class SummaryRequest(TextInput):
    max_sentences: int = Field(default=3, ge=1, le=10)
    style: Literal["concise", "bullet", "paragraph"] = "concise"
    language_hint: Optional[str] = None


class SummaryResponse(BaseModel):
    summary: str
    original_length: int
    summary_length: int
    provider: ProviderInfo


class AnalyzeRequest(TextInput):
    include_sentiment: bool = True
    include_summary: bool = True
    max_sentences: int = Field(default=3, ge=1, le=10)
    language_hint: Optional[str] = None


class AnalyzeResponse(BaseModel):
    sentiment: Optional[SentimentResponse] = None
    summary: Optional[SummaryResponse] = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    llm_configured: bool
