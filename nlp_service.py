from __future__ import annotations

from typing import Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app.config import Settings
from app.schemas.models import (
    AnalyzeResponse,
    ProviderInfo,
    SentimentLabel,
    SentimentResponse,
    SummaryResponse,
)
from app.services.llm_client import LLMClient, LLMClientError

_analyzer = SentimentIntensityAnalyzer()


class TextTooLongError(ValueError):
    pass


class NLPService:
    def __init__(self, settings: Settings, llm: LLMClient) -> None:
        self._settings = settings
        self._llm = llm

    def _validate_length(self, text: str) -> None:
        if len(text) > self._settings.max_text_length:
            raise TextTooLongError(
                f"Texte trop long ({len(text)} car.). Maximum: {self._settings.max_text_length}."
            )

    async def analyze_sentiment(
        self,
        text: str,
        *,
        language_hint: Optional[str] = None,
    ) -> SentimentResponse:
        self._validate_length(text)
        if self._settings.llm_enabled:
            return await self._sentiment_llm(text, language_hint)
        return self._sentiment_vader(text)

    async def summarize(
        self,
        text: str,
        *,
        max_sentences: int = 3,
        style: str = "concise",
        language_hint: Optional[str] = None,
    ) -> SummaryResponse:
        self._validate_length(text)
        # pas de clé API -> résumé maison (extractif), ça suffit pour la démo
        if not self._settings.llm_enabled:
            return self._summarize_extractive(text, max_sentences)

        system = (
            "Tu es un assistant NLP. Réponds uniquement en JSON avec les clés: "
            '"summary" (string), "language" (string ISO court). '
            "Le résumé doit être fidèle au texte source, sans inventer de faits."
        )
        lang = language_hint or "auto"
        style_hint = {
            "concise": "phrases courtes et directes",
            "bullet": "puces markdown (- item)",
            "paragraph": "un paragraphe fluide",
        }.get(style, "concise")

        user = (
            f"Langue cible: {lang}. Style: {style_hint}. "
            f"Longueur max: {max_sentences} phrase(s) ou équivalent.\n\n"
            f"Texte:\n{text}"
        )

        try:
            data = await self._llm.chat_json(system, user, temperature=0.3)
            summary = str(data.get("summary", "")).strip()
            if not summary:
                raise LLMClientError("Résumé vide.")
        except LLMClientError:
            return self._summarize_extractive(text, max_sentences)

        return SummaryResponse(
            summary=summary,
            original_length=len(text),
            summary_length=len(summary),
            provider=ProviderInfo(engine="llm", model=self._llm.model_name),
        )

    async def analyze_full(
        self,
        text: str,
        *,
        include_sentiment: bool = True,
        include_summary: bool = True,
        max_sentences: int = 3,
        language_hint: Optional[str] = None,
    ) -> AnalyzeResponse:
        sentiment = None
        summary = None
        if include_sentiment:
            sentiment = await self.analyze_sentiment(text, language_hint=language_hint)
        if include_summary:
            summary = await self.summarize(
                text,
                max_sentences=max_sentences,
                language_hint=language_hint,
            )
        return AnalyzeResponse(sentiment=sentiment, summary=summary)

    def _sentiment_vader(self, text: str) -> SentimentResponse:
        scores = _analyzer.polarity_scores(text)
        compound = scores["compound"]
        label, confidence = _label_from_compound(compound, scores)
        return SentimentResponse(
            label=label,
            score=round(compound, 4),
            confidence=round(confidence, 4),
            provider=ProviderInfo(engine="vader"),
        )

    async def _sentiment_llm(
        self,
        text: str,
        language_hint: Optional[str],
    ) -> SentimentResponse:
        system = (
            "Analyse le sentiment du texte. JSON uniquement avec: "
            '"label" ("positive"|"neutral"|"negative"), '
            '"score" (float entre -1 et 1), '
            '"confidence" (float entre 0 et 1).'
        )
        lang = language_hint or "auto"
        user = f"Langue: {lang}\n\nTexte:\n{text}"

        try:
            data = await self._llm.chat_json(system, user, temperature=0.0)
            label = SentimentLabel(str(data.get("label", "neutral")).lower())
            score = float(data.get("score", 0.0))
            confidence = float(data.get("confidence", 0.5))
            score = max(-1.0, min(1.0, score))
            confidence = max(0.0, min(1.0, confidence))
        except (LLMClientError, ValueError, KeyError):
            return self._sentiment_vader(text)

        return SentimentResponse(
            label=label,
            score=round(score, 4),
            confidence=round(confidence, 4),
            provider=ProviderInfo(engine="llm", model=self._llm.model_name),
        )

    def _summarize_extractive(self, text: str, max_sentences: int) -> SummaryResponse:
        sentences = _split_sentences(text)
        if not sentences:
            summary = text[:500]
        elif len(sentences) <= max_sentences:
            summary = " ".join(sentences)
        else:
            ranked = sorted(
                sentences,
                key=lambda s: (_word_score(s), len(s)),
                reverse=True,
            )
            top = ranked[:max_sentences]
            summary = " ".join(top)

        return SummaryResponse(
            summary=summary.strip(),
            original_length=len(text),
            summary_length=len(summary),
            provider=ProviderInfo(engine="vader", model="extractive-heuristic"),
        )


def _split_sentences(text: str) -> list[str]:
    import re

    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 10]


def _word_score(sentence: str) -> int:
    words = sentence.lower().split()
    return sum(1 for w in words if len(w) > 5 and w.isalpha())


def _label_from_compound(compound: float, scores: dict) -> tuple:
    if compound >= 0.05:
        label = SentimentLabel.POSITIVE
        confidence = scores["pos"]
    elif compound <= -0.05:
        label = SentimentLabel.NEGATIVE
        confidence = scores["neg"]
    else:
        label = SentimentLabel.NEUTRAL
        confidence = scores["neu"]
    return label, confidence
