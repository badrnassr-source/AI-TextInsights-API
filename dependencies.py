from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.services.llm_client import LLMClient
from app.services.nlp_service import NLPService


@lru_cache
def get_llm_client() -> LLMClient:
    return LLMClient(get_settings())


def get_nlp_service() -> NLPService:
    return NLPService(get_settings(), get_llm_client())
