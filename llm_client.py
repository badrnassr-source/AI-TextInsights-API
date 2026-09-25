from __future__ import annotations

import json
import re
from typing import Any, Dict

import httpx

from app.config import Settings


class LLMClientError(Exception):
    pass


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def chat_json(self, system: str, user: str, *, temperature: float = 0.2) -> Dict[str, Any]:
        if not self.settings.llm_enabled:
            raise LLMClientError("pas de OPENAI_API_KEY")

        url = f"{self.settings.openai_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.openai_model,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
            except httpx.HTTPError as err:
                raise LLMClientError(str(err)) from err

        content = resp.json()["choices"][0]["message"]["content"]
        return _load_json(content)

    @property
    def model_name(self) -> str:
        return self.settings.openai_model


def _load_json(content: str) -> Dict[str, Any]:
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise LLMClientError("réponse LLM illisible")
        return json.loads(match.group())
