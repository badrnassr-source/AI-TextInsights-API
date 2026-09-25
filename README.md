# AI-TextInsights-API
API REST FastAPI pour l'analyse de sentiment (VADER) et le résumé automatique de texte via LLM/NLP
# NoteAI — Microservice NLP (Sentiment & Résumé)

API REST légère en **FastAPI** pour l’analyse de sentiment et le résumé automatique de textes, avec **VADER** en local et **LLM** (API compatible OpenAI) lorsque une clé est configurée.

## Démarrage rapide

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Optionnel : renseigner OPENAI_API_KEY dans .env
uvicorn app.main:app --reload
