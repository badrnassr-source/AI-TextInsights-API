# TextKit API

Petit backend que j'ai codé pour m'entrainer sur **FastAPI** + un peu de **NLP/LLM**.  
Tu envoies du texte, tu récupères le sentiment (positif / neutre / negatif) et/ou un résumé.

> Projet portfolio — Nassr Badr, L2 informatique (2025–2026)  
> Stack : Python, FastAPI, VADER, API OpenAI (optionnelle)

## Pourquoi ce projet ?

J'voulais un truc concret à montrer en entretien : une **API REST** propre, pas juste un script Jupyter.  
Sans clé OpenAI ça tourne quand même (sentiment avec VADER, résumé basique). Avec une clé, le résumé passe par un LLM.

## Install & run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

Swagger : http://127.0.0.1:8000/docs

Tests (optionnel) :

```bash
pip install pytest httpx
pytest -q
```

## Routes

- `GET /health` — ping + est-ce que le LLM est configuré
- `POST /api/v1/analyze/sentiment` — body JSON `{ "text": "..." }`
- `POST /api/v1/analyze/summary` — `{ "text": "...", "max_sentences": 3 }`
- `POST /api/v1/analyze` — sentiment + résumé d'un coup

Exemple :

```bash
curl -X POST http://127.0.0.1:8000/api/v1/analyze/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "The internship interview went really well!"}'
```

## Structure du repo

```
app/
  main.py              # entry FastAPI
  config.py            # variables .env
  routers/analyze.py   # endpoints
  schemas/models.py    # pydantic
  services/            # logique NLP + appels LLM
```

## Docker

```bash
docker build -t textkit-api .
docker run -p 8000:8000 --env-file .env textkit-api
```

## Idées d'amélioration (TODO perso)

- [ ] rate limiting basique
- [ ] meilleur résumé FR (modèle ou fine-tuning plus tard)
- [ ] petit front React pour la démo live
