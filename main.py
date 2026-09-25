from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.config import Settings, get_settings
from app.routers import analyze
from app.schemas.models import HealthResponse

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description="API perso — sentiment & résumé de texte (NLP + LLM).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router, prefix="/api/v1")


@app.get("/health", response_model=HealthResponse)
def health(cfg: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(version=__version__, llm_configured=cfg.llm_enabled)
