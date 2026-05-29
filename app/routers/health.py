"""GET /health — liveness + readiness probe."""

from fastapi import APIRouter, Request

from app.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health(request: Request) -> HealthResponse:
    """Returns API status and list of loaded models.

    Use as a Kubernetes/Docker liveness probe:
        GET /health  →  200 {"status": "ok", ...}
    """
    registry = request.app.state.registry
    return HealthResponse(
        status="ok",
        models_loaded=registry.loaded_names,
        version=request.app.state.config.version,
    )
