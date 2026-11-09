"""FastAPI app: ``GET /health`` and ``POST /predict``.

The model is loaded once at startup (lifespan) from the registry alias named in the config
and kept in ``app.state``. Routes get it through the ``get_service`` dependency, which tests
replace with a tiny model (``app.dependency_overrides``), so no real MLflow is needed there.

Start: ``scripts/serve.sh``. Interactive docs: ``http://127.0.0.1:8000/docs``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request

from failsense.api.schemas import HealthResponse, PredictRequest, PredictResponse
from failsense.api.service import ModelService
from failsense.config import load_config

logger = logging.getLogger(__name__)


def _load_from_config() -> ModelService:
    return ModelService.from_config(load_config())


def create_app(loader: Callable[[], ModelService] = _load_from_config) -> FastAPI:
    """Build the app. ``loader`` returns the service at startup (default: from the config)."""
    if not logging.getLogger().handlers:  # uvicorn configures only its own loggers
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # If this raises (e.g. ModelConfigError), the server does not start.
        app.state.service = loader()
        yield

    app = FastAPI(
        title="FailSense API",
        description="Calibrated 48-step repair risk of a truck from its latest readout(s).",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health(service: ModelService = Depends(get_service)) -> HealthResponse:  # noqa: B008
        """Status and the loaded model (name, registry version, alias)."""
        return HealthResponse(
            status="ok",
            model_name=service.info.name,
            model_version=service.info.version,
            model_alias=service.info.alias,
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(
        request: PredictRequest,
        service: ModelService = Depends(get_service),  # noqa: B008
    ) -> PredictResponse:
        """Calibrated risk and alert for the current readout.

        `previous` must be the readout of the same truck immediately before `current`.
        """
        result = service.predict(
            current=request.current,
            previous=request.previous,
            specifications=request.specifications,
            vehicle_id=request.vehicle_id,
        )
        return PredictResponse(
            risk=result.risk,
            alert=result.alert,
            threshold=service.threshold,
            model_name=service.info.name,
            model_version=service.info.version,
        )

    return app


def get_service(request: Request) -> ModelService:
    """The loaded service; 503 while none is loaded."""
    service: ModelService | None = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return service


app = create_app()
