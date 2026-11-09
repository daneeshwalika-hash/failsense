"""FastAPI app: ``GET /health``, ``POST /predict`` and ``GET /predictions/latest``.

The model is loaded once at startup (lifespan) from the registry alias named in the config
and kept in ``app.state``. Routes get it through the ``get_service`` dependency, which tests
replace with a tiny model (``app.dependency_overrides``), so no real MLflow is needed there.

Start: ``scripts/serve.sh``. Interactive docs: ``http://127.0.0.1:8000/docs``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, Request

from failsense.api.prediction_log import MAX_LATEST, PredictionLog, PredictionRecord
from failsense.api.schemas import (
    HealthResponse,
    LoggedPrediction,
    PredictRequest,
    PredictResponse,
)
from failsense.api.service import ModelService
from failsense.config import load_config

logger = logging.getLogger(__name__)


def _load_from_config() -> ModelService:
    return ModelService.from_config(load_config())


def _log_from_config() -> PredictionLog | None:
    return PredictionLog.from_config(load_config())


def create_app(
    loader: Callable[[], ModelService] = _load_from_config,
    log_factory: Callable[[], PredictionLog | None] = lambda: None,
) -> FastAPI:
    """Build the app.

    Args:
        loader: Returns the model service at startup (default: from the config).
        log_factory: Returns the DuckDB prediction log at startup, or ``None`` for no log. The
            default is NO log, so tests never write files; the module-level ``app`` passes the
            log named in the config.
    """
    if not logging.getLogger().handlers:  # uvicorn configures only its own loggers
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # If this raises (e.g. ModelConfigError), the server does not start.
        app.state.service = loader()
        app.state.prediction_log = log_factory()
        try:
            yield
        finally:
            if app.state.prediction_log is not None:
                app.state.prediction_log.close()  # releases the DuckDB writer lock

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
        http_request: Request,
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
        response = PredictResponse(
            risk=result.risk,
            alert=result.alert,
            threshold=service.threshold,
            model_name=service.info.name,
            model_version=service.info.version,
        )
        _write_log(http_request, request, response)
        return response

    @app.get("/predictions/latest", response_model=list[LoggedPrediction])
    def latest_predictions(
        limit: int = Query(20, ge=1, le=MAX_LATEST),
        log: PredictionLog = Depends(get_log),  # noqa: B008
    ) -> list[dict]:
        """The most recent logged predictions, newest first (from the DuckDB log)."""
        return log.latest(limit)

    return app


def _write_log(http_request: Request, request: PredictRequest, response: PredictResponse) -> None:
    """Append the prediction to the log. A logging problem must never break the prediction."""
    log: PredictionLog | None = getattr(http_request.app.state, "prediction_log", None)
    if log is None:
        return
    try:
        log.log(
            PredictionRecord(
                vehicle_id=request.vehicle_id,
                time_step=float(request.current["time_step"]),
                has_previous=request.previous is not None,
                risk=response.risk,
                alert=response.alert,
                threshold=response.threshold,
                model_name=response.model_name,
                model_version=response.model_version,
            )
        )
    except Exception:  # noqa: BLE001 - the answer is already computed; log and carry on
        logger.exception("could not write the prediction log")


def get_log(request: Request) -> PredictionLog:
    """The prediction log; 503 if the log is disabled."""
    log: PredictionLog | None = getattr(request.app.state, "prediction_log", None)
    if log is None:
        raise HTTPException(status_code=503, detail="prediction log is disabled")
    return log


def get_service(request: Request) -> ModelService:
    """The loaded service; 503 while none is loaded."""
    service: ModelService | None = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return service


app = create_app(log_factory=_log_from_config)
