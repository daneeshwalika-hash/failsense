"""The model service: builds the features of one request, scores it, calibrates, decides.

Kept free of FastAPI so it can be tested with a tiny pipeline and used by other callers
(Streamlit never loads it; scripts could). ``ModelService.from_config`` is the only place that
touches MLflow, and it imports MLflow lazily.

Pipeline of one prediction (same code path as training and validation):

1. rows: [previous (counters only), current]  ->  ``build_model_features`` (deltas from the
   previous readout, histogram features from the current row), the specifications attached,
2. ``select_model_inputs``: the registered input columns in the registered order, nothing else
   (``vehicle_id`` and labels cannot reach the model),
3. ``predict_proba`` -> raw score -> sigmoid calibration -> risk, alert = risk >= threshold.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from failsense.config import resolve_path
from failsense.data.labels import TIME_COL, VEHICLE_COL
from failsense.data.schema import SPEC_COLUMNS, counter_columns, histogram_groups
from failsense.features.build import (
    attach_specifications,
    build_model_features,
    features_params_from_config,
)
from failsense.models.calibration import apply_sigmoid_parameters
from failsense.models.validation_data import registry_input_columns, select_model_inputs

logger = logging.getLogger(__name__)

_INTERNAL_TRUCK_ID = 0  # placeholder join key; the caller's vehicle_id is never used as a feature


class ModelConfigError(RuntimeError):
    """The deployed model, calibrator and threshold do not belong together. Fail fast."""


@dataclass(frozen=True)
class ModelInfo:
    """Which registered model the service runs."""

    name: str
    version: str
    alias: str


@dataclass(frozen=True)
class Prediction:
    """Result of one request. ``score`` (raw) is for logging only and is not in the response."""

    risk: float
    alert: bool
    score: float


def known_spec_categories(pipeline: Any) -> dict[str, set[str]]:
    """Spec categories the fitted one-hot encoder has seen (empty if it cannot be found)."""
    try:
        encoder = pipeline.named_steps["preprocess"].named_transformers_["cat"]
        return {
            str(c): {str(v) for v in cats}
            for c, cats in zip(encoder.feature_names_in_, encoder.categories_, strict=True)
        }
    except (AttributeError, KeyError):
        return {}


class ModelService:
    """A loaded pipeline plus its calibrator, threshold and registered input columns."""

    def __init__(
        self,
        pipeline: Any,
        calibrator: Mapping[str, Any],
        threshold: float,
        input_columns: Sequence[str],
        info: ModelInfo,
        top_fraction: float,
    ) -> None:
        if calibrator.get("kind") != "sigmoid":
            raise ModelConfigError(f"only a sigmoid calibrator is supported, got {calibrator!r}")
        if not 0 < threshold < 1:
            raise ModelConfigError(f"threshold must be in (0, 1), got {threshold}")
        self.pipeline = pipeline
        self.calibrator = dict(calibrator)
        self.threshold = float(threshold)
        self.input_columns = list(input_columns)
        self.info = info
        self.top_fraction = top_fraction
        self._known_specs = known_spec_categories(pipeline)

    # --- features ---------------------------------------------------------------------
    def build_inputs(
        self,
        current: Mapping[str, float | None],
        previous: Mapping[str, float | None] | None,
        specifications: Mapping[str, str],
    ) -> pd.DataFrame:
        """One model-input row, built with the same function as training and validation.

        ``previous`` must be the readout of the same truck immediately before ``current``.
        """
        rows = [] if previous is None else [{VEHICLE_COL: _INTERNAL_TRUCK_ID, **previous}]
        rows.append({VEHICLE_COL: _INTERNAL_TRUCK_ID, **current})
        readouts = pd.DataFrame(rows).astype(
            {c: "float64" for c in pd.DataFrame(rows).columns if c != VEHICLE_COL}
        )  # None -> NaN
        readouts[VEHICLE_COL] = readouts[VEHICLE_COL].astype("int64")
        last = pd.Index([len(readouts) - 1])
        features = build_model_features(
            readouts, last, counter_columns(), histogram_groups(), self.top_fraction
        )
        specs = pd.DataFrame([{VEHICLE_COL: _INTERNAL_TRUCK_ID, **specifications}])
        frame = attach_specifications(features, specs)
        self._warn_unseen_specs(specifications)
        return select_model_inputs(frame, self.input_columns).reset_index(drop=True)

    def _warn_unseen_specs(self, specifications: Mapping[str, str]) -> None:
        for column, seen in self._known_specs.items():
            if column in specifications and specifications[column] not in seen:
                logger.warning(
                    "%s=%r was not seen in training (the one-hot encoder maps it to all zeros)",
                    column,
                    specifications[column],
                )

    # --- prediction -------------------------------------------------------------------
    def predict(
        self,
        current: Mapping[str, float | None],
        previous: Mapping[str, float | None] | None,
        specifications: Mapping[str, str],
        vehicle_id: int | None = None,
    ) -> Prediction:
        """Calibrated risk and alert decision for one readout.

        ``vehicle_id`` is only written to the log, never used as a feature.
        """
        inputs = self.build_inputs(current, previous, specifications)
        score = float(self.pipeline.predict_proba(inputs)[0, 1])
        risk = float(apply_sigmoid_parameters(self.calibrator, [score])[0])
        alert = risk >= self.threshold
        logger.info(
            "predict vehicle=%s time_step=%s previous=%s score=%.5f risk=%.5f alert=%s",
            vehicle_id, current[TIME_COL], previous is not None, score, risk, alert,
        )  # fmt: skip
        return Prediction(risk=risk, alert=alert, score=score)

    def smoke_test(self) -> None:
        """Run one synthetic request; raises if the registered inputs cannot be built.

        Called at startup, so a mismatch between the registry file and the feature code stops
        the service immediately instead of failing on the first real request.
        """
        from failsense.api.schemas import previous_keys, readout_keys

        current = {k: 1.0 for k in readout_keys()}
        previous = {k: 0.5 for k in previous_keys()}
        specs = {c: "Cat0" for c in SPEC_COLUMNS}
        result = self.predict(current, previous, specs)
        if not np.isfinite(result.risk):
            raise ModelConfigError("smoke test produced a non-finite risk")

    # --- loading ----------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ModelService:
        """Load model, calibrator and threshold named in the config.

        Raises:
            ModelConfigError: If the summary file's model URI differs from
                ``models:/<tuning.registered_model>@<tuning.alias>`` (the calibrator and the
                threshold were fitted for exactly one model), or the summary is incomplete.
        """
        import mlflow
        import mlflow.sklearn
        from mlflow import MlflowClient

        from failsense.tracking import tracking_params_from_config

        name, alias = config["tuning"]["registered_model"], config["tuning"]["alias"]
        uri = f"models:/{name}@{alias}"
        deployment = load_deployment(config)
        if deployment["model_uri"] != uri:
            raise ModelConfigError(
                f"summary model_uri {deployment['model_uri']!r} != configured {uri!r}: the "
                "calibrator and threshold belong to another model. Re-run the calibration "
                "for this model or fix tuning.registered_model / tuning.alias."
            )
        db_path, _, _ = tracking_params_from_config(config)
        mlflow.set_tracking_uri(f"sqlite:///{resolve_path(db_path)}")
        pipeline = mlflow.sklearn.load_model(uri)
        version = MlflowClient().get_model_version_by_alias(name, alias).version
        service = cls(
            pipeline=pipeline,
            calibrator=deployment["calibrator"],
            threshold=float(deployment["threshold"]),
            input_columns=registry_input_columns(config),
            info=ModelInfo(name=name, version=str(version), alias=alias),
            top_fraction=features_params_from_config(config),
        )
        service.smoke_test()
        logger.info("Loaded %s version %s, threshold %.6f", uri, version, service.threshold)
        return service


def load_deployment(config: Mapping[str, Any]) -> dict[str, Any]:
    """The ``deployment_choice`` block of the calibration summary named in ``api.*``."""
    section = config["api"]
    path = resolve_path(section["summary_path"])
    if not path.is_file():
        raise ModelConfigError(f"calibration summary not found: {path}")
    block = json.loads(path.read_text(encoding="utf-8")).get(section["summary_key"])
    needed = ("model_uri", "calibrator", "threshold")
    if not isinstance(block, dict) or any(k not in block for k in needed):
        raise ModelConfigError(f"{path}: block {section['summary_key']!r} needs {needed}")
    return block
