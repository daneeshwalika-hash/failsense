"""Request and response schemas of the prediction API (the API's data contract).

The API is **stateless**: the caller sends the current readout of a truck, optionally the
readout *immediately before it* (``previous``), and the truck's specifications. Nothing is
stored between calls.

Why ``previous`` is "the readout immediately before ``current``"
----------------------------------------------------------------
In training the counter delta of a readout is ``value - value of the previous readout of the
same truck`` (``groupby(truck).diff()``), i.e. between *consecutive* readouts. A ``previous``
from three readouts earlier would give a delta (and a rate) on a different time scale than the
model has seen and silently shift the risk. The API cannot verify that ``previous`` belongs to
the same truck or is really the neighbour, only that its ``time_step`` is strictly smaller; the
caller owns that contract. Without ``previous`` the deltas are NaN, exactly like the first
readout of a truck in training.

Validation rules (all violations are HTTP 422):

* every one of the 105 raw readout columns plus ``time_step`` must be present (a missing key
  is an error), its value may be ``null`` (a sensor gap, imputed by the model's pipeline);
* values are JSON numbers >= 0 (strings and booleans are rejected, no silent coercion);
* unknown keys are rejected, label columns (``class_label``, ``in_study_repair`` ...) with an
  explicit message; ``vehicle_id`` is a top-level field only and never reaches the model.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    model_validator,
)

from failsense.data.labels import LABEL_COLUMNS, TIME_COL, VEHICLE_COL
from failsense.data.schema import SPEC_COLUMNS, counter_columns, feature_columns

# Label-like names that must be rejected with a clear message. LABEL_COLUMNS holds the label
# columns of the training tables; the other two are the challenge's file name / derived name.
FORBIDDEN_KEYS: frozenset[str] = LABEL_COLUMNS | {"class_label", "proximity_class", VEHICLE_COL}

# A readout value: a JSON number >= 0 (strict: "5" and true are errors), or null.
Reading = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)] | None
# time_step: a JSON number >= 0, never null.
TimeStep = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


def _check_keys(values: Mapping[str, Any], required: set[str], what: str) -> None:
    """Raise ValueError (-> HTTP 422) unless ``values`` has exactly the ``required`` keys."""
    forbidden = sorted(set(values) & FORBIDDEN_KEYS)
    if forbidden:
        raise ValueError(
            f"{what}: label or identifier columns are not allowed here: {forbidden} "
            f"(vehicle_id is a top-level field)"
        )
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"{what}: {len(missing)} missing column(s), e.g. {missing[:5]}")
    unknown = sorted(set(values) - required)
    if unknown:
        raise ValueError(f"{what}: unknown column(s), e.g. {unknown[:5]}")


def readout_keys() -> set[str]:
    """Keys of a full readout: ``time_step`` and the 105 raw feature columns."""
    return {TIME_COL, *feature_columns()}


def previous_keys() -> set[str]:
    """Keys of the previous readout: ``time_step`` and the 8 counters (deltas need only these)."""
    return {TIME_COL, *counter_columns()}


def _validate_current(values: dict[str, Any]) -> dict[str, Any]:
    _check_keys(values, readout_keys(), "current")
    if values[TIME_COL] is None:
        raise ValueError(f"current: {TIME_COL} must not be null")
    return values


def _validate_previous(values: dict[str, Any]) -> dict[str, Any]:
    _check_keys(values, previous_keys(), "previous")
    if values[TIME_COL] is None:
        raise ValueError(f"previous: {TIME_COL} must not be null")
    return values


def _validate_specifications(values: dict[str, str]) -> dict[str, str]:
    _check_keys(values, set(SPEC_COLUMNS), "specifications")
    return values


CurrentReadout = Annotated[dict[str, Reading], AfterValidator(_validate_current)]
PreviousReadout = Annotated[dict[str, Reading], AfterValidator(_validate_previous)]
Specifications = Annotated[dict[str, StrictStr], AfterValidator(_validate_specifications)]


class PredictRequest(BaseModel):
    """One prediction request: current readout, optional previous readout, truck specs."""

    model_config = ConfigDict(extra="forbid")

    vehicle_id: StrictInt | None = Field(
        default=None, ge=0, description="Only for logging. Never a model input."
    )
    current: CurrentReadout = Field(
        description="time_step and the 105 raw readout columns (value or null)."
    )
    previous: PreviousReadout | None = Field(
        default=None,
        description=(
            "The readout of the SAME truck IMMEDIATELY BEFORE `current` (consecutive readouts, "
            "as in training): time_step and the 8 counter columns. Omit it for a truck's first "
            "readout; the deltas are then NaN."
        ),
    )
    specifications: Specifications = Field(description="Spec_0 ... Spec_7 of the truck (strings).")

    @model_validator(mode="after")
    def _previous_is_earlier(self) -> PredictRequest:
        if self.previous is not None:
            now, before = self.current[TIME_COL], self.previous[TIME_COL]
            if not before < now:  # both are non-null floats (checked above)
                raise ValueError(
                    f"previous.{TIME_COL} ({before}) must be strictly smaller than "
                    f"current.{TIME_COL} ({now})"
                )
        return self


class PredictResponse(BaseModel):
    """Calibrated risk and the alert decision. The raw model score is never returned."""

    risk: float = Field(ge=0, le=1, description="Calibrated probability of a repair within 48.")
    alert: bool = Field(description="True if risk >= threshold.")
    threshold: float = Field(description="Alert threshold on the calibrated risk (cost-based).")
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    """Service status and the loaded model."""

    status: str
    model_name: str
    model_version: str
    model_alias: str


class LoggedPrediction(BaseModel):
    """One row of the prediction log (what the API answered, newest first)."""

    id: int
    logged_at: str
    vehicle_id: int | None
    time_step: float
    has_previous: bool
    risk: float
    alert: bool
    threshold: float
    model_name: str
    model_version: str
