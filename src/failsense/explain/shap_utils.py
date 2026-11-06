"""SHAP values of the Q1 pipeline and the reusable ``top_shap_factors``.

The registered model is ``Pipeline(preprocess -> XGBoost)``. SHAP explains the **XGBoost step**,
so the Pipeline's preprocessing (median imputer + one-hot of the 8 specs; 239 input columns ->
321 model columns) is applied first and the SHAP values of the one-hot columns are summed back
to their input column (``Spec_7``), so every explanation speaks the language of the 239 inputs.

Units: SHAP values are additive contributions to the model's **raw log-odds** (the margin of the
UNCALIBRATED model): ``base_value + sum(shap) = margin`` and ``p_raw = sigmoid(margin)``. The
deployed sigmoid calibrator is linear in the margin (``logit(p_cal) = intercept + slope * margin``),
so the contribution in calibrated log-odds is exactly ``slope * shap``; ``top_shap_factors``
takes that factor as ``scale``.

Limitation: ``shap.TreeExplainer`` runs with the default
``feature_perturbation="tree_path_dependent"`` (no background data; the tree's own cover
statistics stand in for the data distribution). With correlated features this can split credit
between them in a way that is not intuitive. The ``interventional`` variant with a training
background sample is the alternative; it is not used for runtime reasons.
"""

from __future__ import annotations

import functools
import json
import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from failsense.data.schema import SPEC_COLUMNS

logger = logging.getLogger(__name__)

ADDITIVITY_TOLERANCE = 1e-3  # |base + sum(shap) - margin| in log-odds


def origin_columns(feature_names: Sequence[str], input_columns: Sequence[str]) -> list[str]:
    """Input column each model column comes from (``cat__Spec_7_Cat3`` -> ``Spec_7``).

    Raises:
        ValueError: If a model column cannot be traced back to an input column.
    """
    inputs = sorted(input_columns, key=len, reverse=True)  # longest match first
    origins = []
    for name in feature_names:
        prefix, _, rest = name.partition("__")
        if prefix == "num" and rest in input_columns:
            origins.append(rest)
        elif prefix == "cat":
            match = next((c for c in inputs if rest.startswith(c + "_")), None)
            if match is None:
                raise ValueError(f"cannot trace {name!r} back to an input column")
            origins.append(match)
        else:
            raise ValueError(f"cannot trace {name!r} back to an input column")
    return origins


class ShapExplainer:
    """SHAP for a fitted ``Pipeline(preprocess, model)`` with a tree model."""

    def __init__(self, pipeline: Pipeline, input_columns: Sequence[str]) -> None:
        self.input_columns = list(input_columns)
        self.preprocess = pipeline.named_steps["preprocess"]
        self.model = pipeline.named_steps["model"]
        self.feature_names = [str(n) for n in self.preprocess.get_feature_names_out()]
        self.origin = origin_columns(self.feature_names, self.input_columns)
        self.explainer = shap.TreeExplainer(self.model)  # tree_path_dependent, see module docstring
        # ``expected_value`` is only final after the first ``shap_values`` call (it moved by 1e-2
        # on a small model), so run one dummy row now and read it afterwards.
        self.explainer.shap_values(np.zeros((1, len(self.feature_names))), check_additivity=False)
        self.base_value = self._current_base_value()

    def _current_base_value(self) -> float:
        return float(np.ravel(self.explainer.expected_value)[0])

    def prepare(self, rows: pd.DataFrame | pd.Series) -> pd.DataFrame:
        """Input frame in the model's column order, numbers as float, specs as text.

        Raises:
            ValueError: If an input column is missing or the input is empty.
        """
        frame = rows.to_frame().T if isinstance(rows, pd.Series) else rows
        missing = [c for c in self.input_columns if c not in frame.columns]
        if missing:
            raise ValueError(f"{len(missing)} input columns missing, e.g. {missing[:5]}")
        if frame.empty:
            raise ValueError("no rows to explain")
        out = frame[self.input_columns].copy()
        for column in self.input_columns:
            if column in SPEC_COLUMNS:
                out[column] = out[column].astype(str)
            else:
                out[column] = pd.to_numeric(out[column], errors="raise").astype(float)
        return out.reset_index(drop=True)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Preprocessed model input (imputed, one-hot)."""
        return np.asarray(self.preprocess.transform(frame))

    def margin(self, frame: pd.DataFrame) -> np.ndarray:
        """Raw log-odds of the uncalibrated model."""
        return self.model.predict(self.transform(frame), output_margin=True)

    def shap_values(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """SHAP matrix on the model columns and the preprocessed data it belongs to.

        Raises:
            AssertionError: If ``base_value + sum(shap)`` differs from the margin
                (the explanation would not add up).
        """
        z = self.transform(frame)
        values = np.asarray(self.explainer.shap_values(z))
        self.base_value = self._current_base_value()
        error = np.abs(values.sum(axis=1) + self.base_value - self.margin(frame)).max()
        if error > ADDITIVITY_TOLERANCE:
            raise AssertionError(f"SHAP values do not add up to the margin (max error {error:.2e})")
        return values, z

    def by_input(self, values: np.ndarray) -> pd.DataFrame:
        """SHAP per INPUT column: the one-hot columns of a spec are summed (same row sums)."""
        summed = pd.DataFrame(values, columns=self.origin).T.groupby(level=0, sort=False).sum().T
        return summed[self.input_columns]


@functools.lru_cache(maxsize=1)
def default_explainer() -> ShapExplainer:
    """Explainer of the registered ``failsense-q1@candidate`` model (loaded once per process)."""
    import mlflow

    from failsense.config import load_config, resolve_path
    from failsense.tracking import setup_mlflow

    config = load_config()
    setup_mlflow(config)
    registry = json.loads(
        (resolve_path(config["paths"]["reports"]) / "registry_q1.json").read_text(encoding="utf-8")
    )
    pipeline = mlflow.sklearn.load_model(registry["model_uri"])
    return ShapExplainer(pipeline, registry["input_columns"])


def top_shap_factors(
    row: pd.DataFrame | pd.Series,
    k: int = 5,
    explainer: ShapExplainer | None = None,
    scale: float = 1.0,
) -> list[tuple[str, Any, float]]:
    """The ``k`` input features that moved one truck's score most, by absolute SHAP value.

    Args:
        row: ONE readout row (a Series or a one-row frame) with the 239 input columns.
        k: Number of factors (>= 1; at most the number of input columns is returned).
        explainer: A ``ShapExplainer``; ``None`` loads the registered model once.
        scale: Multiplier for the SHAP values: ``1.0`` = raw log-odds of the uncalibrated model;
            the slope of the sigmoid calibrator gives calibrated log-odds contributions.

    Returns:
        ``[(feature, value, shap), ...]`` sorted by ``|shap|`` descending. ``value`` is the input
        value (float, NaN if missing; the category text for a spec); ``shap`` > 0 pushed the
        risk up, < 0 pushed it down. The SHAP values of the one-hot columns of a spec are summed.

    Raises:
        ValueError: If ``k < 1``, ``row`` has not exactly one row, or input columns are missing.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    explainer = explainer or default_explainer()
    frame = explainer.prepare(row)
    if len(frame) != 1:
        raise ValueError(f"top_shap_factors explains exactly one row, got {len(frame)}")
    values, _ = explainer.shap_values(frame)
    per_input = explainer.by_input(values).iloc[0]
    order = per_input.abs().sort_values(ascending=False, kind="stable").index[:k]
    return [
        (c, frame[c].iloc[0] if c in SPEC_COLUMNS else float(frame[c].iloc[0]),
         float(per_input[c] * scale))
        for c in order
    ]  # fmt: skip
