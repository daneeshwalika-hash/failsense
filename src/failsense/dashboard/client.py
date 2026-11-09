"""HTTP client for the FailSense API (the dashboard's only way to get a prediction)."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

import requests

logger = logging.getLogger(__name__)

API_URL_ENV = "FAILSENSE_API_URL"


class ApiError(RuntimeError):
    """The API could not be reached or answered with an error status.

    ``status_code`` is ``None`` when there was no answer at all (connection refused, timeout).
    """

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def api_url_from_config(config: Mapping[str, Any]) -> str:
    """``FAILSENSE_API_URL`` if set, else ``dashboard.api_url`` (trailing slash removed)."""
    return (os.environ.get(API_URL_ENV) or config["dashboard"]["api_url"]).rstrip("/")


def _detail(response: requests.Response) -> str:
    """A short human-readable reason from an error response (FastAPI's ``detail``)."""
    try:
        detail = response.json().get("detail", response.text)
    except (ValueError, AttributeError):
        return response.text[:300]
    if isinstance(detail, list):  # 422: one entry per violation
        parts = [
            f"{'.'.join(str(p) for p in d.get('loc', []))}: {d.get('msg', '')}" for d in detail
        ]
        return "; ".join(parts[:5])
    return str(detail)


class ApiClient:
    """Thin wrapper around ``requests`` for the API's health, predict and log endpoints."""

    def __init__(
        self, base_url: str, timeout: float = 10.0, session: requests.Session | None = None
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        try:
            response = self.session.request(method, url, timeout=self.timeout, **kwargs)
        except requests.RequestException as exc:
            raise ApiError(f"API not reachable at {self.base_url}: {exc}") from exc
        if response.status_code >= 400:
            raise ApiError(
                f"{method} {path} -> HTTP {response.status_code}: {_detail(response)}",
                status_code=response.status_code,
            )
        return response.json()

    def health(self) -> dict[str, Any]:
        """Status and loaded model (name, version, alias)."""
        return self._request("GET", "/health")

    def latest_predictions(self, limit: int = 20) -> list[dict[str, Any]]:
        """The most recent rows of the API's prediction log, newest first."""
        return self._request("GET", "/predictions/latest", params={"limit": limit})

    def explain(
        self, payload: Mapping[str, Any], top_k: int = 5, with_report: bool = False
    ) -> dict[str, Any]:
        """Risk, the ``top_k`` SHAP factors and (optionally) the German maintenance report."""
        body = {**payload, "top_k": top_k, "with_report": with_report}
        return self._request("POST", "/explain", json=body)

    def predict(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Calibrated risk, alert flag, threshold and model version for one request body."""
        return self._request("POST", "/predict", json=payload)
