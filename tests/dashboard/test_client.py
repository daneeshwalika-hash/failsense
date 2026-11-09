"""ApiClient: URL handling, error mapping (no real server; a fake session)."""

from __future__ import annotations

from typing import Any

import pytest
import requests

from failsense.dashboard.client import ApiClient, ApiError, api_url_from_config


class FakeResponse:
    def __init__(self, status: int, body: Any, text: str = "") -> None:
        self.status_code, self._body, self.text = status, body, text

    def json(self) -> Any:
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class FakeSession:
    def __init__(self, response: FakeResponse | Exception) -> None:
        self.response, self.calls = response, []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((method, url, kwargs))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_health_and_predict_call_the_right_urls() -> None:
    session = FakeSession(FakeResponse(200, {"status": "ok"}))
    client = ApiClient("http://api:8000/", timeout=3, session=session)  # type: ignore[arg-type]
    assert client.health() == {"status": "ok"}
    client.predict({"a": 1})
    assert session.calls[0][:2] == ("GET", "http://api:8000/health")
    assert session.calls[1][:2] == ("POST", "http://api:8000/predict")
    assert session.calls[1][2]["json"] == {"a": 1} and session.calls[1][2]["timeout"] == 3


def test_422_detail_is_summarised() -> None:
    detail = [{"loc": ["body", "current"], "msg": "Value error, missing column"}]
    client = ApiClient("http://x", session=FakeSession(FakeResponse(422, {"detail": detail})))  # type: ignore[arg-type]
    with pytest.raises(ApiError, match="HTTP 422.*body.current: Value error") as info:
        client.predict({})
    assert info.value.status_code == 422


def test_string_detail_and_non_json_error_body() -> None:
    client = ApiClient(
        "http://x", session=FakeSession(FakeResponse(503, {"detail": "model not loaded"}))
    )  # type: ignore[arg-type]
    with pytest.raises(ApiError, match="model not loaded"):
        client.health()
    broken = FakeResponse(500, ValueError("no json"), text="Internal Server Error")
    with pytest.raises(ApiError, match="Internal Server Error"):
        ApiClient("http://x", session=FakeSession(broken)).health()  # type: ignore[arg-type]


@pytest.mark.parametrize("error", [requests.ConnectionError("refused"), requests.Timeout("slow")])
def test_unreachable_api_gives_an_api_error_without_status(error: Exception) -> None:
    with pytest.raises(ApiError, match="not reachable") as info:
        ApiClient("http://x", session=FakeSession(error)).health()  # type: ignore[arg-type]
    assert info.value.status_code is None


def test_api_url_environment_overrides_config(monkeypatch: pytest.MonkeyPatch) -> None:
    config = {"dashboard": {"api_url": "http://127.0.0.1:8000/"}}
    monkeypatch.delenv("FAILSENSE_API_URL", raising=False)
    assert api_url_from_config(config) == "http://127.0.0.1:8000"
    monkeypatch.setenv("FAILSENSE_API_URL", "http://api:8000/")
    assert api_url_from_config(config) == "http://api:8000"


def test_latest_predictions_sends_the_limit() -> None:
    session = FakeSession(FakeResponse(200, [{"id": 1}]))
    client = ApiClient("http://x", session=session)  # type: ignore[arg-type]
    assert client.latest_predictions(5) == [{"id": 1}]
    assert session.calls[0][:2] == ("GET", "http://x/predictions/latest")
    assert session.calls[0][2]["params"] == {"limit": 5}


def test_explain_sends_top_k_and_the_report_flag() -> None:
    session = FakeSession(FakeResponse(200, {"risk": 0.04}))
    client = ApiClient("http://x", session=session)  # type: ignore[arg-type]
    assert client.explain({"current": {}}, top_k=3, with_report=True) == {"risk": 0.04}
    method, url, kwargs = session.calls[0]
    assert (method, url) == ("POST", "http://x/explain")
    assert kwargs["json"] == {"current": {}, "top_k": 3, "with_report": True}
    client.explain({"current": {}})
    assert session.calls[1][2]["json"]["with_report"] is False
