"""Streamlit page: pick a demo truck, ask the API for its risk, show alert and trajectory.

Run with ``scripts/dashboard.sh`` (``make dashboard``). The API must be running
(``make serve``). This page never loads the model: every number comes from ``POST /predict``.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from failsense.config import load_config
from failsense.dashboard import data
from failsense.dashboard.client import ApiClient, ApiError, api_url_from_config
from failsense.dashboard.payloads import build_payload, risk_history, sorted_readouts
from failsense.data.labels import TIME_COL, VEHICLE_COL
from failsense.data.schema import SPEC_COLUMNS

logger = logging.getLogger(__name__)

CLASS_TEXT = {
    0: "no repair within the next 48 time steps",
    1: "repair in 25-48 time steps",
    2: "repair in 13-24 time steps",
    3: "repair in 7-12 time steps",
    4: "repair within 6 time steps",
}


@st.cache_data(show_spinner="Loading demo trucks (validation split) ...")
def _load_demo(_config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    return data.load_demo_trucks(_config)


@st.cache_data(ttl=60, show_spinner=False)
def _score(
    url: str, timeout: float, truck: int, position: int, use_previous: bool, n_history: int,
    _readouts: pd.DataFrame, _specs: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, dict]:  # fmt: skip
    """Score one truck through the API; cached for a minute per (truck, readout, switch).

    Streamlit reruns the whole script on every click. Without the cache each rerun would send
    16 identical requests and fill the prediction log with duplicates. (Underscore arguments
    are not part of the cache key; the data is fixed for the session.)
    """
    client = ApiClient(url, timeout=timeout)
    truck_readouts = sorted_readouts(_readouts[_readouts[VEHICLE_COL] == truck])
    truck_specs = _specs[_specs[VEHICLE_COL] == truck].iloc[0].to_dict()
    body = build_payload(truck_readouts, truck_specs, position, use_previous, vehicle_id=truck)
    answer = client.predict(body)
    history = risk_history(
        client, truck_readouts.iloc[: position + 1], truck_specs, n_history, vehicle_id=truck
    )
    return answer, history, body


def _format_value(value: object) -> str:
    """An input value as table text: numbers with 4 significant digits, ``fehlt`` if missing."""
    if value is None or (isinstance(value, float) and value != value):
        return "fehlt"
    return f"{value:.4g}" if isinstance(value, int | float) else str(value)


def _show_report(data: dict) -> None:
    """The factors and the German maintenance report returned by ``POST /explain``."""
    report = data["report"]
    levels = {"niedrig": "success", "erhöht": "warning", "hoch": "error"}
    getattr(st, levels[report["risk_level"]])(f"Risikolevel: {report['risk_level']}")
    st.markdown(f"**Zusammenfassung.** {report['summary']}")
    st.markdown(f"**Empfehlung.** {report['recommendation']}")
    st.markdown("**Begründung (Faktoren des Modells):**")
    for reason in report["reasons"]:
        st.markdown(f"- `{reason['factor']}`: {reason['explanation']}")
    st.caption(report["caveat"])
    if report["source"] == "llm":
        st.caption(f"Quelle: lokales Sprachmodell ({report['llm_model']}), automatisch geprüft.")
    else:
        st.caption("Quelle: Textvorlage (das Sprachmodell war nicht verfügbar oder lieferte "
                   "keinen gültigen Bericht).")  # fmt: skip
    if report["warnings"]:
        with st.expander("Hinweise zur Prüfung"):
            for warning in report["warnings"]:
                st.write(warning)
    factors = pd.DataFrame(data["factors"])[["feature", "value", "contribution", "direction"]]
    # numbers and category text (a specification) in one column cannot be shown as one Arrow type
    factors["value"] = factors["value"].map(_format_value)
    st.dataframe(factors, hide_index=True, width="stretch")
    st.caption(
        f"Startwert des Modells (SHAP-Basiswert): {data['baseline_risk']:.1%}; das ist nicht das "
        "mittlere Risiko der Flotte. Beiträge in kalibrierten Log-Odds; die Variablen sind "
        "anonymisiert."
    )


@st.fragment(run_every=float(load_config()["dashboard"]["refresh_seconds"]))
def _latest_predictions(client: ApiClient, limit: int) -> None:
    """The API's most recent logged predictions (every client, e.g. the stream simulator)."""
    try:
        rows = client.latest_predictions(limit)
    except ApiError as exc:
        st.info(f"Prediction log not available: {exc}")
        return
    if not rows:
        st.info("No predictions logged yet. Start `make simulate` or pick a truck above.")
        return
    frame = pd.DataFrame(rows)
    left, right = st.columns(2)
    left.metric(f"Alerts in the last {len(frame)} predictions", f"{frame['alert'].mean():.0%}")
    right.metric("Mean risk", f"{frame['risk'].mean():.1%}")
    shown = frame[["logged_at", "vehicle_id", "time_step", "risk", "alert", "model_version"]]
    st.dataframe(
        shown, hide_index=True, width="stretch",
        column_config={"risk": st.column_config.NumberColumn(format="percent")},
    )  # fmt: skip


def main() -> None:
    config = load_config()
    section = config["dashboard"]
    st.set_page_config(page_title="FailSense", page_icon=None, layout="wide")
    st.title("FailSense - repair risk per truck")
    st.caption(
        "Calibrated probability that Component X needs a repair within 48 time steps. "
        "Demo trucks are sampled from the validation split; all numbers come from the API."
    )

    # --- sidebar: API connection ---------------------------------------------------------
    url = st.sidebar.text_input("API URL", api_url_from_config(config))
    client = ApiClient(url, timeout=float(section["timeout_seconds"]))
    try:
        health = client.health()
        st.sidebar.success(
            f"{health['model_name']} v{health['model_version']} (@{health['model_alias']})"
        )
    except ApiError as exc:
        st.sidebar.error("API unavailable")
        st.error(f"{exc}\n\nStart it with `make serve`.")
        return

    # --- truck selection -------------------------------------------------------------------
    readouts, specs, label = _load_demo(config)
    truck = st.sidebar.selectbox("Truck (vehicle_id)", sorted(label.index.tolist()))
    use_previous = st.sidebar.checkbox("Send the previous readout (deltas)", value=True)
    truck_readouts = sorted_readouts(readouts[readouts[VEHICLE_COL] == truck])
    truck_specs = specs[specs[VEHICLE_COL] == truck].iloc[0].to_dict()
    position = st.sidebar.slider(
        "Readout to score (position in the truck's history)",
        0, len(truck_readouts) - 1, len(truck_readouts) - 1,
    )  # fmt: skip

    # --- prediction ------------------------------------------------------------------------
    try:
        answer, history, body = _score(
            url, float(section["timeout_seconds"]), int(truck), position, use_previous,
            int(section["history_readouts"]), readouts, specs,
        )  # fmt: skip
    except ApiError as exc:
        st.error(str(exc))
        return

    risk, threshold = answer["risk"], answer["threshold"]
    left, middle, right = st.columns(3)
    left.metric("Calibrated risk", f"{risk:.1%}")
    middle.metric("Alert threshold", f"{threshold:.2%}")
    right.metric("Risk / threshold", f"{risk / threshold:.1f} x")
    if answer["alert"]:
        st.error("ALERT: risk is at or above the cost-based threshold - schedule a workshop check.")
    else:
        st.success("No alert: risk is below the threshold.")
    st.caption(
        "The threshold is chosen by cost (a missed failure is far more expensive than a false "
        "alarm), so a truck can trigger an alert at a risk of only a few percent."
    )

    st.subheader("Risk of the latest readouts")
    chart = history.set_index(TIME_COL)[["risk", "threshold"]]
    st.line_chart(chart, y_label="calibrated risk", x_label="time_step")

    with st.expander("Known outcome (validation label, not sent to the API)"):
        cls = int(label.loc[truck])
        st.write(f"class {cls}: {CLASS_TEXT[cls]}")
    with st.expander("Truck specifications"):
        st.json({c: truck_specs[c] for c in SPEC_COLUMNS})
    with st.expander("Request sent to POST /predict (current readout shortened)"):
        shown = {k: v for k, v in body.items() if k != "current"}
        shown["current"] = {k: v for k, v in list(body["current"].items())[:6]} | {"...": "..."}
        st.json(shown)

    st.subheader("Wartungsbericht (Deutsch)")
    key = [int(truck), int(position), bool(use_previous)]
    if st.button("Wartungsbericht erzeugen"):
        report_client = ApiClient(url, timeout=float(section["report_timeout_seconds"]))
        with st.spinner("Das lokale Sprachmodell schreibt den Bericht (kann dauern) ..."):
            try:
                st.session_state["report"] = {
                    "key": key,
                    "data": report_client.explain(body, top_k=5, with_report=True),
                }
            except ApiError as exc:
                st.session_state.pop("report", None)
                st.error(str(exc))
    stored = st.session_state.get("report")
    if stored and stored["key"] == key:
        _show_report(stored["data"])

    st.subheader("Latest predictions of the API (all clients)")
    _latest_predictions(client, int(section["latest_rows"]))


main()
