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
    body = build_payload(truck_readouts, truck_specs, position, use_previous, vehicle_id=truck)
    try:
        answer = client.predict(body)
        history = risk_history(
            client, truck_readouts.iloc[: position + 1], truck_specs,
            int(section["history_readouts"]), vehicle_id=truck,
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


main()
