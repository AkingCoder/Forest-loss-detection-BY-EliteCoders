from __future__ import annotations

import io
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from detector import (
    detect_forest_loss,
    DetectionConfig,
    make_demo_scene,
    summarize_alerts,
)


def _scene_fingerprint(scene: dict[str, np.ndarray | None]) -> str:
    digest = hashlib.sha256()
    for name in ("before", "after", "before_cloud", "after_cloud"):
        value = scene.get(name)
        digest.update(name.encode("utf-8"))
        if value is not None:
            digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def _load_npz_scene(data: bytes) -> dict[str, np.ndarray | None]:
    with np.load(io.BytesIO(data)) as loaded:
        required = {"before", "after"}
        if not required.issubset(loaded.files):
            raise ValueError("The .npz must contain before and after arrays.")
        return {
            "before": loaded["before"],
            "after": loaded["after"],
            "before_cloud": loaded["before_cloud"] if "before_cloud" in loaded else None,
            "after_cloud": loaded["after_cloud"] if "after_cloud" in loaded else None,
        }


st.set_page_config(
    page_title="VanRakshak | Forest-loss alerts",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    .hero {background: linear-gradient(120deg,#063b2a,#0c6b4a); color:white;
           padding:1.25rem 1.5rem; border-radius:16px; margin-bottom:1rem;}
    .hero h1 {margin:0; font-size:2.15rem;}
    .hero p {margin:.35rem 0 0; color:#d8f4e7;}
    div[data-testid="stMetric"] {background:#f5faf7; border-radius:12px; padding:8px 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>🌳 VanRakshak</h1>'
    "<p>Satellite forest-loss alerts for India · cloud-aware · 10 m prototype</p></div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Analysis setup")
    source = st.radio(
        "Scene source",
        ["Demo scene", "Downloaded Sentinel-2 scene", "Upload .npz scene"],
        index=0,
    )
    if source == "Demo scene":
        region = st.selectbox(
            "Test region",
            ["Odisha (held-out)", "Jharkhand (training)", "Madhya Pradesh (training)"],
        )
        seed = {"Odisha (held-out)": 23, "Jharkhand (training)": 11, "Madhya Pradesh (training)": 7}[region]
        before_date, after_date = "2023-01-15", "2024-01-15"
        scene = make_demo_scene(seed=seed, height=120, width=160)
        st.caption("Demo data is deterministic and contains simulated forest clearing, clouds and seasonal effects.")
    elif source == "Downloaded Sentinel-2 scene":
        downloaded_path = Path("sentinel_scene.npz")
        if not downloaded_path.exists():
            st.info(
                "No downloaded scene found. Run `python download_scene.py`, "
                "then refresh this page."
            )
            st.stop()
        try:
            scene = _load_npz_scene(downloaded_path.read_bytes())
        except (OSError, ValueError) as error:
            st.error(f"Could not load {downloaded_path}: {error}")
            st.stop()
        before_date = st.text_input("Before date", "2023-02-07")
        after_date = st.text_input("After date", "2024-02-07")
        st.caption(f"Loaded {downloaded_path} ({scene['before'].shape[0]} × {scene['before'].shape[1]} pixels).")
    else:
        uploaded = st.file_uploader(
            "Upload a NumPy scene (.npz)",
            type=["npz"],
            help="Required arrays: before (H,W,4), after (H,W,4). Optional: before_cloud and after_cloud (H,W).",
        )
        if uploaded is None:
            st.info("Upload a scene or switch to Demo scene.")
            st.stop()
        try:
            scene = _load_npz_scene(uploaded.getvalue())
        except ValueError as error:
            st.error(str(error))
            st.stop()
        before_date = st.text_input("Before date", "2023-01-15")
        after_date = st.text_input("After date", "2024-01-15")

    st.divider()
    st.subheader("Detector controls")
    sensitivity = st.slider("Alert sensitivity", 0.50, 0.95, 0.72, 0.01)
    min_patch = st.slider("Minimum patch size (pixels)", 4, 150, 18)
    cloud_limit = st.slider("Maximum cloud probability", 0.05, 0.80, 0.45, 0.05)
    run = st.button("Run forest-loss scan", type="primary", width="stretch")

config = DetectionConfig(
    alert_threshold=sensitivity,
    min_patch_pixels=min_patch,
    cloud_probability_limit=cloud_limit,
)
result_key = (
    source,
    before_date,
    after_date,
    sensitivity,
    min_patch,
    cloud_limit,
    _scene_fingerprint(scene),
)
if "result" not in st.session_state or run or st.session_state.get("result_key") != result_key:
    st.session_state.result = detect_forest_loss(scene, config)
    st.session_state.result_key = result_key

result = st.session_state.result
summary = summarize_alerts(result)

st.caption(
    f"Comparing **{before_date}** → **{after_date}** · "
    f"cloud pixels excluded: **{summary['cloud_coverage']:.1%}** · "
    f"method: seasonal-normalized NBR + NDVI change"
)

metric_cols = st.columns(4)
metric_cols[0].metric("Alert patches", summary["patch_count"])
metric_cols[1].metric("Suspected loss", f"{summary['loss_area_ha']:.1f} ha")
metric_cols[2].metric("High-confidence area", f"{summary['high_confidence_area_ha']:.1f} ha")
metric_cols[3].metric("Pixels screened", f"{summary['valid_pixels']:,}")

left, right = st.columns([1.55, 1])
with left:
    st.subheader("Alert map")
    display = result["alert_score"].copy()
    display[result["valid_mask"] == 0] = np.nan
    fig = go.Figure(
        go.Heatmap(
            z=display,
            colorscale=[[0, "#eef7f1"], [0.55, "#f5c542"], [1, "#c92a2a"]],
            zmin=0,
            zmax=1,
            hovertemplate="row=%{y}, col=%{x}<br>risk=%{z:.2f}<extra></extra>",
            colorbar={"title": "Risk"},
        )
    )
    fig.update_layout(
        height=510,
        margin={"l": 0, "r": 0, "t": 8, "b": 0},
        xaxis_title="Easting pixel",
        yaxis_title="Northing pixel",
        template="plotly_white",
    )
    st.plotly_chart(fig, width="stretch")

with right:
    st.subheader("How alerts are scored")
    st.markdown(
        """
        1. **Cloud mask** removes cloudy and invalid pixels from both dates.
        2. **Season normalization** compares each pixel to a robust local baseline,
           reducing leaf-fall and harvest-like seasonal changes.
        3. **Spectral change** combines NBR loss, NDVI loss and SWIR increase.
        4. **Patch filtering** removes isolated noise and groups contiguous loss.
        """
    )
    st.info(
        "Prototype note: Sentinel-2 bands are represented as "
        "[B4 red, B8 NIR, B11 SWIR, B12 SWIR2] reflectance values."
    )

st.subheader("Scene quality and evidence")
evidence_cols = st.columns(3)
for column, title, image in [
    (evidence_cols[0], "Before: vegetation index", result["before_ndvi"]),
    (evidence_cols[1], "After: vegetation index", result["after_ndvi"]),
    (evidence_cols[2], "Change confidence", result["alert_score"]),
]:
    with column:
        shown = image.astype(float).copy()
        shown[result["valid_mask"] == 0] = np.nan
        scale = "RdYlGn" if "index" in title else "YlOrRd"
        chart = go.Figure(go.Heatmap(z=shown, colorscale=scale, showscale=False))
        chart.update_layout(height=220, margin={"l": 0, "r": 0, "t": 28, "b": 0}, title=title)
        st.plotly_chart(chart, width="stretch")

st.subheader("Prioritized alerts")
alerts = result["alerts"].copy()
if alerts.empty:
    st.success("No persistent forest-loss patches exceeded the configured threshold.")
else:
    shown_alerts = alerts[
        ["alert_id", "confidence", "area_ha", "row", "column", "dominant_signal", "cloud_free"]
    ].rename(
        columns={
            "alert_id": "Alert",
            "confidence": "Confidence",
            "area_ha": "Area (ha)",
            "row": "Pixel row",
            "column": "Pixel column",
            "dominant_signal": "Evidence",
            "cloud_free": "Cloud-free",
        }
    )
    st.dataframe(
        shown_alerts.style.format({"Confidence": "{:.0%}", "Area (ha)": "{:.2f}"}),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Download alerts CSV",
        alerts.to_csv(index=False).encode("utf-8"),
        "vanrakshak_alerts.csv",
        "text/csv",
    )

with st.expander("Model card and evaluation plan"):
    st.markdown(
        """
        **Intended use:** rapid prioritization for field verification, not legal
        deforestation certification. The 10 m alert area is an estimate.

        **Training/testing:** use coarse Global Forest Watch loss labels only to
        tune thresholds in selected training regions. Keep one complete Indian
        state (for example Odisha) held out for testing and report precision,
        recall, F1 and hectares error at the patch level.

        **Known limitations:** shadows, plantations, crop harvests, fire scars,
        river migration and unmasked clouds can still create false positives.
        Each alert should be reviewed with a higher-resolution image or field visit.
        """
    )
