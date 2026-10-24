from __future__ import annotations

import io
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import folium
from streamlit_folium import st_folium

from detector import (
    detect_forest_loss,
    DetectionConfig,
    summarize_alerts,
)
from sentinel_download import download_before_after


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
    header[data-testid="stHeader"] {display: none;}
    [data-testid="stToolbar"] {display: none;}
    footer {display: none;}
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    .hero {background: linear-gradient(120deg,#063b2a,#0c6b4a); color:white;
           padding:1.25rem 1.5rem; border-radius:16px; margin-bottom:1rem;}
    .hero h1 {margin:0; font-size:2.15rem;}
    .hero p {margin:.35rem 0 0; color:#d8f4e7;}
    div[data-testid="stMetric"] {background:#f5faf7; border-radius:12px; padding:8px 12px;}
    .conclusion-card {border-radius:14px; padding:1rem 1.1rem; min-height:108px;
                      border:1px solid #b7d9c7; background:#f1fbf5;}
    .conclusion-card.alert {border-color:#f0b4ad; background:#fff4f2;}
    .conclusion-label {font-size:.9rem; font-weight:600; color:#315443; margin-bottom:.35rem;}
    .conclusion-card.alert .conclusion-label {color:#8b2e25;}
    .conclusion-value {font-size:1.65rem; font-weight:700; color:#176b3a;}
    .conclusion-card.alert .conclusion-value {color:#b42318;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>🌳 VanRakshak</h1>'
    "<p>Find possible forest clearing from satellite images · choose a place, compare two dates, and review the alerts</p></div>",
    unsafe_allow_html=True,
)

st.info(
    "Start by choosing a location or loading a scene. VanRakshak compares "
    "satellite images from two dates and highlights areas that may need field verification.",
    title="How this works",
)

with st.sidebar:
    st.header("1. Choose satellite data")
    source = st.radio(
        "Where should the images come from?",
        ["Pick a location on the map", "Use downloaded scene", "Upload a scene"],
        index=0,
    )
    if source == "Pick a location on the map":
        st.subheader("Compare dates")
        before_date = st.text_input(
            "Earlier image date",
            "2023-02-07",
            help="The older satellite image used as the baseline.",
        )
        after_date = st.text_input(
            "Later image date",
            "2024-02-07",
            help="The newer satellite image compared with the baseline.",
        )
        st.caption("Click the map once to choose the centre of the analysis area.")

        map_state = st.session_state.get("location_map_state")
        map_center = map_state or {"lat": 22.5, "lng": 79.0}
        map_view = folium.Map(
            location=[map_center["lat"], map_center["lng"]],
            zoom_start=5,
            control_scale=True,
        )
        if map_state:
            folium.Marker(
                [map_state["lat"], map_state["lng"]],
                tooltip="Selected location",
            ).add_to(map_view)
        map_result = st_folium(
            map_view,
            height=360,
            width=None,
            key="location_selector",
            returned_objects=["last_clicked"],
        )
        clicked = map_result.get("last_clicked")
        if clicked:
            map_state = {"lat": float(clicked["lat"]), "lng": float(clicked["lng"])}
            st.session_state.location_map_state = map_state
        if not map_state:
            st.info("No location selected yet. Click anywhere on the map to begin.")
            st.stop()
        selected_lat = map_state["lat"]
        selected_lng = map_state["lng"]
        aoi_half_size = 0.005
        aoi = (
            selected_lng - aoi_half_size,
            selected_lat - aoi_half_size,
            selected_lng + aoi_half_size,
            selected_lat + aoi_half_size,
        )
        st.caption(
            f"Selected location: `{selected_lat:.6f}, {selected_lng:.6f}` "
            f"(latitude, longitude). Analysis area: about 1 km × 1 km."
        )
        scene_key = (selected_lat, selected_lng, before_date, after_date)
        if st.session_state.get("download_key") != scene_key:
            try:
                with st.spinner("Downloading Sentinel-2 scenes..."):
                    st.session_state.downloaded_scene = download_before_after(
                        aoi,
                        before_date,
                        after_date,
                        resolution_m=10,
                        max_cloud_coverage=100,
                        search_window_days=60,
                    )
                    st.session_state.download_key = scene_key
            except (RuntimeError, ValueError, OSError) as error:
                st.error(f"Could not download Sentinel-2 data: {error}")
                st.stop()
        scene = st.session_state.downloaded_scene
    elif source == "Use downloaded scene":
        downloaded_path = Path("sentinel_scene.npz")
        if not downloaded_path.exists():
            st.info(
                "No downloaded scene found. Run `python download_scene.py` "
                "from the project folder, then refresh this page."
            )
            st.stop()
        try:
            scene = _load_npz_scene(downloaded_path.read_bytes())
        except (OSError, ValueError) as error:
            st.error(f"Could not load {downloaded_path}: {error}")
            st.stop()
        before_date = st.text_input("Earlier image date", "2023-02-07")
        after_date = st.text_input("Later image date", "2024-02-07")
        st.caption(
            f"Loaded local scene: {scene['before'].shape[0]} × "
            f"{scene['before'].shape[1]} pixels."
        )
    else:
        uploaded = st.file_uploader(
            "Choose a NumPy scene (.npz)",
            type=["npz"],
            help="The file must contain before and after four-band satellite images. "
            "Cloud arrays are optional.",
        )
        if uploaded is None:
            st.info("Choose a .npz file or switch to map selection.")
            st.stop()
        try:
            scene = _load_npz_scene(uploaded.getvalue())
        except ValueError as error:
            st.error(str(error))
            st.stop()
        before_date = st.text_input("Earlier image date", "2023-01-15")
        after_date = st.text_input("Later image date", "2024-01-15")

    st.divider()
    st.header("2. Configure the scan")
    sensitivity = st.slider(
        "How sensitive should alerts be?",
        0.15,
        0.95,
        0.20,
        0.01,
        help="Lower values find more possible changes but may include more false alarms. "
        "Higher values show only stronger changes.",
    )
    min_patch = st.slider(
        "Ignore patches smaller than (pixels)",
        4,
        150,
        8,
        help="Small isolated changes are ignored as likely noise.",
    )
    cloud_limit = st.slider(
        "Allow pixels with up to this cloud probability",
        0.05,
        0.80,
        0.45,
        0.05,
        help="Pixels that are too cloudy in either image are excluded from the analysis.",
    )
    run = st.button("Run forest-loss analysis", type="primary", width="stretch")

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

st.header("1. Analysis results")
st.caption(
    f"Comparing the image from **{before_date}** with **{after_date}**. "
    f"Cloudy or invalid pixels ignored: **{summary['cloud_coverage']:.1%}**."
)

alerts = result["alerts"].copy()
deforestation_detected = not alerts.empty

st.header("2. Final conclusion")
st.caption("This analysis gives you the two key answers first.")
with st.container(border=True):
    conclusion_cols = st.columns(2)
    with conclusion_cols[0]:
        st.markdown(
            f'<div class="conclusion-card"><div class="conclusion-label">'
            f"1. Affected area</div><div class=\"conclusion-value\">"
            f"{summary['loss_area_ha']:.2f} ha</div></div>",
            unsafe_allow_html=True,
        )
    with conclusion_cols[1]:
        card_class = "conclusion-card alert" if deforestation_detected else "conclusion-card"
        result_text = "Yes - likely" if deforestation_detected else "No - not detected"
        st.markdown(
            f'<div class="{card_class}"><div class="conclusion-label">'
            f"2. Deforestation detected?</div><div class=\"conclusion-value\">"
            f"{result_text}</div></div>",
            unsafe_allow_html=True,
        )
    if deforestation_detected:
        st.warning(
            "The imagery shows a likely forest-loss change. Confirm it with "
            "higher-resolution imagery or a field visit.",
            icon="⚠️",
        )
    else:
        st.success(
            "No significant deforestation was detected with the current settings.",
            icon="✅",
        )

left, right = st.columns([1.55, 1])
with left:
    st.subheader("Where might forest loss have occurred?")
    st.caption(
        "Green areas show stable vegetation, while yellow and red areas show "
        "increasing evidence of change. Grey areas were too cloudy to assess."
    )
    display = result["alert_score"].copy()
    display[result["valid_mask"] == 0] = np.nan
    fig = go.Figure(
        go.Heatmap(
            z=display,
            colorscale=[
                [0, "#198754"],
                [0.45, "#8bc34a"],
                [0.68, "#f6c343"],
                [1, "#c62828"],
            ],
            zmin=0,
            zmax=1,
            hovertemplate="pixel row=%{y}, pixel column=%{x}<br>change score=%{z:.2f}<extra></extra>",
            colorbar={"title": "Change score"},
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
    st.subheader("How the result is created")
    st.markdown(
        """
        1. **Remove clouds** — cloudy pixels in either image are left out.
        2. **Compare vegetation** — the images are checked for changes in
           vegetation and moisture.
        3. **Reduce seasonal noise** — broad changes such as leaf fall are
           reduced before scoring.
        4. **Group evidence** — nearby changed pixels become one review area.
        """
    )
    st.info(
        "These are screening alerts, not proof of illegal deforestation. "
        "Review each alert with a higher-resolution image or a field visit.",
        title="Important",
    )

st.header("3. Review the evidence")
st.caption(
    "Use these maps to understand why an area was flagged. "
    "Green indicates healthier vegetation; red indicates weaker vegetation."
)
evidence_cols = st.columns(3)
for column, title, image in [
    (evidence_cols[0], "Vegetation before", result["before_ndvi"]),
    (evidence_cols[1], "Vegetation after", result["after_ndvi"]),
    (evidence_cols[2], "Change strength", result["alert_score"]),
]:
    with column:
        shown = image.astype(float).copy()
        shown[result["valid_mask"] == 0] = np.nan
        scale = "RdYlGn" if title.startswith("Vegetation") else "YlOrRd"
        chart = go.Figure(go.Heatmap(z=shown, colorscale=scale, showscale=False))
        chart.update_layout(height=220, margin={"l": 0, "r": 0, "t": 28, "b": 0}, title=title)
        st.plotly_chart(chart, width="stretch")

with st.expander("Image change details"):
    if alerts.empty:
        st.info("There are no detected image changes to show.")
    else:
        change_table = alerts[
            ["alert_id", "confidence", "area_ha", "dominant_signal", "cloud_free"]
        ].rename(
            columns={
                "alert_id": "Review ID",
                "confidence": "Change strength",
                "area_ha": "Estimated area (ha)",
                "dominant_signal": "Strongest signal",
                "cloud_free": "Clear imagery",
            }
        )
        change_table.insert(
            2,
            "Confidence",
            np.where(
                change_table["Change strength"] >= 0.85,
                "High",
                np.where(change_table["Change strength"] >= 0.72, "Medium", "Review"),
            ),
        )
        st.caption(
            "Each row is a connected area where the satellite images showed "
            "possible forest change. Start with rows marked High."
        )
        st.dataframe(
            change_table.style.format(
                {
                    "Change strength": "{:.0%}",
                    "Estimated area (ha)": "{:.2f}",
                    "Clear imagery": "{:.0%}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
