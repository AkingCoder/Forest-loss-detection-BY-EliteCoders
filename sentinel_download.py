from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any

import numpy as np
from dotenv import load_dotenv


load_dotenv()


EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B04", "B08", "B11", "B12", "SCL", "dataMask"],
      units: ["REFLECTANCE", "REFLECTANCE", "REFLECTANCE",
              "REFLECTANCE", "DN", "DN"]
    }],
    output: { bands: 6, sampleType: "FLOAT32" }
  };
}

function evaluatePixel(sample) {
  return [sample.B04, sample.B08, sample.B11, sample.B12,
          sample.SCL, sample.dataMask];
}
"""

# SCL classes retained as usable observations. Classes 8, 9 and 10 are
# cloud-related; class 3 is cloud shadow and is excluded as well.
VALID_SCL_CLASSES = (4, 5, 6, 7, 11)


def _parse_date(value: str | date) -> str:
    if isinstance(value, date):
        return value.isoformat()
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as error:
        raise ValueError(f"Invalid date {value!r}; use YYYY-MM-DD.") from error


def create_sentinelhub_config() -> Any:
    """Create Sentinel Hub credentials from SENTINELHUB_CLIENT_ID/SECRET."""
    from sentinelhub import SHConfig

    client_id = os.getenv("SENTINELHUB_CLIENT_ID")
    client_secret = os.getenv("SENTINELHUB_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "Set SENTINELHUB_CLIENT_ID and SENTINELHUB_CLIENT_SECRET before downloading data."
        )
    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    return config


def download_sentinel2_scene(
    bbox_wgs84: tuple[float, float, float, float],
    scene_date: str | date,
    *,
    resolution_m: int = 10,
    max_cloud_coverage: float = 30.0,
    search_window_days: int = 15,
    config: Any | None = None,
) -> dict[str, np.ndarray | None]:
    """Download one Sentinel-2 L2A scene in detector-compatible format.

    ``bbox_wgs84`` is ``(min_longitude, min_latitude, max_longitude,
    max_latitude)``. Returned reflectance bands are float32 arrays in the
    order B04, B08, B11, B12 and are scaled from Sentinel Hub reflectance.
    """
    if resolution_m <= 0:
        raise ValueError("resolution_m must be greater than zero.")
    if search_window_days < 0:
        raise ValueError("search_window_days must not be negative.")
    min_lng, min_lat, max_lng, max_lat = bbox_wgs84
    if not min_lng < max_lng or not min_lat < max_lat:
        raise ValueError("bbox must be (min_lng, min_lat, max_lng, max_lat).")

    from sentinelhub import (
        BBox,
        CRS,
        DataCollection,
        MimeType,
        MosaickingOrder,
        SentinelHubRequest,
        bbox_to_dimensions,
    )

    config = config or create_sentinelhub_config()
    bbox = BBox(bbox=bbox_wgs84, crs=CRS.WGS84)
    size = bbox_to_dimensions(bbox, resolution=resolution_m)
    day = _parse_date(scene_date)
    requested_day = date.fromisoformat(day)
    interval_start = (requested_day - timedelta(days=search_window_days)).isoformat()
    interval_end = (requested_day + timedelta(days=search_window_days)).isoformat()
    request = SentinelHubRequest(
        evalscript=EVALSCRIPT,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(interval_start, interval_end),
                mosaicking_order=MosaickingOrder.LEAST_CC,
                maxcc=max_cloud_coverage / 100,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )
    downloaded = request.get_data()
    if not downloaded:
        raise RuntimeError(f"No Sentinel-2 L2A data was returned for {day}.")

    raw = np.asarray(downloaded[0], dtype=np.float32)
    if raw.ndim != 3 or raw.shape[2] != 6:
        raise RuntimeError(f"Unexpected Sentinel Hub response shape: {raw.shape}.")

    scl = raw[:, :, 4].astype(np.uint8)
    data_mask = raw[:, :, 5] > 0
    cloud_free = data_mask & np.isin(scl, VALID_SCL_CLASSES)
    if not np.any(cloud_free):
        raise RuntimeError(
            f"No valid cloud-free pixels found near {day}. "
            f"Try a wider search window or a less restrictive cloud limit."
        )
    bands = np.clip(raw[:, :, :4], 0, 1).astype(np.float32)
    cloud_probability = np.where(cloud_free, 0.0, 1.0).astype(np.float32)
    return {
        "before": bands,
        "after": bands.copy(),
        "before_cloud": cloud_probability,
        "after_cloud": cloud_probability.copy(),
    }


def download_before_after(
    bbox_wgs84: tuple[float, float, float, float],
    before_date: str | date,
    after_date: str | date,
    **kwargs: Any,
) -> dict[str, np.ndarray | None]:
    """Download two dates and return a scene ready for detect_forest_loss."""
    before = download_sentinel2_scene(bbox_wgs84, before_date, **kwargs)
    after = download_sentinel2_scene(bbox_wgs84, after_date, **kwargs)
    if before["before"].shape != after["before"].shape:
        raise RuntimeError("Before and after downloads do not have matching shapes.")
    return {
        "before": before["before"],
        "after": after["before"],
        "before_cloud": before["before_cloud"],
        "after_cloud": after["before_cloud"],
    }
