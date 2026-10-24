from pathlib import Path

import numpy as np

from detector import detect_forest_loss, summarize_alerts
from sentinel_download import download_before_after


# Center point in (latitude, longitude), converted to a small WGS84 bbox.
CENTER_LAT = 26.9255
CENTER_LNG = 75.7488
AOI_HALF_SIZE_DEGREES = 0.005
AOI = (
    CENTER_LNG - AOI_HALF_SIZE_DEGREES,
    CENTER_LAT - AOI_HALF_SIZE_DEGREES,
    CENTER_LNG + AOI_HALF_SIZE_DEGREES,
    CENTER_LAT + AOI_HALF_SIZE_DEGREES,
)
BEFORE_DATE = "2023-02-07"
AFTER_DATE = "2024-02-07"
OUTPUT_FILE = Path("sentinel_scene.npz")


def main() -> None:
    print("Downloading Sentinel-2 L2A scenes...")
    scene = download_before_after(
        AOI,
        BEFORE_DATE,
        AFTER_DATE,
        resolution_m=10,
        max_cloud_coverage=100,
        search_window_days=60,
    )

    np.savez_compressed(
        OUTPUT_FILE,
        before=scene["before"],
        after=scene["after"],
        before_cloud=scene["before_cloud"],
        after_cloud=scene["after_cloud"],
    )

    result = detect_forest_loss(scene)
    print(f"Saved: {OUTPUT_FILE.resolve()}")
    print(f"Array shape: {scene['before'].shape}")
    print("Summary:", summarize_alerts(result))
    print(result["alerts"].to_string(index=False))


if __name__ == "__main__":
    main()
