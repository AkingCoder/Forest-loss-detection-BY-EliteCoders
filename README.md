# VanRakshak: Satellite Forest-Loss Alerts

An offline-first hackathon prototype for finding likely forest loss between two Sentinel-2 observation dates in India.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The app lets you click a location on a map, downloads Sentinel-2 data for a
small bounding box around that point, and runs the forest-loss detector. An
`.npz` upload option is also available.

## Download Sentinel-2 L2A data

Install the additional dependency and set Sentinel Hub credentials from the
Sentinel Hub Dashboard:

```powershell
python -m pip install -r requirements.txt
$env:SENTINELHUB_CLIENT_ID = "your-client-id"
$env:SENTINELHUB_CLIENT_SECRET = "your-client-secret"
```

Alternatively, create a local `.env` file (never commit it):

```text
SENTINELHUB_CLIENT_ID=your-client-id
SENTINELHUB_CLIENT_SECRET=your-client-secret
```

The downloader loads this file automatically.

Then download a before/after scene:

```python
from detector import detect_forest_loss
from sentinel_download import download_before_after

scene = download_before_after(
    (12.814865, 45.83978, 12.860184, 45.866098),
    "2023-02-07",
    "2024-02-07",
)
result = detect_forest_loss(scene)
```

Coordinates are `(min_longitude, min_latitude, max_longitude, max_latitude)`.
The downloader requests Sentinel-2 L2A B04, B08, B11 and B12 at 10 m output
resolution, and uses the SCL band to exclude cloud and cloud-shadow classes.

## Upload your own scene

The app accepts a NumPy `.npz` file with four-band arrays in this order:

```text
before: (height, width, 4) -> [B4 red, B8 NIR, B11 SWIR1, B12 SWIR2]
after:  (height, width, 4)
before_cloud: (height, width), optional cloud probabilities from 0 to 1
after_cloud:  (height, width), optional cloud probabilities from 0 to 1
```

Example:

```python
import numpy as np
np.savez("scene.npz", before=before, after=after,
         before_cloud=before_cloud, after_cloud=after_cloud)
```

## Select a location in the app

Choose **Select location on map**, click a point in the map, and the app will
use the selected latitude and longitude to create a WGS84 bounding box and
download before/after Sentinel-2 scenes. The default box extends `0.005`
degrees in each direction from the selected point. Sentinel Hub credentials
are required for this mode.

## Approach

1. Exclude pixels cloudy in either date.
2. Derive NDVI and NBR from Sentinel-2 bands.
3. Reduce scene-wide seasonal shifts before scoring persistent change.
4. Combine NBR loss, NDVI loss and SWIR increase into a 0–1 risk score.
5. Keep contiguous patches above the threshold and remove small noise.
6. Export prioritized alert centroids and confidence as CSV.

The prototype is a decision-support tool, not a legal deforestation map. For a competition submission, use Sentinel-2 L2A composites and tune thresholds using Global Forest Watch labels in training states, then report patch-level precision, recall and F1 on a fully held-out state.

## Suggested demo pitch

“VanRakshak does not confuse one cloudy image or a seasonal leaf-fall event with forest loss. It compares cloud-free, season-normalized spectral evidence, groups only persistent changes into patches, and sends the highest-confidence locations to a field team.”
