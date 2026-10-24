from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import numpy as np
import pandas as pd

BANDS = ("red", "nir", "swir1", "swir2")
PIXEL_AREA_HA = 0.01


@dataclass(frozen=True)
class DetectionConfig:
    alert_threshold: float = 0.20
    min_patch_pixels: int = 8
    cloud_probability_limit: float = 0.45


def _index(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return numerator / (denominator + 1e-6)


def _components(mask: np.ndarray) -> tuple[np.ndarray, list[dict[str, float]]]:
    labels = np.zeros(mask.shape, dtype=np.int32)
    patches: list[dict[str, float]] = []
    next_label = 0
    height, width = mask.shape
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or labels[row, col]:
                continue
            next_label += 1
            queue = deque([(row, col)])
            labels[row, col] = next_label
            pixels: list[tuple[int, int]] = []
            while queue:
                current_row, current_col = queue.popleft()
                pixels.append((current_row, current_col))
                for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    neighbor_row, neighbor_col = current_row + d_row, current_col + d_col
                    if (
                        0 <= neighbor_row < height
                        and 0 <= neighbor_col < width
                        and mask[neighbor_row, neighbor_col]
                        and labels[neighbor_row, neighbor_col] == 0
                    ):
                        labels[neighbor_row, neighbor_col] = next_label
                        queue.append((neighbor_row, neighbor_col))
            rows, cols = zip(*pixels)
            patches.append(
                {
                    "label": next_label,
                    "size": len(pixels),
                    "row": float(np.mean(rows)),
                    "column": float(np.mean(cols)),
                }
            )
    return labels, patches


def _cloud_mask(scene: dict[str, np.ndarray | None], config: DetectionConfig) -> np.ndarray:
    before_cloud = scene.get("before_cloud")
    after_cloud = scene.get("after_cloud")
    shape = scene["before"].shape[:2]
    before_valid = np.ones(shape, dtype=bool) if before_cloud is None else before_cloud <= config.cloud_probability_limit
    after_valid = np.ones(shape, dtype=bool) if after_cloud is None else after_cloud <= config.cloud_probability_limit
    return before_valid & after_valid


def detect_forest_loss(
    scene: dict[str, np.ndarray | None], config: DetectionConfig = DetectionConfig()
) -> dict[str, np.ndarray | pd.DataFrame]:
    before = np.asarray(scene["before"], dtype=np.float32)
    after = np.asarray(scene["after"], dtype=np.float32)
    if before.ndim != 3 or after.ndim != 3 or before.shape != after.shape or before.shape[2] != 4:
        raise ValueError("before and after must both have shape (height, width, 4).")
    if not 0 < config.alert_threshold < 1:
        raise ValueError("alert_threshold must be between 0 and 1.")
    if config.min_patch_pixels < 1:
        raise ValueError("min_patch_pixels must be at least 1.")
    if not 0 <= config.cloud_probability_limit <= 1:
        raise ValueError("cloud_probability_limit must be between 0 and 1.")

    before_ndvi = _index(before[:, :, 1] - before[:, :, 0], before[:, :, 1] + before[:, :, 0])
    after_ndvi = _index(after[:, :, 1] - after[:, :, 0], after[:, :, 1] + after[:, :, 0])
    before_nbr = _index(before[:, :, 1] - before[:, :, 3], before[:, :, 1] + before[:, :, 3])
    after_nbr = _index(after[:, :, 1] - after[:, :, 3], after[:, :, 1] + after[:, :, 3])
    valid = _cloud_mask(scene, config) & np.isfinite(before_ndvi) & np.isfinite(after_ndvi)

    ndvi_loss = np.clip((before_ndvi - after_ndvi) / 0.65, 0, 1)
    nbr_loss = np.clip((before_nbr - after_nbr) / 0.70, 0, 1)
    swir_increase = np.clip((after[:, :, 2] - before[:, :, 2]) / 0.35, 0, 1)
    baseline = float(np.median(ndvi_loss[valid])) if np.any(valid) else 0.0
    seasonal_residual = np.clip(ndvi_loss - max(0, baseline * 0.55), 0, 1)
    score = np.clip(0.50 * nbr_loss + 0.30 * seasonal_residual + 0.20 * swir_increase, 0, 1)
    score[~valid] = 0

    candidate = valid & (score >= config.alert_threshold)
    labels, patches = _components(candidate)
    rows = []
    keep_labels = set()
    for patch in patches:
        if patch["size"] < config.min_patch_pixels:
            continue
        keep_labels.add(int(patch["label"]))
        center_row, center_col = int(round(patch["row"])), int(round(patch["column"]))
        evidence = {
            "NBR loss": float(nbr_loss[center_row, center_col]),
            "NDVI loss": float(seasonal_residual[center_row, center_col]),
            "SWIR increase": float(swir_increase[center_row, center_col]),
        }
        rows.append(
            {
                "alert_id": f"VR-{len(rows) + 1:03d}",
                "confidence": float(np.mean(score[labels == patch["label"]])),
                "area_ha": patch["size"] * PIXEL_AREA_HA,
                "row": round(patch["row"], 1),
                "column": round(patch["column"], 1),
                "dominant_signal": max(evidence, key=evidence.get),
                "cloud_free": float(np.mean(valid[labels == patch["label"]])),
            }
        )
    filtered_labels = np.where(np.isin(labels, list(keep_labels)), labels, 0)
    alert_columns = [
        "alert_id",
        "confidence",
        "area_ha",
        "row",
        "column",
        "dominant_signal",
        "cloud_free",
    ]
    alerts = pd.DataFrame(rows, columns=alert_columns).sort_values(
        "confidence", ascending=False, ignore_index=True
    )
    return {
        "alert_score": score,
        "before_ndvi": before_ndvi,
        "after_ndvi": after_ndvi,
        "valid_mask": valid,
        "labels": filtered_labels,
        "alerts": alerts,
    }


def summarize_alerts(result: dict[str, np.ndarray | pd.DataFrame]) -> dict[str, float | int]:
    alerts = result["alerts"]
    valid_mask = result["valid_mask"]
    return {
        "patch_count": int(len(alerts)),
        "loss_area_ha": float(alerts["area_ha"].sum()) if not alerts.empty else 0.0,
        "high_confidence_area_ha": float(alerts.loc[alerts["confidence"] >= 0.85, "area_ha"].sum())
        if not alerts.empty
        else 0.0,
        "valid_pixels": int(np.count_nonzero(valid_mask)),
        "cloud_coverage": float(1 - np.mean(valid_mask)),
        "max_score": float(np.max(result["alert_score"])) if np.any(valid_mask) else 0.0,
    }


def make_demo_scene(seed: int = 23, height: int = 120, width: int = 160) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:height, :width]
    texture = rng.normal(0, 0.018, (height, width))
    forest = 0.72 + texture + 0.04 * np.sin(xx / 13) * np.cos(yy / 17)
    before = np.stack(
        [0.07 + texture, 0.72 + texture, 0.16 + texture, 0.10 + texture], axis=-1
    ).astype(np.float32)
    before += rng.normal(0, 0.012, before.shape).astype(np.float32)
    after = before.copy()
    # Simulated seasonal signal affects the whole scene; loss patches are persistent.
    after[:, :, 0] += 0.018
    after[:, :, 1] -= 0.035
    after[:, :, 2] += 0.012
    loss_mask = ((xx - width * 0.68) / 18) ** 2 + ((yy - height * 0.42) / 24) ** 2 < 1
    loss_mask |= ((xx - width * 0.30) / 11) ** 2 + ((yy - height * 0.70) / 16) ** 2 < 1
    after[loss_mask, 0] = 0.25
    after[loss_mask, 1] = 0.18
    after[loss_mask, 2] = 0.42
    after[loss_mask, 3] = 0.38
    clouds = rng.random((height, width))
    before_cloud = np.where(clouds > 0.91, 0.85, rng.uniform(0, 0.2, (height, width)))
    after_cloud = np.where(rng.random((height, width)) > 0.90, 0.85, rng.uniform(0, 0.2, (height, width)))
    return {
        "before": np.clip(before, 0.01, 1),
        "after": np.clip(after, 0.01, 1),
        "before_cloud": before_cloud,
        "after_cloud": after_cloud,
    }
