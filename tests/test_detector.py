import numpy as np

from detector import DetectionConfig, detect_forest_loss, make_demo_scene, summarize_alerts


def test_demo_produces_alerts_and_summary():
    result = detect_forest_loss(make_demo_scene(seed=23), DetectionConfig(min_patch_pixels=10))
    summary = summarize_alerts(result)
    assert summary["patch_count"] >= 1
    assert summary["loss_area_ha"] > 0
    assert result["alert_score"].shape == (120, 160)


def test_cloud_mask_excludes_cloudy_pixels():
    scene = make_demo_scene(seed=7, height=20, width=20)
    scene["before_cloud"] = np.ones((20, 20))
    result = detect_forest_loss(scene)
    assert np.count_nonzero(result["valid_mask"]) == 0
    assert result["alerts"].empty


def test_invalid_band_shape_is_explicit():
    scene = {"before": np.zeros((4, 4, 3)), "after": np.zeros((4, 4, 3))}
    try:
        detect_forest_loss(scene)
    except ValueError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError("Expected a shape validation error")
