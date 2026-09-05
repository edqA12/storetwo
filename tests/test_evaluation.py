from __future__ import annotations

import pytest

from core.evaluation import binary_metrics, evaluate_ablation, temporal_event_metrics


def test_binary_metrics_include_false_alarms_per_hour_and_delay():
    metrics = binary_metrics(
        [True, True, False, False],
        [True, False, True, False],
        observed_duration_s=3600.0,
        delays_s=[0.8],
    )
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["false_alarms_per_hour"] == 1.0
    assert metrics["mean_detection_delay_s"] == 0.8


def test_ablation_reports_near_fall_and_missing_variant_without_inventing_metrics():
    rows = [
        {
            "ground_truth": "NORMAL",
            "duration_s": "60",
            "rgb_only_prediction": "NORMAL",
            "depth_only_prediction": "",
        },
        {
            "ground_truth": "NEAR_FALL",
            "duration_s": "60",
            "event_time_s": "10",
            "rgb_only_prediction": "NEAR_FALL",
            "rgb_only_detection_time_s": "10.5",
            "depth_only_prediction": "",
        },
    ]
    report = evaluate_ablation(rows, variants=("rgb_only", "depth_only"))
    assert report["variants"]["rgb_only"]["overall_alert"]["f1"] == 1.0
    assert report["variants"]["rgb_only"]["near_fall"]["recall"] == 1.0
    assert report["variants"]["depth_only"]["status"] == "INSUFFICIENT_DATA"


def test_ablation_requires_real_ground_truth_labels():
    with pytest.raises(ValueError, match="真实标签"):
        evaluate_ablation([{"ground_truth": ""}], variants=("rgb_only",))


def test_temporal_event_metrics_match_once_and_report_delay_and_confusion():
    truths = [
        {"clip_id": "positive", "event_id": "nf-1", "event_type": "NEAR_FALL", "start_s": 2, "end_s": 4},
    ]
    predictions = [
        {"clip_id": "positive", "event_type": "NEAR_FALL", "detection_time_s": 3},
        {"clip_id": "negative", "event_type": "NEAR_FALL", "detection_time_s": 1},
    ]
    metrics = temporal_event_metrics(
        truths, predictions, event_type="NEAR_FALL", clip_ids=["positive", "negative", "clean"]
    )
    assert metrics["tp"] == 1 and metrics["fp"] == 1 and metrics["fn"] == 0
    assert metrics["precision"] == 0.5 and metrics["recall"] == 1.0
    assert metrics["warning_delay_s"]["mean"] == 1.0
    assert metrics["recovery_margin_s"]["mean"] == 1.0
    assert metrics["clip_level_confusion_matrix"]["matrix"] == [[1, 1], [0, 1]]


def test_temporal_event_metrics_do_not_invent_scores_without_positive_truth():
    metrics = temporal_event_metrics(
        [],
        [{"clip_id": "negative", "event_type": "NEAR_FALL", "detection_time_s": 1}],
        event_type="NEAR_FALL",
        clip_ids=["negative", "clean"],
    )
    assert metrics["status"] == "INSUFFICIENT_POSITIVES"
    assert metrics["precision"] == 0.0
    assert metrics["recall"] is None and metrics["f1"] is None
    assert metrics["clip_level_confusion_matrix"]["matrix"] == [[1, 1], [0, 0]]
