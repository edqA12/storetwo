from __future__ import annotations

from collections import Counter
from statistics import fmean, median
from typing import Any, Iterable


ABLATION_VARIANTS = (
    "rgb_only",
    "depth_only",
    "rgb_depth",
    "rgb_depth_baseline",
)


def _safe_divide(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def _rounded(value: float | None, digits: int = 4) -> float | None:
    return round(value, digits) if value is not None else None


def binary_metrics(
    expected: Iterable[bool],
    predicted: Iterable[bool],
    *,
    observed_duration_s: float = 0.0,
    delays_s: Iterable[float] = (),
) -> dict[str, Any]:
    pairs = list(zip(expected, predicted, strict=True))
    true_positive = sum(1 for truth, guess in pairs if truth and guess)
    true_negative = sum(1 for truth, guess in pairs if not truth and not guess)
    false_positive = sum(1 for truth, guess in pairs if not truth and guess)
    false_negative = sum(1 for truth, guess in pairs if truth and not guess)
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    false_alarm_rate = _safe_divide(false_positive, false_positive + true_negative)
    false_alarms_per_hour = (
        false_positive / (observed_duration_s / 3600.0)
        if observed_duration_s > 0 else None
    )
    delays = [float(value) for value in delays_s if float(value) >= 0.0]
    return {
        "samples": len(pairs),
        "tp": true_positive,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
        "precision": _rounded(precision),
        "recall": _rounded(recall),
        "f1": _rounded(f1),
        "false_alarm_rate": _rounded(false_alarm_rate),
        "false_alarms_per_hour": _rounded(false_alarms_per_hour),
        "mean_detection_delay_s": _rounded(fmean(delays) if delays else None, 3),
    }


def evaluate_ablation(
    records: Iterable[dict[str, Any]],
    variants: Iterable[str] = ABLATION_VARIANTS,
) -> dict[str, Any]:
    rows = list(records)
    if not rows:
        raise ValueError("评测清单为空，不能生成准确率指标。")
    allowed_labels = {"NORMAL", "NEAR_FALL", "FALL"}
    labels = [str(row.get("ground_truth", "")).strip().upper() for row in rows]
    unknown = sorted({label for label in labels if label not in allowed_labels})
    if unknown:
        raise ValueError(f"存在不支持的真实标签：{', '.join(unknown)}")

    report: dict[str, Any] = {
        "dataset": {
            "samples": len(rows),
            "label_distribution": dict(Counter(labels)),
            "has_near_fall_ground_truth": "NEAR_FALL" in labels,
        },
        "variants": {},
        "warnings": [],
    }
    if "NEAR_FALL" not in labels:
        report["warnings"].append(
            "数据集中没有近跌倒真值，不输出近跌倒识别准确率；不得据此宣称近跌倒预测性能。"
        )

    for variant in variants:
        prediction_key = f"{variant}_prediction"
        detection_key = f"{variant}_detection_time_s"
        evaluated: list[tuple[dict[str, Any], str, str]] = []
        for row, truth in zip(rows, labels, strict=True):
            prediction = str(row.get(prediction_key, "")).strip().upper()
            if not prediction:
                continue
            if prediction not in allowed_labels:
                raise ValueError(f"{prediction_key} 存在不支持的预测标签：{prediction}")
            evaluated.append((row, truth, prediction))
        if not evaluated:
            report["variants"][variant] = {
                "status": "INSUFFICIENT_DATA",
                "samples": 0,
                "message": "没有该组预测结果，未计算指标。",
            }
            continue

        duration = sum(max(0.0, float(row.get("duration_s", 0.0) or 0.0)) for row, _, _ in evaluated)
        event_delays: list[float] = []
        confirmation_delays: list[float] = []
        confirmation_key = f"{variant}_confirmation_time_s"
        for row, truth, prediction in evaluated:
            if truth == prediction and truth in {"NEAR_FALL", "FALL"}:
                event_time = row.get("event_time_s")
                detection_time = row.get(detection_key)
                if event_time not in (None, "") and detection_time not in (None, ""):
                    event_delays.append(float(detection_time) - float(event_time))
            if truth == "FALL":
                event_time = row.get("event_time_s")
                confirmation_time = row.get(confirmation_key)
                if event_time not in (None, "") and confirmation_time not in (None, ""):
                    confirmation_delays.append(float(confirmation_time) - float(event_time))

        overall = binary_metrics(
            [truth != "NORMAL" for _, truth, _ in evaluated],
            [prediction != "NORMAL" for _, _, prediction in evaluated],
            observed_duration_s=duration,
            delays_s=event_delays,
        )
        near_fall = None
        if any(truth == "NEAR_FALL" for _, truth, _ in evaluated):
            near_fall = binary_metrics(
                [truth == "NEAR_FALL" for _, truth, _ in evaluated],
                [prediction == "NEAR_FALL" for _, _, prediction in evaluated],
                observed_duration_s=duration,
            )
        fall = None
        if any(truth == "FALL" for _, truth, _ in evaluated):
            fall = binary_metrics(
                [truth == "FALL" for _, truth, _ in evaluated],
                [prediction == "FALL" for _, _, prediction in evaluated],
                observed_duration_s=duration,
                delays_s=[
                    float(row[detection_key]) - float(row["event_time_s"])
                    for row, truth, prediction in evaluated
                    if truth == prediction == "FALL"
                    and row.get("event_time_s") not in (None, "")
                    and row.get(detection_key) not in (None, "")
                ],
            )
        degraded = [item for item in evaluated if str(item[0].get("depth_degraded", "")).lower() in {"1", "true", "yes"}]
        degraded_metrics = None
        if degraded:
            degraded_duration = sum(
                max(0.0, float(row.get("duration_s", 0.0) or 0.0)) for row, _, _ in degraded
            )
            degraded_metrics = binary_metrics(
                [truth != "NORMAL" for _, truth, _ in degraded],
                [prediction != "NORMAL" for _, _, prediction in degraded],
                observed_duration_s=degraded_duration,
            )
        report["variants"][variant] = {
            "status": "OK",
            "samples": len(evaluated),
            "coverage": round(len(evaluated) / len(rows), 4),
            "overall_alert": overall,
            "near_fall": near_fall,
            "fall": fall,
            "depth_degraded_subset": degraded_metrics,
            "mean_fall_confirmation_delay_s": _rounded(
                fmean(value for value in confirmation_delays if value >= 0.0)
                if any(value >= 0.0 for value in confirmation_delays) else None,
                3,
            ),
        }
    return report


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * min(1.0, max(0.0, percentile))
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def temporal_event_metrics(
    truth_events: Iterable[dict[str, Any]],
    predicted_events: Iterable[dict[str, Any]],
    *,
    event_type: str,
    clip_ids: Iterable[str] = (),
    tolerance_before_s: float = 0.5,
    tolerance_after_s: float = 0.5,
) -> dict[str, Any]:
    """按视频内时间段进行一对一事件匹配，并输出事件级与视频级指标。"""

    target = str(event_type).strip().upper()
    truths = [
        {
            **item,
            "clip_id": str(item["clip_id"]),
            "start_s": float(item["start_s"]),
            "end_s": float(item["end_s"]),
        }
        for item in truth_events
        if str(item.get("event_type", "")).strip().upper() == target
    ]
    predictions = [
        {
            **item,
            "clip_id": str(item["clip_id"]),
            "detection_time_s": float(item["detection_time_s"]),
        }
        for item in predicted_events
        if str(item.get("event_type", "")).strip().upper() == target
    ]
    for truth in truths:
        if truth["end_s"] < truth["start_s"]:
            raise ValueError(f"事件结束时间早于开始时间：{truth.get('event_id', '')}")

    unmatched_predictions = set(range(len(predictions)))
    matches: list[dict[str, Any]] = []
    for truth in sorted(truths, key=lambda item: (item["clip_id"], item["start_s"])):
        candidates = [
            index
            for index in unmatched_predictions
            if predictions[index]["clip_id"] == truth["clip_id"]
            and truth["start_s"] - tolerance_before_s
            <= predictions[index]["detection_time_s"]
            <= truth["end_s"] + tolerance_after_s
        ]
        if not candidates:
            continue
        prediction_index = min(
            candidates,
            key=lambda index: predictions[index]["detection_time_s"],
        )
        unmatched_predictions.remove(prediction_index)
        prediction = predictions[prediction_index]
        delay_s = prediction["detection_time_s"] - truth["start_s"]
        recovery_margin_s = truth["end_s"] - prediction["detection_time_s"]
        matches.append({
            "clip_id": truth["clip_id"],
            "event_id": str(truth.get("event_id", "")),
            "truth_start_s": round(truth["start_s"], 3),
            "truth_end_s": round(truth["end_s"], 3),
            "detection_time_s": round(prediction["detection_time_s"], 3),
            "warning_delay_s": round(delay_s, 3),
            "recovery_margin_s": round(recovery_margin_s, 3),
        })

    true_positive = len(matches)
    false_negative = len(truths) - true_positive
    false_positive = len(unmatched_predictions)
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )

    all_clip_ids = {
        *(str(value) for value in clip_ids),
        *(item["clip_id"] for item in truths),
        *(item["clip_id"] for item in predictions),
    }
    truth_positive_clips = {item["clip_id"] for item in truths}
    predicted_positive_clips = {item["clip_id"] for item in predictions}
    clip_tp = len(truth_positive_clips & predicted_positive_clips)
    clip_fn = len(truth_positive_clips - predicted_positive_clips)
    clip_fp = len(predicted_positive_clips - truth_positive_clips)
    clip_tn = len(all_clip_ids - truth_positive_clips - predicted_positive_clips)
    delays = [float(item["warning_delay_s"]) for item in matches]
    margins = [float(item["recovery_margin_s"]) for item in matches]
    status = "OK" if truths else "INSUFFICIENT_POSITIVES"
    return {
        "status": status,
        "event_type": target,
        "truth_events": len(truths),
        "predicted_events": len(predictions),
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": _rounded(precision),
        "recall": _rounded(recall),
        "f1": _rounded(f1),
        "warning_delay_s": {
            "definition": "prediction_emit_time - annotated_event_start; positive means late",
            "mean": _rounded(fmean(delays) if delays else None, 3),
            "median": _rounded(median(delays) if delays else None, 3),
            "p95": _rounded(_percentile(delays, 0.95), 3),
        },
        "recovery_margin_s": {
            "definition": "annotated_event_end - prediction_emit_time; positive means before recovery/end",
            "mean": _rounded(fmean(margins) if margins else None, 3),
        },
        "clip_level_confusion_matrix": {
            "labels": [f"NOT_{target}", target],
            "matrix": [[clip_tn, clip_fp], [clip_fn, clip_tp]],
            "tn": clip_tn,
            "fp": clip_fp,
            "fn": clip_fn,
            "tp": clip_tp,
        },
        "matches": matches,
        "message": (
            "没有真实正事件，Precision、Recall、F1 和预警延迟均不具备统计定义。"
            if not truths else ""
        ),
    }
