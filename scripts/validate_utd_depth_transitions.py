"""用 UTD Kinect v2 原始深度序列验收真实深度特征链路。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import load_config  # noqa: E402
from core.depth_pipeline import DepthPipeline  # noqa: E402


DEFAULT_ROOT = (
    PROJECT_ROOT
    / "test_videos"
    / "external"
    / "utd_continuous_transitions"
    / "ContinuousTransitionAndFallDataset"
)


def _sample_indexes(frame_count: int, stride: int) -> list[int]:
    indexes = list(range(0, frame_count, max(1, stride)))
    if frame_count and (not indexes or indexes[-1] != frame_count - 1):
        indexes.append(frame_count - 1)
    return indexes


def validate(dataset_root: Path, stride: int = 15) -> dict[str, Any]:
    pipeline = DepthPipeline(load_config(PROJECT_ROOT / "config.yaml").multimodal)
    files = sorted(dataset_root.glob("Subject*/Depth/*_depth_K2.mat"))
    trials: list[dict[str, Any]] = []
    total_frames = 0
    metric_frames = 0
    available_frames = 0
    finite_feature_frames = 0
    changed_distance_frames = 0
    depth_medians: list[float] = []

    for path in files:
        state: dict[str, Any] = {}
        trial_metric = 0
        trial_available = 0
        trial_feature = 0
        trial_changed = 0
        max_spatial_score = 0.0
        with h5py.File(path, "r") as handle:
            if "depth_K2" not in handle:
                trials.append({"file": str(path), "passed": False, "reason": "缺少 depth_K2"})
                continue
            frames = handle["depth_K2"]
            source_frame_count = int(frames.shape[0])
            indexes = _sample_indexes(source_frame_count, stride)
            for frame_index in indexes:
                # 文件布局为 frame × width × height，转为 height × width。
                depth = np.asarray(frames[frame_index], dtype=np.float32).T
                timestamp_s = float(frame_index) / 30.0
                result, state, _ = pipeline.process_frame(depth, timestamp_s, state)
                diagnostics = result.diagnostics
                total_frames += 1
                if diagnostics.get("metric_depth_input"):
                    metric_frames += 1
                    trial_metric += 1
                if result.available:
                    available_frames += 1
                    trial_available += 1
                distance_change = float(diagnostics.get("distance_change_ratio", 0.0))
                spatial_score = float(diagnostics.get("pre_fall_spatial_score", 0.0))
                if np.isfinite(distance_change) and np.isfinite(spatial_score):
                    finite_feature_frames += 1
                    trial_feature += 1
                if distance_change > 0.0:
                    changed_distance_frames += 1
                    trial_changed += 1
                median_depth = diagnostics.get("median_depth")
                if median_depth is not None:
                    depth_medians.append(float(median_depth))
                max_spatial_score = max(max_spatial_score, spatial_score)

        sampled = len(indexes)
        passed = (
            trial_metric == sampled
            and trial_feature == sampled
            # UTD 深度为仅保留人体前景的稀疏图，空背景帧仍应被质量门控；
            # 每段至少 20% 抽样帧可用且确实观测到距离变化即可验收信号链路。
            and trial_available / max(1, sampled) >= 0.20
            and trial_changed > 0
        )
        trials.append({
            "subject": path.parents[1].name,
            "trial": path.stem,
            "source_frames": source_frame_count,
            "sampled_frames": sampled,
            "metric_depth_frames": trial_metric,
            "available_frames": trial_available,
            "distance_change_frames": trial_changed,
            "max_pre_fall_spatial_score": round(max_spatial_score, 2),
            "passed": passed,
        })

    subjects = sorted({item.get("subject") for item in trials if item.get("subject")})
    passed_trials = sum(1 for item in trials if item.get("passed"))
    report = {
        "dataset_name": "UTD Continuous Transition Movements Dataset",
        "validation_scope": "真实 Kinect v2 深度读取与前置空间特征链路",
        "action_accuracy_scope": False,
        "reason_action_accuracy_unavailable": "公开下载文件未提供逐动作起止真值",
        "acceptance_criteria": {
            "expected_subjects": 5,
            "expected_trials": 25,
            "metric_and_finite_feature_rate": 1.0,
            "minimum_available_rate_per_sparse_foreground_trial": 0.20,
            "minimum_distance_change_frames_per_trial": 1
        },
        "subjects": len(subjects),
        "trials": len(trials),
        "passed_trials": passed_trials,
        "sampled_frames": total_frames,
        "metric_input_rate": round(metric_frames / max(1, total_frames), 4),
        "depth_available_rate": round(available_frames / max(1, total_frames), 4),
        "finite_feature_rate": round(finite_feature_frames / max(1, total_frames), 4),
        "distance_change_frame_rate": round(changed_distance_frames / max(1, total_frames), 4),
        "median_depth": round(float(np.median(depth_medians)), 2) if depth_medians else None,
        "acceptance_passed": (
            len(subjects) == 5
            and len(trials) == 25
            and passed_trials == 25
            and metric_frames == total_frames
            and finite_feature_frames == total_frames
        ),
        "trials_detail": trials,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="UTD 原始深度专项验收")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "exports" / "utd_depth_acceptance_report.json",
    )
    args = parser.parse_args()
    if not args.dataset_root.exists():
        raise SystemExit(f"数据目录不存在：{args.dataset_root}")
    report = validate(args.dataset_root, args.stride)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"真实深度序列：{report['passed_trials']}/{report['trials']} 通过")
    print(f"抽样帧：{report['sampled_frames']}，深度可用率：{report['depth_available_rate']:.1%}")
    print(f"报告：{args.output}")
    return 0 if report["acceptance_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
