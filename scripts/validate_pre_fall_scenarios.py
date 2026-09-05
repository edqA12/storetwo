"""运行近跌倒与 Sit-to-Stand 的带真值工程场景验收。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import PreFallConfig  # noqa: E402
from core.pre_fall import PreFallRiskAnalyzer  # noqa: E402
from core.types import Decision, PoseFeatures  # noqa: E402


def _features(frame: dict[str, Any]) -> PoseFeatures:
    return PoseFeatures(
        valid=True,
        torso_angle_deg=float(frame.get("angle", 5.0)),
        box_aspect=0.45,
        hip_y_norm=0.55,
        center_x_norm=float(frame.get("x", 0.5)),
        center_y_norm=float(frame.get("y", 0.45)),
        center_speed=float(frame.get("speed", 0.04)),
        down_velocity=float(frame.get("down", 0.0)),
        angle_velocity=float(frame.get("rotation", 0.0)),
        motion_norm=float(frame.get("motion", 0.02)),
        lying_score=float(frame.get("lying", 0.05)),
        knee_angle_deg=(float(frame["knee"]) if frame.get("knee") is not None else None),
        quality=0.9,
    )


def validate(dataset: dict[str, Any]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for scenario in dataset.get("scenarios", []):
        config = PreFallConfig(
            baseline_min_samples=1,
            near_fall_min_stability_score=0.0,
            near_fall_min_torso_angle=20.0,
            score_smoothing_alpha=1.0,
            window_sample_interval_s=0.0,
        )
        analyzer = PreFallRiskAnalyzer(config)
        state: dict[str, Any] = {}
        near_falls: list[dict[str, Any]] = []
        sit_to_stand: list[dict[str, Any]] = []
        for frame in scenario["frames"]:
            decision = Decision(1, str(frame.get("state", "NORMAL")), "测试", 10.0)
            _, new_near_falls = analyzer.update(
                1, _features(frame), decision, state, float(frame["t"]), 0.9,
                observed_at_s=1700000000.0 + float(frame["t"]),
            )
            near_falls.extend(new_near_falls)
            sit_to_stand.extend(state.get("new_sit_to_stand_events", []))
        expected_sts = scenario.get("expected_sit_to_stand")
        actual_sts = None
        if sit_to_stand:
            actual_sts = "success" if sit_to_stand[-1]["succeeded"] else "failure"
        passed = (
            len(near_falls) == int(scenario.get("expected_near_falls", 0))
            and actual_sts == expected_sts
        )
        results.append({
            "id": scenario["id"],
            "category": scenario["category"],
            "expected_near_falls": int(scenario.get("expected_near_falls", 0)),
            "actual_near_falls": len(near_falls),
            "expected_sit_to_stand": expected_sts,
            "actual_sit_to_stand": actual_sts,
            "passed": passed,
        })
    passed_count = sum(1 for item in results if item["passed"])
    return {
        "dataset_name": dataset.get("dataset_name"),
        "dataset_type": dataset.get("dataset_type"),
        "medical_accuracy_claim_allowed": False,
        "scenarios": len(results),
        "passed": passed_count,
        "failed": len(results) - passed_count,
        "engineering_acceptance_rate": round(passed_count / max(1, len(results)), 4),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="近跌倒与起身专项工程验收")
    parser.add_argument(
        "dataset", type=Path,
        nargs="?",
        default=PROJECT_ROOT / "test_videos" / "pre_fall_specialized_scenarios.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "data" / "exports" / "pre_fall_specialized_report.json",
    )
    args = parser.parse_args()
    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    report = validate(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"专项场景：{report['passed']}/{report['scenarios']} 通过")
    print(f"报告：{args.output}")
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
