"""从同一批带真值的预测清单自动生成四组消融指标。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.evaluation import ABLATION_VARIANTS, evaluate_ablation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="评估 RGB Only、Depth Only、RGB+Depth、RGB+Depth+Baseline 四组结果"
    )
    parser.add_argument("manifest", type=Path, help="带真值和四组预测列的CSV")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "exports" / "ablation_report.json")
    args = parser.parse_args()
    if not args.manifest.exists():
        raise SystemExit(f"未找到评测清单：{args.manifest}")
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    report = evaluate_ablation(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"消融报告已生成：{args.output}")
    for variant in ABLATION_VARIANTS:
        result = report["variants"][variant]
        if result["status"] != "OK":
            print(f"- {variant}: 数据不足，未计算")
            continue
        metrics = result["overall_alert"]
        print(
            f"- {variant}: P={metrics['precision']} R={metrics['recall']} "
            f"F1={metrics['f1']} 误报/小时={metrics['false_alarms_per_hour']}"
        )
    for warning in report["warnings"]:
        print(f"注意：{warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
