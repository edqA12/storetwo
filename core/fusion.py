from __future__ import annotations

from .config import MultimodalConfig
from .types import FusedDecision, ModalityResult


FUSION_LABELS = {
    "NORMAL": "多模态判断正常",
    "DESCENDING": "多模态检测到快速变化",
    "LYING": "多模态检测到异常躺卧",
    "SUSPECT": "多模态疑似跌倒",
    "FALLEN": "多模态确认跌倒",
    "UNKNOWN": "多模态证据不足",
}


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


class FusionEngine:
    """以质量加权和可解释规则融合视觉与深度证据。"""

    def __init__(self, config: MultimodalConfig) -> None:
        self.config = config

    def fuse(
        self,
        visual: ModalityResult,
        depth: ModalityResult,
        timestamp_s: float,
    ) -> FusedDecision:
        depth_available = depth.available and depth.quality >= self.config.min_depth_quality
        visual_available = visual.available and visual.quality > 0.0
        if not depth_available:
            return FusedDecision(
                timestamp_s=float(timestamp_s),
                risk=visual.risk,
                quality=visual.quality,
                state=visual.state,
                label=visual.label,
                method="visual_fallback",
                modality_weights={"vision": 1.0, "depth": 0.0},
                explanation=["深度画面质量不足，已自动退化为视觉判断"],
            )

        visual_weight = self.config.visual_weight * max(visual.quality, 0.05)
        depth_weight = self.config.depth_weight * depth.quality
        total_weight = max(visual_weight + depth_weight, 1e-6)
        weights = {
            "vision": round(visual_weight / total_weight, 3),
            "depth": round(depth_weight / total_weight, 3),
        }
        weighted_risk = (
            visual.risk * visual_weight + depth.risk * depth_weight
        ) / total_weight
        quality = min(1.0, (visual.quality + depth.quality) / 2.0)
        explanation = [
            f"视觉权重 {weights['vision']:.2f}，深度权重 {weights['depth']:.2f}"
        ]

        if not visual_available:
            if depth.risk >= 70.0 and depth.quality >= 0.45:
                state, risk = "SUSPECT", max(75.0, weighted_risk)
                explanation.append("视觉不可用，深度强变化触发保守疑似告警")
            else:
                state, risk = "UNKNOWN", min(59.0, depth.risk)
                explanation.append("视觉不可用，深度证据不足以单独确认跌倒")
        elif visual.state == "FALLEN":
            if depth.risk >= self.config.depth_support_threshold:
                state, risk = "FALLEN", max(85.0, visual.risk, weighted_risk)
                explanation.append("视觉跌倒证据得到深度变化支持")
            elif depth.risk <= self.config.depth_contradiction_threshold:
                state, risk = "SUSPECT", 79.0
                explanation.append("视觉与深度证据不一致，降级为疑似并等待人工确认")
            else:
                state, risk = "FALLEN", max(85.0, weighted_risk)
                explanation.append("视觉证据明确，深度证据中性")
        elif visual.state in {"SUSPECT", "LYING"}:
            if depth.risk >= self.config.depth_support_threshold:
                state, risk = "SUSPECT", max(75.0, weighted_risk)
                explanation.append("异常姿态得到深度时序证据支持")
            elif depth.risk <= self.config.depth_contradiction_threshold:
                state, risk = visual.state, min(74.0, weighted_risk)
                explanation.append("深度画面未发现同步变化，降低告警等级")
            else:
                state, risk = visual.state, weighted_risk
                explanation.append("两种模态证据暂时中性")
        elif visual.state == "DESCENDING" and depth.risk >= self.config.depth_support_threshold:
            state, risk = "SUSPECT", max(75.0, weighted_risk)
            explanation.append("视觉下降与深度强变化同时出现")
        elif depth.risk >= 75.0 and depth.quality >= 0.55:
            state, risk = "DESCENDING", min(69.0, max(55.0, weighted_risk))
            explanation.append("仅深度出现强变化，保持观察并等待视觉证据")
        else:
            state, risk = visual.state, weighted_risk
            explanation.append("融合结果以视觉状态为主，深度用于校正风险")

        state = state if state in FUSION_LABELS else "UNKNOWN"
        return FusedDecision(
            timestamp_s=float(timestamp_s),
            risk=round(_clamp(risk), 1),
            quality=round(max(0.0, min(1.0, quality)), 3),
            state=state,
            label=FUSION_LABELS[state],
            method="quality_weighted_rules",
            modality_weights=weights,
            explanation=explanation,
        )
