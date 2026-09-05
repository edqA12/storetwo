from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime
from statistics import fmean, median, pstdev
from typing import Any

from .config import PreFallConfig
from .types import Decision, PoseFeatures, PreFallRiskResult


LOW = "LOW"
MEDIUM = "MEDIUM"
HIGH = "HIGH"

LEVEL_LABELS = {
    LOW: "低风险",
    MEDIUM: "中风险",
    HIGH: "高风险",
}


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _scaled(value: float, start: float, full: float) -> float:
    if full <= start:
        return 100.0 if value >= full else 0.0
    return _clamp((value - start) / (full - start) * 100.0)


def _initial_state(timestamp_s: float) -> dict[str, Any]:
    return {
        "history": [],
        "baseline_speed": None,
        "baseline_stability": None,
        "baseline_samples": 0,
        "near_falls": [],
        "near_fall_candidate": None,
        "sit_to_stand_phase": "UNKNOWN",
        "sit_to_stand_seated_since": None,
        "sit_to_stand_seated_samples": [],
        "sit_to_stand_rising_candidate_since": None,
        "sit_to_stand_transition": None,
        "sit_to_stand_observations": [],
        "sit_to_stand_baseline_s": None,
        "sit_to_stand_baseline_samples": 0,
        "sit_to_stand_last_duration_s": None,
        "sit_to_stand_last_status": "未检测",
        "sit_to_stand_count": 0,
        "previous_knee_angle_deg": None,
        "previous_sts_timestamp_s": None,
        "new_sit_to_stand_events": [],
        "new_sit_to_stand_attempts": [],
        "hourly_windows": {},
        "daily_windows": {},
        "last_window_sample_at": None,
        "profile_loaded": False,
        "profile_dirty": False,
        "smoothed_score": 0.0,
        "level": LOW,
        "pending_level": None,
        "pending_since": None,
        "last_level_change_ts": float(timestamp_s),
        "last_result": None,
    }


class PreFallRiskAnalyzer:
    """连续行为工程评分器；结果用于风险分级，不表示医学跌倒概率。"""

    def __init__(self, config: PreFallConfig) -> None:
        self.config = config

    def update(
        self,
        track_id: int,
        features: PoseFeatures,
        event_decision: Decision,
        state: dict[str, Any],
        timestamp_s: float,
        rgb_quality: float,
        depth_quality: float = 0.0,
        depth_available: bool = False,
        depth_features: dict[str, Any] | None = None,
        observed_at_s: float | None = None,
    ) -> tuple[PreFallRiskResult, list[dict[str, Any]]]:
        if not state:
            state.update(_initial_state(timestamp_s))
        state["new_sit_to_stand_events"] = []
        state["new_sit_to_stand_attempts"] = []
        wall_timestamp_s = float(observed_at_s if observed_at_s is not None else timestamp_s)

        if not features.valid or features.center_x_norm is None or features.center_y_norm is None:
            previous = state.get("last_result")
            if isinstance(previous, PreFallRiskResult):
                result = PreFallRiskResult(
                    **{
                        **previous.to_record(),
                        "timestamp_s": float(timestamp_s),
                        "rgb_quality": float(rgb_quality),
                        "depth_quality": float(depth_quality),
                        "depth_degraded": not bool(depth_available),
                        "level_changed": False,
                    }
                )
            else:
                result = PreFallRiskResult(
                    timestamp_s=float(timestamp_s),
                    track_id=int(track_id),
                    movement_score=0.0,
                    stability_score=0.0,
                    near_fall_count=0,
                    sit_to_stand_score=0.0,
                    baseline_deviation=0.0,
                    pre_fall_risk_score=0.0,
                    pre_fall_risk_level=LOW,
                    risk_factors=["关键点质量不足，前置风险暂不更新"],
                    rgb_quality=float(rgb_quality),
                    depth_quality=float(depth_quality),
                    depth_degraded=not bool(depth_available),
                )
            state["last_result"] = result
            return result, []

        history: list[dict[str, float]] = state.setdefault("history", [])
        history.append({
            "ts": float(timestamp_s),
            "center_x": float(features.center_x_norm),
            "center_y": float(features.center_y_norm),
            "speed": max(0.0, float(features.center_speed)),
            "motion": max(0.0, float(features.motion_norm)),
            "angle": float(features.torso_angle_deg or 0.0),
        })
        minimum_ts = float(timestamp_s) - float(self.config.short_window_s)
        history[:] = [sample for sample in history if sample["ts"] >= minimum_ts]

        stability_score, sway_score = self._stability_scores(history)
        depth_spatial_score = _clamp(float((depth_features or {}).get("pre_fall_spatial_score", 0.0)))
        depth_fusion_applied = bool(depth_available and depth_features)
        if depth_fusion_applied:
            depth_weight = min(1.0, max(0.0, float(self.config.pre_fall_depth_weight)))
            stability_score = _clamp(
                (1.0 - depth_weight) * stability_score + depth_weight * depth_spatial_score
            )
        sit_to_stand_phase_before = str(state.get("sit_to_stand_phase", "UNKNOWN"))
        sit_to_stand_events = self._advance_sit_to_stand(
            track_id, features, state, timestamp_s, stability_score, rgb_quality,
            depth_quality, depth_available,
        )
        state["new_sit_to_stand_events"] = sit_to_stand_events
        active_speeds = [
            sample["speed"] for sample in history
            if sample["motion"] >= self.config.min_active_motion_norm
            and sample["speed"] >= self.config.min_active_speed
        ]
        current_speed = fmean(active_speeds) if active_speeds else 0.0

        baseline_ready = int(state.get("baseline_samples", 0)) >= self.config.baseline_min_samples
        baseline_speed = state.get("baseline_speed")
        speed_drop_ratio = 0.0
        if baseline_ready and baseline_speed and active_speeds:
            speed_drop_ratio = max(0.0, (float(baseline_speed) - current_speed) / max(float(baseline_speed), 1e-6))
        movement_score = _scaled(
            speed_drop_ratio,
            self.config.movement_drop_start_ratio,
            self.config.movement_drop_full_ratio,
        )
        if depth_fusion_applied:
            movement_score = _clamp(
                (1.0 - depth_weight) * movement_score + depth_weight * depth_spatial_score
            )

        near_fall_events = self._advance_near_fall(
            track_id,
            features,
            event_decision,
            state,
            timestamp_s,
            stability_score,
            sway_score,
            rgb_quality,
            depth_quality,
            depth_available,
            sit_to_stand_phase_before,
        )
        if near_fall_events:
            state["new_sit_to_stand_events"] = []
            state["new_sit_to_stand_attempts"] = []
            state["sit_to_stand_phase"] = "STANDING"
            state["sit_to_stand_seated_since"] = None
            state["sit_to_stand_seated_samples"] = []
            state["sit_to_stand_rising_candidate_since"] = None
            state["sit_to_stand_transition"] = None
        near_falls: list[float] = state.setdefault("near_falls", [])
        near_falls[:] = [
            value for value in near_falls
            if float(timestamp_s) - float(value) <= self.config.near_fall_window_s
        ]
        near_fall_count = len(near_falls)
        near_fall_score = _clamp(
            near_fall_count / max(1, int(self.config.near_fall_count_full)) * 100.0
        )

        baseline_stability = state.get("baseline_stability")
        stability_deviation = 0.0
        if baseline_ready and baseline_stability is not None:
            stability_deviation = _clamp(
                max(0.0, stability_score - float(baseline_stability))
                / max(self.config.baseline_stability_full_delta, 1e-6)
                * 100.0
            )
        baseline_deviation = max(movement_score, stability_deviation)
        sit_to_stand_score = self._current_sit_to_stand_score(state, timestamp_s)

        weights = (
            self.config.movement_weight,
            self.config.stability_weight,
            self.config.near_fall_weight,
            self.config.sit_to_stand_weight,
            self.config.baseline_weight,
        )
        weight_sum = max(sum(weights), 1e-6)
        short_term_score = (
            movement_score * weights[0]
            + stability_score * weights[1]
            + near_fall_score * weights[2]
            + sit_to_stand_score * weights[3]
            + baseline_deviation * weights[4]
        ) / weight_sum
        self._update_time_windows(
            state,
            wall_timestamp_s,
            current_speed,
            stability_score,
            short_term_score,
            len(near_fall_events),
            sit_to_stand_events,
        )
        medium_term_score, long_term_score, persistent_near_falls = self._time_window_scores(
            state, wall_timestamp_s, baseline_speed, state.get("baseline_stability")
        )
        near_fall_count = max(near_fall_count, persistent_near_falls)
        window_weights = (
            self.config.short_term_weight,
            self.config.medium_term_weight,
            self.config.long_term_weight,
        )
        window_weight_sum = max(sum(window_weights), 1e-6)
        raw_score = (
            short_term_score * window_weights[0]
            + medium_term_score * window_weights[1]
            + long_term_score * window_weights[2]
        ) / window_weight_sum
        previous_score = float(state.get("smoothed_score", 0.0))
        alpha = min(1.0, max(0.0, float(self.config.score_smoothing_alpha)))
        smoothed_score = raw_score if not history[:-1] else alpha * raw_score + (1.0 - alpha) * previous_score
        smoothed_score = round(_clamp(smoothed_score), 1)
        state["smoothed_score"] = smoothed_score

        previous_level = str(state.get("level", LOW))
        level = self._advance_level(state, smoothed_score, timestamp_s)
        level_changed = level != previous_level
        factors = self._risk_factors(
            movement_score,
            speed_drop_ratio,
            stability_score,
            near_fall_count,
            baseline_deviation,
            baseline_ready,
            depth_available,
            sit_to_stand_score,
            state.get("sit_to_stand_last_duration_s"),
            str(state.get("sit_to_stand_last_status", "未检测")),
            depth_spatial_score,
            depth_fusion_applied,
            medium_term_score,
            long_term_score,
        )

        result = PreFallRiskResult(
            timestamp_s=float(timestamp_s),
            track_id=int(track_id),
            movement_score=round(movement_score, 1),
            stability_score=round(stability_score, 1),
            near_fall_count=near_fall_count,
            sit_to_stand_score=round(sit_to_stand_score, 1),
            baseline_deviation=round(baseline_deviation, 1),
            pre_fall_risk_score=smoothed_score,
            pre_fall_risk_level=level,
            risk_factors=factors,
            rgb_quality=float(rgb_quality),
            depth_quality=float(depth_quality),
            depth_degraded=not bool(depth_available),
            level_changed=level_changed,
            previous_level=previous_level if level_changed else None,
            movement_speed=round(current_speed, 4),
            baseline_speed=(round(float(baseline_speed), 4) if baseline_speed is not None else None),
            speed_change_pct=round(-speed_drop_ratio * 100.0, 1),
            sit_to_stand_duration_s=state.get("sit_to_stand_last_duration_s"),
            sit_to_stand_status=str(state.get("sit_to_stand_last_status", "未检测")),
            sit_to_stand_count=int(state.get("sit_to_stand_count", 0)),
            person_id=str(self.config.person_id),
            short_term_score=round(short_term_score, 1),
            medium_term_score=round(medium_term_score, 1),
            long_term_score=round(long_term_score, 1),
            depth_spatial_score=round(depth_spatial_score, 1),
            depth_fusion_applied=depth_fusion_applied,
        )
        state["last_result"] = result

        candidate_active = state.get("near_fall_candidate") is not None
        safe_for_baseline = (
            event_decision.state == "NORMAL"
            and not candidate_active
            and stability_score <= self.config.baseline_max_stability_score
        )
        if safe_for_baseline:
            self._update_baseline(state, current_speed, stability_score, bool(active_speeds))
            state["profile_dirty"] = True
        return result, near_fall_events

    def restore_persistent_profile(
        self,
        state: dict[str, Any],
        profile: dict[str, Any] | None,
        timestamp_s: float,
    ) -> None:
        if not state:
            state.update(_initial_state(timestamp_s))
        if profile:
            for key in (
                "baseline_speed", "baseline_stability", "baseline_samples",
                "sit_to_stand_baseline_s", "sit_to_stand_baseline_samples",
                "hourly_windows", "daily_windows",
            ):
                if key in profile:
                    state[key] = profile[key]
        state["profile_loaded"] = True
        state["profile_dirty"] = False

    @staticmethod
    def export_persistent_profile(state: dict[str, Any]) -> dict[str, Any]:
        return {
            key: state.get(key)
            for key in (
                "baseline_speed", "baseline_stability", "baseline_samples",
                "sit_to_stand_baseline_s", "sit_to_stand_baseline_samples",
                "hourly_windows", "daily_windows",
            )
        }

    def _update_time_windows(
        self,
        state: dict[str, Any],
        observed_at_s: float,
        movement_speed: float,
        stability_score: float,
        short_term_score: float,
        near_fall_delta: int,
        sit_to_stand_events: list[dict[str, Any]],
    ) -> None:
        hourly: dict[str, dict[str, Any]] = state.setdefault("hourly_windows", {})
        daily: dict[str, dict[str, Any]] = state.setdefault("daily_windows", {})
        moment = datetime.fromtimestamp(max(0.0, observed_at_s), UTC)
        hour_key = moment.strftime("%Y-%m-%dT%H:00Z")
        day_key = moment.strftime("%Y-%m-%d")
        hour_bucket = hourly.setdefault(hour_key, self._new_window_bucket(observed_at_s))
        day_bucket = daily.setdefault(day_key, self._new_window_bucket(observed_at_s))
        last_sample = state.get("last_window_sample_at")
        should_sample = (
            last_sample is None
            or observed_at_s - float(last_sample) >= self.config.window_sample_interval_s
        )
        for bucket in (hour_bucket, day_bucket):
            bucket["last_ts"] = observed_at_s
            bucket["near_fall_count"] += int(near_fall_delta)
            for event in sit_to_stand_events:
                bucket["sit_to_stand_score_sum"] += float(event.get("sit_to_stand_score", 0.0))
                bucket["sit_to_stand_count"] += 1
            if should_sample:
                bucket["samples"] += 1
                bucket["speed_sum"] += float(movement_speed)
                bucket["stability_sum"] += float(stability_score)
                bucket["risk_sum"] += float(short_term_score)
                bucket["max_risk"] = max(float(bucket["max_risk"]), float(short_term_score))
        if should_sample:
            state["last_window_sample_at"] = observed_at_s
        minimum_hour_ts = observed_at_s - self.config.medium_window_s
        hourly_keys = [key for key, value in hourly.items() if float(value.get("last_ts", 0.0)) < minimum_hour_ts]
        for key in hourly_keys:
            del hourly[key]
        minimum_day_ts = observed_at_s - max(1, self.config.long_window_days) * 86400.0
        daily_keys = [key for key, value in daily.items() if float(value.get("last_ts", 0.0)) < minimum_day_ts]
        for key in daily_keys:
            del daily[key]
        state["profile_dirty"] = True

    @staticmethod
    def _new_window_bucket(timestamp_s: float) -> dict[str, Any]:
        return {
            "first_ts": float(timestamp_s), "last_ts": float(timestamp_s),
            "samples": 0, "speed_sum": 0.0, "stability_sum": 0.0,
            "risk_sum": 0.0, "max_risk": 0.0, "near_fall_count": 0,
            "sit_to_stand_score_sum": 0.0, "sit_to_stand_count": 0,
        }

    def _time_window_scores(
        self,
        state: dict[str, Any],
        observed_at_s: float,
        baseline_speed: float | None,
        baseline_stability: float | None,
    ) -> tuple[float, float, int]:
        hourly = [
            value for value in state.get("hourly_windows", {}).values()
            if observed_at_s - float(value.get("last_ts", 0.0)) <= self.config.medium_window_s
        ]
        medium_score, near_falls = self._aggregate_window_score(
            hourly, baseline_speed, baseline_stability
        )
        daily = list(state.get("daily_windows", {}).values())
        daily_scores = [self._aggregate_window_score([bucket], baseline_speed, baseline_stability)[0] for bucket in daily]
        long_score = fmean(daily_scores) if daily_scores else medium_score
        if len(daily_scores) >= 2:
            long_score = max(long_score, daily_scores[-1] - daily_scores[0])
        return _clamp(medium_score), _clamp(long_score), near_falls

    def _aggregate_window_score(
        self,
        buckets: list[dict[str, Any]],
        baseline_speed: float | None,
        baseline_stability: float | None,
    ) -> tuple[float, int]:
        samples = sum(int(bucket.get("samples", 0)) for bucket in buckets)
        near_falls = sum(int(bucket.get("near_fall_count", 0)) for bucket in buckets)
        if not buckets:
            return 0.0, 0
        avg_speed = sum(float(bucket.get("speed_sum", 0.0)) for bucket in buckets) / max(1, samples)
        avg_stability = sum(float(bucket.get("stability_sum", 0.0)) for bucket in buckets) / max(1, samples)
        avg_risk = sum(float(bucket.get("risk_sum", 0.0)) for bucket in buckets) / max(1, samples)
        speed_score = 0.0
        if baseline_speed and avg_speed > 0.0:
            speed_score = _scaled(
                max(0.0, (float(baseline_speed) - avg_speed) / max(float(baseline_speed), 1e-6)),
                self.config.movement_drop_start_ratio,
                self.config.movement_drop_full_ratio,
            )
        stability_deviation = 0.0
        if baseline_stability is not None:
            stability_deviation = _clamp(
                max(0.0, avg_stability - float(baseline_stability))
                / max(self.config.baseline_stability_full_delta, 1e-6) * 100.0
            )
        near_fall_score = _clamp(
            near_falls / max(1, self.config.near_fall_count_full) * 100.0
        )
        sts_count = sum(int(bucket.get("sit_to_stand_count", 0)) for bucket in buckets)
        sts_score = sum(float(bucket.get("sit_to_stand_score_sum", 0.0)) for bucket in buckets) / max(1, sts_count)
        score = (
            speed_score * self.config.movement_weight
            + stability_deviation * self.config.stability_weight
            + near_fall_score * self.config.near_fall_weight
            + sts_score * self.config.sit_to_stand_weight
            + avg_risk * self.config.baseline_weight
        ) / max(
            self.config.movement_weight + self.config.stability_weight
            + self.config.near_fall_weight + self.config.sit_to_stand_weight
            + self.config.baseline_weight,
            1e-6,
        )
        return _clamp(score), near_falls

    def _advance_sit_to_stand(
        self,
        track_id: int,
        features: PoseFeatures,
        state: dict[str, Any],
        timestamp_s: float,
        stability_score: float,
        rgb_quality: float,
        depth_quality: float,
        depth_available: bool,
    ) -> list[dict[str, Any]]:
        knee_angle = features.knee_angle_deg
        previous_knee = state.get("previous_knee_angle_deg")
        previous_timestamp = state.get("previous_sts_timestamp_s")
        state["previous_knee_angle_deg"] = knee_angle
        state["previous_sts_timestamp_s"] = float(timestamp_s)
        if knee_angle is None:
            return []

        if (
            float(features.box_area_ratio) < self.config.sit_to_stand_min_box_area_ratio
            or float(rgb_quality) < self.config.sit_to_stand_min_pose_quality
        ):
            state["sit_to_stand_rising_candidate_since"] = None
            return []

        torso_angle = float(features.torso_angle_deg or 0.0)
        seated = (
            knee_angle <= self.config.sit_to_stand_seated_knee_max_deg
            and features.lying_score < self.config.near_fall_max_lying_score
        )
        standing = (
            knee_angle >= self.config.sit_to_stand_standing_knee_min_deg
            and torso_angle <= self.config.sit_to_stand_max_torso_angle_deg
            and features.lying_score < self.config.near_fall_max_lying_score
        )
        phase = str(state.get("sit_to_stand_phase", "UNKNOWN"))

        def remember_seated_sample() -> None:
            samples: list[dict[str, float]] = state.setdefault(
                "sit_to_stand_seated_samples", []
            )
            samples.append({
                "timestamp_s": float(timestamp_s),
                "knee_angle_deg": float(knee_angle),
                "center_y_norm": float(features.center_y_norm or 0.0),
                "torso_angle_deg": torso_angle,
            })
            minimum_ts = float(timestamp_s) - max(
                2.0, self.config.sit_to_stand_min_sit_hold_s * 3.0
            )
            samples[:] = [
                item for item in samples if item["timestamp_s"] >= minimum_ts
            ]

        def seated_reference() -> tuple[float, float, float]:
            samples: list[dict[str, float]] = state.setdefault(
                "sit_to_stand_seated_samples", []
            )
            if not samples:
                return float(knee_angle), float(features.center_y_norm or 0.0), torso_angle
            return (
                float(median(item["knee_angle_deg"] for item in samples)),
                float(median(item["center_y_norm"] for item in samples)),
                float(median(item["torso_angle_deg"] for item in samples)),
            )

        if phase == "UNKNOWN":
            if seated:
                state["sit_to_stand_phase"] = "SEATED"
                state["sit_to_stand_seated_since"] = float(timestamp_s)
                state["sit_to_stand_seated_samples"] = []
                remember_seated_sample()
            elif standing:
                state["sit_to_stand_phase"] = "STANDING"
            return []

        if phase == "STANDING":
            if seated:
                state["sit_to_stand_phase"] = "SEATED"
                state["sit_to_stand_seated_since"] = float(timestamp_s)
                state["sit_to_stand_seated_samples"] = []
                remember_seated_sample()
            return []

        if phase == "SEATED":
            if seated:
                remember_seated_sample()
                state["sit_to_stand_rising_candidate_since"] = None
                return []
            seated_since = float(state.get("sit_to_stand_seated_since", timestamp_s))
            held_long_enough = timestamp_s - seated_since >= self.config.sit_to_stand_min_sit_hold_s
            reference_knee, reference_center_y, reference_torso = seated_reference()
            knee_change = float(knee_angle) - reference_knee
            center_rise = reference_center_y - float(features.center_y_norm or reference_center_y)
            torso_change = torso_angle - reference_torso
            kinematic_rise = (
                center_rise >= self.config.sit_to_stand_min_center_rise_norm
                and (
                    knee_change >= self.config.sit_to_stand_min_cumulative_knee_change_deg
                    or torso_change >= self.config.sit_to_stand_min_torso_change_deg
                )
            )
            standing_fallback = (
                standing
                and center_rise >= -self.config.sit_to_stand_max_center_drop_norm
            )
            rising_motion = bool(kinematic_rise or standing_fallback)
            if not held_long_enough or not rising_motion:
                state["sit_to_stand_rising_candidate_since"] = None
                return []
            candidate_since = state.get("sit_to_stand_rising_candidate_since")
            if candidate_since is None:
                candidate_since = float(previous_timestamp) if previous_timestamp is not None else float(timestamp_s)
                state["sit_to_stand_rising_candidate_since"] = candidate_since
            if (
                float(timestamp_s) - float(candidate_since)
                < self.config.sit_to_stand_attempt_hold_s
            ):
                return []
            started_at = float(candidate_since)
            state["sit_to_stand_phase"] = "RISING"
            state["sit_to_stand_rising_candidate_since"] = None
            state["sit_to_stand_transition"] = {
                "started_at": started_at,
                "standing_since": float(timestamp_s) if standing else None,
                "max_stability_score": float(stability_score),
                "start_knee_angle": reference_knee,
                "max_knee_angle": float(knee_angle),
                "reference_center_y": reference_center_y,
                "reference_torso_angle": reference_torso,
            }
            state["new_sit_to_stand_attempts"] = [{
                "event_key": uuid.uuid4().hex,
                "start_time_s": round(started_at, 3),
                "timestamp_s": round(float(timestamp_s), 3),
                "track_id": int(track_id),
                "status": "起身动作开始",
                "knee_extension_deg": round(knee_change, 2),
                "center_rise_norm": round(center_rise, 4),
                "torso_change_deg": round(torso_change, 2),
                "box_area_ratio": round(float(features.box_area_ratio), 4),
                "rgb_quality": round(float(rgb_quality), 3),
                "depth_quality": round(float(depth_quality), 3),
                "depth_degraded": not bool(depth_available),
            }]
            return []

        transition = state.get("sit_to_stand_transition")
        if phase != "RISING" or not isinstance(transition, dict):
            state["sit_to_stand_phase"] = "UNKNOWN"
            state["sit_to_stand_transition"] = None
            return []
        transition["max_stability_score"] = max(
            float(transition["max_stability_score"]), float(stability_score)
        )
        transition["max_knee_angle"] = max(float(transition["max_knee_angle"]), float(knee_angle))
        duration = float(timestamp_s) - float(transition["started_at"])
        reference_center_y = float(
            transition.get("reference_center_y", features.center_y_norm or 0.0)
        )
        reference_torso = float(transition.get("reference_torso_angle", torso_angle))
        center_rise = reference_center_y - float(features.center_y_norm or reference_center_y)
        relative_standing = (
            center_rise >= self.config.sit_to_stand_relative_standing_rise_norm
            and torso_angle <= self.config.sit_to_stand_max_torso_angle_deg
            and knee_angle >= self.config.sit_to_stand_relative_standing_knee_min_deg
            and features.lying_score < self.config.near_fall_max_lying_score
        )
        standing = bool(standing or relative_standing)

        if standing:
            if transition.get("standing_since") is None:
                transition["standing_since"] = float(timestamp_s)
            standing_hold = float(timestamp_s) - float(transition["standing_since"])
            if (
                duration >= self.config.sit_to_stand_min_transition_s
                and standing_hold >= self.config.sit_to_stand_standing_hold_s
            ):
                event = self._finish_sit_to_stand(
                    track_id, state, timestamp_s, transition, True, rgb_quality,
                    depth_quality, depth_available,
                )
                state["sit_to_stand_phase"] = "STANDING"
                state["sit_to_stand_transition"] = None
                return [event]
        else:
            transition["standing_since"] = None

        returned_to_seat = (
            seated
            and abs(center_rise) <= self.config.sit_to_stand_return_center_tolerance_norm
            and abs(float(knee_angle) - float(transition["start_knee_angle"]))
            <= self.config.sit_to_stand_return_knee_tolerance_deg
            and abs(torso_angle - reference_torso)
            <= self.config.sit_to_stand_return_torso_tolerance_deg
        )
        failed = returned_to_seat and duration >= self.config.sit_to_stand_min_transition_s
        timed_out = duration >= self.config.sit_to_stand_max_transition_s
        if failed or timed_out:
            event = self._finish_sit_to_stand(
                track_id, state, timestamp_s, transition, False, rgb_quality,
                depth_quality, depth_available,
            )
            state["sit_to_stand_phase"] = "SEATED" if seated else "UNKNOWN"
            state["sit_to_stand_seated_since"] = float(timestamp_s) if seated else None
            state["sit_to_stand_transition"] = None
            return [event]
        return []

    def _finish_sit_to_stand(
        self,
        track_id: int,
        state: dict[str, Any],
        timestamp_s: float,
        transition: dict[str, Any],
        succeeded: bool,
        rgb_quality: float,
        depth_quality: float,
        depth_available: bool,
    ) -> dict[str, Any]:
        duration = max(0.0, float(timestamp_s) - float(transition["started_at"]))
        baseline = state.get("sit_to_stand_baseline_s")
        baseline_samples = int(state.get("sit_to_stand_baseline_samples", 0))
        fixed_duration_score = _scaled(
            duration, self.config.sit_to_stand_slow_start_s, self.config.sit_to_stand_slow_full_s
        )
        relative_duration_score = 0.0
        if baseline is not None and baseline_samples >= self.config.sit_to_stand_baseline_min_events:
            slowdown = max(0.0, (duration - float(baseline)) / max(float(baseline), 1e-6))
            relative_duration_score = _scaled(
                slowdown,
                self.config.sit_to_stand_baseline_slow_start_ratio,
                self.config.sit_to_stand_baseline_slow_full_ratio,
            )
        instability_score = _clamp(
            float(transition["max_stability_score"])
            / max(self.config.sit_to_stand_instability_full_score, 1e-6) * 100.0
        )
        score = max(fixed_duration_score, relative_duration_score, instability_score)
        status = "成功"
        if not succeeded:
            score = max(score, self.config.sit_to_stand_failure_score)
            status = "失败或重新坐下"
        observation = {
            "timestamp_s": float(timestamp_s),
            "score": round(_clamp(score), 1),
            "succeeded": bool(succeeded),
        }
        state.setdefault("sit_to_stand_observations", []).append(observation)
        state["sit_to_stand_last_duration_s"] = round(duration, 3)
        state["sit_to_stand_last_status"] = status
        state["sit_to_stand_count"] = int(state.get("sit_to_stand_count", 0)) + 1

        if succeeded and score < self.config.risk_factor_min_score:
            alpha = min(1.0, max(0.0, self.config.sit_to_stand_baseline_alpha))
            if baseline is None:
                state["sit_to_stand_baseline_s"] = duration
            elif baseline_samples < self.config.sit_to_stand_baseline_min_events:
                state["sit_to_stand_baseline_s"] = (
                    float(baseline) * baseline_samples + duration
                ) / (baseline_samples + 1)
            else:
                state["sit_to_stand_baseline_s"] = (1.0 - alpha) * float(baseline) + alpha * duration
            state["sit_to_stand_baseline_samples"] = baseline_samples + 1

        return {
            "event_key": uuid.uuid4().hex,
            "start_time_s": round(float(transition["started_at"]), 3),
            "timestamp_s": round(float(timestamp_s), 3),
            "track_id": int(track_id),
            "status": status,
            "succeeded": bool(succeeded),
            "duration_s": round(duration, 3),
            "max_stability_score": round(float(transition["max_stability_score"]), 1),
            "start_knee_angle_deg": round(float(transition["start_knee_angle"]), 1),
            "max_knee_angle_deg": round(float(transition["max_knee_angle"]), 1),
            "sit_to_stand_score": round(_clamp(score), 1),
            "rgb_quality": round(float(rgb_quality), 3),
            "depth_quality": round(float(depth_quality), 3),
            "depth_degraded": not bool(depth_available),
        }

    def _current_sit_to_stand_score(self, state: dict[str, Any], timestamp_s: float) -> float:
        observations: list[dict[str, Any]] = state.setdefault("sit_to_stand_observations", [])
        observations[:] = [
            item for item in observations
            if float(timestamp_s) - float(item["timestamp_s"]) <= self.config.sit_to_stand_repeat_window_s
        ]
        if not observations:
            return 0.0
        repeat_score = _clamp(
            len(observations) / max(1, self.config.sit_to_stand_repeat_count_full) * 100.0
        )
        return _clamp(max(float(item["score"]) for item in observations) * 0.8 + repeat_score * 0.2)

    def _stability_scores(self, history: list[dict[str, float]]) -> tuple[float, float]:
        if len(history) < 3:
            return 0.0, 0.0
        x_steps = [history[index]["center_x"] - history[index - 1]["center_x"] for index in range(1, len(history))]
        sway = pstdev(x_steps) if len(x_steps) >= 2 else 0.0
        angles = [sample["angle"] for sample in history]
        torso_variability = pstdev(angles) if len(angles) >= 2 else 0.0
        sway_score = _scaled(sway, self.config.sway_start_norm, self.config.sway_full_norm)
        torso_score = _scaled(
            torso_variability,
            self.config.torso_variability_start_deg,
            self.config.torso_variability_full_deg,
        )
        return _clamp(0.55 * sway_score + 0.45 * torso_score), sway_score

    def _advance_near_fall(
        self,
        track_id: int,
        features: PoseFeatures,
        event_decision: Decision,
        state: dict[str, Any],
        timestamp_s: float,
        stability_score: float,
        sway_score: float,
        rgb_quality: float,
        depth_quality: float,
        depth_available: bool,
        sit_to_stand_phase_before: str,
    ) -> list[dict[str, Any]]:
        candidate = state.get("near_fall_candidate")
        knee_angle = features.knee_angle_deg
        state["near_fall_previous_knee_angle_deg"] = knee_angle
        if (
            knee_angle is not None
            and (
                sit_to_stand_phase_before == "STANDING"
                or str(state.get("sit_to_stand_phase")) == "STANDING"
            )
            and float(knee_angle) >= self.config.near_fall_knee_recovery_min_deg
        ):
            state["near_fall_upright_knee_angle_deg"] = float(knee_angle)
        upright_knee = state.get("near_fall_upright_knee_angle_deg")
        dynamic_trigger = (
            features.down_velocity >= self.config.near_fall_min_down_velocity
            or features.angle_velocity >= self.config.near_fall_min_angle_velocity
        )
        knee_collapse_trigger = bool(
            knee_angle is not None
            and upright_knee is not None
            and sit_to_stand_phase_before == "STANDING"
            and float(knee_angle) <= self.config.near_fall_knee_collapse_max_deg
            and float(upright_knee) >= self.config.near_fall_knee_recovery_min_deg
            and float(upright_knee) - float(knee_angle)
            >= self.config.near_fall_min_knee_drop_deg
            and float(features.box_area_ratio) >= self.config.sit_to_stand_min_box_area_ratio
            and float(rgb_quality) >= self.config.sit_to_stand_min_pose_quality
        )
        if candidate is None and (
            (dynamic_trigger and event_decision.state == "DESCENDING")
            or knee_collapse_trigger
        ):
            candidate = {
                "started_at": float(timestamp_s),
                "trigger_kind": "knee_collapse" if knee_collapse_trigger else "dynamic_descent",
                "max_angle": float(features.torso_angle_deg or 0.0),
                "max_down_velocity": float(features.down_velocity),
                "max_angle_velocity": float(features.angle_velocity),
                "start_knee_angle": float(upright_knee) if upright_knee is not None else None,
                "min_knee_angle": float(knee_angle) if knee_angle is not None else None,
                "max_center_speed": float(features.center_speed),
                "max_abs_lateral_velocity": abs(float(features.lateral_velocity)),
                "lowest_center_y": float(features.center_y_norm or 0.0),
                "max_lying_score": float(features.lying_score),
                "max_stability_score": float(stability_score),
                "max_sway_score": float(sway_score),
                "rgb_quality": float(rgb_quality),
                "depth_quality": float(depth_quality),
                "depth_degraded": not bool(depth_available),
                "developed_into_fall": False,
            }
            state["near_fall_candidate"] = candidate

        if not isinstance(candidate, dict):
            return []

        candidate["max_angle"] = max(float(candidate["max_angle"]), float(features.torso_angle_deg or 0.0))
        candidate["max_down_velocity"] = max(float(candidate["max_down_velocity"]), float(features.down_velocity))
        candidate["max_angle_velocity"] = max(float(candidate["max_angle_velocity"]), float(features.angle_velocity))
        if knee_angle is not None:
            minimum_knee = candidate.get("min_knee_angle")
            candidate["min_knee_angle"] = (
                float(knee_angle)
                if minimum_knee is None
                else min(float(minimum_knee), float(knee_angle))
            )
        candidate["max_center_speed"] = max(
            float(candidate.get("max_center_speed", 0.0)), float(features.center_speed)
        )
        candidate["max_abs_lateral_velocity"] = max(
            float(candidate.get("max_abs_lateral_velocity", 0.0)),
            abs(float(features.lateral_velocity)),
        )
        candidate["lowest_center_y"] = max(float(candidate["lowest_center_y"]), float(features.center_y_norm or 0.0))
        candidate["max_lying_score"] = max(float(candidate["max_lying_score"]), float(features.lying_score))
        candidate["max_stability_score"] = max(float(candidate["max_stability_score"]), float(stability_score))
        candidate["max_sway_score"] = max(float(candidate["max_sway_score"]), float(sway_score))
        candidate["rgb_quality"] = max(float(candidate["rgb_quality"]), float(rgb_quality))
        candidate["depth_quality"] = max(float(candidate["depth_quality"]), float(depth_quality))
        candidate["depth_degraded"] = bool(candidate["depth_degraded"] and not depth_available)
        if event_decision.state in {"SUSPECT", "FALLEN", "LYING"}:
            candidate["developed_into_fall"] = True

        duration = float(timestamp_s) - float(candidate["started_at"])
        knee_collapse_candidate = candidate.get("trigger_kind") == "knee_collapse"
        knee_recovered = bool(
            knee_angle is not None
            and float(knee_angle) >= self.config.near_fall_knee_recovery_min_deg
        )
        collapse_motion_confirmed = (
            float(candidate.get("max_center_speed", 0.0))
            >= self.config.near_fall_knee_collapse_min_speed
        )
        recovered = (
            event_decision.state == "NORMAL"
            and duration >= self.config.near_fall_recovery_min_s
            and (
                not knee_collapse_candidate
                or (knee_recovered and collapse_motion_confirmed)
            )
        )
        expired = duration > self.config.near_fall_recovery_max_s
        if not recovered and not expired:
            return []

        state["near_fall_candidate"] = None
        valid_dynamic_near_fall = (
            recovered
            and not bool(candidate["developed_into_fall"])
            and float(candidate["max_angle"]) >= self.config.near_fall_min_torso_angle
            and float(candidate["max_lying_score"]) < self.config.near_fall_max_lying_score
            and float(candidate["max_stability_score"]) >= self.config.near_fall_min_stability_score
            and (
                float(candidate["max_sway_score"]) >= self.config.near_fall_min_sway_score
                or (
                    float(candidate["max_down_velocity"]) >= self.config.near_fall_min_down_velocity
                    and float(candidate["max_angle_velocity"]) >= self.config.near_fall_min_angle_velocity
                )
            )
        )
        start_knee = candidate.get("start_knee_angle")
        minimum_knee = candidate.get("min_knee_angle")
        valid_knee_collapse = bool(
            recovered
            and knee_collapse_candidate
            and not bool(candidate["developed_into_fall"])
            and start_knee is not None
            and minimum_knee is not None
            and float(minimum_knee) <= self.config.near_fall_knee_collapse_max_deg
            and float(start_knee) - float(minimum_knee)
            >= self.config.near_fall_min_knee_drop_deg
            and collapse_motion_confirmed
            and float(candidate["max_angle"])
            <= self.config.near_fall_knee_collapse_max_torso_deg
            and float(candidate["max_lying_score"]) < self.config.near_fall_max_lying_score
        )
        valid_near_fall = valid_dynamic_near_fall or valid_knee_collapse
        if not valid_near_fall:
            return []

        state.setdefault("near_falls", []).append(float(timestamp_s))
        return [{
            "event_key": uuid.uuid4().hex,
            "start_time_s": round(float(candidate["started_at"]), 3),
            "timestamp_s": round(float(timestamp_s), 3),
            "track_id": int(track_id),
            "max_torso_angle": round(float(candidate["max_angle"]), 2),
            "max_down_velocity": round(float(candidate["max_down_velocity"]), 4),
            "trigger_kind": str(candidate.get("trigger_kind", "dynamic_descent")),
            "start_knee_angle_deg": (
                round(float(start_knee), 2) if start_knee is not None else None
            ),
            "min_knee_angle_deg": (
                round(float(minimum_knee), 2) if minimum_knee is not None else None
            ),
            "max_center_speed": round(float(candidate.get("max_center_speed", 0.0)), 4),
            "lowest_center_y": round(float(candidate["lowest_center_y"]), 4),
            "recovery_time_s": round(duration, 3),
            "rgb_quality": round(float(candidate["rgb_quality"]), 3),
            "depth_quality": round(float(candidate["depth_quality"]), 3),
            "depth_degraded": bool(candidate["depth_degraded"]),
            "developed_into_fall": False,
        }]

    def _update_baseline(
        self,
        state: dict[str, Any],
        current_speed: float,
        stability_score: float,
        active_motion: bool,
    ) -> None:
        count = int(state.get("baseline_samples", 0))
        alpha = min(1.0, max(0.0, float(self.config.baseline_alpha)))
        if active_motion and current_speed > 0.0:
            previous_speed = state.get("baseline_speed")
            if previous_speed is None:
                state["baseline_speed"] = float(current_speed)
            elif count < self.config.baseline_min_samples:
                state["baseline_speed"] = (float(previous_speed) * count + current_speed) / (count + 1)
            else:
                state["baseline_speed"] = (1.0 - alpha) * float(previous_speed) + alpha * current_speed
        previous_stability = state.get("baseline_stability")
        if previous_stability is None:
            state["baseline_stability"] = float(stability_score)
        elif count < self.config.baseline_min_samples:
            state["baseline_stability"] = (float(previous_stability) * count + stability_score) / (count + 1)
        else:
            state["baseline_stability"] = (1.0 - alpha) * float(previous_stability) + alpha * stability_score
        state["baseline_samples"] = count + 1

    def _desired_level(self, current: str, score: float) -> str:
        if current == LOW:
            if score >= self.config.high_threshold:
                return HIGH
            if score >= self.config.medium_threshold:
                return MEDIUM
            return LOW
        if current == MEDIUM:
            if score >= self.config.high_threshold:
                return HIGH
            if score <= self.config.medium_exit_threshold:
                return LOW
            return MEDIUM
        if score <= self.config.medium_exit_threshold:
            return LOW
        if score <= self.config.high_exit_threshold:
            return MEDIUM
        return HIGH

    def _advance_level(self, state: dict[str, Any], score: float, timestamp_s: float) -> str:
        current = str(state.get("level", LOW))
        desired = self._desired_level(current, score)
        if desired == current:
            state["pending_level"] = None
            state["pending_since"] = None
            return current
        if state.get("pending_level") != desired:
            state["pending_level"] = desired
            state["pending_since"] = float(timestamp_s)
            return current
        pending_since = float(state.get("pending_since", timestamp_s))
        if desired == HIGH:
            required = self.config.high_upgrade_hold_s
        elif current == LOW and desired == MEDIUM:
            required = self.config.upgrade_hold_s
        else:
            required = self.config.downgrade_hold_s
        if float(timestamp_s) - pending_since < required:
            return current
        state["level"] = desired
        state["last_level_change_ts"] = float(timestamp_s)
        state["pending_level"] = None
        state["pending_since"] = None
        return desired

    def _risk_factors(
        self,
        movement_score: float,
        speed_drop_ratio: float,
        stability_score: float,
        near_fall_count: int,
        baseline_deviation: float,
        baseline_ready: bool,
        depth_available: bool,
        sit_to_stand_score: float,
        sit_to_stand_duration_s: float | None,
        sit_to_stand_status: str,
        depth_spatial_score: float,
        depth_fusion_applied: bool,
        medium_term_score: float,
        long_term_score: float,
    ) -> list[str]:
        factors: list[str] = []
        threshold = self.config.risk_factor_min_score
        if movement_score >= threshold:
            factors.append(f"运动速度较个人基线下降 {speed_drop_ratio * 100.0:.0f}%")
        if stability_score >= threshold:
            factors.append(f"身体摆动与躯干变化异常（{stability_score:.0f}/100）")
        if near_fall_count:
            factors.append(f"当前统计窗口检测到 {near_fall_count} 次近跌倒")
        if baseline_deviation >= threshold:
            factors.append(f"当前行为相对个人基线偏离 {baseline_deviation:.0f}/100")
        if sit_to_stand_score >= threshold:
            duration_text = (
                f"，耗时 {float(sit_to_stand_duration_s):.1f} 秒"
                if sit_to_stand_duration_s is not None else ""
            )
            factors.append(f"起身能力异常：{sit_to_stand_status}{duration_text}")
        if depth_fusion_applied and depth_spatial_score >= threshold:
            factors.append(f"深度空间运动异常（{depth_spatial_score:.0f}/100）")
        if medium_term_score >= threshold:
            factors.append(f"当天/24小时风险持续偏高（{medium_term_score:.0f}/100）")
        if long_term_score >= threshold:
            factors.append(f"多天风险趋势偏高（{long_term_score:.0f}/100）")
        if not baseline_ready:
            factors.append("正在建立监测对象的持久化个人行为基线")
        if not depth_available:
            factors.append("深度信息不可用，前置风险按 RGB 模式继续评估")
        if not factors:
            factors.append("连续行为未见明显异常")
        return factors
