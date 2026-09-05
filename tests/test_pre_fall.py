from __future__ import annotations

from core.config import PreFallConfig
from core.pre_fall import LOW, MEDIUM, PreFallRiskAnalyzer
from core.types import Decision, PoseFeatures


def feature(
    *,
    x: float = 0.50,
    y: float = 0.45,
    angle: float = 5.0,
    speed: float = 0.04,
    lateral: float = 0.0,
    motion: float = 0.02,
    down: float = 0.0,
    rotation: float = 0.0,
    lying: float = 0.05,
    knee: float | None = None,
    area: float = 0.12,
) -> PoseFeatures:
    return PoseFeatures(
        valid=True,
        torso_angle_deg=angle,
        box_aspect=0.45,
        hip_y_norm=0.55,
        center_x_norm=x,
        center_y_norm=y,
        center_speed=speed,
        lateral_velocity=lateral,
        down_velocity=down,
        angle_velocity=rotation,
        motion_norm=motion,
        lying_score=lying,
        knee_angle_deg=knee,
        box_area_ratio=area,
        quality=0.9,
    )


def decision(state: str = "NORMAL") -> Decision:
    return Decision(track_id=1, state=state, label=state, risk=10.0)


def test_stable_activity_builds_baseline_and_stays_low():
    config = PreFallConfig(baseline_min_samples=3, score_smoothing_alpha=1.0)
    analyzer = PreFallRiskAnalyzer(config)
    state: dict = {}
    result = None
    for index in range(8):
        result, events = analyzer.update(
            1,
            feature(x=0.45 + index * 0.004),
            decision(),
            state,
            index * 0.2,
            rgb_quality=0.9,
        )
        assert events == []
    assert result is not None
    assert state["baseline_samples"] >= 3
    assert result.pre_fall_risk_level == LOW
    assert result.pre_fall_risk_score < config.medium_threshold


def test_imbalance_then_recovery_records_near_fall_not_confirmed_fall():
    config = PreFallConfig(
        baseline_min_samples=1,
        near_fall_min_stability_score=0.0,
        near_fall_min_torso_angle=20.0,
        score_smoothing_alpha=1.0,
    )
    analyzer = PreFallRiskAnalyzer(config)
    state: dict = {}
    analyzer.update(1, feature(), decision(), state, 0.0, 0.9)
    analyzer.update(
        1,
        feature(x=0.56, angle=22.0, down=0.22, rotation=45.0),
        decision("DESCENDING"),
        state,
        0.3,
        0.9,
    )
    result, events = analyzer.update(
        1,
        feature(x=0.51, angle=6.0),
        decision("NORMAL"),
        state,
        0.8,
        0.9,
    )
    assert len(events) == 1
    assert events[0]["developed_into_fall"] is False
    assert result.near_fall_count == 1
    assert any("近跌倒" in factor for factor in result.risk_factors)


def test_candidate_that_becomes_fall_is_not_counted_as_near_fall():
    config = PreFallConfig(near_fall_min_stability_score=0.0)
    analyzer = PreFallRiskAnalyzer(config)
    state: dict = {}
    analyzer.update(
        1,
        feature(angle=25.0, down=0.25, rotation=50.0),
        decision("DESCENDING"),
        state,
        0.0,
        0.9,
    )
    analyzer.update(
        1,
        feature(angle=70.0, lying=0.8),
        decision("SUSPECT"),
        state,
        0.3,
        0.9,
    )
    result, events = analyzer.update(
        1,
        feature(),
        decision("NORMAL"),
        state,
        0.8,
        0.9,
    )
    assert events == []
    assert result.near_fall_count == 0


def test_fast_bend_without_lateral_instability_is_not_near_fall():
    config = PreFallConfig(near_fall_min_stability_score=0.0)
    analyzer = PreFallRiskAnalyzer(config)
    state: dict = {}
    analyzer.update(1, feature(), decision(), state, 0.0, 0.9)
    analyzer.update(
        1,
        feature(angle=42.0, rotation=48.0, down=0.02),
        decision("DESCENDING"),
        state,
        0.3,
        0.9,
    )
    result, events = analyzer.update(
        1,
        feature(),
        decision("NORMAL"),
        state,
        0.8,
        0.9,
    )
    assert events == []
    assert result.near_fall_count == 0


def test_upright_knee_collapse_then_moving_recovery_records_near_fall():
    config = PreFallConfig(
        baseline_min_samples=1,
        near_fall_knee_collapse_min_speed=0.04,
    )
    analyzer = PreFallRiskAnalyzer(config)
    state: dict = {}
    analyzer.update(1, feature(knee=166.0), decision(), state, 0.0, 0.9)
    analyzer.update(
        1, feature(knee=92.0, speed=0.02, lateral=0.01),
        decision(), state, 0.1, 0.9,
    )
    analyzer.update(
        1, feature(knee=125.0, speed=0.03, lateral=0.02),
        decision(), state, 0.2, 0.9,
    )
    result, events = analyzer.update(
        1, feature(knee=160.0, speed=0.06, lateral=0.05),
        decision(), state, 0.5, 0.9,
    )
    assert len(events) == 1
    assert events[0]["trigger_kind"] == "knee_collapse"
    assert result.near_fall_count == 1
    assert state["new_sit_to_stand_attempts"] == []
    assert state["sit_to_stand_phase"] == "STANDING"


def test_knee_collapse_without_recovery_motion_is_not_near_fall():
    config = PreFallConfig(
        baseline_min_samples=1,
        near_fall_recovery_max_s=0.8,
        near_fall_knee_collapse_min_speed=0.04,
    )
    analyzer = PreFallRiskAnalyzer(config)
    state: dict = {}
    analyzer.update(1, feature(knee=166.0), decision(), state, 0.0, 0.9)
    analyzer.update(
        1, feature(knee=92.0, speed=0.01), decision(), state, 0.1, 0.9,
    )
    analyzer.update(
        1, feature(knee=160.0, speed=0.01), decision(), state, 0.5, 0.9,
    )
    result, events = analyzer.update(
        1, feature(knee=162.0, speed=0.01), decision(), state, 1.0, 0.9,
    )
    assert events == []
    assert result.near_fall_count == 0


def test_knee_collapse_with_large_torso_bend_is_not_near_fall():
    config = PreFallConfig(baseline_min_samples=1)
    analyzer = PreFallRiskAnalyzer(config)
    state: dict = {}
    analyzer.update(1, feature(knee=166.0), decision(), state, 0.0, 0.9)
    analyzer.update(
        1, feature(knee=95.0, angle=24.0, speed=0.05),
        decision(), state, 0.1, 0.9,
    )
    result, events = analyzer.update(
        1, feature(knee=160.0, angle=20.0, speed=0.06),
        decision(), state, 0.5, 0.9,
    )
    assert events == []
    assert result.near_fall_count == 0


def test_risk_level_uses_hold_and_hysteresis():
    config = PreFallConfig(
        baseline_min_samples=1,
        medium_threshold=5.0,
        high_threshold=95.0,
        medium_exit_threshold=2.0,
        upgrade_hold_s=0.5,
        score_smoothing_alpha=1.0,
        sway_start_norm=0.0,
        sway_full_norm=0.001,
        torso_variability_start_deg=0.0,
        torso_variability_full_deg=1.0,
    )
    analyzer = PreFallRiskAnalyzer(config)
    state: dict = {}
    analyzer.update(1, feature(x=0.45), decision(), state, 0.0, 0.9)
    analyzer.update(1, feature(x=0.60, angle=18.0), decision(), state, 0.2, 0.9)
    pending, _ = analyzer.update(1, feature(x=0.42, angle=2.0), decision(), state, 0.4, 0.9)
    promoted, _ = analyzer.update(1, feature(x=0.63, angle=20.0), decision(), state, 1.0, 0.9)
    assert pending.pre_fall_risk_level == LOW
    assert promoted.pre_fall_risk_level == MEDIUM


def test_sit_to_stand_success_records_duration_and_updates_risk_component():
    config = PreFallConfig(
        baseline_min_samples=1,
        sit_to_stand_min_sit_hold_s=0.4,
        sit_to_stand_standing_hold_s=0.3,
        score_smoothing_alpha=1.0,
    )
    analyzer = PreFallRiskAnalyzer(config)
    state: dict = {}
    analyzer.update(1, feature(knee=105.0), decision(), state, 0.0, 0.9)
    analyzer.update(1, feature(knee=105.0), decision(), state, 0.5, 0.9)
    analyzer.update(1, feature(y=0.40, knee=140.0, down=-0.08), decision(), state, 0.8, 0.9)
    assert len(state["new_sit_to_stand_attempts"]) == 1
    analyzer.update(1, feature(knee=160.0, down=-0.04), decision(), state, 1.2, 0.9)
    result, _ = analyzer.update(1, feature(knee=165.0), decision(), state, 1.6, 0.9)
    events = state["new_sit_to_stand_events"]
    assert len(events) == 1
    assert events[0]["succeeded"] is True
    assert 0.9 <= events[0]["duration_s"] <= 1.2
    assert result.sit_to_stand_status == "成功"
    assert result.sit_to_stand_count == 1


def test_sit_to_stand_return_to_seat_is_recorded_as_failure():
    config = PreFallConfig(
        baseline_min_samples=1,
        sit_to_stand_min_sit_hold_s=0.4,
        sit_to_stand_min_transition_s=0.2,
        score_smoothing_alpha=1.0,
    )
    analyzer = PreFallRiskAnalyzer(config)
    state: dict = {}
    analyzer.update(1, feature(knee=100.0), decision(), state, 0.0, 0.9)
    analyzer.update(1, feature(knee=100.0), decision(), state, 0.5, 0.9)
    analyzer.update(1, feature(y=0.40, knee=138.0, down=-0.06), decision(), state, 0.8, 0.9)
    result, _ = analyzer.update(1, feature(knee=105.0), decision(), state, 1.2, 0.9)
    events = state["new_sit_to_stand_events"]
    assert len(events) == 1
    assert events[0]["succeeded"] is False
    assert result.sit_to_stand_score >= config.sit_to_stand_failure_score * 0.8
    assert any("起身能力异常" in factor for factor in result.risk_factors)


def test_sit_to_stand_ignores_single_frame_seated_knee_jitter():
    config = PreFallConfig(
        baseline_min_samples=1,
        sit_to_stand_min_sit_hold_s=0.4,
        sit_to_stand_attempt_hold_s=0.1,
        score_smoothing_alpha=1.0,
    )
    analyzer = PreFallRiskAnalyzer(config)
    state: dict = {}
    analyzer.update(1, feature(knee=105.0), decision(), state, 0.0, 0.9)
    analyzer.update(1, feature(knee=106.0), decision(), state, 0.5, 0.9)
    analyzer.update(1, feature(knee=165.0), decision(), state, 0.6, 0.9)
    analyzer.update(1, feature(knee=106.0), decision(), state, 0.7, 0.9)
    assert state["new_sit_to_stand_attempts"] == []
    assert state["new_sit_to_stand_events"] == []
    assert state["sit_to_stand_phase"] == "SEATED"


def test_sit_to_stand_requires_rise_plus_extension_or_torso_change():
    config = PreFallConfig(
        baseline_min_samples=1,
        sit_to_stand_min_sit_hold_s=0.4,
        sit_to_stand_attempt_hold_s=0.1,
        score_smoothing_alpha=1.0,
    )
    analyzer = PreFallRiskAnalyzer(config)
    state: dict = {}
    analyzer.update(1, feature(knee=105.0), decision(), state, 0.0, 0.9)
    analyzer.update(1, feature(knee=105.0), decision(), state, 0.5, 0.9)
    analyzer.update(
        1, feature(y=0.40, knee=108.0, down=-0.20), decision(), state, 0.7, 0.9,
    )
    analyzer.update(
        1, feature(y=0.39, knee=109.0, down=-0.20), decision(), state, 0.9, 0.9,
    )
    assert state["new_sit_to_stand_attempts"] == []
    assert state["sit_to_stand_phase"] == "SEATED"


def test_sit_to_stand_rejects_standing_knee_jitter_while_center_drops():
    config = PreFallConfig(
        baseline_min_samples=1,
        sit_to_stand_min_sit_hold_s=0.4,
        sit_to_stand_attempt_hold_s=0.1,
        score_smoothing_alpha=1.0,
    )
    analyzer = PreFallRiskAnalyzer(config)
    state: dict = {}
    analyzer.update(1, feature(y=0.50, knee=105.0), decision(), state, 0.0, 0.9)
    analyzer.update(1, feature(y=0.50, knee=105.0), decision(), state, 0.5, 0.9)
    analyzer.update(1, feature(y=0.55, knee=165.0), decision(), state, 0.7, 0.9)
    analyzer.update(1, feature(y=0.56, knee=165.0), decision(), state, 0.9, 0.9)
    assert state["new_sit_to_stand_attempts"] == []
    assert state["sit_to_stand_phase"] == "SEATED"


def test_sit_to_stand_ignores_small_screen_person():
    config = PreFallConfig(
        baseline_min_samples=1,
        sit_to_stand_min_sit_hold_s=0.4,
        score_smoothing_alpha=1.0,
    )
    analyzer = PreFallRiskAnalyzer(config)
    state: dict = {}
    analyzer.update(1, feature(knee=100.0, area=0.01), decision(), state, 0.0, 0.9)
    analyzer.update(1, feature(knee=100.0, area=0.01), decision(), state, 0.5, 0.9)
    analyzer.update(
        1, feature(y=0.35, knee=165.0, down=-0.1, area=0.01),
        decision(), state, 0.8, 0.9,
    )
    assert state["new_sit_to_stand_attempts"] == []
    assert state["new_sit_to_stand_events"] == []


def test_three_time_windows_survive_profile_export_and_restore():
    config = PreFallConfig(
        baseline_min_samples=1,
        window_sample_interval_s=0.0,
        score_smoothing_alpha=1.0,
    )
    analyzer = PreFallRiskAnalyzer(config)
    state: dict = {}
    analyzer.restore_persistent_profile(
        state,
        {"baseline_speed": 0.08, "baseline_stability": 0.0, "baseline_samples": 10},
        0.0,
    )
    first, _ = analyzer.update(
        1, feature(speed=0.035), decision(), state, 0.0, 0.9,
        observed_at_s=1_700_000_000.0,
    )
    second, _ = analyzer.update(
        1, feature(speed=0.030), decision(), state, 1.0, 0.9,
        observed_at_s=1_700_086_500.0,
    )
    assert len(state["daily_windows"]) == 2
    assert first.short_term_score >= 0.0
    assert second.medium_term_score >= 0.0
    profile = analyzer.export_persistent_profile(state)
    restored: dict = {}
    analyzer.restore_persistent_profile(restored, profile, 0.0)
    assert restored["baseline_samples"] >= 10
    assert len(restored["daily_windows"]) == 2


def test_depth_spatial_evidence_is_fused_into_pre_fall_score():
    config = PreFallConfig(
        baseline_min_samples=1,
        pre_fall_depth_weight=0.5,
        score_smoothing_alpha=1.0,
        window_sample_interval_s=0.0,
    )
    analyzer = PreFallRiskAnalyzer(config)
    rgb_state: dict = {}
    depth_state: dict = {}
    rgb, _ = analyzer.update(
        1, feature(), decision(), rgb_state, 0.0, 0.9,
        observed_at_s=1_700_000_000.0,
    )
    fused, _ = analyzer.update(
        1, feature(), decision(), depth_state, 0.0, 0.9,
        depth_quality=0.9,
        depth_available=True,
        depth_features={"pre_fall_spatial_score": 90.0},
        observed_at_s=1_700_000_000.0,
    )
    assert fused.depth_fusion_applied is True
    assert fused.depth_spatial_score == 90.0
    assert fused.pre_fall_risk_score > rgb.pre_fall_risk_score
