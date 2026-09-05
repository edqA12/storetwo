from __future__ import annotations

import uuid
from typing import Any

from .config import DetectionConfig
from .types import Decision, PoseFeatures


NORMAL = "NORMAL"
DESCENDING = "DESCENDING"
SUSPECT = "SUSPECT"
FALLEN = "FALLEN"
LYING = "LYING"
UNKNOWN = "UNKNOWN"

LABELS = {
    NORMAL: "正常",
    DESCENDING: "快速下降",
    SUSPECT: "疑似跌倒",
    FALLEN: "已检测到跌倒",
    LYING: "躺卧/姿态异常",
    UNKNOWN: "姿态不完整",
}

REASON_LABELS = {
    "rapid_down": "人体中心快速下降",
    "rapid_rotation": "躯干快速旋转",
    "lying": "身体持续接近水平",
    "low_motion": "姿态运动幅度较低",
    "poor_pose": "关键点不完整",
    "unexplained_lying": "未观察到快速下降的躺卧",
}


def _initial_machine(timestamp_s: float) -> dict[str, Any]:
    return {
        "state": NORMAL,
        "state_since": float(timestamp_s),
        "lying_since": None,
        "upright_since": float(timestamp_s),
        "descent_evidence_until": 0.0,
        "last_event_ts": -1e12,
        "active_event_key": None,
        "invalid_since": None,
    }


def _transition(machine: dict[str, Any], state: str, timestamp_s: float) -> None:
    if machine.get("state") != state:
        machine["state"] = state
        machine["state_since"] = float(timestamp_s)


def advance_state_machine(
    track_id: int,
    features: PoseFeatures,
    machine: dict[str, Any],
    timestamp_s: float,
    config: DetectionConfig,
) -> Decision:
    if not machine:
        machine.update(_initial_machine(timestamp_s))

    if not features.valid:
        if machine.get("invalid_since") is None:
            machine["invalid_since"] = float(timestamp_s)
        return Decision(
            track_id=track_id,
            state=UNKNOWN,
            label=LABELS[UNKNOWN],
            risk=0.0,
            reason_codes=["poor_pose"],
        )

    invalid_since = machine.get("invalid_since")
    if invalid_since is not None:
        invalid_duration = timestamp_s - float(invalid_since)
        if invalid_duration >= config.missing_tolerance_s:
            last_event_ts = float(machine.get("last_event_ts", -1e12))
            machine.clear()
            machine.update(_initial_machine(timestamp_s))
            machine["last_event_ts"] = last_event_ts
        else:
            if machine.get("lying_since") is not None:
                machine["lying_since"] = float(machine["lying_since"]) + max(0.0, invalid_duration)
            if machine.get("upright_since") is not None:
                machine["upright_since"] = float(machine["upright_since"]) + max(0.0, invalid_duration)
        machine["invalid_since"] = None

    angle = float(features.torso_angle_deg or 0.0)
    lying = features.lying_score >= config.lying_enter_score or angle >= config.lying_enter_angle_deg
    upright = features.lying_score <= config.lying_exit_score and angle <= config.lying_exit_angle_deg
    rapid_down = features.down_velocity >= config.fast_down_velocity
    rapid_rotation = features.angle_velocity >= config.fast_angle_velocity
    transition_evidence = rapid_down or rapid_rotation

    reasons: list[str] = []
    if rapid_down:
        reasons.append("rapid_down")
    if rapid_rotation:
        reasons.append("rapid_rotation")
    if lying:
        reasons.append("lying")
    if features.motion_norm < 0.018:
        reasons.append("low_motion")

    if lying:
        if machine.get("lying_since") is None:
            machine["lying_since"] = float(timestamp_s)
        machine["upright_since"] = None
    elif upright:
        machine["lying_since"] = None
        if machine.get("upright_since") is None:
            machine["upright_since"] = float(timestamp_s)

    if transition_evidence:
        machine["descent_evidence_until"] = float(timestamp_s + config.descent_timeout_s)

    state = str(machine.get("state", NORMAL))
    evidence_active = timestamp_s <= float(machine.get("descent_evidence_until", 0.0))
    lying_duration = (
        timestamp_s - float(machine["lying_since"])
        if machine.get("lying_since") is not None else 0.0
    )
    upright_duration = (
        timestamp_s - float(machine["upright_since"])
        if machine.get("upright_since") is not None else 0.0
    )

    if state == NORMAL:
        if transition_evidence:
            _transition(machine, DESCENDING, timestamp_s)
            if lying:
                machine["lying_since"] = float(timestamp_s)
        elif lying and lying_duration >= config.unexplained_lying_s:
            _transition(machine, LYING, timestamp_s)
    elif state == LYING:
        if upright and upright_duration >= config.recovery_upright_s:
            _transition(machine, NORMAL, timestamp_s)
            machine["active_event_key"] = None
    elif state == DESCENDING:
        if lying and evidence_active:
            _transition(machine, SUSPECT, timestamp_s)
        elif upright and timestamp_s - float(machine["state_since"]) >= 0.35:
            _transition(machine, NORMAL, timestamp_s)
        elif timestamp_s - float(machine["state_since"]) > config.descent_timeout_s:
            _transition(machine, LYING if lying else NORMAL, timestamp_s)
    elif state == SUSPECT:
        # 快速下降/旋转证据只负责进入SUSPECT。一旦进入疑似状态，持续
        # 横卧本身即可完成确认，避免低帧率或短暂遮挡令证据窗口先过期，
        # 从而长期卡在79分。
        if lying and lying_duration >= config.lying_hold_s:
            _transition(machine, FALLEN, timestamp_s)
        elif upright:
            _transition(machine, NORMAL, timestamp_s)
        elif not lying and timestamp_s - float(machine["state_since"]) > config.descent_timeout_s:
            _transition(machine, NORMAL, timestamp_s)
    elif state == FALLEN:
        if upright and upright_duration >= config.recovery_upright_s:
            _transition(machine, NORMAL, timestamp_s)
            machine["active_event_key"] = None

    state = str(machine.get("state", NORMAL))
    suspected_fall = (
        state == LYING
        and lying_duration >= config.suspected_fall_hold_s
        and features.quality >= config.suspected_fall_min_quality
    )
    risk = 8.0
    risk += min(38.0, features.lying_score * 38.0)
    risk += min(24.0, max(0.0, features.down_velocity) / max(config.fast_down_velocity, 1e-6) * 18.0)
    risk += min(18.0, max(0.0, features.angle_velocity) / max(config.fast_angle_velocity, 1e-6) * 14.0)
    if evidence_active:
        risk += 12.0
    if lying and features.motion_norm < 0.018:
        risk += 8.0
    if state == NORMAL:
        risk = min(risk, 39.0)
    elif state == DESCENDING:
        risk = max(45.0, min(risk, 69.0))
    elif state in (SUSPECT, LYING):
        risk = max(60.0, min(risk, 79.0))
    elif state == FALLEN:
        risk = max(85.0, risk)
    if suspected_fall:
        risk = max(79.0, risk)
    risk = round(min(100.0, risk), 1)

    event_started = False
    event_key = machine.get("active_event_key")
    confirmed_fall = state == FALLEN
    if (confirmed_fall or suspected_fall) and event_key is None:
        if timestamp_s - float(machine.get("last_event_ts", -1e12)) >= config.event_cooldown_s:
            event_key = uuid.uuid4().hex
            machine["active_event_key"] = event_key
            machine["last_event_ts"] = float(timestamp_s)
            event_started = True

    if state == LYING and "unexplained_lying" not in reasons:
        reasons.append("unexplained_lying")

    return Decision(
        track_id=track_id,
        state=state,
        label=LABELS[state],
        risk=risk,
        event_started=event_started,
        event_key=str(event_key) if event_key else None,
        reason_codes=reasons,
    )
