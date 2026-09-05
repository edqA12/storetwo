from __future__ import annotations

from core.config import DetectionConfig
from core.state_machine import DESCENDING, FALLEN, LYING, NORMAL, SUSPECT, advance_state_machine
from core.types import PoseFeatures


def feature(
    angle: float = 5.0,
    lying_score: float = 0.05,
    down: float = 0.0,
    rotation: float = 0.0,
    motion: float = 0.03,
) -> PoseFeatures:
    return PoseFeatures(
        valid=True,
        torso_angle_deg=angle,
        box_aspect=0.45 if angle < 40 else 1.8,
        hip_y_norm=0.55,
        center_y_norm=0.45,
        down_velocity=down,
        angle_velocity=rotation,
        motion_norm=motion,
        lying_score=lying_score,
        quality=0.9,
    )


def test_fall_evidence_chain_emits_exactly_one_event():
    config = DetectionConfig()
    machine: dict = {}
    normal = advance_state_machine(1, feature(), machine, 0.0, config)
    descending = advance_state_machine(1, feature(down=0.35), machine, 0.2, config)
    suspect = advance_state_machine(1, feature(angle=80, lying_score=0.95), machine, 0.4, config)
    fallen = advance_state_machine(1, feature(angle=82, lying_score=0.98, motion=0.01), machine, 1.3, config)
    repeated = advance_state_machine(1, feature(angle=82, lying_score=0.98, motion=0.01), machine, 1.5, config)
    assert normal.state == NORMAL
    assert descending.state == DESCENDING
    assert suspect.state == SUSPECT
    assert fallen.state == FALLEN and fallen.event_started
    assert repeated.state == FALLEN and not repeated.event_started
    assert fallen.event_key == repeated.event_key


def test_initial_lying_is_yellow_not_red_fall():
    config = DetectionConfig()
    machine: dict = {}
    first = advance_state_machine(1, feature(angle=85, lying_score=0.98), machine, 0.0, config)
    later = advance_state_machine(1, feature(angle=85, lying_score=0.98), machine, 2.1, config)
    assert first.state == NORMAL
    assert later.state == LYING
    assert not later.event_started
    assert later.risk < 80


def test_motion_while_already_lying_does_not_become_confirmed_fall():
    config = DetectionConfig()
    machine: dict = {}
    advance_state_machine(1, feature(angle=85, lying_score=0.98), machine, 0.0, config)
    lying = advance_state_machine(1, feature(angle=85, lying_score=0.98), machine, 2.1, config)
    turning = advance_state_machine(
        1,
        feature(angle=78, lying_score=0.92, down=0.4, rotation=100.0, motion=0.08),
        machine,
        2.3,
        config,
    )
    later = advance_state_machine(1, feature(angle=82, lying_score=0.96), machine, 3.5, config)
    assert lying.state == LYING
    assert turning.state == LYING
    assert later.state == LYING
    assert not turning.event_started and later.event_started


def test_recovery_returns_to_normal():
    config = DetectionConfig()
    machine: dict = {}
    advance_state_machine(1, feature(down=0.4), machine, 0.0, config)
    advance_state_machine(1, feature(angle=80, lying_score=0.95), machine, 0.2, config)
    fallen = advance_state_machine(1, feature(angle=80, lying_score=0.95), machine, 1.0, config)
    assert fallen.state == FALLEN
    advance_state_machine(1, feature(), machine, 1.2, config)
    recovered = advance_state_machine(1, feature(), machine, 2.3, config)
    assert recovered.state == NORMAL


def test_long_invalid_pose_breaks_continuous_fall_confirmation():
    config = DetectionConfig()
    machine: dict = {}
    advance_state_machine(1, feature(down=0.4), machine, 0.0, config)
    suspect = advance_state_machine(1, feature(angle=80, lying_score=0.95), machine, 0.2, config)
    invalid = PoseFeatures(valid=False)
    advance_state_machine(1, invalid, machine, 0.3, config)
    recovered = advance_state_machine(1, feature(angle=82, lying_score=0.97), machine, 1.0, config)
    assert suspect.state == SUSPECT
    assert recovered.state != FALLEN
    assert not recovered.event_started


def test_suspect_confirms_even_after_descent_evidence_expires():
    config = DetectionConfig(descent_timeout_s=0.5, lying_hold_s=0.8)
    machine: dict = {}
    advance_state_machine(1, feature(down=0.4), machine, 0.0, config)
    suspect = advance_state_machine(1, feature(angle=80, lying_score=0.95), machine, 0.2, config)
    fallen = advance_state_machine(1, feature(angle=82, lying_score=0.98), machine, 1.1, config)
    assert suspect.state == SUSPECT
    assert fallen.state == FALLEN
    assert fallen.event_started


def test_prolonged_unexplained_lying_emits_one_suspected_event():
    config = DetectionConfig(suspected_fall_hold_s=3.0, suspected_fall_min_quality=0.55)
    machine: dict = {}
    advance_state_machine(1, feature(angle=85, lying_score=0.98), machine, 0.0, config)
    lying = advance_state_machine(1, feature(angle=85, lying_score=0.98), machine, 2.1, config)
    suspected = advance_state_machine(1, feature(angle=85, lying_score=0.98), machine, 3.1, config)
    repeated = advance_state_machine(1, feature(angle=85, lying_score=0.98), machine, 3.5, config)
    assert lying.state == LYING and not lying.event_started
    assert suspected.state == LYING and suspected.risk == 79.0
    assert suspected.event_started
    assert not repeated.event_started
