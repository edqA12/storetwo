from __future__ import annotations

import numpy as np

from core.tracker import assign_track_ids
from core.types import PersonPose


def pose(bbox):
    return PersonPose(0, bbox, 0.9, np.zeros((17, 2)), np.ones(17))


def test_tracker_keeps_id_for_nearby_box_and_expires_old_track():
    state = {"tracks": {}, "next_track_id": 1}
    first = assign_track_ids([pose((10, 10, 50, 100))], state, 0.0, 0.2, 0.4, 0.5)
    second = assign_track_ids([pose((12, 11, 52, 101))], state, 0.2, 0.2, 0.4, 0.5)
    assert first[0].track_id == second[0].track_id == 1
    third = assign_track_ids([pose((200, 20, 250, 110))], state, 1.0, 0.2, 0.4, 0.5)
    assert third[0].track_id == 2
