from __future__ import annotations

import numpy as np

from tempograph.lodge_protocol import _beat_alignment


def test_out_of_duration_music_beats_must_not_enter_bas_denominator() -> None:
    music = np.asarray([10, 20, 30, 150, 180])
    motion = np.asarray([10, 20, 30])
    unbounded = _beat_alignment(music, motion)
    bounded = _beat_alignment(music[music < 100], motion)
    assert bounded == 1.0
    assert unbounded < bounded
