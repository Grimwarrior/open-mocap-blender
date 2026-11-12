"""Timing between two videos: found from the move, not from the actor walking to or from a camera."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import combine  # noqa: E402
import synthetic_combine  # noqa: E402


def walk(frames, start, end, horizon):
    """The actor's picture as he walks: his size goes from `start` to `end` times, around the horizon."""
    p = combine._as_array(frames).copy()
    s = np.linspace(start, end, len(p))[:, None]
    p[:, :, 0] *= s
    p[:, :, 1] = horizon + (p[:, :, 1] - horizon) * s
    return p


class Sync(unittest.TestCase):
    def test_walking_toward_one_camera_and_away_from_the_other(self):
        # The katana Front/Back takes: raw heights matched at -0.05 and the Back video was left out.
        take = synthetic_combine.make_take(seed=0, extra_angles=(180.0,))
        F, back = take["front"], take["views"][1]
        horizon = float(combine._as_array(F)[:, :, 1].min()) - 1.0
        F_walk = walk(F, 0.6, 1.4, horizon)
        B_walk = walk(back["frames"], 1.4, 0.6, horizon)
        tF, tB = np.asarray(take["tF"]), np.asarray(back["times"])
        offset, match = combine.find_time_offset(tF, F_walk, tB, B_walk)
        expected, _ = combine.find_time_offset(tF, combine._as_array(F), tB, combine._as_array(back["frames"]))
        self.assertGreaterEqual(match, combine.MIN_SYNC_MATCH)
        self.assertAlmostEqual(offset, expected, delta=1.5 / 30)
        # raw picture heights see the walk: the motion match drops, so the views do not pass as one performance
        self.assertLess(combine.motion_match(tF, F_walk, tB, B_walk, offset), combine.FUSE_SYNCED_MATCH)


if __name__ == "__main__":
    unittest.main()
