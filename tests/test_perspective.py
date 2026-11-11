"""Perspective correction of a combined take: both feet on one floor, nothing moved where nothing is nearer."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import combine  # noqa: E402
import synthetic_combine  # noqa: E402

UNIT = synthetic_combine.UNIT


def ankle_gap(frames):
    c = np.asarray(frames).reshape(len(frames), 40, 3)
    return float(np.abs(c[:, combine.L_ANKLE, 1] - c[:, combine.R_ANKLE, 1]).mean())


class Perspective(unittest.TestCase):
    def combined(self, distance, seed=0):
        take = synthetic_combine.make_take(seed=seed, camera_distance=distance)
        frames, _ = combine.combine_improved(take["front"], take["side"], take["tF"], take["tS"],
                                             take["visF"], take["visS"])
        return take, frames

    def test_nearer_foot_comes_back_up_to_the_floor(self):
        take, frames = self.combined(4.0)
        fixed, report = combine.correct_perspective(frames, 4.0 * UNIT, horizon=-1.3 * UNIT)
        self.assertLess(ankle_gap(fixed), 0.6 * ankle_gap(frames))
        lo, hi = report["perspective_factor"]
        self.assertLess(lo, 1.0)
        self.assertGreater(hi, 1.0)

    def test_points_at_the_hip_plane_stay_put(self):
        frames = np.zeros((3, 40, 3))
        frames[:, :, 0] = np.linspace(-3, 3, 40)
        frames[:, :, 1] = np.linspace(-9, -1, 40)
        fixed, report = combine.correct_perspective(frames.reshape(3, -1).tolist(), 17.0, horizon=-5.4)
        np.testing.assert_allclose(np.asarray(fixed).reshape(3, 40, 3)[:, :33], frames[:, :33])   # 33+ are means
        self.assertEqual(report["perspective_factor"], (1.0, 1.0))

    def test_nearer_point_moves_toward_the_picture_centre_and_horizon(self):
        frames = np.zeros((1, 40, 3))
        frames[0, 27] = [2.0, -8.0, -1.7]      # an ankle 1.7 units nearer the camera (f = 17: 10% closer)
        fixed, _ = combine.correct_perspective(frames.reshape(1, -1).tolist(), 17.0, horizon=-5.4)
        ankle = np.asarray(fixed).reshape(40, 3)[27]
        self.assertAlmostEqual(ankle[0], 2.0 * 0.9)
        self.assertAlmostEqual(ankle[1], -5.4 + (-8.0 + 5.4) * 0.9)
        self.assertEqual(ankle[2], -1.7)       # depth itself is not changed

    def test_off_and_unknown_focal_leave_the_take_alone(self):
        take, frames = self.combined(5.0)
        self.assertEqual(combine.correct_perspective(frames, 0.0)[0], frames)

    def test_focal_from_field_of_view(self):
        # 1920 px wide, 60 degrees across: f = 960 / tan(30 deg) = 1663 px
        self.assertAlmostEqual(combine.focal_from_fov(1920, 60.0), 16.628, places=2)


if __name__ == "__main__":
    unittest.main()
