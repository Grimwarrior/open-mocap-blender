"""The retarget's leg solver against a synthetic actor whose foot contacts are known."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synthetic_retarget as bench  # noqa: E402

R = bench.R


class LegSolver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loose = bench.run(plant=False)
        cls.held = bench.run()

    def test_standing_feet_are_held(self):
        self.assertGreater(self.loose["slide"], 15.0)          # the benchmark does make unheld feet slide
        self.assertLess(self.held["slide"], 1.5)

    def test_contacts_are_found(self):
        self.assertGreater(self.held["contact"], 93.0)

    def test_foot_turning_on_its_ball_is_not_frozen(self):
        self.assertLess(self.held["pivot"], 5.0)

    def test_feet_in_the_air_follow_the_mocap(self):
        self.assertLess(self.held["air"], self.loose["air"] + 2.0)

    def test_knees_point_the_right_way(self):
        self.assertLess(self.held["knee"], 4.0)
        self.assertLess(self.held["knee_straight"], 5.0)
        self.assertLess(self.held["flips"], 1.0)

    def test_straight_legs_keep_their_knee_direction_under_jitter(self):
        noisy = bench.run(noise=0.012)
        self.assertLess(noisy["knee_straight"], 9.0)             # 15.7 before bend_direction
        self.assertLess(noisy["flips"], 1.0)


class BendDirection(unittest.TestCase):
    def test_straight_stretch_keeps_the_bend_from_before_and_after(self):
        n = 90
        along = np.tile([0.0, 0.0, -1.0], (n, 1))
        rng = np.random.default_rng(1)
        offset = np.tile([0.0, -0.12, -0.4], (n, 1))             # bent forward (-y) ...
        offset[30:60, 1] = 0.0                                   # ... but straight in the middle,
        offset[30:60, :2] += rng.normal(size=(30, 2)) * 0.01     # where only jitter is left
        hint = np.tile([1.0, 0.0, 0.0], (n, 1))                  # a hint that points elsewhere
        way = R.bend_direction(offset, along, 0.45, hint, 30.0)
        self.assertGreater(float(np.min(way[:, 1] * -1.0)), 0.9)

    def test_always_straight_limb_uses_the_hint(self):
        n = 60
        along = np.tile([0.0, 0.0, -1.0], (n, 1))
        offset = np.tile([0.0, 0.0, -0.4], (n, 1))
        hint = np.tile([0.0, -1.0, 0.0], (n, 1))
        way = R.bend_direction(offset, along, 0.45, hint, 30.0)
        self.assertGreater(float(np.min(way @ np.array([0.0, -1.0, 0.0]))), 0.99)


class FootStates(unittest.TestCase):
    def setUp(self):
        self.fps = 30.0
        t = np.arange(0, 4, 1 / self.fps)
        self.n = len(t)
        self.t = t

    def foot(self, ankle):
        return ankle, ankle + np.array([0.0, 0.05, -0.06]), ankle + np.array([0.0, -0.17, -0.06])

    def test_jitter_around_the_threshold_does_not_flicker(self):
        rng = np.random.default_rng(0)
        ankle = np.zeros((self.n, 3)) + np.array([0.0, 0.0, 0.07])
        ankle[:, 0] += 0.012 * np.sin(2 * np.pi * 1.7 * self.t)               # slow sway near the speed limit
        ankle += rng.normal(size=ankle.shape) * 0.003
        states = R.foot_states(*self.foot(ankle), 0.88, self.fps)
        self.assertEqual(len(R._runs(states["down"])), 1)
        self.assertGreater(states["flat"].mean(), 0.9)

    def test_floor_that_moves_in_the_picture(self):
        # a foot standing first far from the camera (high in the picture), then near (low)
        ankle = np.zeros((self.n, 3)) + np.array([0.0, 0.0, 0.25])
        step = (self.t > 1.8) & (self.t < 2.2)
        ankle[self.t >= 2.2, 2] = 0.07
        ankle[step, 2] = 0.25 + 0.15 * np.sin(np.pi * (self.t[step] - 1.8) / 0.4)
        ankle[step, 1] = -1.5 * (self.t[step] - 1.8)
        ankle[self.t >= 2.2, 1] = -0.6
        states = R.foot_states(*self.foot(ankle), 0.88, self.fps)
        self.assertTrue(states["down"][:40].all())               # the first stand is a contact too
        self.assertTrue(states["down"][-40:].all())
        self.assertFalse(states["down"][57:63].any())            # and the step between is not

    def test_unreliable_landmarks_are_not_contacts(self):
        ankle = np.zeros((self.n, 3)) + np.array([0.0, 0.0, 0.07])
        confidence = np.ones(self.n)
        confidence[30:60] = 0.1
        states = R.foot_states(*self.foot(ankle), 0.88, self.fps, confidence)
        self.assertFalse(states["down"][35:55].any())


if __name__ == "__main__":
    unittest.main()
