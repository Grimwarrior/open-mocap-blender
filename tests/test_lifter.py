"""Depth Lifter: feeding a take to the 2D-to-3D model and mixing its depth back in (lifter.py)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lifter  # noqa: E402


def take(count, seed=0):
    """(count, 40, 2) points of a swaying stick figure about 5 units tall, and visibility."""
    rng = np.random.default_rng(seed)
    base = np.zeros((40, 2))
    base[:, 1] = np.linspace(-5.0, 0.0, 40)                  # spread from head (-5) to feet (0)
    base[23], base[24] = [0.3, -2.5], [-0.3, -2.5]
    base[11], base[12] = [0.5, -4.0], [-0.5, -4.0]
    base[0], base[27], base[28] = [0.0, -5.0], [0.3, 0.0], [-0.3, 0.0]
    t = np.arange(count)[:, None, None]
    points = base[None] + 0.2 * np.sin(t / 10.0) + rng.normal(0, 0.01, (count, 40, 2))
    return points, np.full((count, 40), 0.9)


def height_model(x):
    """Stand-in for the network: 'depth' = each joint's height in model units (flip-invariant)."""
    out = np.array(x, dtype=np.float32)
    out[..., 2] = x[..., 1]
    return out


class FeedTest(unittest.TestCase):
    def test_h36m_joints_come_from_the_right_points(self):
        points, vis = take(3)
        h = lifter.to_h36m(points, vis)
        np.testing.assert_allclose(h[:, 0, :2], (points[:, 23] + points[:, 24]) / 2)    # pelvis
        np.testing.assert_allclose(h[:, 1, :2], points[:, 24])                           # right hip
        np.testing.assert_allclose(h[:, 11, :2], points[:, 11])                          # left shoulder
        np.testing.assert_allclose(h[:, 7, :2], (h[:, 0, :2] + h[:, 8, :2]) / 2)        # spine
        np.testing.assert_allclose(h[:, :, 2], 0.9)

    def test_actor_is_scaled_to_the_training_size(self):
        points, vis = take(50)
        n, unit = lifter.normalize(lifter.to_h36m(points, vis))
        self.assertAlmostEqual(float(np.median(np.ptp(n[:, :, 1], axis=1))), lifter.TARGET_HEIGHT, places=6)
        self.assertLess(float(np.abs(np.median(n[:, 0, :2], axis=0)).max()), 1e-9)
        self.assertGreater(unit, 0)

    def test_windows_cover_every_frame(self):
        (idx, w), = lifter.windows(120)
        self.assertEqual(len(idx), lifter.WINDOW)
        self.assertEqual(set(idx.tolist()), set(range(120)))
        wins = lifter.windows(600)
        covered = np.zeros(600)
        for idx, w in wins:
            self.assertEqual(len(idx), lifter.WINDOW)
            covered[idx] += w
        self.assertTrue(np.all(covered > 0))
        self.assertEqual(wins[-1][0][-1], 599)

    def test_mirroring_twice_is_no_change(self):
        x = np.random.default_rng(1).normal(size=(1, 243, 17, 3)).astype(np.float32)
        np.testing.assert_allclose(lifter.flip(lifter.flip(x)), x)
        self.assertEqual(float(lifter.flip(x)[0, 0, 1, 0]), -float(x[0, 0, 4, 0]))      # right hip <- -left hip


class LiftTest(unittest.TestCase):
    def check(self, count):
        points, vis = take(count)
        depth = lifter.lift(points, vis, height_model)
        h = lifter.to_h36m(points, vis)
        expected = h[:, :, 1] - h[:, :1, 1]                    # heights about the pelvis, in take units
        np.testing.assert_allclose(depth, expected, atol=1e-4)

    def test_short_take_comes_back_in_its_own_units(self):
        self.check(120)

    def test_long_take_is_read_in_windows(self):
        self.check(700)


class ApplyTest(unittest.TestCase):
    def setUp(self):
        self.count = 4
        points, self.vis = take(self.count)
        self.old_z = np.tile(np.linspace(-1.0, 1.0, 40), (self.count, 1))
        self.frames = [np.column_stack([p, z]).ravel().tolist() for p, z in zip(points, self.old_z)]
        self.meta = [{'depth_confidence': [1.0] * 40, 'depth_source': 'mediapipe_world'} for _ in range(self.count)]
        self.depth = np.full((self.count, 17), 0.0)
        self.depth[:, 13] = 2.0                                # lifted left wrist 2 units behind the pelvis

    def z(self, t):
        return np.asarray(self.frames[t][2::3])

    def test_lifted_joint_is_mixed_and_followers_move_with_it(self):
        lifter.apply(self.frames, self.vis, self.meta, self.depth, share=0.5)
        hips = (self.old_z[0, 23] + self.old_z[0, 24]) / 2
        wrist = 0.5 * (hips + 2.0) + 0.5 * self.old_z[0, 15]
        self.assertAlmostEqual(self.z(0)[15], wrist)
        shift = wrist - self.old_z[0, 15]
        for knuckle in (17, 19, 21):
            self.assertAlmostEqual(self.z(0)[knuckle], self.old_z[0, knuckle] + shift)
        self.assertAlmostEqual(self.z(0)[36], (self.z(0)[17] + self.z(0)[19]) / 2)       # left hand centre
        self.assertEqual(self.meta[0]['depth_source'], 'mediapipe_world+lifter')

    def test_share_zero_changes_nothing_and_missing_depth_takes_the_lifter(self):
        lifter.apply(self.frames, self.vis, self.meta, self.depth, share=0.0)
        np.testing.assert_allclose(self.z(1)[:33], self.old_z[1, :33])
        self.meta[2]['depth_confidence'] = [0.0] * 40          # tracker had no depth in this frame
        frames = [self.frames[2]]
        lifter.apply(frames, self.vis[2:3], self.meta[2:3], self.depth[2:3], share=0.3)
        hips = (self.old_z[2, 23] + self.old_z[2, 24]) / 2
        self.assertAlmostEqual(np.asarray(frames[0][2::3])[15], hips + 2.0)
        self.assertGreater(self.meta[2]['depth_confidence'][15], 0.0)


if __name__ == "__main__":
    unittest.main()
