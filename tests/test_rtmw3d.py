"""RTMW3D (experimental tracker): depth decoding, score calibration and the actor's size in pixels."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rtmw3d  # noqa: E402


def standing_actor(scale=200.0, turn_deg=0.0):
    """33 points of an actor built from the template bones, turned about the vertical axis.
    :return: (xy pixels, depth metres) as an orthographic camera at `scale` px/m sees them."""
    p = np.zeros((33, 3))
    p[23], p[24] = [0.125, 0, 0], [-0.125, 0, 0]               # hips 0.25 apart
    p[11], p[12] = [0.18, -0.48, 0], [-0.18, -0.48, 0]          # shoulders 0.36 apart, ~0.5 above hips
    p[13], p[14] = p[11] + [0, 0.29, 0], p[12] + [0, 0.29, 0]
    p[15], p[16] = p[13] + [0, 0.26, 0], p[14] + [0, 0.26, 0]
    p[25], p[26] = p[23] + [0, 0.43, 0], p[24] + [0, 0.43, 0]
    p[27], p[28] = p[25] + [0, 0.42, 0], p[26] + [0, 0.42, 0]
    t = np.radians(turn_deg)
    rot = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
    p = p @ rot.T
    return p[:, :2] * scale + [500.0, 400.0], p[:, 2]


class DecodeTest(unittest.TestCase):
    def test_depth_axis_size_comes_from_the_model(self):
        self.assertEqual(rtmw3d.depth_bins([[1, 133, 576], [1, 133, 768], [1, 133, 576]]), 288.0)
        self.assertEqual(rtmw3d.depth_bins([[1, 133, 576], [1, 133, 768], [1, 133, "z"]]), 288.0)
        self.assertEqual(rtmw3d.depth_bins([]), 288.0)

    def test_centre_of_the_depth_axis_is_the_root(self):
        np.testing.assert_allclose(rtmw3d.decode_depth([0.0, 144.0, 288.0], 288.0),
                                   [-rtmw3d.Z_RANGE, 0.0, rtmw3d.Z_RANGE])

    def test_visibility_matches_mediapipe_scale(self):
        v = rtmw3d.visibility([0.3, 0.5, 0.6, 0.73, 0.9])
        self.assertEqual(v[0], 0.0)
        self.assertAlmostEqual(v[1], 0.30, places=2)
        self.assertTrue(0.5 < v[2] < 0.7)
        self.assertAlmostEqual(v[3], 1.0, places=6)
        self.assertEqual(v[4], 1.0)


class ScaleTest(unittest.TestCase):
    def test_front_and_turned_actor_give_the_true_scale(self):
        for turn in (0.0, 35.0, 80.0):
            xy, z = standing_actor(200.0, turn)
            self.assertAlmostEqual(rtmw3d.pixels_per_metre(xy, z), 200.0, delta=200.0 * 0.03, msg=turn)

    def test_depth_in_pixels_is_about_the_hips(self):
        xy, z = standing_actor(150.0, 60.0)
        depth, scale = rtmw3d.depth_pixels(xy, z + 3.0, np.ones(33))       # model depth has an offset
        self.assertAlmostEqual(scale, 150.0, delta=4.0)
        self.assertAlmostEqual((depth[23] + depth[24]) / 2, 0.0, places=6)
        np.testing.assert_allclose(depth, (z - (z[23] + z[24]) / 2) * scale, rtol=1e-6)

    def test_scale_is_smoothed_and_kept_when_the_actor_is_hidden(self):
        xy, z = standing_actor(300.0)
        _depth, scale = rtmw3d.depth_pixels(xy, z, np.ones(33), previous_scale=200.0)
        self.assertAlmostEqual(scale, 0.8 * 200.0 + 0.2 * 300.0, delta=3.0)
        depth, kept = rtmw3d.depth_pixels(xy, z, np.zeros(33), previous_scale=200.0)
        self.assertEqual(kept, 200.0)
        self.assertIsNotNone(depth)
        depth, none = rtmw3d.depth_pixels(xy, z, np.zeros(33))
        self.assertIsNone(depth)
        self.assertIsNone(none)


class BlendTest(unittest.TestCase):
    def setUp(self):
        self.own = np.linspace(-50.0, 50.0, 33)
        self.mp = np.linspace(40.0, -40.0, 33)
        self.vis = np.full(33, 0.8)

    def test_share_mixes_the_depths(self):
        depth, conf, source = rtmw3d.blend_depth(self.own, self.vis, self.mp, np.ones(33), share=0.1)
        np.testing.assert_allclose(depth, 0.9 * self.mp + 0.1 * self.own)
        np.testing.assert_allclose(conf, 0.9 * 0.8 + 0.1 * 0.8 * rtmw3d.DEPTH_TRUST)
        self.assertEqual(source, 'rtmw3d+mediapipe')
        depth, _conf, _source = rtmw3d.blend_depth(self.own, self.vis, self.mp, np.ones(33), share=0.0)
        np.testing.assert_allclose(depth, self.mp)

    def test_mediapipe_confidence_caps_its_part(self):
        _depth, conf, _source = rtmw3d.blend_depth(self.own, self.vis, self.mp, np.full(33, 0.2), share=0.0)
        np.testing.assert_allclose(conf, 0.2)

    def test_one_missing_depth_leaves_the_other(self):
        depth, conf, source = rtmw3d.blend_depth(self.own, self.vis, None, None, share=0.1)
        np.testing.assert_allclose(depth, self.own)
        np.testing.assert_allclose(conf, 0.8 * rtmw3d.DEPTH_TRUST)
        self.assertEqual(source, 'rtmw3d')
        depth, conf, source = rtmw3d.blend_depth(None, self.vis, self.mp, np.ones(33), share=0.1)
        np.testing.assert_allclose(depth, self.mp)
        self.assertEqual(source, 'mediapipe')
        depth, conf, source = rtmw3d.blend_depth(None, self.vis, None, None)
        self.assertIsNone(depth)
        self.assertEqual(source, 'missing')
        self.assertFalse(np.any(conf))


if __name__ == "__main__":
    unittest.main()
