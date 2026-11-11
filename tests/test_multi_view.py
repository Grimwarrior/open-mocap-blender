"""More than two views: further cameras must help, and views that do not fit must be left out."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import combine  # noqa: E402
import synthetic_combine  # noqa: E402


def hidden_joint_error(frames, take):
    """Width/height error (% of body height) of the joints the front view cannot see."""
    c = np.asarray(frames).reshape(len(frames), 40, 3)
    t = take["truth"][:len(c)] * synthetic_combine.UNIT
    hidden = take["visF"][:len(c), :33] < 0.5
    error = np.hypot(c[:, :33, 0] - t[:, :33, 0], -c[:, :33, 1] - t[:, :33, 2])[hidden]
    return float(error.mean() / 5.95 * 100)


class MultiView(unittest.TestCase):
    def test_one_view_is_the_two_video_combine(self):
        take = synthetic_combine.make_take(seed=0)
        frames, report = combine.combine_multi(take["front"], take["views"], take["tF"], take["visF"])
        expected, _ = combine.combine_improved(take["front"], take["side"], take["tF"], take["tS"],
                                               take["visF"], take["visS"])
        self.assertEqual(frames, expected)
        self.assertEqual(len(report["views"]), 1)

    def test_second_side_view_lowers_the_depth_error(self):
        two, three = [], []
        for seed in range(3):
            take = synthetic_combine.make_take(seed=seed, extra_angles=(270.0,))
            two.append(synthetic_combine.evaluate(
                combine.combine_multi(take["front"], take["views"][:1], take["tF"], take["visF"])[0], take)["depth"])
            frames, report = combine.combine_multi(take["front"], take["views"], take["tF"], take["visF"])
            three.append(synthetic_combine.evaluate(frames, take)["depth"])
            self.assertIn("depth", report["views"][1]["use"])
            self.assertAlmostEqual(abs(report["views"][1]["angle"]), 90.0, delta=8.0)
        self.assertLess(np.mean(three), 0.9 * np.mean(two))

    def test_opposite_view_places_hidden_joints(self):
        take = synthetic_combine.make_take(seed=0, extra_angles=(180.0,))
        before = hidden_joint_error(combine.combine_multi(take["front"], take["views"][:1], take["tF"], take["visF"])[0], take)
        frames, report = combine.combine_multi(take["front"], take["views"], take["tF"], take["visF"])
        self.assertEqual(report["views"][1]["use"], "hidden joints")
        self.assertLess(hidden_joint_error(frames, take), 0.8 * before)

    def test_opposite_view_of_another_performance_is_left_out(self):
        take = synthetic_combine.make_take(seed=0, separate=True, swap_bursts=1, offset=0.2, extra_angles=(180.0,))
        frames, report = combine.combine_multi(take["front"], take["views"], take["tF"], take["visF"], sync="WARP")
        expected, _ = combine.combine_multi(take["front"], take["views"][:1], take["tF"], take["visF"], sync="WARP")
        self.assertEqual(report["views"][1]["use"], "left out")
        self.assertTrue(report["views"][1]["reason"])
        self.assertLess(np.abs(np.asarray(frames) - np.asarray(expected)).max(), 0.05)

    def test_a_working_view_stands_in_for_a_useless_side_video(self):
        take = synthetic_combine.make_take(seed=0, side_angle=175.0, swap_bursts=0, extra_angles=(90.0,))
        frames, report = combine.combine_multi(take["front"], take["views"], take["tF"], take["visF"])
        self.assertTrue(report["combined"])
        self.assertIn("depth", report["views"][1]["use"])
        self.assertLess(synthetic_combine.evaluate(frames, take)["depth"], 2.5)

    def test_no_view_with_depth_is_refused(self):
        take = synthetic_combine.make_take(seed=0, side_angle=175.0, swap_bursts=0, extra_angles=(5.0,))
        _frames, report = combine.combine_multi(take["front"], take["views"], take["tF"], take["visF"])
        self.assertFalse(report["combined"])
        self.assertEqual(len(report["views"]), 2)


class RealCameras(unittest.TestCase):
    """Cameras with perspective: nearer is bigger and, below the camera, lower in the picture."""

    def test_perspective_is_not_mistaken_for_swapped_legs(self):
        # Each camera sees the foot nearer to it lower, so two cameras disagree about which foot
        # is lower exactly as a left/right label swap would. No labels are swapped here.
        for distance in (5.0, 3.5):
            take = synthetic_combine.make_take(seed=0, swap_bursts=0, camera_distance=distance, extra_angles=(270.0,))
            frames, report = combine.combine_multi(take["front"], take["views"], take["tF"], take["visF"])
            self.assertEqual(report["lr_switches"]["legs"], 0)
            self.assertEqual(report["lr_switches"]["arms"], 0)
            self.assertIn("depth", report["views"][1]["use"])
            self.assertLess(synthetic_combine.evaluate(frames, take)["depth"], 5.0)


class SkeletonSolve(unittest.TestCase):
    """Bone lengths kept by solving all joints together instead of clipping bone by bone."""

    def errors(self, solve, **take_args):
        saved = combine.SKELETON_SOLVE
        combine.SKELETON_SOLVE = solve
        try:
            rows = []
            for seed in range(3):
                take = synthetic_combine.make_take(seed=seed, **take_args)
                frames, _ = combine.combine_improved(take["front"], take["side"], take["tF"], take["tS"],
                                                     take["visF"], take["visS"])
                rows.append(synthetic_combine.evaluate(frames, take))
            return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        finally:
            combine.SKELETON_SOLVE = saved

    def test_steadier_than_clipping_and_no_less_accurate(self):
        for args, slack in ((dict(), 0.95), (dict(noise=0.012, swap_bursts=4), 1.0), (dict(camera_distance=5.0), 1.03)):
            clip, solve = self.errors(False, **args), self.errors(True, **args)
            self.assertLess(solve["depth"], slack * clip["depth"], args)
            self.assertLess(solve["jitter"], 0.9 * clip["jitter"], args)
            self.assertGreater(solve["strike"], clip["strike"] - 3.0, args)

    def test_strength_zero_changes_nothing(self):
        take = synthetic_combine.make_take(seed=0)
        F = combine._as_array(take["front"])
        D = F[:, :, 2].copy()
        self.assertIs(combine._solve_skeleton(F, D, np.ones(D.shape), 0.0), D)

    def test_hips_keep_their_depth(self):
        take = synthetic_combine.make_take(seed=0)
        saved = combine.SKELETON_SOLVE
        combine.SKELETON_SOLVE = False
        try:
            frames, _ = combine.combine_improved(take["front"], take["side"], take["tF"], take["tS"],
                                                 take["visF"], take["visS"], bone_strength=0.0)
        finally:
            combine.SKELETON_SOLVE = saved
        P = combine._as_array(frames)
        D = P[:, :, 2].copy()
        out = combine._solve_skeleton(P, D, np.ones(D.shape), 1.0)
        self.assertLess(float(np.abs(out[:, combine.HIP_C] - D[:, combine.HIP_C]).max()), 0.05)


if __name__ == "__main__":
    unittest.main()
