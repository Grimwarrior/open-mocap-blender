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

    def legs(self, frames):
        c = np.asarray(frames).reshape(len(frames), 40, 3)
        b = combine.to_blender(c[:, :, :2], c[:, :, 2])
        leg = lambda a, k, e: np.linalg.norm(b[:, k] - b[:, a], axis=1) + np.linalg.norm(b[:, e] - b[:, k], axis=1)
        return leg(23, 25, 27), leg(24, 26, 28)

    def test_legs_are_solved_again_after_the_correction(self):
        # 5 m: at 4 m the left/right leg spread is a near tie once the views are synced right
        # (0.031 -> 0.033 while the depth error still drops 2.82 -> 2.74%)
        take = synthetic_combine.make_take(seed=1, camera_distance=5.0)
        frames, rep = combine.combine_improved(take["front"], take["side"], take["tF"], take["tS"],
                                               take["visF"], take["visS"])
        conf = [m["depth_confidence"] for m in rep["metadata"]]
        plain, _ = combine.correct_perspective(frames, 5.0 * UNIT, horizon=-1.3 * UNIT)
        solved, _ = combine.correct_perspective(frames, 5.0 * UNIT, horizon=-1.3 * UNIT, confidence=conf,
                                                bone_strength=0.8)
        spread = lambda f: float(np.std(np.log(np.divide(*self.legs(f)))))
        self.assertLess(spread(solved), spread(plain))
        a, b = np.asarray(plain).reshape(-1, 40, 3), np.asarray(solved).reshape(-1, 40, 3)
        np.testing.assert_allclose(a[:, :, :2], b[:, :, :2])               # only depth is solved

    def test_without_confidence_the_depth_is_unchanged(self):
        take, frames = self.combined(4.0)
        fixed, _ = combine.correct_perspective(frames, 4.0 * UNIT, horizon=-1.3 * UNIT, bone_strength=0.8)
        np.testing.assert_allclose(np.asarray(fixed).reshape(-1, 40, 3)[:, :, 2],
                                   np.asarray(frames).reshape(-1, 40, 3)[:, :, 2])

    def test_focal_from_field_of_view(self):
        # 1920 px wide, 60 degrees across: f = 960 / tan(30 deg) = 1663 px
        self.assertAlmostEqual(combine.focal_from_fov(1920, 60.0), 16.628, places=2)



def standing_take(distance, height, seed=0):
    """The retarget benchmark's actor (steps, lunges, both feet often down) filmed by a pinhole camera
    `distance` m away at `height` m, as combined frames; and the camera's focal length and horizon."""
    import synthetic_retarget as sr
    rng = np.random.default_rng(seed)
    truth = sr.true_motion(np.arange(0, 8.0, 1 / 30))
    P = sr.tracked(truth, rng, camera_distance=distance, camera_height=height)
    C = np.zeros_like(P)
    C[:, :, 0], C[:, :, 1], C[:, :, 2] = P[:, :, 0], -P[:, :, 2], P[:, :, 1]
    focal = (distance + float(np.median(truth["hips"][:, 1]))) * sr.UNIT
    return C.reshape(len(C), -1).tolist(), focal, -height * sr.UNIT


def floor_gap(frames, fps=30.0):
    """Mean height difference of the feet's lowest points where both stand (as fit_horizon finds them)."""
    c = np.asarray(frames).reshape(len(frames), 40, 3)
    leg = float(np.median(np.linalg.norm(c[:, 25] - c[:, 23], axis=1) + np.linalg.norm(c[:, 27] - c[:, 25], axis=1)))
    both = np.ones(len(c), dtype=bool)
    low = []
    for ankle, heel, toe in combine.FOOT_POINTS:
        both &= combine._still(c[:, [ankle, heel, toe]].mean(axis=1), fps, leg)
        low.append(np.maximum(c[:, heel, 1], c[:, toe, 1]))
    return float(np.mean(np.abs(low[0][both] - low[1][both])))


class Horizon(unittest.TestCase):
    def test_horizon_of_a_high_and_a_low_camera_is_found(self):
        for distance, height in ((5.0, 1.3), (3.5, 1.0), (4.0, 1.6)):
            frames, focal, true_horizon = standing_take(distance, height)
            horizon, report = combine.fit_horizon(frames, focal, default=-5.4)
            self.assertTrue(report["horizon_found"])
            self.assertAlmostEqual(horizon, true_horizon, delta=0.15)

    def test_feet_stand_level_even_with_a_wrong_field_of_view(self):
        frames, focal, _ = standing_take(3.5, 1.0)
        for f in (focal * 0.75, focal * 1.25):
            middle = combine.correct_perspective(frames, f, horizon=-5.4)[0]
            found = combine.correct_perspective(frames, f, horizon=combine.fit_horizon(frames, f)[0])[0]
            self.assertLess(floor_gap(found), 0.05)
            self.assertLess(floor_gap(found), 0.5 * floor_gap(middle))

    def test_without_standing_feet_the_default_is_kept(self):
        take = synthetic_combine.make_take(seed=0, camera_distance=5.0)    # an actor whose feet never stand
        frames, _ = combine.combine_improved(take["front"], take["side"], take["tF"], take["tS"],
                                             take["visF"], take["visS"])
        horizon, report = combine.fit_horizon(frames, 17.0, default=-5.4)
        self.assertEqual(horizon, -5.4)
        self.assertFalse(report["horizon_found"])
        self.assertIn("horizon_reason", report)


def standing_spread(frames, fps=30.0):
    """Height spread of the lowest point of every standing foot in every frame."""
    c = np.asarray(frames).reshape(len(frames), 40, 3)
    leg = float(np.median(np.linalg.norm(c[:, 25] - c[:, 23], axis=1) + np.linalg.norm(c[:, 27] - c[:, 25], axis=1)))
    vals = [np.maximum(c[:, heel, 1], c[:, toe, 1])[combine._still(c[:, [ankle, heel, toe]].mean(axis=1), fps, leg)]
            for ankle, heel, toe in combine.FOOT_POINTS]
    return float(np.std(np.concatenate(vals)))


class WholeTake(unittest.TestCase):
    def test_floor_stays_at_one_height_as_the_actor_moves(self):
        for distance, height in ((5.0, 1.3), (3.5, 1.0)):
            frames, focal, true_horizon = standing_take(distance, height)
            per_frame = combine.correct_perspective(frames, focal, combine.fit_horizon(frames, focal)[0])[0]
            horizon, report = combine.fit_horizon(frames, focal, whole_take=True)
            whole = combine.correct_perspective(frames, focal, horizon, whole_take=True)[0]
            self.assertTrue(report["horizon_found"])
            self.assertAlmostEqual(horizon, true_horizon, delta=0.15)
            self.assertLess(standing_spread(whole), 0.05)
            self.assertLess(standing_spread(whole), 0.3 * standing_spread(per_frame))

    def test_body_keeps_one_size_and_the_hips_their_depth(self):
        take = synthetic_combine.make_take(seed=0, camera_distance=3.5)
        frames, rep = combine.combine_improved(take["front"], take["side"], take["tF"], take["tS"],
                                               take["visF"], take["visS"])
        conf = [m["depth_confidence"] for m in rep["metadata"]]
        args = (frames, 3.5 * UNIT, -1.3 * UNIT, conf, 0.8)
        per_frame, whole = combine.correct_perspective(*args)[0], combine.correct_perspective(*args, whole_take=True)[0]
        cv = lambda f: combine.bone_length_cv(combine.to_blender(np.asarray(f).reshape(-1, 40, 3)[:, :, :2],
                                                                 np.asarray(f).reshape(-1, 40, 3)[:, :, 2]))
        self.assertLess(cv(whole), 0.7 * cv(per_frame))
        scaled = combine.correct_perspective(frames, 3.5 * UNIT, -1.3 * UNIT, whole_take=True)[0]    # no leg solve
        np.testing.assert_allclose(np.asarray(scaled).reshape(-1, 40, 3)[:, combine.HIP_C, 2],
                                   np.asarray(frames).reshape(-1, 40, 3)[:, combine.HIP_C, 2], atol=1e-9)

    def test_actor_who_never_stands_keeps_the_default(self):
        take = synthetic_combine.make_take(seed=0, camera_distance=5.0)
        frames, _ = combine.combine_improved(take["front"], take["side"], take["tF"], take["tS"],
                                             take["visF"], take["visS"])
        horizon, report = combine.fit_horizon(frames, 17.0, default=-5.4, whole_take=True)
        self.assertEqual(horizon, -5.4)
        self.assertFalse(report["horizon_found"])

if __name__ == "__main__":
    unittest.main()
