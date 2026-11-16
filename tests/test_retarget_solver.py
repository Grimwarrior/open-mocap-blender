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


class HipJoints(unittest.TestCase):
    """Legs solved from hip joints read off the posed rig, and held feet kept on the floor."""

    @classmethod
    def setUpClass(cls):
        truth = bench.true_motion(np.arange(0, 6.0, 1 / 30.0))
        cls.P = bench.tracked(truth, np.random.default_rng(0))

    def test_measured_hip_joints_are_used_as_given(self):
        n = len(self.P)
        tilt = np.radians(20.0)                                  # the pelvis rolled about the forward axis
        left = np.tile([bench.HIP_HALF * np.cos(tilt), 0.0, bench.HIP_HALF * np.sin(tilt)], (n, 1))
        solved = bench.solve(self.P, 30.0, hip_offsets={"Left": left, "Right": -left})
        apart = solved["Left"]["joint"] - solved["Right"]["joint"]
        self.assertLess(np.abs(apart - 2 * left).max(), 1e-6)
        for side in ("Left", "Right"):                           # and the feet are still held from there
            ankle, down = solved[side]["ankle"], solved["down"][side]
            self.assertTrue(np.isfinite(ankle).all())
            self.assertGreater(down.mean(), 0.2)

    def test_held_feet_do_not_sink_under_the_floor(self):
        solved = bench.solve(self.P, 30.0)
        for side in ("Left", "Right"):
            low = solved["floor"] - solved[side]["ankle"][solved["down"][side], 2]
            self.assertLess(low.max(), 0.003)


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

    def test_frames_without_depth_still_turn_the_knee(self):
        # The first 30 frames have no depth from other views (confidence 0 -> floor), while the
        # knee turns from -y to +x; the held direction of frame 30 must not be copied back over them
        n = 60
        along = np.tile([0.0, 0.0, -1.0], (n, 1))
        turn = np.radians(np.clip(np.arange(n) * 3.0, 0.0, 90.0))
        offset = np.stack([0.15 * np.sin(turn), -0.15 * np.cos(turn), np.full(n, -0.4)], axis=1)
        hint = np.tile([0.0, -1.0, 0.0], (n, 1))
        conf = np.where(np.arange(n) < 30, R.DIRECTION_DEPTH_FLOOR, 1.0)
        way = R.bend_direction(offset, along, 0.45, hint, 30.0, conf)
        true = R._unit(offset - offset * np.array([0.0, 0.0, 1.0]))
        held = R.bend_direction(offset, along, 0.45, hint, 30.0, np.where(np.arange(n) < 30, 0.0, 1.0))
        err = lambda w: np.degrees(np.arccos(np.clip(np.sum(w[:30] * true[:30], axis=1), -1, 1)))
        self.assertLess(float(err(way).max()), 10.0)
        self.assertGreater(float(err(held).max()), 45.0)

    def test_depth_floor_counts_visible_frames_without_depth(self):
        import json
        meta = {"frames": [1.0, 2.0, 3.0], "fps": 30.0, "visibility": [[1.0] * 40, [1.0] * 40, [0.1] * 40],
                "metadata": [{"depth_confidence": [0.0] * 40}, {"depth_confidence": [1.0] * 40},
                             {"depth_confidence": [0.0] * 40}]}

        class Root:
            def get(self, key, default=None):
                return json.dumps(meta) if key == "open_mocap_metadata" else default

        class Empty:
            parent = Root()

        frames = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(R.landmark_confidence([Empty()], frames, (25,)), [0.0, 1.0, 0.0])
        np.testing.assert_allclose(R.landmark_confidence([Empty()], frames, (25,), 0.3), [0.3, 1.0, 0.1])


def heading(v):
    return np.degrees(np.arctan2(v[:, 1], v[:, 0]))


class FootYaw(unittest.TestCase):
    n = 40

    def foot(self, degrees, length=0.2):
        a = np.radians(degrees)
        heel = np.zeros((self.n, 3))
        return heel, heel + np.array([length * np.cos(a), length * np.sin(a), -0.05])

    def knee(self, degrees):
        a = np.radians(degrees)
        return np.tile([np.cos(a), np.sin(a), 0.0], (self.n, 1))

    def test_foot_turned_far_from_the_knee_is_brought_back_to_the_limit(self):
        heel, toe = self.foot(-134.0)
        out = R.limited_toe(heel, toe, self.knee(-108.0), 10.0, 30.0)
        self.assertAlmostEqual(float(heading(out - heel)[20]), -118.0, places=3)

    def test_foot_inside_the_limit_is_untouched(self):
        heel, toe = self.foot(-100.0)
        out = R.limited_toe(heel, toe, self.knee(-108.0), 10.0, 30.0)
        np.testing.assert_allclose(out, toe, atol=1e-9)

    def test_turning_keeps_the_foot_length_and_height(self):
        heel, toe = self.foot(-160.0)
        out = R.limited_toe(heel, toe, self.knee(-108.0), 10.0, 30.0)
        self.assertAlmostEqual(float(np.linalg.norm(out[5] - heel[5])), float(np.linalg.norm(toe[5] - heel[5])), places=9)
        self.assertAlmostEqual(float(out[5, 2]), -0.05, places=9)

    def test_headings_across_the_plus_minus_180_seam(self):
        heel, toe = self.foot(-170.0)                                           # 10 degrees past +180 ...
        out = R.limited_toe(heel, toe, self.knee(174.0), 5.0, 30.0)             # ... of a knee at +174
        diff = (heading(out - heel)[20] - 174.0 + 180) % 360 - 180
        self.assertAlmostEqual(float(diff), 5.0, places=3)


class SpineAndNeck(unittest.TestCase):
    n = 5

    def frame(self, lean_deg):
        """Mocap chest frame of a torso leaning lean_deg forward (forward = -y, left = +x)."""
        a = np.radians(lean_deg)
        up = np.tile([0.0, -np.sin(a), np.cos(a)], (self.n, 1))
        lat = np.tile([1.0, 0.0, 0.0], (self.n, 1))
        return lat, up, np.cross(lat, up)

    def lean(self, v):
        return np.degrees(np.arctan2(-v[:, 1], v[:, 2]))

    def test_neutral_neck_gives_the_rest_neck(self):
        frame = self.frame(10.0)
        a = np.radians(10.0 + R.NECK_NEUTRAL_DEG)
        neck = np.tile([0.0, -np.sin(a), np.cos(a)], (self.n, 1))
        out = R.neck_direction(neck, frame, (0.0, 1.0, 0.0))
        np.testing.assert_allclose(self.lean(out), 10.0, atol=1e-9)        # rest neck: along the torso

    def test_neck_takes_its_share_of_a_lean_beyond_neutral(self):
        frame = self.frame(0.0)
        a = np.radians(R.NECK_NEUTRAL_DEG + 40.0)
        neck = np.tile([0.0, -np.sin(a), np.cos(a)], (self.n, 1))
        out = R.neck_direction(neck, frame, (0.0, 1.0, 0.0), share=0.5)
        np.testing.assert_allclose(self.lean(out), 20.0, atol=1e-9)

    def test_sideways_neck_tilt_turns_the_neck_sideways(self):
        frame = self.frame(0.0)
        tilt = np.radians(30.0)
        neck = np.tile([np.sin(tilt), 0.0, np.cos(tilt)], (self.n, 1))
        out = R.neck_direction(neck, frame, (0.0, 1.0, 0.0), neutral_deg=0.0, share=0.5)
        np.testing.assert_allclose(np.degrees(np.arcsin(out[:, 0])), 15.0, atol=1e-6)

    def test_spine_keeps_its_rest_curve_and_the_upright_actor_gives_the_rest_pose(self):
        n = self.n
        P = {R.HIP_C: np.zeros((n, 3)), R.SHOULDER_C: np.tile([0.0, 0.0, 1.0], (n, 1)),
             R.L_HIP: np.tile([0.2, 0.0, 0.0], (n, 1)), R.R_HIP: np.tile([-0.2, 0.0, 0.0], (n, 1)),
             R.L_SHOULDER: np.tile([0.3, 0.0, 1.0], (n, 1)), R.R_SHOULDER: np.tile([-0.3, 0.0, 1.0], (n, 1)),
             R.EAR_C: np.tile([0.0, -np.sin(np.radians(R.NECK_NEUTRAL_DEG)) * 0.3,
                               1.0 + np.cos(np.radians(R.NECK_NEUTRAL_DEG)) * 0.3], (n, 1))}
        back = np.radians(-7.0)
        rests = {"Spine": (0.0, np.cos(back), np.sin(back)), "Spine2": (0.0, 1.0, 0.0), "Neck": (0.0, 1.0, 0.0)}
        saved = R._true, R.landmark_confidence
        R._true = lambda _e, idx, _f, _m: {i: P[i].copy() for i in idx}
        R.landmark_confidence = lambda _e, f, _i, *_floor: np.ones(len(f))
        try:
            out = R.spine_directions(None, np.arange(n, dtype=float), 30.0, False, rests)
        finally:
            R._true, R.landmark_confidence = saved
        np.testing.assert_allclose(self.lean(out["Spine"][1]), -7.0, atol=1e-6)    # leans back as at rest
        np.testing.assert_allclose(self.lean(out["Spine2"][1]), 0.0, atol=1e-6)
        np.testing.assert_allclose(self.lean(out["Neck"][1]), 0.0, atol=1e-6)      # head over the chest


# SOMA's rest skeleton as GEM-X gives it (metres, Y up, facing +Z, left +X): a gently curved spine
SOMA_REST = {"Hips": (0.0, 0.0, 0.0), "Spine1": (0.0, 0.051, 0.001), "Spine2": (0.0, 0.124, -0.001),
             "Chest": (0.0, 0.195, -0.007), "Neck1": (0.0, 0.45, -0.012), "Head": (0.0, 0.578, 0.028),
             "LeftLeg": (0.114, -0.082, 0.023), "RightLeg": (-0.114, -0.082, 0.022),
             "LeftArm": (0.158, 0.418, -0.01), "RightArm": (-0.158, 0.418, -0.01)}
RIG_RESTS = {"Hips": (0.0, 1.0, 0.0), "Spine": (0.0, np.cos(np.radians(-7)), np.sin(np.radians(-7))),
             "Spine1": (0.0, 1.0, 0.0), "Spine2": (0.0, 1.0, 0.05), "Neck": (0.0, 1.0, 0.1)}


class GemxSpine(unittest.TestCase):
    n = 3

    def take(self, yaw_deg, bend=None):
        """The rest skeleton stood in the take's Z-up space turned by yaw (facing -Y at 0), as (N, 3)
        chain, hip and shoulder lines; bend = (segment's top joint, degrees forward) bends the chain
        above that joint about the left-right axis."""
        rest = {k: np.array(v) for k, v in SOMA_REST.items()}
        to_take = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # Y up -> Z up, +Z face -> -Y
        y = np.radians(yaw_deg)
        turn = np.array([[np.cos(y), -np.sin(y), 0.0], [np.sin(y), np.cos(y), 0.0], [0.0, 0.0, 1.0]])
        M = turn @ to_take
        pos = {k: M @ v for k, v in rest.items()}
        if bend:
            pivot_name, deg = bend
            names = list(R.SPINE_SEGMENTS)
            order = ["Hips", "Spine1", "Spine2", "Chest", "Neck1", "Head"]
            pivot = pos[pivot_name]
            a = np.radians(deg)
            lat = M @ np.array([1.0, 0.0, 0.0])
            K = np.array([[0, -lat[2], lat[1]], [lat[2], 0, -lat[0]], [-lat[1], lat[0], 0]])
            rot = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
            for k in order[order.index(pivot_name) + 1:] + ["LeftArm", "RightArm"]:
                pos[k] = pivot + rot @ (pos[k] - pivot)
        chain = {k: np.tile(pos[k], (self.n, 1)) for k in ("Hips", "Spine1", "Spine2", "Chest", "Neck1", "Head")}
        hips = np.tile(pos["LeftLeg"] - pos["RightLeg"], (self.n, 1))
        shoulders = np.tile(pos["LeftArm"] - pos["RightArm"], (self.n, 1))
        return chain, hips, shoulders, rest, M

    def rig_rest(self, M, short):
        """The rig bone's rest direction on a body frame from the actor's own hip line, as the
        retarget builds it (SOMA's rest hips are not quite level in depth)."""
        line = M @ (np.array(SOMA_REST["LeftLeg"]) - np.array(SOMA_REST["RightLeg"]))
        lat, up, fwd = R._body_frame(line[None], np.array([[0.0, 0.0, 1.0]]))
        return R._unit(R._from_components(RIG_RESTS[short], (lat, up, fwd)))[0]

    def test_actor_standing_as_at_rest_gives_the_rig_rest_spine(self):
        for yaw in (0.0, 70.0, -150.0):
            chain, hips, shoulders, rest, M = self.take(yaw)
            out = R.spine_segment_directions(chain, hips, shoulders, rest, np.array([0.0, 1.0, 0.0]), RIG_RESTS)
            self.assertEqual(set(out), set(RIG_RESTS))
            for short, d in out.items():
                np.testing.assert_allclose(d, np.tile(self.rig_rest(M, short), (self.n, 1)), atol=1e-9)

    def test_a_bent_upper_back_bends_only_the_bones_above_it(self):
        chain, hips, shoulders, rest, M = self.take(30.0, bend=("Chest", 25.0))
        out = R.spine_segment_directions(chain, hips, shoulders, rest, np.array([0.0, 1.0, 0.0]), RIG_RESTS)
        angle = lambda short: float(np.degrees(np.arccos(np.clip(out[short][0] @ self.rig_rest(M, short), -1, 1))))
        for short in ("Hips", "Spine", "Spine1"):
            self.assertLess(angle(short), 1e-6)                # below the bend: rest as before
        for short in ("Spine2", "Neck"):
            self.assertAlmostEqual(angle(short), 25.0, places=4)


class Head(unittest.TestCase):
    def test_hidden_face_turns_with_the_body(self):
        # The actor turns 267 deg, facing the chest's forward all along; the face is unseen while
        # the body turns 195 deg of it. Filled in the world, the head took the short way round.
        n = 90
        turn = np.radians(np.arange(n) * 3.0)
        fwd = np.stack([np.sin(turn), -np.cos(turn), np.zeros(n)], axis=1)
        lat = np.stack([np.cos(turn), np.sin(turn), np.zeros(n)], axis=1)
        up = np.tile([0.0, 0.0, 1.0], (n, 1))
        head = np.tile([0.0, 0.0, 1.6], (n, 1))
        P = {R.HIP_C: np.tile([0.0, 0.0, 1.0], (n, 1)), R.SHOULDER_C: np.tile([0.0, 0.0, 1.45], (n, 1)),
             R.L_SHOULDER: np.tile([0.0, 0.0, 1.45], (n, 1)) + 0.2 * lat,
             R.R_SHOULDER: np.tile([0.0, 0.0, 1.45], (n, 1)) - 0.2 * lat,
             R.EAR_C: head, R.L_EAR: head + 0.08 * lat, R.R_EAR: head - 0.08 * lat, R.NOSE: head + 0.1 * fwd,
             R.EYE_L: head + 0.03 * lat + 0.08 * fwd + 0.03 * up, R.EYE_R: head - 0.03 * lat + 0.08 * fwd + 0.03 * up,
             R.MOUTH_L: head + 0.02 * lat + 0.08 * fwd - 0.05 * up, R.MOUTH_R: head - 0.02 * lat + 0.08 * fwd - 0.05 * up}
        hidden = (np.arange(n) >= 15) & (np.arange(n) <= 80)
        frames = np.arange(n, dtype=float)
        saved = R.local_location, R.landmark_confidence
        R.local_location = lambda e, f: P[e][int(f)]
        R.landmark_confidence = lambda _e, f, _i, *_floor: np.where(hidden, 0.0, 1.0)
        try:
            _, h_up, h_fwd = R.head_directions(list(range(40)), frames, 30.0)
        finally:
            R.local_location, R.landmark_confidence = saved
        off = np.degrees(np.arccos(np.clip(np.sum(h_fwd * fwd, axis=1), -1.0, 1.0)))
        self.assertLess(float(off.max()), 10.0)
        np.testing.assert_allclose(h_up, up, atol=1e-6)


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
