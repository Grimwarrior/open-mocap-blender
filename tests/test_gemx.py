"""GEM-X takes as Open Mocap captures (gemx.py): camera axes, point mapping, face/heel points, 3D hands."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gemx  # noqa: E402

FINGERS = ["Thumb1", "Thumb2", "Thumb3", "ThumbEnd"] + [f"{f}{p}" for f in ("Index", "Middle", "Ring", "Pinky")
                                                        for p in ("1", "2", "3", "4", "End")]
NAMES = (["Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "HeadEnd", "Jaw", "LeftEye", "RightEye"]
         + [f"{s}{n}" for s in ("Left", "Right") for n in ["Shoulder", "Arm", "ForeArm", "Hand"] + [f"Hand{x}" for x in FINGERS]]
         + [f"{s}{n}" for s in ("Left", "Right") for n in ("Leg", "Shin", "Foot", "ToeBase", "ToeEnd")])
LEVEL_CAMERA = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])   # looks along world -Z


def actor(T=4, facing=(0.0, 0.0, 1.0)):
    """(T, 77, 3) joints of a person standing at the world origin (Y up), facing `facing` (toward
    the level camera by default), left = up x facing."""
    f = np.asarray(facing, float) / np.linalg.norm(facing)
    up = np.array([0.0, 1.0, 0.0])
    left = np.cross(up, f)
    P = {"Hips": 1.0 * up, "Head": 1.65 * up, "HeadEnd": 1.82 * up, "Jaw": 1.62 * up + 0.03 * f,
         "LeftEye": 1.66 * up + 0.09 * f + 0.03 * left, "RightEye": 1.66 * up + 0.09 * f - 0.03 * left}
    for s, k in (("Left", 1.0), ("Right", -1.0)):
        L = k * left
        P.update({s + "Shoulder": 1.45 * up + 0.05 * L, s + "Arm": 1.42 * up + 0.18 * L, s + "ForeArm": 1.15 * up + 0.2 * L,
                  s + "Hand": 0.9 * up + 0.2 * L, s + "Leg": 0.95 * up + 0.1 * L, s + "Shin": 0.5 * up + 0.1 * L,
                  s + "Foot": 0.08 * up + 0.1 * L, s + "ToeBase": 0.12 * f + 0.1 * L, s + "ToeEnd": 0.18 * f + 0.1 * L})
        for i, x in enumerate(FINGERS):
            P[s + "Hand" + x] = P[s + "Hand"] - (0.02 + 0.01 * (i % 5)) * up + 0.01 * (i // 5) * f
    J = np.stack([P.get(n, P["Hips"]) for n in NAMES])
    return np.repeat(J[None], T, axis=0)


def take(J, R=LEVEL_CAMERA):
    T = len(J)
    return {"joints_global": J, "joint_names": np.array(["Root"] + NAMES), "R_w2c": np.repeat(R[None], T, axis=0),
            "frame_times": np.arange(T) / 30.0, "fps": np.array(30.0), "width": np.array(1920), "height": np.array(1080)}


class CameraAxes(unittest.TestCase):
    def test_level_camera_keeps_its_axes(self):
        right, down, fwd = gemx.camera_axes(LEVEL_CAMERA)
        np.testing.assert_allclose(right, [1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(down, [0, -1, 0], atol=1e-12)
        np.testing.assert_allclose(fwd, [0, 0, -1], atol=1e-12)

    def test_tilted_camera_is_levelled_and_yaw_kept(self):
        a, yaw = np.radians(25.0), np.radians(30.0)
        pitch = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        turn = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        R = pitch @ LEVEL_CAMERA @ turn                    # world -> camera: turned, then tilted down
        right, down, fwd = gemx.camera_axes(R)
        np.testing.assert_allclose(down, [0, -1, 0], atol=1e-12)
        self.assertAlmostEqual(float(fwd[1]), 0.0)
        np.testing.assert_allclose(np.cross(down, fwd), right, atol=1e-12)          # right-handed
        level = gemx.camera_axes(LEVEL_CAMERA @ turn)[2]
        np.testing.assert_allclose(fwd, level, atol=1e-12)                           # the tilt is gone


class Capture(unittest.TestCase):
    def setUp(self):
        self.J = actor()
        self.cap = gemx.to_capture(take(self.J))
        self.P = np.asarray(self.cap["frames"]).reshape(-1, 40, 3)
        self.idx = {n: i for i, n in enumerate(NAMES)}

    def test_joints_land_on_their_points_in_tracker_units(self):
        # level camera looking along -Z: x right = world x, y down = -height, depth = -z
        for k, n in ((11, "LeftArm"), (16, "RightHand"), (25, "LeftShin"), (28, "RightFoot")):
            w = self.J[0, self.idx[n]]
            np.testing.assert_allclose(self.P[0, k], np.array([w[0], -w[1], -w[2]]) * gemx.UNIT, atol=1e-9)
        np.testing.assert_allclose(self.P[0, 33], (self.P[0, 23] + self.P[0, 24]) / 2)
        self.assertEqual(len(self.cap["times"]), 4)
        self.assertEqual(len(self.cap["hands"]["Left"][0]), 63)

    def test_actor_facing_the_camera_has_his_left_on_the_picture_right(self):
        self.assertGreater(self.P[0, 11, 0], self.P[0, 12, 0])        # left shoulder at larger x
        self.assertLess(self.P[0, 0, 1], self.P[0, 33, 1])            # nose above the hips (y down)

    def test_face_and_heel_points(self):
        P = self.P[0]
        ears = (P[7] + P[8]) / 2
        self.assertLess(P[0, 2], ears[2])                             # nose nearer the camera than the ears
        self.assertGreater(P[7, 0], P[8, 0])                          # left ear on the left
        self.assertGreater(P[9, 1], P[0, 1])                          # mouth below the nose
        for heel, ankle, toe in ((29, 27, 31), (30, 28, 32)):
            self.assertGreater(P[heel, 1], P[ankle, 1])               # heel below the ankle
            self.assertGreater(P[heel, 2], P[toe, 2])                 # and behind the toes

    def test_foot_is_rigid_whatever_the_toes_do(self):
        # heel under the ankle, toe point on the heel -> ball line: turning the toe tips does not turn it
        J = actor()
        k = self.idx["LeftToeEnd"]
        J2 = J.copy()
        J2[:, k] = J[:, self.idx["LeftToeBase"]] + np.array([0.05, 0.0, 0.02])      # toes splayed sideways
        P2 = np.asarray(gemx.to_capture(take(J2))["frames"]).reshape(-1, 40, 3)
        np.testing.assert_allclose(P2[:, [29, 31]], self.P[:, [29, 31]], atol=1e-12)
        heel, ankle, toe = self.P[0, 29], self.P[0, 27], self.P[0, 31]
        self.assertAlmostEqual(float(heel[1] - ankle[1]), gemx.HEEL_BELOW * gemx.UNIT, delta=0.02 * gemx.UNIT)
        w = self.J[0, self.idx["LeftToeBase"]]
        ball = np.array([w[0], -w[1], -w[2]]) * gemx.UNIT                       # level camera: x, -height, -z
        normal = (ankle - heel) / (gemx.HEEL_BELOW * gemx.UNIT)                  # the foot's up, in the picture
        sole_ball = ball - gemx.BALL_ABOVE_SOLE * gemx.UNIT * normal
        cross = np.cross(sole_ball - heel, toe - heel)
        self.assertLess(float(np.linalg.norm(cross)), 1e-9)                      # heel, sole and toe on one line

    def test_foot_pointed_along_the_shin_keeps_its_heel_side(self):
        T = 30
        J = actor(T=T)
        ankle = self.idx["RightFoot"]
        ball = self.idx["RightToeBase"]
        pointed = slice(12, 18)                                    # toes straight down, in line with the shin
        J[pointed, ball] = J[pointed, ankle] + np.array([0.0, -0.13, -0.003])          # a touch behind
        P = np.asarray(gemx.to_capture(take(J))["frames"]).reshape(T, 40, 3)
        # the heel stays on the back of the foot (farther from the camera than the ankle), never in front
        self.assertTrue(np.all(P[:, 30, 2] >= P[:, 28, 2] - 1e-6))

    def test_actor_facing_away_mirrors_left_and_right_in_the_picture(self):
        P = np.asarray(gemx.to_capture(take(actor(facing=(0.0, 0.0, -1.0))))["frames"]).reshape(-1, 40, 3)
        self.assertLess(P[0, 11, 0], P[0, 12, 0])
        self.assertGreater(P[0, 0, 2], ((P[0, 7] + P[0, 8]) / 2)[2])   # nose farther than the ears


class WorldFromCameraJoints(unittest.TestCase):
    def test_rotation_comes_from_the_two_joint_sets_not_the_identity_camera(self):
        # GEM-X's fixed camera has R_w2c = identity; the camera-space joints tell where the camera is
        a, yaw = np.radians(10.0), np.radians(-40.0)
        pitch = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        turn = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        R = pitch @ LEVEL_CAMERA @ turn
        J = actor(T=5)
        J = J + np.linspace(0, 1, 5)[:, None, None] * np.array([0.3, 0.0, 0.5])      # walking
        incam = J @ R.T + np.array([0.0, 0.2, 4.0]) + np.linspace(0, 1, 5)[:, None, None] * 0.7
        np.testing.assert_allclose(gemx.world_to_camera(J, incam), R, atol=1e-9)
        data = take(J, np.eye(3))
        data["joints_incam"] = incam
        P = np.asarray(gemx.to_capture(data)["frames"]).reshape(-1, 40, 3)
        expected = np.asarray(gemx.to_capture(take(J, R))["frames"]).reshape(-1, 40, 3)
        np.testing.assert_allclose(P, expected, atol=1e-9)


class Floor(unittest.TestCase):
    def test_drifting_floor_is_levelled_where_feet_stand(self):
        T = 60
        J = actor(T=T)
        drift = np.linspace(0.0, 0.2, T)                          # GEM-X's world rising 0.2 m
        J[:, :, 1] += drift[:, None]
        jump = slice(25, 35)                                       # both feet off the floor here
        J[jump, :, 1] += 0.3
        static = np.ones((T, 6))
        static[jump, :4] = 0.0
        data = take(J)
        data["static_conf"] = static
        capture = gemx.to_capture(data)
        np.testing.assert_array_equal([row["foot_contact"] for row in capture["meta"]], static[:, :4])
        P = np.asarray(capture["frames"]).reshape(T, 40, 3) / gemx.UNIT
        low = -P[:, [29, 30, 31, 32], 1].max(axis=1)              # lowest foot point's height (y down)
        stand = np.ones(T, bool)
        stand[jump] = False
        stand[20:40] = False                                       # the smoothing reaches a little around it
        self.assertLess(float(np.abs(low[stand]).max()), 0.01)
        self.assertGreater(float(low[30]), 0.25)                   # the jump keeps its height

    def test_each_standing_foot_is_put_on_the_floor(self):
        # GEM-X with the right foot 3 cm above the left while both stand: both end on the floor, the
        # hips where they were, the knee a little lower
        T = 20
        J = actor(T=T)
        idx = {n: i for i, n in enumerate(NAMES)}
        for n in ("RightFoot", "RightToeBase", "RightToeEnd"):
            J[:, idx[n], 1] += 0.03
        data = take(J)
        data["static_conf"] = np.ones((T, 6))
        P = np.asarray(gemx.to_capture(data)["frames"]).reshape(T, 40, 3) / gemx.UNIT
        low = lambda a, b: -P[:, [a, b], 1].max(axis=1)              # lowest of heel and toe (y down)
        np.testing.assert_allclose(low(29, 31), 0.0, atol=1e-3)
        np.testing.assert_allclose(low(30, 32), 0.0, atol=1e-3)
        plain = np.asarray(gemx.to_capture(take(J))["frames"]).reshape(T, 40, 3) / gemx.UNIT
        # the right hip stays where it is against the left one; the right knee comes down (y down)
        np.testing.assert_allclose(P[:, 24] - P[:, 23], plain[:, 24] - plain[:, 23], atol=1e-9)
        self.assertGreater(float((P[0, 26, 1] - P[0, 25, 1]) - (plain[0, 26, 1] - plain[0, 25, 1])), 0.01)

    def test_pinning_holds_through_a_step(self):
        T = 40
        J = actor(T=T)
        idx = {n: i for i, n in enumerate(NAMES)}
        step = slice(15, 25)                                         # the right foot swings 10 cm up
        for n in ("RightFoot", "RightToeBase", "RightToeEnd"):
            J[:, idx[n], 1] += 0.03
            J[step, idx[n], 1] += 0.10
        static = np.ones((T, 6))
        static[step, 2:4] = 0.0
        data = take(J)
        data["static_conf"] = static
        P = np.asarray(gemx.to_capture(data)["frames"]).reshape(T, 40, 3) / gemx.UNIT
        right = -P[:, [30, 32], 1].max(axis=1)
        self.assertAlmostEqual(float(right[20]), 0.10, delta=0.005)  # the step keeps its height above the floor
        np.testing.assert_allclose(right[:10], 0.0, atol=1e-3)

    def test_flat_standing_foot_lies_on_the_floor_and_a_tiptoe_keeps_its_heel_up(self):
        T = 30
        J = actor(T=T)
        idx = {n: i for i, n in enumerate(NAMES)}
        J[:, idx["RightToeBase"], 1] += 0.04                     # GEM-X has the right foot tipped toe-up
        J[:, idx["RightToeEnd"], 1] += 0.05
        static = np.ones((T, 6))
        static[20:, 2] = 0.0                                      # from frame 20 the right ankle lifts: on its ball
        data = take(J)
        data["static_conf"] = static
        P = np.asarray(gemx.to_capture(data)["frames"]).reshape(T, 40, 3) / gemx.UNIT
        h = -P[..., 1]
        np.testing.assert_allclose(h[:12, [30, 32]], 0.0, atol=1e-3)          # flat: heel and toe on the floor
        self.assertGreater(float(h[28, 32] - h[28, 30]), 0.01)                   # on its ball: tipped as GEM-X has it

    def test_without_standing_feet_nothing_moves(self):
        J = actor(T=10)
        data = take(J)
        data["static_conf"] = np.zeros((10, 6))
        a = gemx.to_capture(data)["frames"]
        np.testing.assert_allclose(a, gemx.to_capture(take(J))["frames"])


class Spine(unittest.TestCase):
    def test_spine_chain_and_rest_skeleton_go_with_the_take(self):
        J = actor(T=4)
        data = take(J)
        data["joints_rest"] = J[0]
        spine = gemx.to_capture(data)["spine"]
        self.assertEqual(spine["names"], list(gemx.SPINE_CHAIN))
        self.assertEqual(np.asarray(spine["points"]).shape, (4, 3 * len(gemx.SPINE_CHAIN)))
        self.assertEqual(set(spine["rest"]), set(gemx.SPINE_CHAIN + gemx.SPINE_REST_EXTRA))
        hips = np.asarray(spine["points"])[0, :3]
        np.testing.assert_allclose(hips, np.asarray(gemx.to_capture(take(J))["frames"]).reshape(-1, 40, 3)[0, 33]
                                   * 0 + np.array([J[0, 0, 0], -J[0, 0, 1], -J[0, 0, 2]]) * gemx.UNIT, atol=1e-9)

    def test_no_rest_skeleton_no_spine(self):
        self.assertIsNone(gemx.to_capture(take(actor(T=2)))["spine"])


class Fingers3D(unittest.TestCase):
    def test_finger_curls_take_depth_and_match_flat_hands(self):
        from test_capture_batch import retarget as R          # the add-on's retarget with mocked Blender
        rng = np.random.default_rng(0)
        flat = rng.normal(size=(12, 21, 2))
        deep = np.concatenate([flat, np.zeros((12, 21, 1))], axis=2)
        frames = np.arange(12, dtype=float)
        a = R.finger_curls(flat, frames, frames, 30.0)
        b = R.finger_curls(deep, frames, frames, 30.0)
        for finger in a:
            np.testing.assert_allclose(a[finger], b[finger])


if __name__ == "__main__":
    unittest.main()
