"""GEM-X takes as Open Mocap captures (gemx.py): camera axes, point mapping, face/heel points, 3D hands."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gemx  # noqa: E402
import gemx_runner  # noqa: E402

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


class RunnerErrors(unittest.TestCase):
    # The last lines GEM-X printed when libx264 could not start under low memory (2026-11-24).
    TAIL = [
        "Traceback (most recent call last):",
        '    subprocess.run(command, check=True, capture_output=True, text=True)',
        r'  File "D:\mocap\envs\gemx\Lib\subprocess.py", line 571, in run',
        "    raise CalledProcessError(retcode, process.args,",
        "subprocess.CalledProcessError: Command '['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', "
        "'-i', 'C:\\clip.mp4', '-c:v', 'libx264', 'C:\\input.mp4']' returned non-zero exit status 3752568763.",
    ]

    def test_ffmpeg_codes(self):
        self.assertIn("external library", gemx_runner.ffmpeg_meaning(3752568763))
        self.assertIn("memory", gemx_runner.ffmpeg_meaning(2**32 - 12))
        self.assertIsNone(gemx_runner.ffmpeg_meaning(1))

    def test_explains_ffmpeg_failure(self):
        msg = gemx_runner.explain_failure(self.TAIL, 2.5)
        self.assertIn("0xDFABA7BB", msg)
        self.assertIn("2.5 GB", msg)
        self.assertIn("Close other Blender instances", msg)
        self.assertIsNone(gemx_runner.explain_failure(["RuntimeError: CUDA out of memory"]))

    def test_free_memory_is_readable(self):
        free = gemx_runner.free_commit_gb()
        if sys.platform == "win32":
            self.assertGreater(free, 0)


def _rot(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


class SwordFromOneVideo(unittest.TestCase):
    """gemx.sword_axis: a grip fitted to the picture where the tracker is right, the 3D blade everywhere."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(5)
        T = 120
        t = np.arange(T) / 30.0
        # the right hand swings and turns in front of the camera (camera space: x right, y down, z forward)
        frames = np.array([_rot([0.3, 1.0, 0.2], 2.2 * np.sin(1.3 * s)) @ _rot([1.0, 0.0, 0.4], 1.1 * np.sin(0.9 * s + 1))
                           for s in t])
        offset = _rot([0.0, 0.2, 1.0], 0.5)                      # the left hand's frame against the right's
        cls.grip_r = np.array([0.15, 0.95, 0.27]) / np.linalg.norm([0.15, 0.95, 0.27])
        names = ["Root"] + [s + n for s in ("Right", "Left") for n in ("Hand", "HandIndex2", "HandPinky2")]
        J = np.zeros((T, 6, 3))
        centre = np.stack([0.3 * np.sin(1.1 * t), -0.2 + 0.1 * np.cos(t), np.full(T, 3.0)], axis=1)
        for k, (side, F) in enumerate((("Right", frames), ("Left", frames @ offset))):
            wrist = centre + (0.05 if side == "Right" else -0.05) * frames[:, :, 0]
            J[:, 3 * k] = wrist
            J[:, 3 * k + 1] = wrist + 0.09 * F[:, :, 1] + 0.04 * F[:, :, 0]      # index knuckle
            J[:, 3 * k + 2] = wrist + 0.09 * F[:, :, 1] - 0.04 * F[:, :, 0]      # pinky knuckle
        K = np.array([[1400.0, 0, 960], [0, 1400.0, 540], [0, 0, 1]])
        cls.blade = np.einsum('tij,j->ti', frames, cls.grip_r)
        pic = gemx._picture_dirs(J[:, 0], cls.blade, K)
        noise = np.radians(rng.normal(0, 2.0, T))
        c, s = np.cos(noise), np.sin(noise)
        pic = np.stack([c * pic[:, 0] - s * pic[:, 1], s * pic[:, 0] + c * pic[:, 1]], axis=1)
        conf = np.full(T, 0.9)
        cls.wrong = np.zeros(T, bool)                            # the tracker is sure of a wrong blade here
        cls.wrong[20:35] = cls.wrong[60:72] = True
        turn = rng.uniform(np.radians(60), np.radians(180), cls.wrong.sum())
        c, s = np.cos(turn), np.sin(turn)
        p = pic[cls.wrong]
        pic[cls.wrong] = np.stack([c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1]], axis=1)
        cls.lost = np.zeros(T, bool)
        cls.lost[90:105] = True
        conf[cls.lost] = 0.2                                     # and blurred away here
        rows = np.concatenate([pic, conf[:, None]], axis=1).tolist()
        cls.data = {"joints_incam": J, "joints_global": J, "joint_names": np.array(names), "K": K,
                    "fps": np.array(30.0)}
        cls.rows, cls.report = gemx.sword_axis(cls.data, rows)

    def test_grip_is_found_despite_the_wrong_frames(self):
        grip = np.array(self.report["Right"]["grip"])
        truth = self.grip_r * np.array([1.0, 1.0, -1.0])       # hand_frames' third axis is along x across = -z here
        self.assertLess(np.degrees(np.arccos(np.clip(grip @ truth, -1, 1))), 5.0)
        self.assertLess(self.report["Right"]["inliers"], 0.85 * self.report["fit_frames"])   # wrong frames left out

    def test_blade_in_3d_where_the_picture_is_sure_or_lost(self):
        cap = np.array(self.rows)[:, :3]
        truth = self.blade * np.array([1.0, -1.0, 1.0])          # camera -> capture for a level camera (y down)
        err = np.degrees(np.arccos(np.clip(np.sum(cap * truth, axis=1), -1, 1)))
        good = ~self.wrong & ~self.lost
        self.assertLess(float(np.median(err[good])), 7.0)        # the picture's direction (2 deg noise), the hands' lean
        self.assertLess(float(np.median(err[self.lost])), 8.0)   # the hands alone where the picture lost it

    def test_holder_from_3d_hands_and_the_search(self):
        side = np.array([0, 1, 1, 1, -1, 0, 1])                 # the search's hand (0: both searched together)
        sure = np.array([True, True, False, True, True, True, False])
        apart = np.array([False, True, True, True, True, True, False])
        np.testing.assert_array_equal(gemx.sword_holder(side, sure, apart), [0, 1, 1, 1, -1, -1, 0])
        np.testing.assert_array_equal(gemx.sword_holder(side, np.zeros(7, bool), apart), [0, 1, 1, 1, 1, 1, 0])

    def test_picture_points_from_kp2d(self):
        T = 3
        names = ["Root"] + [n for n in NAMES]
        kp = np.zeros((T, len(NAMES), 3))
        kp[:, NAMES.index("RightHand")] = [1060.0, 440.0, 0.9]
        data = {"kp2d": kp, "joint_names": np.array(names), "width": np.array(1920), "height": np.array(1080)}
        frames, vis = gemx.picture_capture(data)
        P = np.asarray(frames).reshape(T, 40, 3)
        np.testing.assert_allclose(P[:, 16, :2], [[1.0, -6.4]] * T)                  # (px - w/2)/100, (py - h)/100
        self.assertAlmostEqual(float(vis[0, 16]), 0.9)
        self.assertIsNone(gemx.picture_capture({"joint_names": np.array(names)}))


class SecondView(unittest.TestCase):
    """The same move filmed again from 90 degrees: matched in time by the body, it gives the blade's lean."""

    BODY = ["Hips", "Chest", "LeftLeg", "RightLeg", "RightHand", "LeftHand"] + [
        n for n in gemx.POSE_POINTS if n not in ("RightHand", "LeftHand")]

    def body(self, t):
        """(T, joints, 3) a body that turns and moves its limbs (body coordinates: x left, y up, z forward)."""
        yaw = 0.6 * np.sin(0.7 * t)
        J = np.zeros((len(t), len(self.BODY), 3))
        rest = {"Hips": [0, 0, 0], "Chest": [0, 0.45, 0], "LeftLeg": [0.1, -0.05, 0], "RightLeg": [-0.1, -0.05, 0],
                "Head": [0, 0.75, 0.05], "LeftShin": [0.1, -0.5, 0.05], "RightShin": [-0.1, -0.5, 0.05],
                "LeftFoot": [0.1, -0.9, 0], "RightFoot": [-0.1, -0.9, 0]}
        for k, n in enumerate(self.BODY):
            p = np.tile(np.asarray(rest.get(n, [0, 0, 0]), float), (len(t), 1))
            if n in ("RightHand", "LeftHand", "RightForeArm", "LeftForeArm"):
                s = 1 if "Right" in n else -1
                p = np.stack([-0.15 * s + 0.2 * np.sin(2 * t + s), 0.3 + 0.25 * np.cos(1.5 * t), 0.35 + 0.1 * np.sin(t)], 1)
            J[:, k] = p
        c, s = np.cos(yaw), np.sin(yaw)
        turn = np.stack([np.stack([c, 0 * c, s], 1), np.tile([0, 1, 0], (len(t), 1)), np.stack([-s, 0 * c, c], 1)], 1)
        return np.einsum('tij,tkj->tki', turn, J), turn

    def test_warp_finds_a_known_offset(self):
        t = np.arange(150) / 30.0
        a, _ = self.body(t)
        b, _ = self.body(np.arange(160) / 30.0 - 0.2)           # the second take starts 6 frames earlier
        fa = gemx.pose_features(a, self.BODY, 30.0)
        fb = gemx.pose_features(b, self.BODY, 30.0)
        match, _ = gemx.align_takes(fa, fb)
        self.assertLess(float(np.median(np.abs(match[20:-20] - (np.arange(150)[20:-20] + 6)))), 1.0)

    def test_two_pictures_give_the_lean(self):
        T = 90
        t = np.arange(T) / 30.0
        J, turn = self.body(t)
        blade_body = gemx._unit(np.stack([0.3 + 0.5 * np.sin(t), 0.6 * np.cos(1.3 * t), 0.7 + 0.2 * np.sin(2 * t)], 1))
        blade_world = np.einsum('tij,tj->ti', turn, blade_body)
        K = np.array([[1400.0, 0, 960], [0, 1400.0, 540], [0, 0, 1]])
        views = []
        # world -> camera (x right, y down, z forward): in front of the body, then at its left side
        for Rw in (np.array([[1.0, 0, 0], [0, -1, 0], [0, 0, -1]]), np.array([[0, 0, -1.0], [0, -1, 0], [-1, 0, 0]])):
            self.assertAlmostEqual(float(np.linalg.det(Rw)), 1.0)
            Jc = J @ Rw.T + np.array([0, 0.2, 3.5])
            blade_c = blade_world @ Rw.T
            hand = Jc[:, self.BODY.index("RightHand")]
            rows = np.concatenate([gemx._picture_dirs(hand, blade_c, K), np.ones((T, 1))], 1)
            data = {"joints_incam": Jc, "joint_names": np.array(self.BODY), "K": K, "fps": np.array(30.0)}
            views.append((data, rows, hand, blade_c))
        (d1, r1, hand1, blade1), (d2, r2, _, _) = views
        lean, weight, flip, rep = gemx.second_view_lean(d1, self.BODY, hand1, gemx._unit(r1[:, :2]), K, d2, r2.tolist())
        true_lean = gemx._lean(hand1, blade1, gemx._unit(r1[:, :2]), K)
        used = weight > 0.5
        self.assertGreater(used.mean(), 0.5)
        self.assertLess(float(np.degrees(np.median(np.abs(lean[used] - true_lean[used])))), 3.0)
        self.assertFalse(flip.any())


if __name__ == "__main__":
    unittest.main()
