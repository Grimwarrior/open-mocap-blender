"""Sword tracking: the blade's direction in a picture, and in 3D from several views."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import combine  # noqa: E402
import sword  # noqa: E402
import synthetic_combine  # noqa: E402

W, H = 640, 480


def stick_figure(grip):
    """40 landmarks (pixels) of an actor about 300 px tall holding both hands at `grip`."""
    p = np.zeros((40, 2))
    hip, chest = np.array([320.0, 300.0]), np.array([320.0, 200.0])
    p[0] = [320, 160]
    p[11], p[12] = chest + [25, 0], chest - [25, 0]
    p[23], p[24] = hip + [15, 0], hip - [15, 0]
    p[25], p[26] = hip + [18, 70], hip - [18, 70]
    p[27], p[28] = hip + [20, 140], hip - [20, 140]
    p[31], p[32] = p[27] + [10, 5], p[28] - [10, -5]
    for side, (s, e, w) in {1: (11, 13, 15), -1: (12, 14, 16)}.items():
        p[e] = (p[s] + grip) / 2 + [8 * side, 10]
        p[w] = grip + [4 * side, 0]
    for i in (17, 19, 21):
        p[i] = p[15] + [3, -3]
    for i in (18, 20, 22):
        p[i] = p[16] + [-3, -3]
    for c, (i, j) in combine.CENTROIDS.items():
        p[c] = (p[i] + p[j]) / 2
    return p


def draw_line(img, a, b, half, value):
    """Thick line (half-width in pixels) on a 2D/3D array."""
    Y, X = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    d = b - a
    t = np.clip(((X - a[0]) * d[0] + (Y - a[1]) * d[1]) / max(float(d @ d), 1e-9), 0, 1)
    img[np.hypot(X - a[0] - t * d[0], Y - a[1] - t * d[1]) <= half] = value


def video(n=30, turn_deg=5.0, blade=True):
    """Frames of an actor turning a sword; :return: (read_frames, capture frames, vis, times, true angles)."""
    rng = np.random.default_rng(0)
    stage = (40 + 25 * rng.random((H, W, 1))).repeat(3, axis=2).astype(np.uint8)
    angles = np.radians(-60.0 + turn_deg * np.arange(n))
    imgs, frames = [], []
    for k in range(n):
        grip = np.array([300.0 + 2 * k, 230.0])
        p = stick_figure(grip)
        img = stage.copy()
        for a, b in sword.LIMBS:                                   # dark clothes, wide limbs
            draw_line(img, p[a], p[b], 12, (15, 15, 15))
        draw_line(img, p[13], p[15], 1.2, (235, 235, 235))         # white piping along a sleeve
        if blade:
            tip = grip + 140 * np.array([np.cos(angles[k]), np.sin(angles[k])])
            draw_line(img, grip, tip, 2.5, (210, 215, 220))
        imgs.append(img)
        flat = np.zeros((40, 3))
        flat[:, 0] = (p[:, 0] - W / 2) / 100
        flat[:, 1] = (p[:, 1] - H) / 100
        frames.append(flat.reshape(-1).tolist())

    def read_frames():
        yield from enumerate(imgs)

    times = (np.arange(n) / 30.0).tolist()
    return read_frames, frames, np.full((n, 40), 0.9).tolist(), times, angles


class Detection(unittest.TestCase):
    def test_finds_a_turning_blade_and_not_the_sleeve(self):
        read_frames, frames, vis, times, truth = video()
        rows = np.array(sword.track_frames(read_frames, frames, vis, times, 30.0, W, H))
        found = np.arctan2(rows[:, 1], rows[:, 0])
        err = np.degrees(np.abs((found - truth + np.pi) % (2 * np.pi) - np.pi))
        self.assertLess(np.median(err), 5.0)
        self.assertGreater(np.mean(err < 12.0), 0.9)
        self.assertGreater(np.median(rows[:, 2]), 0.5)

    def test_no_blade_no_confidence(self):
        read_frames, frames, vis, times, _ = video(blade=False)
        rows = np.array(sword.track_frames(read_frames, frames, vis, times, 30.0, W, H))
        self.assertLess(np.median(rows[:, 2]), 0.2)

    def test_opening_removes_only_what_is_narrower(self):
        a = np.zeros((40, 40))
        a[5:35, 10:12] = 1.0          # 2 px line
        a[5:35, 20:30] = 1.0          # 10 px bar
        o = sword.opening(a, 5)
        self.assertEqual(o[5:35, 10:12].max(), 0.0)
        self.assertEqual(o[8:32, 20:30].min(), 1.0)


def blade(t):
    """A blade swinging in 3D (world x, depth y, up z), unit vectors."""
    yaw, pitch = 1.3 * np.sin(1.1 * t), 0.6 * np.cos(0.8 * t)
    return np.stack([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)], axis=-1)


def picture(v, angle_deg):
    """Picture direction (x right, y down) of world directions seen from a camera turned about the actor."""
    a = np.radians(angle_deg)
    u = np.stack([np.cos(a) * v[:, 0] - np.sin(a) * v[:, 1], -v[:, 2]], axis=1)
    return u / np.linalg.norm(u, axis=1, keepdims=True)


class Fusion(unittest.TestCase):
    def fused(self, extra_angles=(), separate=False):
        take = synthetic_combine.make_take(seed=0, extra_angles=extra_angles, separate=separate)
        frames, report = combine.combine_multi(take["front"], take["views"], take["tF"], take["visF"])
        tF = np.asarray(take["tF"])
        rows = lambda u: np.concatenate([u, np.ones((len(u), 1))], axis=1).tolist()
        front = rows(picture(blade(tF), 0.0))
        views = []
        for k, v in enumerate(take["views"]):
            angle = 90.0 if k == 0 else extra_angles[k - 1]
            start = 0.4 if k == 0 else 0.15 * k * (-1) ** (k - 1)      # each view's start, as make_take films it
            views.append(dict(v, sword=rows(picture(blade(np.asarray(v["times"]) + start), angle))))
        axis, srep = combine.sword_axis(take["front"], take["tF"], front, views, frames, report)
        out = np.asarray(axis)[:, :3]
        truth = blade(tF)
        expected = np.stack([truth[:, 0], -truth[:, 2], truth[:, 1]], axis=1)    # combine: x, y down, depth
        err = np.degrees(np.arccos(np.clip(np.sum(out * expected, axis=1), -1, 1)))
        mirrored = expected * [1, 1, -1]
        err_m = np.degrees(np.arccos(np.clip(np.sum(out * mirrored, axis=1), -1, 1)))
        return min(err, err_m, key=np.median), srep

    def test_side_view_gives_the_blade_depth(self):
        err, srep = self.fused()
        self.assertEqual(len(srep["views"]), 1)
        self.assertLess(np.median(err), 8.0)

    def test_without_further_views_the_blade_stays_in_the_front_picture(self):
        take = synthetic_combine.make_take(seed=0)
        frames, report = combine.combine_multi(take["front"], take["views"], take["tF"], take["visF"])
        u = picture(blade(np.asarray(take["tF"])), 0.0)
        front = np.concatenate([u, np.ones((len(u), 1))], axis=1).tolist()
        axis, srep = combine.sword_axis(take["front"], take["tF"], front, [dict(take["views"][0])], frames, report)
        out = np.asarray(axis)
        self.assertEqual(srep["views"], [])
        np.testing.assert_allclose(out[:, 2], 0.0, atol=1e-6)
        self.assertGreater(np.min(np.sum(out[:, :2] * u, axis=1)), 0.99)


class Grip(unittest.TestCase):
    def take(self, n=60):
        """Both hands on a handle in front of the body; the left one hidden (and wrong) in some frames."""
        rng = np.random.default_rng(1)
        t = np.arange(n) / 30.0
        P = np.zeros((n, 40, 3))
        P[:, 11], P[:, 12] = [0.6, -1.4, 0.0], [-0.6, -1.4, 0.0]          # shoulders (x, y down, depth)
        P[:, 23], P[:, 24] = [0.3, 0.0, 0.0], [-0.3, 0.0, 0.0]
        handle = np.stack([np.cos(t), 0.3 * np.sin(t), np.sin(t)], axis=1)
        handle /= np.linalg.norm(handle, axis=1, keepdims=True)
        grip = np.stack([0.2 * np.sin(2 * t), -0.6 + 0.1 * t, -1.0 + 0 * t], axis=1)
        P[:, 16] = grip + 0.24 * handle
        P[:, 15] = grip - 0.24 * handle
        for s, e, w in ((11, 13, 15), (12, 14, 16)):
            P[:, e] = (P[:, s] + P[:, w]) / 2 + [0.0, 0.3, 0.0]
        for c, (i, j) in combine.CENTROIDS.items():
            P[:, c] = (P[:, i] + P[:, j]) / 2
        rel = np.full((n, 40), 0.9)
        hidden = rng.random(n) < 0.4
        P[hidden, 15] = P[hidden, 23] + [0.1, 0.0, 0.2]                    # guessed at the hip
        rel[hidden, 15] = 0.05
        axis = np.concatenate([handle, np.ones((n, 1))], axis=1)            # the blade along the handle
        return P, rel, axis, hidden

    def test_hidden_hand_goes_back_on_the_handle(self):
        P, rel, axis, hidden = self.take()
        out, rep = combine.two_hand_grip(P.reshape(len(P), -1).tolist(), rel, axis.tolist())
        Q = combine._as_array(out)
        true_left = P[:, 16] - 0.48 * axis[:, :3]
        self.assertLess(np.max(np.linalg.norm(Q[hidden, 15] - true_left[hidden], axis=1)), 0.05)
        np.testing.assert_allclose(np.linalg.norm(Q[:, 16] - Q[:, 15], axis=1), 0.48, atol=0.02)
        np.testing.assert_allclose(Q[:, 16], P[:, 16], atol=0.05)          # the seen hand stays
        fore = np.linalg.norm(Q[:, 15] - Q[:, 13], axis=1)
        self.assertLess(np.std(fore) / np.mean(fore), 0.02)                 # the elbow keeps the forearm's length

    def test_without_a_blade_the_line_between_the_wrists_is_used(self):
        P, rel, _axis, hidden = self.take()
        out, _ = combine.two_hand_grip(P.reshape(len(P), -1).tolist(), rel, None)
        Q = combine._as_array(out)
        self.assertLess(np.median(np.linalg.norm(Q[hidden, 15] - (P[hidden, 16] - 0.48 * _axis[hidden, :3]), axis=1)), 0.15)


class Capture(unittest.TestCase):
    def test_sword_rows_are_checked_and_kept(self):
        import capture_data
        frame = [0.0] * 120
        data = {"positionList": [frame, frame], "position_times": [0.0, 0.033], "position_vis": [[0.5] * 40] * 2,
                "sword_front": [[1.0, 0.0, 0.9], [0.0, 1.0, 0.8]], "sword_axis": [[1.0, 0.0, 0.0, 0.5]] * 2}
        capture_data.validate_pose_capture(data)
        with self.assertRaises(ValueError):
            capture_data.validate_pose_capture(dict(data, sword_front=[[1.0, 0.0, 0.9]]))
        view = {"frames": [frame], "times": [0.0], "vis": [[0.5] * 40], "sword": [[0.0, -1.0, 1.0]]}
        self.assertEqual(capture_data.validate_extra_views([view])[0]["sword"], [[0.0, -1.0, 1.0]])
        with self.assertRaises(ValueError):
            capture_data.validate_extra_views([dict(view, sword=[[0.0, -1.0, 1.0]] * 2)])


if __name__ == "__main__":
    unittest.main()
