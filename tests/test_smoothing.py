"""Live hand smoothing (smoothing.py): a One Euro filter tuned on a synthetic hand with known motion."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smoothing as S  # noqa: E402


def _ease(t, a, b):
    x = np.clip((t - a) / (b - a), 0.0, 1.0)
    return x * x * (3 - 2 * x)


def synthetic_hand(seed, noise=0.02, fps=30.0):
    """
        A hand 1 long (21 points, wrist 0, middle knuckle 9) that holds still, drifts slowly, curls its
        fingers and swipes fast (about 10 hand lengths / s), with `noise` hand lengths of jitter per
        point (depth 2.5 times as much). :return: times, true points, tracked points, wrist speed.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, 8.0, 1.0 / fps)
    shape = rng.normal(size=(21, 3)) * 0.4
    shape[0], shape[9] = 0.0, [0.0, 1.0, 0.0]
    path = np.zeros((len(t), 3))
    path[:, 0] = 0.3 * np.sin(0.8 * t) * _ease(t, 2.0, 3.0)
    path[:, 0] += 3.0 * (_ease(t, 4.0, 4.4) - _ease(t, 5.5, 5.9))
    path[:, 1] += 2.0 * _ease(t, 6.5, 6.8)
    curl = 0.3 * _ease(t, 1.0, 1.5) * np.sin(3 * t)
    truth = path[:, None, :] + shape[None] * (1 + curl[:, None, None])
    tracked = truth + rng.normal(size=truth.shape) * noise * np.array([1.0, 1.0, 2.5])
    speed = np.linalg.norm(np.gradient(path, axis=0), axis=1) * fps
    return t, truth, tracked, speed


def measure(strength):
    """Mean over four hands: error still / slow / fast (hand lengths) and jitter while still."""
    rows = []
    for seed in range(4):
        t, truth, tracked, speed = synthetic_hand(seed)
        if strength:
            f = S.OneEuro(strength)
            out = np.array([f.update(x, ti, S.hand_length(x)) for x, ti in zip(tracked, t)])
        else:
            out = tracked
        err = np.linalg.norm(out - truth, axis=-1).mean(axis=1)
        jitter = np.linalg.norm(out[2:] - 2 * out[1:-1] + out[:-2], axis=-1).mean(axis=1)
        rows.append((err[speed < 0.05].mean(), err[(speed > 0.05) & (speed < 1)].mean(), err[speed > 4].mean(),
                     np.median(jitter[speed[1:-1] < 0.05])))
    return np.mean(rows, axis=0)


class OneEuroHands(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = measure(0)
        cls.normal = measure(50)

    def test_still_hands_steady(self):
        still, _slow, _fast, jitter = self.normal
        self.assertLess(still, 0.65 * self.raw[0])         # 0.051 -> 0.028 hand lengths
        self.assertLess(jitter, 0.25 * self.raw[3])        # 0.109 -> 0.020

    def test_slow_moves_not_worse(self):
        self.assertLess(self.normal[1], self.raw[1])

    def test_fast_swipes_lag_little(self):
        self.assertLess(self.normal[2], 0.25)              # behind by under a quarter of the hand
        self.assertLess(measure(25)[2], self.normal[2])    # less strength, less lag

    def test_strength_trades_lag_for_steadiness(self):
        self.assertLess(measure(75)[3], self.normal[3])


class LiveHandsBehaviour(unittest.TestCase):
    def setUp(self):
        self.hand = np.zeros((21, 3))
        self.hand[9] = [0.0, 1.0, 0.0]

    def test_strength_zero_passes_points_through(self):
        live = S.LiveHands(0)
        pts = (self.hand + 0.3).tolist()
        self.assertIs(live.points("R", pts, 0.0), pts)

    def test_same_detection_is_filtered_once(self):
        live = S.LiveHands(50)
        live.points("R", self.hand.tolist(), 0.0)
        a = live.points("R", (self.hand + 0.1).tolist(), 1 / 30)
        b = live.points("R", (self.hand + 0.1).tolist(), 1 / 30)   # Blender redraws before the next detection
        self.assertEqual(a, b)

    def test_gap_and_jump_start_again(self):
        live = S.LiveHands(50)
        live.points("R", self.hand.tolist(), 0.0)
        far = (self.hand + [5.0, 0.0, 0.0]).tolist()
        np.testing.assert_allclose(live.points("R", far, 1 / 30), far)        # jumped 5 hand lengths
        back = self.hand.tolist()
        np.testing.assert_allclose(live.points("R", back, 1.0), back)        # lost for most of a second

    def test_small_move_is_smoothed(self):
        live = S.LiveHands(50)
        live.points("R", self.hand.tolist(), 0.0)
        moved = live.points("R", (self.hand + [0.05, 0.0, 0.0]).tolist(), 1 / 30)
        self.assertTrue(0.0 < moved[0][0] < 0.05)

    def test_strength_can_change_while_running(self):
        live = S.LiveHands(50)
        live.points("R", self.hand.tolist(), 0.0)
        live.set_strength(90)
        self.assertAlmostEqual(live.filters["R"].min_cutoff, S.MIN_CUTOFF / S.strength_scale(90))
        live.set_strength(0)
        self.assertEqual(live.filters, {})


class OfflineRotations(unittest.TestCase):
    """The offline One Euro on bone directions (Combine Data's ONE_EURO smoothing)."""

    def test_slerp_turns_along_the_great_circle(self):
        a, b = np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]])
        np.testing.assert_allclose(S.slerp(a, b, np.array([[0.5]])), [[np.sqrt(0.5), np.sqrt(0.5), 0]], atol=1e-12)
        np.testing.assert_allclose(S.slerp(a, a, np.array([[0.3]])), a)

    def test_two_passes_leave_no_lag(self):
        t = np.arange(0, 3, 1 / 30)
        angle = 2.0 * t                                              # turning at 2 rad/s
        way = np.stack([np.cos(angle), np.sin(angle), 0 * t], axis=1)[:, None]
        rng = np.random.default_rng(0)
        noisy = S._unit(way + rng.normal(size=way.shape) * 0.03)
        cut = (4.0, 0.5, 60.0)
        f = S.OneEuroDirections(cut)
        forward = np.array([f.update(v, ti) for v, ti in zip(noisy, t)])
        both = S.zero_phase(lambda: S.OneEuroDirections(cut), noisy, t, directions=True)
        err = lambda w: np.degrees(np.arccos(np.clip(np.sum(w[15:-15] * way[15:-15], axis=-1), -1, 1))).mean()
        self.assertLess(err(both), 0.5 * err(forward))               # the forward pass alone trails
        self.assertLess(err(both), np.degrees(0.03) * 1.2)
        np.testing.assert_allclose(np.linalg.norm(both, axis=-1), 1.0, atol=1e-9)

    def test_combine_with_one_euro_bones(self):
        import combine
        import synthetic_combine as bench
        take = bench.make_take(seed=0)
        rows = {}
        for method in ("PEAKS", "ONE_EURO"):
            frames, report = combine.combine_improved(take["front"], take["side"], take["tF"], take["tS"], take["visF"],
                                                      take["visS"], smoothing_method=method)
            self.assertTrue(report.get("combined", True))
            rows[method] = bench.evaluate(frames, take)
        euro, peaks = rows["ONE_EURO"], rows["PEAKS"]
        self.assertLess(euro["depth"], 1.3 * peaks["depth"])          # 1.20 against 1.08 % (three seeds)
        self.assertGreater(euro["strike"], 80.0)
        self.assertEqual(euro["xy"], peaks["xy"])                    # width/height smoothed the same way

    def test_off_and_short_takes_pass_through(self):
        import combine
        P = np.random.default_rng(1).normal(size=(10, 40, 3))
        t = np.arange(10) / 30.0
        np.testing.assert_array_equal(combine.euro_skeleton(P, t, 0.0), P)
        np.testing.assert_array_equal(combine.euro_skeleton(P[:2], t[:2], 1.5), P[:2])


if __name__ == "__main__":
    unittest.main()
