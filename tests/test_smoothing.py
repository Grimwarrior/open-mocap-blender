"""Live hand smoothing (smoothing.py): a One Euro filter tuned on a synthetic hand with known motion."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
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


if __name__ == "__main__":
    unittest.main()
