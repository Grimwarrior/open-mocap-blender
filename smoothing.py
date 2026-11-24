"""
    Causal, speed-adaptive smoothing for live tracking (no future frames): the One Euro filter
    (Casiez, Roussel and Vogel, CHI 2012) as RGB MoCap's PEAR live preview uses it
    (github.com/toyxyz/toyxyz_RGBMocap, mocap/pear/temporal_filter.py, MIT).

    Each point is smoothed by a low-pass filter whose cutoff rises with the point's speed: a hand
    held still is smoothed strongly (its jitter goes), a fast move hardly at all (little lag). Speed
    is measured in `scale` units per second, for hands the hand's own length, so the same settings
    hold near and far from the camera.
"""
import math

import numpy as np

# Cutoffs in Hz, BETA per (hand length / s). On a synthetic hand (tests/test_smoothing.py) with jitter of
# 2% of its length, at strength 50: position error while still 0.051 -> 0.028 hand lengths, jitter (frame
# to frame acceleration) 0.121 -> 0.022; the cost is lag in a fast swipe (10 hand lengths / s): 0.05 ->
# 0.17 hand lengths. Strength 25: lag 0.10, jitter 0.037; 75: lag 0.30, jitter 0.012. On the body
# tracker's hands in five recorded sword takes, the slowest third of frames: jitter 0.131 -> 0.081.
MIN_CUTOFF = 1.0
BETA = 1.5
MAX_CUTOFF = 15.0
DERIVATIVE_CUTOFF = 1.0
MAX_GAP = 0.25          # longer between two samples (s): start again from the new sample
JUMP = 2.0              # a live hand that moves this many hand lengths in one sample starts again too


def alpha(cutoff, dt):
    """Share of the new sample a first-order low-pass filter with this cutoff (Hz) takes after dt seconds."""
    return 1.0 / (1.0 + 1.0 / (2.0 * math.pi * np.asarray(cutoff) * dt))


def strength_scale(strength):
    """Strength 1..100 (50 the default) as a divisor of the cutoffs: each 25 more halves them."""
    value = float(strength)
    if not math.isfinite(value):
        raise ValueError("Invalid smoothing strength")
    return 2.0 ** ((min(100.0, max(1.0, value)) - 50.0) / 25.0)


class OneEuro:
    """One Euro filter for a set of points (..., 3): one speed per point."""

    def __init__(self, strength=50.0):
        self.set_strength(strength)
        self.reset()

    def set_strength(self, strength):
        scale = strength_scale(strength)
        self.min_cutoff = MIN_CUTOFF / scale
        self.beta = BETA / scale
        self.max_cutoff = MAX_CUTOFF / scale

    def reset(self):
        self.raw = None
        self.value = None
        self.derivative = None
        self.time = None

    def update(self, points, time, scale=1.0):
        """
            :param points: (..., 3) this sample's points.
            :param time: when it was captured, in seconds.
            :param scale: length that speeds are measured in (the hand's length).
            :return: the filtered points, same shape.
        """
        x = np.asarray(points, dtype=float)
        dt = None if self.time is None else time - self.time
        if dt is not None and dt == 0 and self.raw is not None and self.raw.shape == x.shape:
            return self.value.copy()                  # the same sample again
        if dt is None or not 0 < dt <= MAX_GAP or self.raw.shape != x.shape:
            self.raw, self.value, self.derivative = x.copy(), x.copy(), np.zeros_like(x)
            self.time = time
            return self.value.copy()
        self.derivative += alpha(DERIVATIVE_CUTOFF, dt) * ((x - self.raw) / dt - self.derivative)
        speed = np.linalg.norm(self.derivative, axis=-1, keepdims=True) / max(float(scale), 1e-9)
        # Still points get the low cutoff; a cap keeps a spike in the speed from switching smoothing off
        cutoff = np.minimum(self.max_cutoff, self.min_cutoff + self.beta * speed)
        self.value += alpha(cutoff, dt) * (x - self.value)
        self.raw = x.copy()
        self.time = time
        return self.value.copy()


def hand_length(points):
    """Wrist -> middle-finger knuckle (MediaPipe hand points 0 and 9), the speed unit for a hand."""
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    return float(np.linalg.norm(p[9] - p[0])) if len(p) > 9 else 1.0


class LiveHands:
    """One filter per hand side for the live hand tracker; strength 0 passes the points through."""

    def __init__(self, strength=50.0):
        self.strength = float(strength)
        self.filters = {}
        self.last = {}

    def set_strength(self, strength):
        """Change the strength while tracking runs; 0 switches the filter off (and forgets its state)."""
        strength = float(strength)
        if strength == self.strength:
            return
        self.strength = strength
        if strength <= 0:
            self.filters.clear()
            self.last.clear()
        for f in self.filters.values():
            f.set_strength(strength)

    def points(self, side, points, time):
        """The points to show for this hand, filtered once per new sample (`time` changes)."""
        if self.strength <= 0:
            return points
        if side in self.last and self.last[side][0] == time:
            return self.last[side][1]
        f = self.filters.setdefault(side, OneEuro(self.strength))
        length = hand_length(points)
        if f.raw is not None and f.raw.shape == np.shape(points) and \
                np.linalg.norm(np.asarray(points) - f.raw, axis=-1).mean() > JUMP * length:
            f.reset()                   # another hand took this label: show it where it is, no glide
        out = f.update(points, time, length).tolist()
        self.last[side] = (time, out)
        return out
