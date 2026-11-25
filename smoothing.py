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

    def __init__(self, strength=50.0, cutoffs=None):
        """:param cutoffs: (min cutoff, beta, max cutoff) instead of the live hand defaults at `strength`."""
        if cutoffs is None:
            self.set_strength(strength)
        else:
            self.min_cutoff, self.beta, self.max_cutoff = cutoffs
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


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def slerp(a, b, t):
    """Unit vectors a -> b (..., 3) turned a share t (..., 1) of the way along the great circle."""
    dot = np.clip(np.sum(a * b, axis=-1, keepdims=True), -1.0, 1.0)
    theta = np.arccos(dot)
    s = np.sin(theta)
    small = s < 1e-6
    w1 = np.where(small, 1.0 - t, np.sin((1.0 - t) * theta) / np.where(small, 1.0, s))
    w2 = np.where(small, t, np.sin(t * theta) / np.where(small, 1.0, s))
    return _unit(w1 * a + w2 * b)


class OneEuroDirections:
    """
        One Euro filter for unit vectors (..., 3), a bone's direction: it turns along the great circle
        toward each new direction, by a share set from the direction's angular speed (rad/s), as PEAR's
        filter turns rotation matrices geodesically instead of averaging their entries.
    """

    def __init__(self, cutoffs):
        self.min_cutoff, self.beta, self.max_cutoff = cutoffs
        self.raw = self.value = self.derivative = self.time = None

    def update(self, units, time):
        x = _unit(np.asarray(units, dtype=float))
        dt = None if self.time is None else time - self.time
        if dt is None or not 0 < dt <= MAX_GAP or self.raw.shape != x.shape:
            self.raw, self.value = x.copy(), x.copy()
            self.derivative = np.zeros(x.shape[:-1] + (1,))
            self.time = time
            return self.value.copy()
        turn = np.arccos(np.clip(np.sum(self.raw * x, axis=-1, keepdims=True), -1.0, 1.0)) / dt
        self.derivative += alpha(DERIVATIVE_CUTOFF, dt) * (turn - self.derivative)
        cutoff = np.minimum(self.max_cutoff, self.min_cutoff + self.beta * self.derivative)
        self.value = slerp(self.value, x, alpha(cutoff, dt))
        self.raw, self.time = x, time
        return self.value.copy()


ZERO_PHASE = 'CASCADE'      # 'CASCADE': backward over the forward pass's output; 'AVERAGE': the two passes averaged


def zero_phase(make, values, times, scale=1.0, directions=False):
    """
        Offline, without lag: a filter from `make()` run forward and backward over the take. CASCADE
        filters the forward result backward (as filtfilt does: the lags cancel, the smoothing adds up);
        AVERAGE averages the two passes (the forward one trails a move as much as the backward one leads).
        :param values: (N, ..., 3); times (N,) seconds. :return: filtered values, same shape.
    """
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)

    def run(v, t):
        f = make()
        return np.array([f.update(a, b) if directions else f.update(a, b, scale) for a, b in zip(v, t)])

    forward = run(values, times)
    if ZERO_PHASE == 'CASCADE':
        return run(forward[::-1], -times[::-1])[::-1]
    both = forward + run(values[::-1], -times[::-1])[::-1]
    return _unit(both) if directions else both / 2.0


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
