"""
    Ground-truth test of Combine Data (numpy only, no Blender):

        python tests/synthetic_combine.py

    A stick figure with constant bone lengths walks, turns its torso and strikes with one arm.
    Two cameras film it: the front one, and a side one with its own start time, distance, angle,
    noise, hidden far-side limbs and bursts of swapped left/right labels. The combined take is
    compared with the true 3D motion, which real footage can never give:

        depth    mean error of the depth of every joint relative to the hips, % of body height
        xy       mean error of width/height, % of body height
        strike   peak wrist speed of the strike kept by the combine, % of the true peak
        jitter   mean frame-to-frame acceleration, relative to the true motion's (1 = as smooth)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import combine  # noqa: E402

UNIT = 3.4          # world metres -> tracked units (a 1.75 m actor is ~6 units = 600 px tall)


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def true_motion(t, variation=0.0):
    """
        :param t: times (s). :param variation: 0..1, how differently this performance is done
                  (a second take of the same move).
        :return: (N, 40, 3) joints in metres: x = actor's left, y = away from the front camera,
                 z = up. The actor faces the front camera (-y).
    """
    n = len(t)
    P = np.zeros((n, 40, 3))
    v = variation
    for k, tk in enumerate(t):
        ph = 2 * np.pi * 0.9 * tk
        root = np.array([0.15 * np.sin(0.5 * ph), -0.25 * tk + 0.1 * v * np.sin(ph), 0.95 + 0.03 * np.cos(2 * ph)])
        pelvis = _rot_z(0.25 * np.sin(ph) * (1 + 0.3 * v))
        torso = pelvis @ _rot_z(0.5 * np.sin(ph + 0.6)) @ _rot_x(-0.15 + 0.1 * np.sin(ph))
        down, fwd, left = np.array([0, 0, -1.0]), np.array([0, -1.0, 0]), np.array([1.0, 0, 0])
        J = {}
        J[23], J[24] = root + pelvis @ (0.1 * left), root + pelvis @ (-0.1 * left)
        for side, (hip, knee, ankle, heel, toe) in ((1, (23, 25, 27, 29, 31)), (-1, (24, 26, 28, 30, 32))):
            swing = 0.5 * np.sin(ph) * side
            bend = 0.6 * max(0.0, np.sin(ph * 1.0 + (0 if side > 0 else np.pi)))
            thigh = pelvis @ _rot_x(swing)
            shin = thigh @ _rot_x(-bend)
            J[knee] = J[hip] + thigh @ (0.45 * down)
            J[ankle] = J[knee] + shin @ (0.43 * down)
            J[heel] = J[ankle] + shin @ (0.06 * down - 0.05 * fwd)
            J[toe] = J[ankle] + shin @ (0.07 * down + 0.17 * fwd)
        chest = root + torso @ np.array([0, 0, 0.5])
        J[11], J[12] = chest + torso @ (0.19 * left), chest + torso @ (-0.19 * left)
        # left arm swings; right arm lifts and strikes down fast once per 2.2 s
        cyc = (tk * (1 + 0.1 * v)) % 2.2
        lift = np.interp(cyc, [0, 1.0, 1.3, 1.42, 2.2], [0.3, 0.3, 2.6, 0.5, 0.3])
        arms = {1: (_rot_x(0.6 * np.sin(ph + np.pi)), 0.5 + 0.3 * np.sin(ph)),
                -1: (_rot_x(-lift) @ _rot_y(0.3), 0.4 + 0.25 * lift)}
        for side, (sh, el, wr, pinky, index, thumb) in ((1, (11, 13, 15, 17, 19, 21)), (-1, (12, 14, 16, 18, 20, 22))):
            R, elbow = arms[side]
            upper = torso @ R
            fore = upper @ _rot_x(-elbow)
            J[el] = J[sh] + upper @ (0.3 * down)
            J[wr] = J[el] + fore @ (0.27 * down)
            J[pinky] = J[wr] + fore @ (0.08 * down - 0.03 * side * left)
            J[index] = J[wr] + fore @ (0.09 * down + 0.03 * side * left)
            J[thumb] = J[wr] + fore @ (0.05 * down + 0.05 * fwd)
        head = chest + torso @ np.array([0, 0, 0.22])
        H = torso @ _rot_z(0.3 * np.sin(ph * 0.7))
        face = {0: (0, 0.1, 0.0), 1: (0.02, 0.08, 0.04), 2: (0.035, 0.08, 0.04), 3: (0.05, 0.07, 0.04),
                4: (-0.02, 0.08, 0.04), 5: (-0.035, 0.08, 0.04), 6: (-0.05, 0.07, 0.04),
                7: (0.08, 0, 0.02), 8: (-0.08, 0, 0.02), 9: (0.025, 0.08, -0.04), 10: (-0.025, 0.08, -0.04)}
        for i, (lx, f, up) in face.items():
            J[i] = head + H @ (lx * left + f * fwd + np.array([0, 0, up]))
        for i, p in J.items():
            P[k, i] = p
    for c, (a, b) in combine.CENTROIDS.items():
        P[:, c] = (P[:, a] + P[:, b]) / 2
    return P


def film(P, rng, angle=0.0, scale=1.0, noise=0.006, shift=(0.0, 0.0), swap_bursts=0, z_quality=0.25):
    """
        Orthographic camera turned `angle` degrees about the actor (0 = front, 90 = side).
        :return: (frames as flat lists like the tracker stores them, visibility (N, 40))
    """
    a = np.radians(angle)
    n = len(P)
    X = (np.cos(a) * P[..., 0] - np.sin(a) * P[..., 1]) * UNIT / scale + shift[0]
    depth = np.sin(a) * P[..., 0] + np.cos(a) * P[..., 1]              # away from this camera
    Y = -P[..., 2] * UNIT / scale + shift[1]
    # joints of the far arm / leg that lie behind the torso are hidden: low visibility, more noise
    vis = np.full((n, 40), 0.95)
    torso_d = depth[:, [33, 34]].mean(axis=1, keepdims=True)
    torso_x = X[:, [33, 34]].mean(axis=1, keepdims=True)
    limbs = list(range(13, 33))
    hidden = (depth[:, limbs] > torso_d + 0.05) & (np.abs(X[:, limbs] - torso_x) < 0.16 * UNIT / scale)
    vis[:, limbs] = np.where(hidden, 0.25, 0.95)
    sigma = noise * UNIT / scale * np.where(vis < 0.5, 4.0, 1.0)
    X = X + rng.normal(size=X.shape) * sigma
    Y = Y + rng.normal(size=Y.shape) * sigma
    # the tracker's own depth: the right shape, too small, noisy
    Z = (depth - depth[:, [33]]) * UNIT / scale * z_quality + rng.normal(size=X.shape) * 6 * noise * UNIT / scale
    out = np.stack([X, Y, Z], axis=2)
    for _ in range(swap_bursts):                                       # near/far legs or arms mislabelled
        L, R = combine.LR_GROUPS["legs" if rng.random() < 0.5 else "arms"]
        s = int(rng.integers(0, max(1, n - 20)))
        e = s + int(rng.integers(6, 20))
        out[s:e, L], out[s:e, R] = out[s:e, R].copy(), out[s:e, L].copy()
    for c, (i, j) in combine.CENTROIDS.items():
        out[:, c] = (out[:, i] + out[:, j]) / 2
        vis[:, c] = np.minimum(vis[:, i], vis[:, j])
    return out.reshape(n, -1).tolist(), vis


def make_take(seed=0, fps=30.0, duration=6.0, offset=0.4, side_angle=90.0, side_scale=1.3,
              swap_bursts=3, separate=False, noise=0.006, extra_angles=()):
    """
        :param extra_angles: further cameras (degrees about the actor, 270 = the other side, 180 =
                             behind), each with its own start time and distance; with `separate`
                             each films its own performance.
        :return: dict with the tracker-style inputs of the views and the true motion at the front
                 frames. "views" lists the side view and the further ones as combine_multi takes them.
    """
    rng = np.random.default_rng(seed)
    tF = np.arange(0, duration, 1.0 / fps)
    tS = np.arange(0, duration + 1.0, 1.0 / fps)
    truth = true_motion(tF)
    if separate:        # a second performance: other details, timing drifting by a few tenths of a second
        side_motion = true_motion(tS + offset + 0.15 * np.sin(tS * 1.3), variation=1.0)
    else:
        side_motion = true_motion(tS + offset)
    front, visF = film(truth, rng, 0.0, 1.0, noise)
    side, visS = film(side_motion, rng, side_angle, side_scale, noise, shift=(0.7, 0.3), swap_bursts=swap_bursts)
    views = [dict(frames=side, times=tS + 0.0, vis=visS, name=f"side {side_angle:.0f}")]
    for k, angle in enumerate(extra_angles):
        start = 0.15 * (k + 1) * (-1) ** k
        if separate:
            motion = true_motion(tS + start + 0.12 * np.sin(tS * 1.1 + k + 1), variation=0.6 + 0.3 * k)
        else:
            motion = true_motion(tS + start)
        frames, vis = film(motion, rng, angle, 1.15 + 0.1 * k, noise, shift=(-0.4 + 0.3 * k, 0.2), swap_bursts=1)
        views.append(dict(frames=frames, times=tS + 0.0, vis=vis, name=f"view {angle:.0f}"))
    return dict(front=front, side=side, tF=tF, tS=tS + 0.0, visF=visF, visS=visS, truth=truth, fps=fps,
                offset=-offset, views=views)


def evaluate(frames, take):
    """:return: dict of the errors described at the top of this file."""
    C = np.asarray(frames).reshape(len(frames), 40, 3)
    T = take["truth"][:len(C)] * UNIT
    body = float(np.median(T[:, 0, 2] - T[:, [27, 28], 2].mean(axis=1)))
    true_d = T[:, :33, 1] - T[:, [33], 1]
    d = C[:, :33, 2] - C[:, [33], 2]
    if np.mean(np.abs(-d - true_d)) < np.mean(np.abs(d - true_d)):
        d = -d                                              # mirrored in depth (the retarget un-mirrors)
    xy = np.stack([C[:, :33, 0], -C[:, :33, 1]], axis=2)
    true_xy = np.stack([T[:, :33, 0], T[:, :33, 2]], axis=2)
    out = np.stack([C[:, :, 0], C[:, :, 2], -C[:, :, 1]], axis=2)
    true3 = T
    fps = take["fps"]
    wrist_true = np.linalg.norm(np.diff(true3[:, 16] - true3[:, 33], axis=0), axis=1) * fps
    wrist_out = np.linalg.norm(np.diff(out[:, 16] - out[:, 33], axis=0), axis=1) * fps
    k = int(np.argmax(wrist_true))
    lo, hi = max(0, k - 3), k + 4
    acc_out = np.abs(np.diff(out[:, 11:29], n=2, axis=0)).mean()
    acc_true = np.abs(np.diff(true3[:, 11:29], n=2, axis=0)).mean()
    return dict(depth=float(np.mean(np.abs(d - true_d)) / body * 100),
                xy=float(np.mean(np.linalg.norm(xy - true_xy, axis=2)) / body * 100),
                strike=float(wrist_out[lo:hi].max() / wrist_true[lo:hi].max() * 100),
                jitter=float(acc_out / acc_true))


SCENARIOS = {
    "two cameras": dict(),
    "side at 70 deg": dict(side_angle=70.0, swap_bursts=2),
    "noisy tracker": dict(noise=0.012, swap_bursts=4),
    "separate takes": dict(separate=True, swap_bursts=1, offset=0.2),
}


def run(method="improved", seeds=(0, 1, 2), **kwargs):
    """:return: {scenario: mean metrics over the seeds}"""
    table = {}
    for name, sc in SCENARIOS.items():
        rows = []
        for seed in seeds:
            take = make_take(seed=seed, **sc)
            if method == "simple":
                frames = combine.combine_simple(take["front"], take["side"])
            else:
                frames, _ = combine.combine_improved(
                    take["front"], take["side"], take["tF"], take["tS"], take["visF"], take["visS"],
                    sync="WARP" if sc.get("separate") else "OFFSET", **kwargs)
            rows.append(evaluate(frames, take))
        table[name] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    return table


def show(title, table):
    print(f"\n{title}")
    print(f"  {'scenario':<16}{'depth %':>9}{'xy %':>8}{'strike %':>10}{'jitter':>8}")
    for name, m in table.items():
        print(f"  {name:<16}{m['depth']:>9.2f}{m['xy']:>8.2f}{m['strike']:>10.0f}{m['jitter']:>8.2f}")


if __name__ == "__main__":
    simple = run("simple")
    improved = run("improved")
    show("Original method", simple)
    show("Improved method", improved)
    failed = []
    for name in SCENARIOS:
        if name != "separate takes" and improved[name]["depth"] > 0.8 * simple[name]["depth"]:
            failed.append(f"{name}: improved depth error {improved[name]['depth']:.2f}% is not clearly below "
                          f"the original's {simple[name]['depth']:.2f}%")
        if improved[name]["strike"] < 70:
            failed.append(f"{name}: only {improved[name]['strike']:.0f}% of the strike speed is kept")
    print("\n" + ("FAILED\n  " + "\n  ".join(failed) if failed else "OK"))
    sys.exit(1 if failed else 0)
