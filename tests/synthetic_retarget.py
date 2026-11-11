"""
    Ground-truth test of the retarget's leg solver (numpy and mocked Blender, no scene):

        python -B tests/synthetic_retarget.py

    An actor steps, lunges, crouches and turns on the ball of one foot. Where each foot stands,
    when it pivots and where the knees point is known exactly. The "mocap" the solver gets is that
    motion as a tracker would deliver it: in picture units, growing as the actor comes nearer,
    with jitter on every landmark. The solver's legs are then compared with the truth:

        slide     speed of the solved ankle while the foot truly stands flat, cm/s (0 = planted)
        pivot     how far the solved ankle strays from its true arc while the foot turns on its ball, cm
        air       how far the solved ankle is from the true path while the foot is in the air, cm
                  (holding a foot that should move shows here)
        contact   share of frames whose foot-down / foot-up state is found correctly, %
        knee      mean error of the knee direction, degrees
        flips     knee direction jumps of more than 45 degrees between frames
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_capture_batch import retarget as R  # noqa: E402  (the add-on's retarget with mocked Blender)

UNIT = 3.4                       # metres -> tracked units at the actor's usual distance
THIGH, SHIN, HIP_HALF, FOOT = 0.45, 0.43, 0.1, 0.17
HEEL_BACK, SOLE = 0.05, 0.07     # heel behind and sole below the ankle


def _ease(t, t0, t1):
    x = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    return x * x * (3 - 2 * x)


def true_motion(t):
    """
        :return: dict of (N, 3) arrays in metres (x actor's left, y away from the camera, z up):
                 hips, shoulders, per side hip/knee/ankle/heel/toe; and per side the masks flat,
                 pivot (on the ball of the foot) and air.
    """
    n = len(t)
    out = {"t": t}
    # feet: where the ankle stands, moves and turns
    yaw = {"Left": np.zeros(n), "Right": np.zeros(n)}                # heading of the foot, 0 = toward the camera
    ankle = {}
    state = {}
    plan = {"Left": [((0.10, 0.00), 0.0), ((0.12, -0.55), 2.0), ((0.12, -0.55), 9.0)],      # (x, y) until time
            "Right": [((-0.10, 0.00), 0.0), ((-0.10, 0.00), 3.6), ((-0.30, -0.20), 9.0)]}
    steps = {"Left": [(1.2, 1.7)], "Right": [(3.6, 4.1)]}            # in the air between these times
    for side in ("Left", "Right"):
        p = np.zeros((n, 3))
        p[:, 2] = SOLE
        a, b = plan[side][0][0], plan[side][1][0]
        if side == "Right":
            a, b = plan[side][1][0], plan[side][2][0]
        (t0, t1), = steps[side]
        s = _ease(t, t0, t1)
        p[:, 0] = a[0] + (b[0] - a[0]) * s
        p[:, 1] = a[1] + (b[1] - a[1]) * s
        p[:, 2] += 0.14 * np.sin(np.pi * np.clip((t - t0) / (t1 - t0), 0, 1)) ** 2
        air = (t > t0) & (t < t1)
        ankle[side] = p
        state[side] = np.where(air, 2, 0)                            # 0 flat, 1 pivot, 2 air
    # the right foot turns 70 degrees on its ball between 5.0 and 5.8 s: the ball stays, the ankle swings
    turn = np.radians(70.0) * _ease(t, 5.0, 5.8)
    yaw["Right"] = turn
    pivoting = (t > 5.0) & (t < 5.8)
    state["Right"] = np.where(pivoting, 1, state["Right"])
    fwd0 = np.array([0.0, -1.0, 0.0])
    feet = {}
    for side in ("Left", "Right"):
        c, s_ = np.cos(yaw[side]), np.sin(yaw[side])
        fwd = np.stack([s_, -c, np.zeros(n)], axis=1)                 # heading turned about z
        a = ankle[side].copy()
        if side == "Right":
            # keep the ball where it stood before the turn: ball = ankle0 + FOOT * fwd0, ankle = ball - FOOT * fwd
            lifted = 0.04 * np.sin(np.pi * np.clip((t - 5.0) / 0.8, 0, 1)) ** 2          # heel comes up a little
            ball = a + np.array([0.0, -FOOT, -SOLE])[None, :] * np.array([1, 1, 0]) + np.array([0, 0, 0])
            a = ball - FOOT * fwd
            a[:, 2] = SOLE + lifted
        feet[side] = dict(ankle=a, toe=a + FOOT * fwd + np.array([0, 0, -SOLE + 0.01]),
                          heel=a - HEEL_BACK * fwd + np.array([0, 0, -SOLE + 0.01]))
        if side == "Right":
            feet[side]["toe"][:, 2] -= lifted                 # the ball of the foot stays on the floor
            feet[side]["heel"][:, 2] += 2.0 * lifted
    # hips: over the feet, crouching in the lunge, standing tall (legs nearly straight) at the end
    mid = (feet["Left"]["ankle"] + feet["Right"]["ankle"]) / 2
    hips = mid.copy()
    hips[:, 0] += 0.03 * np.sin(1.3 * t)
    hips[:, 1] += 0.04 * np.sin(0.9 * t + 0.4)
    stance = np.linalg.norm(feet["Left"]["ankle"][:, :2] - feet["Right"]["ankle"][:, :2], axis=1)
    tall = np.sqrt(np.maximum((THIGH + SHIN) ** 2 - (stance / 2) ** 2, 0.2)) + SOLE
    hips[:, 2] = tall * (0.9 - 0.12 * _ease(t, 2.0, 2.6) + 0.12 * _ease(t, 3.2, 3.6) + 0.097 * _ease(t, 6.5, 7.2))
    body_yaw = 0.3 * np.sin(0.7 * t) + 0.5 * turn
    lat = np.stack([np.cos(body_yaw), np.sin(body_yaw), np.zeros(n)], axis=1)
    out["hips"] = hips
    out["shoulders"] = hips + np.array([0.0, -0.03, 0.5])
    out["lat"] = lat
    for side, sign in (("Left", 1.0), ("Right", -1.0)):
        hip = hips + sign * HIP_HALF * lat
        an = feet[side]["ankle"]
        T = an - hip
        d = np.clip(np.linalg.norm(T, axis=1), 0.05, (THIGH + SHIN) * 0.9995)
        tdir = T / np.linalg.norm(T, axis=1, keepdims=True)
        c_, s_ = np.cos(yaw[side] + 0.25 * sign), np.sin(yaw[side] + 0.25 * sign)
        knee_way = np.stack([s_, -c_, np.zeros(n)], axis=1)          # knees point where the feet point, a little out
        pole = knee_way - np.sum(knee_way * tdir, axis=1, keepdims=True) * tdir
        pole /= np.linalg.norm(pole, axis=1, keepdims=True)
        cos_a = np.clip((THIGH ** 2 + d ** 2 - SHIN ** 2) / (2 * THIGH * d), -1, 1)
        knee = hip + THIGH * (cos_a[:, None] * tdir + np.sqrt(1 - cos_a ** 2)[:, None] * pole)
        out[side] = dict(hip=hip, knee=knee, ankle=hip + tdir * d[:, None], heel=feet[side]["heel"], toe=feet[side]["toe"],
                         pole=pole, bend=np.degrees(np.arccos(np.clip((THIGH ** 2 + SHIN ** 2 - d ** 2) / (2 * THIGH * SHIN), -1, 1))),
                         flat=state[side] == 0, pivot=state[side] == 1, air=state[side] == 2)
    return out


def tracked(truth, rng, noise=0.006, camera_distance=5.0, camera_height=1.3):
    """
        The motion as landmark positions in the empties' space (N, 40, 3): picture width, depth, picture
        height, in tracked units; bigger and lower in the picture the nearer the actor is.
    """
    n = len(truth["t"])
    P = np.zeros((n, 40, 3))
    put = {R.HIP_C: truth["hips"], R.SHOULDER_C: truth["shoulders"]}
    for side, (h, k, a, heel, toe) in (("Left", (R.L_HIP, R.L_KNEE, R.L_ANKLE, R.L_HEEL, R.L_TOE)),
                                       ("Right", (R.R_HIP, R.R_KNEE, R.R_ANKLE, R.R_HEEL, R.R_TOE))):
        for i, key in ((h, "hip"), (k, "knee"), (a, "ankle"), (heel, "heel"), (toe, "toe")):
            put[i] = truth[side][key]
    for i, p in put.items():
        near = camera_distance / np.maximum(camera_distance + p[:, 1], 0.5) if camera_distance else 1.0
        P[:, i, 0] = p[:, 0] * near * UNIT
        P[:, i, 1] = p[:, 1] * UNIT                      # depth as a second camera would give it
        P[:, i, 2] = (camera_height + (p[:, 2] - camera_height) * near) * UNIT
        P[:, i] += rng.normal(size=(n, 3)) * noise * UNIT * np.array([1.0, 2.5, 1.0])   # depth is the noisy one
    return P


def solve(P, fps, plant=True, hip_height=True):
    """Run the retarget's floor and leg solve on landmark positions. :return: solver output in metres."""
    frames = np.arange(len(P), dtype=float)
    saved = R._true, R.landmark_confidence
    R._true = lambda _e, idx, _f, _m: {i: P[:, i].copy() for i in idx}
    R.landmark_confidence = lambda _e, f, _i: np.ones(len(f))
    try:
        floor = R.mocap_floor(None, frames, fps, False)
        leg = float(np.median(np.linalg.norm(P[:, R.L_HIP] - P[:, R.L_KNEE], axis=1)
                              + np.linalg.norm(P[:, R.L_KNEE] - P[:, R.L_ANKLE], axis=1)))
        ratio = leg / (THIGH + SHIN)
        offsets = {"Left": (HIP_HALF, 0.0, 0.0), "Right": (-HIP_HALF, 0.0, 0.0)}
        legs, shift, share = R.leg_directions(None, frames, fps, False, ratio, THIGH, SHIN, offsets, 0.0, plant,
                                              floor if hip_height else None)
    finally:
        R._true, R.landmark_confidence = saved
    out = {"down": floor[1], "share": share, "ratio": ratio}
    for side, (joint, thigh, knee, shin) in legs.items():
        ankle = knee + SHIN * ratio * shin
        an, toe_i = (R.L_ANKLE, R.L_TOE) if side == "Left" else (R.R_ANKLE, R.R_TOE)
        way = P[:, toe_i] - P[:, an]
        way = R._gauss_np(way, 0.1 * fps)
        way /= np.linalg.norm(way, axis=1, keepdims=True)
        out[side] = dict(joint=joint / ratio, knee=knee / ratio, ankle=ankle / ratio, toe=ankle / ratio + FOOT * way,
                         pole=_perp(knee - joint, ankle - joint))
    return out


def _perp(v, along):
    along = along / np.maximum(np.linalg.norm(along, axis=1, keepdims=True), 1e-9)
    v = v - np.sum(v * along, axis=1, keepdims=True) * along
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def evaluate(solved, truth, fps):
    rows = {k: [] for k in ("slide", "pivot", "air", "contact", "knee", "knee_straight", "flips")}
    for side in ("Left", "Right"):
        tr, so = truth[side], solved[side]
        speed = lambda p: np.linalg.norm(np.gradient(p, axis=0), axis=1) * fps * 100
        inner = lambda m: m & np.roll(m, 3) & np.roll(m, -3)             # away from the changes of state
        flat, pivot, air = inner(tr["flat"]), inner(tr["pivot"]), inner(tr["air"])
        rows["slide"].append(speed(so["ankle"])[flat].mean())
        if pivot.any():
            k = int(np.argmax(pivot))                                    # both paths from where the turn starts
            stray = (so["ankle"] - so["ankle"][k]) - (tr["ankle"] - tr["ankle"][k])
            rows["pivot"].append(np.linalg.norm(stray, axis=1)[pivot].mean() * 100)
        if air.any():
            rel_true = tr["ankle"] - truth["hips"]
            rel = so["ankle"] - (so["joint"] - (1 if side == "Left" else -1) * HIP_HALF * truth["lat"])
            rows["air"].append(np.linalg.norm(rel - rel_true, axis=1)[air].mean() * 100)
        down_true = tr["flat"] | tr["pivot"]
        rows["contact"].append(np.mean(solved["down"][side] == down_true) * 100)
        ang = np.degrees(np.arccos(np.clip(np.sum(so["pole"] * tr["pole"], axis=1), -1, 1)))
        rows["knee"].append(ang.mean())
        straight = tr["bend"] > 160
        if straight.any():
            rows["knee_straight"].append(ang[straight].mean())
        step = np.degrees(np.arccos(np.clip(np.sum(so["pole"][1:] * so["pole"][:-1], axis=1), -1, 1)))
        rows["flips"].append(float(np.sum(step > 45)))
    return {k: (float(np.sum(v)) if k == "flips" else float(np.mean(v)) if v else float("nan")) for k, v in rows.items()}


def run(seeds=(0, 1, 2), fps=30.0, duration=8.0, noise=0.006, **kwargs):
    """:return: mean metrics over the seeds."""
    rows = []
    for seed in seeds:
        truth = true_motion(np.arange(0, duration, 1.0 / fps))
        P = tracked(truth, np.random.default_rng(seed), noise)
        rows.append(evaluate(solve(P, fps, **kwargs), truth, fps))
    return {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}


def show(title, m):
    print(f"{title:<28} slide {m['slide']:5.1f} cm/s  pivot {m['pivot']:4.1f} cm  air {m['air']:4.1f} cm  "
          f"contact {m['contact']:3.0f}%  knee {m['knee']:4.1f} deg (straight legs {m['knee_straight']:4.1f})  flips {m['flips']:.1f}")


if __name__ == "__main__":
    show("feet not held", run(plant=False))
    planted = run()
    show("feet held", planted)
    show("feet held, noisy tracker", run(noise=0.012))
    failed = []
    if planted["slide"] > 3.0:
        failed.append(f"standing feet slide {planted['slide']:.1f} cm/s")
    if planted["contact"] < 85:
        failed.append(f"only {planted['contact']:.0f}% of the foot-down frames found")
    print("\n" + ("FAILED\n  " + "\n  ".join(failed) if failed else "OK"))
    sys.exit(1 if failed else 0)
