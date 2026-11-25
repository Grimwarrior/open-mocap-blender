"""
    NVIDIA GEM-X takes (single-video whole-body motion, SOMA 77 joints) as an Open Mocap capture.

    GEM-X runs in its own Python environment (PyTorch, CUDA), outside Blender: gemx_runner.py tracks
    the video there and writes the joints to an .npz file. This module (numpy only) turns that file
    into what pose tracking plus Combine Data would leave in the capture cache: 40 points per frame
    as [x, y, depth] (x right, y down, depth away from the camera, in tracker units), times,
    visibility and depth metadata, and 21 points per hand. Animate and Retarget then work as for any
    other take; Combine Data and its perspective step are not needed, GEM-X's joints are 3D already.

    The world-space joints are used (gravity up, the floor at height 0, feet held where GEM-X's
    contact clean-up found them standing), turned so that x / depth are the camera's horizontal
    right / forward. SOMA has no ears, nose, mouth or heels: they are placed from the head (eyes, top
    of the head) and the sole (ball -> toe tip), at average adult offsets.
"""
import numpy as np

UNIT = 3.4                  # tracker units per metre: a 1.75 m actor is about 6 units (600 px) tall
CENTROIDS = {33: (23, 24), 34: (11, 12), 35: (18, 20), 36: (17, 19), 37: (7, 8), 38: (9, 10), 39: (1, 4)}

# Open Mocap / MediaPipe pose point <- SOMA joint
POSE_FROM_SOMA = {
    2: "LeftEye", 5: "RightEye",
    11: "LeftArm", 12: "RightArm", 13: "LeftForeArm", 14: "RightForeArm", 15: "LeftHand", 16: "RightHand",
    17: "LeftHandPinky2", 18: "RightHandPinky2", 19: "LeftHandIndex2", 20: "RightHandIndex2",
    21: "LeftHandThumb3", 22: "RightHandThumb3",
    23: "LeftLeg", 24: "RightLeg", 25: "LeftShin", 26: "RightShin", 27: "LeftFoot", 28: "RightFoot",
}
# MediaPipe hand point (0 wrist, then 4 per finger from the knuckle out) <- SOMA joint name (after the side)
HAND_FROM_SOMA = ["Hand", "HandThumb1", "HandThumb2", "HandThumb3", "HandThumbEnd"] + [
    f"Hand{finger}{part}" for finger in ("Index", "Middle", "Ring", "Pinky") for part in ("2", "3", "4", "End")]

# The spine: SOMA's chain from the hips to the head, kept with the take (Animate) so the retarget can
# give each Mixamo spine bone its SOMA segment's bend (retarget.spine_segment_directions), plus the
# actor's skeleton at rest for the chain's neutral curve and the hip / shoulder lines it is turned by.
SPINE_CHAIN = ("Hips", "Spine1", "Spine2", "Chest", "Neck1", "Head")
SPINE_REST_EXTRA = ("LeftLeg", "RightLeg", "LeftArm", "RightArm")

# Face and heel points at average adult offsets (metres) in the head's / foot's own frame
EAR_OUT, EAR_BACK = 0.075, 0.01            # from the head joint, which sits between the ears
EYE_CORNER = 0.015                          # inner / outer eye corner from the eye centre
NOSE_FWD, NOSE_DOWN = 0.025, 0.035          # from the point between the eyes
MOUTH_DOWN, MOUTH_FWD, MOUTH_OUT = 0.075, 0.005, 0.025
# The sole. SOMA's rest mesh (heavy_backtofront's actor): the ankle joint 0.077 m over the floor, the ball
# joint (ToeBase) 0.021 m over the sole under it, the ankle -> ball line falling 21 deg. The heel goes
# HEEL_BELOW from the ankle across that line, away from the knee (0.075 m down, 0.03 m back on a standing
# foot: at the sole), and the sole line runs from it to BALL_ABOVE_SOLE under the ball joint. Through
# the ball joint itself the line rose toward the toes, and a flat-standing foot had its toe point
# 4.6-5.2 cm over the floor (motion_02_LeftToRight). Placed along the toes, the foot turned with them
# (they pointed 20-45 deg off the foot's own axis on heavy_backtofront).
HEEL_BELOW = 0.08
BALL_ABOVE_SOLE = 0.02
# Which side of the foot the heel is on comes from the shin; a foot pointed in line with the shin (less
# than about 30 deg, HEEL_RELIABLE_SINE) counts less and is held from the frames around it, over
# HEEL_HOLD_FRAMES (4 x that where few frames are sure). motion_02_LeftToRight, right foot pointed
# 71-86 deg down in frames 19 and 61-64: the heel flipped above the ankle and the foot turned ~160 deg.
HEEL_RELIABLE_SINE = 0.5
HEEL_HOLD_FRAMES = 3.0
# The toe point (MediaPipe's foot index) goes TOE_AHEAD past the ball along the sole line, so the foot
# reads as one rigid piece: SOMA's toe tips bend and turn on their own, up to 36 deg off the foot's axis
# (95th percentile, heavy_backtofront), and the retarget turns the whole foot with its toe point.
TOE_AHEAD = 0.05


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def camera_axes(R_w2c, up=(0.0, 1.0, 0.0)):
    """
        (right, down, forward) unit vectors in GEM-X world coordinates: the camera's own right and
        forward turned level about the world's up, and down = -up. Right-handed like the camera.
        :param R_w2c: (3, 3) world -> camera rotation (camera x right, y down, z forward).
    """
    R = np.asarray(R_w2c, dtype=float)
    up = np.asarray(up, dtype=float)
    down = -up
    fwd = R[2] - np.dot(R[2], up) * up
    if np.linalg.norm(fwd) < 1e-6:                     # camera looking straight down/up: use its y axis
        fwd = R[1] - np.dot(R[1], up) * up
    fwd = _unit(fwd)
    right = np.cross(down, fwd)                        # x = y cross z for right-handed (x, y, z)
    if np.dot(right, R[0]) < 0:                        # a mirrored world frame: keep the camera's right
        right = -right
    return right, down, fwd


def world_to_camera(joints_global, joints_incam):
    """
        (3, 3) rotation taking GEM-X world directions to camera directions, fitted (Kabsch) on every
        joint relative to its frame's first joint. GEM-X's own R_w2c is the identity for a fixed camera
        and says nothing about how the world frame sits; the two joint sets do. The world path and
        the camera-space path differ (contact clean-up, distances), so positions are taken per frame.
    """
    G = np.asarray(joints_global, dtype=float)
    C = np.asarray(joints_incam, dtype=float)
    g = (G - G[:, :1]).reshape(-1, 3)
    c = (C - C[:, :1]).reshape(-1, 3)
    U, _, Vt = np.linalg.svd(g.T @ c)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def face_and_heels(J, names):
    """
        Ears, eye corners, nose, mouth corners, heels and toes (MediaPipe 0, 1, 3, 4, 6, 7, 8, 9, 10, 29-32)
        from the SOMA joints. :param J: (T, 77, 3) joints (any right-handed metric frame).
        :param names: the 77 joint names. :return: {MediaPipe index: (T, 3)}.
    """
    idx = {n: i for i, n in enumerate(names)}
    g = lambda n: J[:, idx[n]]
    head, top = g("Head"), g("HeadEnd")
    eye_l, eye_r = g("LeftEye"), g("RightEye")
    up = _unit(top - head)
    left = eye_l - eye_r
    left = _unit(left - np.sum(left * up, axis=1, keepdims=True) * up)
    eyes = (eye_l + eye_r) / 2
    fwd = eyes - head
    fwd = _unit(fwd - np.sum(fwd * up, axis=1, keepdims=True) * up - np.sum(fwd * left, axis=1, keepdims=True) * left)
    out = {
        7: head + EAR_OUT * left - EAR_BACK * fwd, 8: head - EAR_OUT * left - EAR_BACK * fwd,
        1: eye_l - EYE_CORNER * left, 3: eye_l + EYE_CORNER * left,
        4: eye_r + EYE_CORNER * left, 6: eye_r - EYE_CORNER * left,
        0: eyes + NOSE_FWD * fwd - NOSE_DOWN * up,
        9: eyes - MOUTH_DOWN * up + MOUTH_FWD * fwd + MOUTH_OUT * left,
        10: eyes - MOUTH_DOWN * up + MOUTH_FWD * fwd - MOUTH_OUT * left,
    }
    for side, k in (("Left", 29), ("Right", 30)):
        knee, ankle, ball = g(f"{side}Shin"), g(f"{side}Foot"), g(f"{side}ToeBase")
        along = _unit(ball - ankle)                    # the foot's own axis, ankle -> ball
        shin = knee - ankle
        across = shin - np.sum(shin * along, axis=1, keepdims=True) * along      # across it, knee side
        # A foot pointed in line with the shin (toes down in a kick) leaves next to no "across": such
        # frames count less and take the side from the frames around them, or the heel flips over
        sine = np.linalg.norm(across, axis=1) / np.maximum(np.linalg.norm(shin, axis=1), 1e-9)
        weight = np.clip(sine / HEEL_RELIABLE_SINE, 0.0, 1.0) ** 2 + 1e-4
        normal = _unit(across)
        held = np.stack([_gauss(normal[:, i] * weight, HEEL_HOLD_FRAMES) for i in range(3)], axis=1)
        held /= _gauss(weight, HEEL_HOLD_FRAMES)[:, None]
        wide = np.stack([_gauss(normal[:, i] * weight, 4 * HEEL_HOLD_FRAMES) for i in range(3)], axis=1)
        wide /= _gauss(weight, 4 * HEEL_HOLD_FRAMES)[:, None]
        sure = np.clip(_gauss(weight, HEEL_HOLD_FRAMES) / 0.3, 0.0, 1.0)[:, None]
        normal = sure * held + (1.0 - sure) * wide
        normal = _unit(normal - np.sum(normal * along, axis=1, keepdims=True) * along)
        out[k] = ankle - HEEL_BELOW * normal           # under the ankle, in the foot's own plane
        sole_ball = ball - BALL_ABOVE_SOLE * normal    # the sole under the ball joint
        out[k + 2] = sole_ball + TOE_AHEAD * _unit(sole_ball - out[k])   # toe on the sole line (31, 32)
    return out


# The floor. GEM-X's world heights drift: on heavy_backtofront (3.4 s) standing feet were at the floor in
# the first frames and 0.19 m above it in the last (camera space was worse, 0.14 m spread against
# 0.05). GEM-X also says which feet stand still: static_conf, per frame, for the left ankle and ball,
# the right ankle and ball, then the hands. In every frame where a foot stands, its lowest point
# (heel, ball or toe tip) is on the floor; the drift between those frames is interpolated and
# smoothed over FLOOR_SMOOTH_S, and taken off every joint's height. Frames with no standing foot
# (a jump) keep the floor of the frames around them.
FLOOR_STATIC = 0.5
FLOOR_SMOOTH_S = 0.3
FOOT_CHANNELS = {"Left": (0, 1), "Right": (2, 3)}


def _gauss(x, sigma):
    if sigma <= 0 or len(x) < 3:
        return x
    r = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    padded = np.pad(x, r, mode="reflect", reflect_type="odd")   # keeps a steady drift straight at the ends
    return np.convolve(padded, k / k.sum(), mode="valid")


def floor_drift(J, names, static, fps):
    """
        (T,) height (metres, up = +Y) the floor has drifted to in each frame, from the feet GEM-X says
        stand still (see above); None if no foot ever stands.
    """
    idx = {n: i for i, n in enumerate(names)}
    heels = face_and_heels(J, names)
    static = np.asarray(static, dtype=float)
    lowest = np.full(len(J), np.nan)
    for side, heel in (("Left", 29), ("Right", 30)):
        low = np.minimum.reduce([J[:, idx[side + "ToeBase"], 1], heels[heel + 2][:, 1], heels[heel][:, 1]])
        stands = static[:, list(FOOT_CHANNELS[side])].max(axis=1) > FLOOR_STATIC
        lowest = np.where(stands, np.fmin(lowest, low), lowest)
    known = np.isfinite(lowest)
    if not known.any():
        return None
    t = np.arange(len(J), dtype=float)
    drift = np.interp(t, t[known], lowest[known])
    return _gauss(drift, FLOOR_SMOOTH_S * fps)


# Each foot on the floor. The floor above is one height per frame, so where both feet stand GEM-X can
# still have one a few centimetres above it (motion_02_LeftToRight: ankles 3.0 cm apart in height,
# median, over the 19 frames both stood; the toe points 5.6 cm). Every frame a foot stands, its lowest
# point is put on the floor: the foot (ankle, ball, toe tip) moves down by that much and the knee by
# PIN_KNEE_SHARE of it, the hip stays. Between its standing frames a foot keeps the correction of the
# frames around it, smoothed over PIN_SMOOTH_S, so it does not jump at lift-off and landing; at most
# PIN_MAX either way.
PIN_KNEE_SHARE = 0.5
PIN_SMOOTH_S = 0.1
PIN_MAX = 0.08


def pin_feet(J, names, static, fps):
    """:return: (J with every standing foot's lowest point on the floor, {side: (T,) correction})."""
    idx = {n: i for i, n in enumerate(names)}
    static = np.asarray(static, dtype=float)
    J = J.copy()
    lift = {}
    heels = face_and_heels(J, names)
    for side, heel in (("Left", 29), ("Right", 30)):
        low = np.minimum.reduce([J[:, idx[side + "ToeBase"], 1], heels[heel + 2][:, 1], heels[heel][:, 1]])
        stands = static[:, list(FOOT_CHANNELS[side])].max(axis=1) > FLOOR_STATIC
        if not stands.any():
            continue
        t = np.arange(len(J), dtype=float)
        off = np.clip(_gauss(np.interp(t, t[stands], low[stands]), PIN_SMOOTH_S * fps), -PIN_MAX, PIN_MAX)
        for n, share in ((side + "Foot", 1.0), (side + "ToeBase", 1.0), (side + "ToeEnd", 1.0),
                         (side + "Shin", PIN_KNEE_SHARE)):
            J[:, idx[n], 1] -= share * off
        lift[side] = off
    return J, lift


# A flat foot. Where GEM-X says both the ankle and the ball of a foot stand still, the foot stands flat,
# but its joints can still have it tipped: motion_02_LeftToRight's right foot, flat by that measure,
# had its toe point 4.4 cm over the floor (median, the frames both feet stood). There the heel and toe
# point are put on the floor (height 0 after the levelling above), eased in and out over
# FLAT_BLEND_S; a foot on its ball alone (heel up) keeps its pitch.
FLAT_BLEND_S = 0.07


def flatten_soles(derived, static, fps):
    """derived (face_and_heels, world, floor at 0) with the heel / toe of flat-standing feet on the floor."""
    static = np.asarray(static, dtype=float) > FLOOR_STATIC
    out = dict(derived)
    for side, heel in (("Left", 29), ("Right", 30)):
        ankle_ch, ball_ch = FOOT_CHANNELS[side]
        flat = (static[:, ankle_ch] & static[:, ball_ch]).astype(float)
        w = np.clip(_gauss(flat, FLAT_BLEND_S * fps), 0.0, 1.0)[:, None] if flat.any() else None
        if w is None:
            continue
        for k in (heel, heel + 2):
            p = out[k].copy()
            p[:, 1:2] = (1.0 - w) * p[:, 1:2]
            out[k] = p
    return out


def to_capture(data, unit=UNIT):
    """
        :param data: dict-like of the runner's .npz: joints_global (T, 77, 3) metres, Y up;
                     joints_incam (T, 77, 3) the same in camera space (orients the world, see
                     world_to_camera; without it R_w2c (3, 3) or (T, 3, 3) is used);
                     joint_names (77 or 78 with 'Root' first); static_conf (T, 6) optional,
                     levels the floor (floor_drift) and puts each standing foot on it (pin_feet);
                     frame_times (T,) seconds; fps; width, height.
        :return: dict with the capture cache's lists: frames (T x 120 floats), times, vis, meta,
                 hands {"Left"/"Right": (T x 63 floats)}, spine (None, or points: T x 18 floats of
                 SPINE_CHAIN in the same units, rest: {name: SOMA rest position}, rest_up), fps,
                 width, height.
    """
    J = np.asarray(data["joints_global"], dtype=float)
    names = [str(n) for n in data["joint_names"]]
    if len(names) == J.shape[1] + 1:
        names = names[1:]                              # the joints come without 'Root'
    if len(names) != J.shape[1]:
        raise ValueError(f"{len(names)} joint names for {J.shape[1]} joints")
    idx = {n: i for i, n in enumerate(names)}
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    if "static_conf" in data:
        drift = floor_drift(J, names, data["static_conf"], fps)
        if drift is not None:
            J = J.copy()
            J[:, :, 1] -= drift[:, None]
            J, _ = pin_feet(J, names, data["static_conf"], fps)
    if "joints_incam" in data:
        R = world_to_camera(np.asarray(data["joints_global"], dtype=float), data["joints_incam"])
    else:
        R = np.asarray(data["R_w2c"], dtype=float).reshape(-1, 3, 3)[0]   # a fixed camera: one rotation
    right, down, fwd = camera_axes(R)
    M = np.stack([right, down, fwd])                   # world -> (x right, y down, depth forward)
    cv = J @ M.T * unit
    T = len(cv)
    pts = np.zeros((T, 40, 3))
    for k, n in POSE_FROM_SOMA.items():
        pts[:, k] = cv[:, idx[n]]
    derived = face_and_heels(J, names)
    if "static_conf" in data:
        derived = flatten_soles(derived, data["static_conf"], fps)
    for k, p in derived.items():
        pts[:, k] = p @ M.T * unit
    for c, (a, b) in CENTROIDS.items():
        pts[:, c] = (pts[:, a] + pts[:, b]) / 2
    hands = {side: cv[:, [idx[side + n] for n in HAND_FROM_SOMA]].reshape(T, 63) for side in ("Left", "Right")}
    times = np.asarray(data["frame_times"], dtype=float).reshape(-1)[:T]
    spine = None
    if "joints_rest" in data and all(n in idx for n in SPINE_CHAIN + SPINE_REST_EXTRA):
        rest = np.asarray(data["joints_rest"], dtype=float).reshape(-1, 3)
        if len(rest) == len(names):
            spine = {"names": list(SPINE_CHAIN), "points": cv[:, [idx[n] for n in SPINE_CHAIN]].reshape(T, -1).tolist(),
                     "rest": {n: rest[idx[n]].tolist() for n in SPINE_CHAIN + SPINE_REST_EXTRA},
                     "rest_up": [0.0, 1.0, 0.0]}
    meta = [{"depth_confidence": [1.0] * 40, "depth_source": "gemx"} for _ in range(T)]
    if "static_conf" in data:
        for row, confidence in zip(meta, data["static_conf"]):
            row["foot_contact"] = [float(x) for x in confidence[:4]]
    return {"frames": pts.reshape(T, 120).tolist(), "times": times.tolist(), "vis": np.ones((T, 40)).tolist(),
            "meta": meta, "hands": {s: h.tolist() for s, h in hands.items()}, "spine": spine, "fps": fps,
            "width": int(np.asarray(data["width"]).reshape(-1)[0]),
            "height": int(np.asarray(data["height"]).reshape(-1)[0])}


# -------------------------------------------------------------
# THE SWORD FROM ONE VIDEO
# -------------------------------------------------------------
# A katana does not move in the hands that hold it: its blade is one fixed direction in each hand's
# own frame. GEM-X gives each hand's frame (wrist, index and pinky knuckles, all 3D), and Track Sword
# finds the blade's direction in the picture where it is plain to see. So the blade's direction in
# each hand is fitted once per take, to the frames where the picture shows the blade (the median
# miss there is what is minimised, so the frames the tracker gets wrong count little), and then gives
# the blade in 3D in every frame, also where the picture loses it (blurred, dark, behind the body).
# Where the picture is sure, it sets the blade's direction across the picture and the hands only its
# lean toward or away from the camera. Both hands on the handle are averaged, each by how well its fit
# matched; held in one hand (HANDS_APART_M), only that hand counts. KatanaKomibination_RTL (one camera,
# 183 frames): the rig's blade as retargeted before missed the picture's by a median 92 deg; fitted
# grips alone 9-14 deg (14-15 on frames left out of the fit); with the picture's direction and the
# rig's own sword laid along it, 3.3 deg (all frames the tracker is sure of), 7 % over 45 deg.
# The fitted blade ran nearly along GEM-X's hand (wrist -> knuckles), tilted 12-15 deg to the palm:
# GEM-X closes its hands round a handle loosely. GRIP_PRIOR is that, for takes the picture cannot fit.
GRIP_FIT_CONF = 0.5           # tracked blade confidence from which a frame takes part in the fit ...
GRIP_FIT_MIN_FRAMES = 12      # ... and how many such frames a fit needs (else GRIP_PRIOR)
GRIP_INLIER_DEG = 30.0        # the fit is repeated on frames this close to it
GRIP_PRIOR = {"Right": (0.0, 0.965, 0.26), "Left": (0.0, 0.965, -0.26)}   # (across, along, palm normal)
SWORD_PICTURE_CONF = (0.3, 0.6)   # the picture's blade sets its direction from this confidence (all of it from the second)
SWORD_MAX_LEAN = 80.0         # degrees the blade may lean toward / away from the camera there
SWORD_FILL_S = 0.1            # unsure frames take the picture's direction from sure frames this near ...
SWORD_HANDS_FILL = 0.05       # ... the hands' with this much weight against them (all of it where none is near)
SWORD_SMOOTH_S = 0.05
# One hand or both. KatanaKomibination_RTL: both hands on the handle 0.19-0.20 m apart (wrists, median
# over frames 1-95), the one-handed overhead and side cuts 0.66-1.20 m. Apart, only the hand holding the
# sword says where the blade is (the picture's search tells which, sword.GRIP_SPLIT).
HANDS_APART_M = 0.4


def picture_capture(data):
    """
        The 40 capture points as GEM-X saw them in the picture: (frames (T, 120) in picture coordinates,
        x = (px - w/2) / 100, y = (py - h) / 100, depth 0; vis (T, 40)). None without 2D keypoints.
    """
    if "kp2d" not in data:
        return None
    kp = np.asarray(data["kp2d"], dtype=float)
    names = [str(n) for n in data["joint_names"]]
    if len(names) == kp.shape[1] + 1:
        names = names[1:]
    idx = {n: i for i, n in enumerate(names)}
    w = float(np.asarray(data["width"]).reshape(-1)[0])
    h = float(np.asarray(data["height"]).reshape(-1)[0])
    T = len(kp)
    P, V = np.zeros((T, 40, 3)), np.zeros((T, 40))
    joints = dict(POSE_FROM_SOMA)
    joints.update({0: "Head", 29: "LeftFoot", 30: "RightFoot", 31: "LeftToeEnd", 32: "RightToeEnd"})
    for k, n in joints.items():
        if n in idx:
            P[:, k, 0] = (kp[:, idx[n], 0] - w / 2) / 100.0
            P[:, k, 1] = (kp[:, idx[n], 1] - h) / 100.0
            V[:, k] = kp[:, idx[n], 2]
    for c, (a, b) in CENTROIDS.items():
        P[:, c] = (P[:, a] + P[:, b]) / 2
        V[:, c] = np.minimum(V[:, a], V[:, b])
    return P.reshape(T, 120), V


def hand_frames(J, names, side):
    """(T, 3, 3) the hand's frame, columns (across: pinky -> index, along: wrist -> knuckles, palm normal)."""
    idx = {n: i for i, n in enumerate(names)}
    wrist, index, pinky = (J[:, idx[side + n]] for n in ("Hand", "HandIndex2", "HandPinky2"))
    along = _unit((index + pinky) / 2 - wrist)
    across = _unit(index - pinky)
    normal = _unit(np.cross(along, across))
    across = np.cross(normal, along)
    return np.stack([across, along, normal], axis=2)


def _project(P, K):
    q = P @ np.asarray(K, dtype=float).T
    return q[..., :2] / np.maximum(q[..., 2:3], 1e-6)


def _picture_dirs(anchor, dirs, K, length=0.4):
    """Unit picture directions (pixels, x right, y down) of 3D directions from 3D anchors (camera metres)."""
    return _unit(_project(anchor + length * _unit(dirs), K) - _project(anchor, K))


def _sphere(step_deg):
    t = np.radians(np.arange(0.0, 180.0 + 1e-9, step_deg))
    p = np.radians(np.arange(0.0, 360.0, step_deg))
    tt, pp = np.meshgrid(t, p, indexing="ij")
    grid = np.stack([np.sin(tt) * np.cos(pp), np.sin(tt) * np.sin(pp), np.cos(tt)], -1).reshape(-1, 3)
    return np.unique(np.round(grid, 9), axis=0)


def _grip_search(frames, anchor, K, blade2d, mask, best=None):
    for step, around in (((4.0, None),) if best is None else ()) + ((0.75, 6.0),):
        cand = _sphere(step)
        if around is not None:
            cand = cand[np.degrees(np.arccos(np.clip(cand @ best[0], -1, 1))) <= around]
        dirs = np.einsum('tij,cj->cti', frames[mask], cand)                        # (C, M, 3)
        pic = _picture_dirs(anchor[mask][None], dirs, K)
        miss = np.degrees(np.arccos(np.clip(np.sum(pic * blade2d[mask][None], -1), -1, 1)))
        med = np.median(miss, axis=1)
        k = int(np.argmin(med))
        best = (cand[k], float(med[k]))
    return best


def grip_miss(frames, anchor, K, blade2d, b):
    """(T,) degrees between the picture's blade and the one this grip gives."""
    pic = _picture_dirs(anchor, np.einsum('tij,j->ti', frames, b), K)
    return np.degrees(np.arccos(np.clip(np.sum(pic * blade2d, 1), -1, 1)))


def fit_grip(frames, anchor, K, blade2d, mask):
    """
        The blade's direction in a hand's frame that best matches the picture: (unit (3,), median miss deg,
        inlier mask). Fitted on all confident frames, then again on those within GRIP_INLIER_DEG of it: a
        tracker sure of a wrong blade (dark against the wall, behind the body) otherwise pulls the grip.
        :param frames: (T, 3, 3) hand frames (camera); anchor (T, 3) the wrist; blade2d (T, 2) picture units.
    """
    best = _grip_search(frames, anchor, K, blade2d, mask)
    for _ in range(2):
        inliers = mask & (grip_miss(frames, anchor, K, blade2d, best[0]) < GRIP_INLIER_DEG)
        if inliers.sum() < GRIP_FIT_MIN_FRAMES:
            break
        best = _grip_search(frames, anchor, K, blade2d, inliers, best)
    inliers = mask & (grip_miss(frames, anchor, K, blade2d, best[0]) < GRIP_INLIER_DEG)
    return best[0], best[1], inliers


def sword_axis(data, blade_rows, other=None):
    """
        :param data: the runner's .npz (dict-like). :param blade_rows: Track Sword's rows for its picture
                     ([dx, dy, confidence] per frame).
        :param other: (data, blade_rows) of a second GEM-X take of the same move from another side
                      (filmed at another time is fine): it sets the blade's lean toward the camera.
        :return: (rows [x, y, depth, confidence] per frame, a unit blade direction in capture coordinates,
                 report dict).
    """
    J = np.asarray(data["joints_incam"], dtype=float)
    names = [str(n) for n in data["joint_names"]]
    if len(names) == J.shape[1] + 1:
        names = names[1:]
    idx = {n: i for i, n in enumerate(names)}
    K = np.asarray(data["K"], dtype=float).reshape(3, 3)
    rows = np.asarray(blade_rows, dtype=float).reshape(len(J), -1)
    blade2d, conf = _unit(rows[:, :2]), np.clip(rows[:, 2], 0, 1)
    mask = conf > GRIP_FIT_CONF
    apart = np.linalg.norm(J[:, idx["RightHand"]] - J[:, idx["LeftHand"]], axis=1) > HANDS_APART_M
    holder = sword_holder(rows[:, 3] if rows.shape[1] > 3 else np.zeros(len(J)), mask, apart)
    report = {"fit_frames": int(mask.sum()), "one_handed": float(np.mean(holder != 0))}
    preds, weights, wrists = {}, {}, {}
    for side, sign in (("Right", 1), ("Left", -1)):
        F = hand_frames(J, names, side)
        wrists[sign] = J[:, idx[side + "Hand"]]
        own = mask & ((holder == 0) | (holder == sign))      # a free hand says nothing about the blade
        if own.sum() >= GRIP_FIT_MIN_FRAMES:
            b, miss, inliers = fit_grip(F, wrists[sign], K, blade2d, own)
        else:
            b = _unit(np.asarray(GRIP_PRIOR[side], dtype=float))
            miss = float(np.median(grip_miss(F, wrists[sign], K, blade2d, b)[own])) if own.any() else 45.0
            inliers = np.zeros(len(J), dtype=bool)
        report[side] = {"grip": [round(float(x), 3) for x in b], "miss_deg": round(miss, 1),
                        "inliers": int(inliers.sum())}
        preds[sign] = np.einsum('tij,j->ti', F, b)
        weights[sign] = 1.0 / max(miss, 3.0) ** 2
    pred = _unit(weights[1] * preds[1] + weights[-1] * preds[-1])
    for sign in (1, -1):
        pred[holder == sign] = _unit(preds[sign][holder == sign])
    # Where the picture is sure, it sets where the blade points across the picture, and the hands only
    # how far it leans toward or away from the camera, which one camera cannot see: the blade is the
    # direction in the picture's plane through the camera, on the picture's side, at the hands' lean.
    # (Gated on agreeing with the hands, frames 19, 73, 97, 127 and 133 of KatanaKomibination_RTL kept
    # the hands' blade 60-100 deg off a plainly visible one.)
    # The blade as two angles: its direction across the picture, and its lean toward (+) or away from
    # the camera. Smoothing the 3D blade instead pulled sure frames off the picture's line wherever the
    # blade pointed nearly at the camera (KatanaKomibination_RTL frames 72 and 117: leaning 60-80 deg,
    # 17-47 deg off a plainly visible blade).
    wrist = np.where((holder == -1)[:, None], wrists[-1], wrists[1])
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    pic = _picture_dirs(wrist, pred, K)
    agree = np.degrees(np.arccos(np.clip(np.sum(pic * blade2d, 1), -1, 1)))
    lo, hi = SWORD_PICTURE_CONF
    w = np.clip((conf - lo) / (hi - lo), 0, 1)
    z_video = blade2d[:, 0] + 1j * blade2d[:, 1]
    z_hands = pic[:, 0] + 1j * pic[:, 1]
    # frames the picture is unsure of take its direction from sure frames nearby, else from the hands
    sigma = SWORD_FILL_S * fps
    near = _gauss_complex(w * z_video, sigma)
    weight = np.array([_gauss(w, sigma)]).reshape(-1) if len(w) >= 3 else w
    z_fill = (near + SWORD_HANDS_FILL * z_hands) / np.maximum(weight + SWORD_HANDS_FILL, 1e-9)
    z = w * z_video + (1 - w) * z_fill / np.maximum(np.abs(z_fill), 1e-9)
    direction = np.stack([z.real, z.imag], axis=1) / np.maximum(np.abs(z), 1e-9)[:, None]
    lean = _lean(wrist, pred, direction, K)
    if other is not None:
        # a second video of the same move: where it sees the blade, the two pictures' planes cross in it
        other_lean, other_weight, flip, other_report = second_view_lean(data, names, wrist, direction, K, *other,
                                                                        sure=w)
        direction = np.where(flip[:, None], -direction, direction)
        lean = np.where(flip, _lean(wrist, pred, direction, K), lean)
        lean = (1 - other_weight) * lean + other_weight * other_lean
        report["second_view"] = other_report
    if len(lean) >= 3:
        lean = _gauss(lean, SWORD_SMOOTH_S * fps)
    blade = _blade_at(wrist, direction, lean, K)
    report.update(picture_share=float(np.mean(w > 0.5)),
                  agree_median=float(np.median(agree[mask])) if mask.any() else None)
    # camera -> capture (the world turned level to the camera, see to_capture)
    R = world_to_camera(np.asarray(data["joints_global"], dtype=float), J)
    M = np.stack(camera_axes(R))
    cap = _unit(blade @ R @ M.T)
    return [[float(x), float(y), float(z), 0.9, float(h)] for (x, y, z), h in zip(cap, holder)], report


# A second video. One picture leaves the blade's lean toward or away from its camera open; a camera
# from another side sees exactly that. The two takes need not be filmed together: the same move done
# twice is matched in time by the body's own pose (dynamic time warping on the hands, elbows, knees,
# feet and head relative to the hips, in the body's frame, with their velocities), and the two
# pictures' planes are compared in that body frame, which both takes share whatever their cameras.
# Where the second video is sure of the blade and its plane is not nearly the first's, the line the
# two planes share gives the lean.
SECOND_VIEW_CONF = (0.3, 0.6)     # the second picture's confidence, as SWORD_PICTURE_CONF
SECOND_VIEW_PLANES = (0.25, 0.5)  # sine of the angle between the two planes from which their line counts
POSE_POINTS = ("LeftHand", "RightHand", "LeftForeArm", "RightForeArm", "LeftShin", "RightShin", "LeftFoot",
               "RightFoot", "Head")
WARP_VELOCITY = 1.0               # weight of the pose features' velocities next to the poses (DTW)


def body_frames(J, names):
    """(T, 3, 3) the body's frame in the joints' coordinates, columns (left, up, forward)."""
    idx = {n: i for i, n in enumerate(names)}
    hips, chest = J[:, idx["Hips"]], J[:, idx["Chest"]]
    lat = _unit(J[:, idx["LeftLeg"]] - J[:, idx["RightLeg"]])
    up = chest - hips
    up = _unit(up - np.sum(up * lat, 1, keepdims=True) * lat)
    return np.stack([lat, up, np.cross(lat, up)], axis=2)


def pose_features(J, names, fps):
    """(T, F) the move as the body does it, whatever the camera: body-frame poses and their velocities."""
    idx = {n: i for i, n in enumerate(names)}
    B = body_frames(J, names)
    hips = J[:, idx["Hips"]]
    size = max(float(np.median(np.linalg.norm(J[:, idx["Chest"]] - hips, axis=1))), 1e-6)
    rel = J[:, [idx[n] for n in POSE_POINTS]] - hips[:, None]
    local = np.einsum('tij,tki->tkj', B, rel).reshape(len(J), -1) / size
    local = np.stack([_gauss(local[:, k], 0.05 * fps) for k in range(local.shape[1])], 1) if len(J) >= 3 else local
    vel = np.gradient(local, axis=0) * fps / 4.0 if len(J) >= 2 else np.zeros_like(local)
    return np.concatenate([local, WARP_VELOCITY * vel], axis=1)


def align_takes(fa, fb):
    """
        For each frame of take A, the matching frame of take B (float): dynamic time warping, B free to
        start and end anywhere (a take may run on before or after the other). :return: (map, mean cost).
    """
    na, nb = len(fa), len(fb)
    cost = np.sqrt(((fa[:, None, :] - fb[None, :, :]) ** 2).sum(-1))
    acc = np.full((na, nb), np.inf)
    acc[0] = cost[0]
    step = np.zeros((na, nb), dtype=np.int8)                 # 0 diagonal, 1 from above (A moves), 2 from left
    for i in range(1, na):
        prev = acc[i - 1]
        diag = np.concatenate([[np.inf], prev[:-1]])
        best = np.minimum(diag, prev)
        step[i] = np.where(diag <= prev, 0, 1)
        row = cost[i] + best
        for j in range(1, nb):                                # B moves on while A waits
            if row[j - 1] + cost[i, j] < row[j]:
                row[j] = row[j - 1] + cost[i, j]
                step[i, j] = 2
        acc[i] = row
    j = int(np.argmin(acc[-1]))
    i = na - 1
    pairs = []
    while i >= 0:
        pairs.append((i, j))
        if i == 0:
            break
        s = step[i, j]
        if s == 0:
            i, j = i - 1, j - 1
        elif s == 1:
            i -= 1
        else:
            j -= 1
    out = np.zeros(na)
    count = np.zeros(na)
    for i, j in pairs:
        out[i] += j
        count[i] += 1
    out /= np.maximum(count, 1)
    return out, float(acc[-1].min() / max(len(pairs), 1))


def second_view_lean(data, names, wrist, direction, K, other_data, other_rows, sure=None):
    """
        (lean (T,) radians, weight (T,), flip (T,) bool, report) the second take gives the first's blade.
        :param wrist, direction: the first take's grip (camera metres) and the blade's picture direction.
        :param sure: (T,) 0..1 how sure the first picture is of that direction.
        flip: frames where the first picture is unsure and both pictures together say the tip points the
        other way across it (a blade seen nearly end-on there).
    """
    J = np.asarray(data["joints_incam"], dtype=float)
    J2 = np.asarray(other_data["joints_incam"], dtype=float)
    names2 = [str(n) for n in other_data["joint_names"]]
    if len(names2) == J2.shape[1] + 1:
        names2 = names2[1:]
    idx2 = {n: i for i, n in enumerate(names2)}
    K2 = np.asarray(other_data["K"], dtype=float).reshape(3, 3)
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    fps2 = float(np.asarray(other_data["fps"]).reshape(-1)[0])
    match, cost = align_takes(pose_features(J, names, fps), pose_features(J2, names2, fps2))
    j = np.clip(np.rint(match).astype(int), 0, len(J2) - 1)
    rows2 = np.asarray(other_rows, dtype=float).reshape(len(J2), -1)
    dir2, conf2 = _unit(rows2[:, :2]), np.clip(rows2[:, 2], 0, 1)
    side2 = rows2[:, 3] if rows2.shape[1] > 3 else np.zeros(len(J2))
    apart2 = np.linalg.norm(J2[:, idx2["RightHand"]] - J2[:, idx2["LeftHand"]], axis=1) > HANDS_APART_M
    holder2 = sword_holder(side2, conf2 > GRIP_FIT_CONF, apart2)
    wrist2 = np.where((holder2 == -1)[:, None], J2[:, idx2["LeftHand"]], J2[:, idx2["RightHand"]])
    # each picture's plane through its camera, as normals in the body's frame
    sight, across = _sight_and_across(wrist, direction, K)
    n1 = np.einsum('tij,ti->tj', body_frames(J, names), np.cross(sight, across))
    sight2, across2 = _sight_and_across(wrist2[j], dir2[j], K2)
    n2 = np.einsum('tij,ti->tj', body_frames(J2, names2)[j], np.cross(sight2, across2))
    line = np.cross(n1, n2)
    sine = np.linalg.norm(line, axis=1)
    blade = np.einsum('tij,tj->ti', body_frames(J, names), _unit(line))       # back to the first camera
    lo, hi = SECOND_VIEW_CONF
    plo, phi = SECOND_VIEW_PLANES
    sure2 = np.clip((conf2[j] - lo) / (hi - lo), 0, 1)
    sure1 = np.ones(len(J)) if sure is None else np.asarray(sure, dtype=float)
    # which end is the tip: each picture votes by how sure it is and how side-on it sees the blade
    in2 = np.einsum('tij,tj->ti', body_frames(J2, names2)[j], _unit(line))
    vote = sure1 * np.sum(blade * across, 1) + sure2 * np.sum(in2 * across2, 1)
    blade *= np.where(vote < 0, -1.0, 1.0)[:, None]
    weight = sure2 * np.clip((sine - plo) / (phi - plo), 0, 1)
    # Only where the first picture is unsure: carried over from another performance through the body's
    # frame, the second view is good for the lean but outvoted a plainly visible tip (frames 85-87,
    # 121-125 and 168 turned round, with the first video showing them right).
    flip = (np.sum(blade * across, 1) < 0) & (weight > 0.5) & (sure1 < 0.5)
    lean = _lean(wrist, blade, np.where(flip[:, None], -direction, direction), K)
    return lean, weight, flip, {"warp_cost": round(cost, 3), "used": float(np.mean(weight > 0.5)),
                                "flipped": int(flip.sum()),
                                "offset_frames": float(np.median(match - np.arange(len(match))))}


def _gauss_complex(z, sigma):
    if len(z) < 3:
        return z
    return _gauss(z.real, sigma) + 1j * _gauss(z.imag, sigma)


def _sight_and_across(wrist, direction, K):
    """Per frame: unit ray from the camera through the grip, and the picture direction as a 3D unit across it."""
    Kinv = np.linalg.inv(K)
    g = _project(wrist, K)
    ray = lambda p: np.concatenate([p, np.ones((len(p), 1))], 1) @ Kinv.T
    sight = _unit(ray(g))
    across = ray(g + 50.0 * direction) - ray(g)
    return sight, _unit(across - np.sum(across * sight, 1, keepdims=True) * sight)


def _lean(wrist, blade, direction, K):
    """(T,) radians a 3D blade leans toward the camera, measured on the picture direction's side."""
    sight, across = _sight_and_across(wrist, direction, K)
    lean = np.arctan2(np.sum(blade * sight, 1), np.abs(np.sum(blade * across, 1)))
    return np.clip(lean, -np.radians(SWORD_MAX_LEAN), np.radians(SWORD_MAX_LEAN))


def _blade_at(wrist, direction, lean, K):
    """(T, 3) the 3D blade (camera) that points along `direction` in the picture and leans `lean` (rad)."""
    sight, across = _sight_and_across(wrist, direction, K)
    return _unit(np.cos(lean)[:, None] * across + np.sin(lean)[:, None] * sight)


def sword_holder(side, sure, apart):
    """
        (T,) which hand holds the sword per frame (1 right, -1 left, 0 both on the handle). Both, where the
        wrists are close in 3D (apart False); elsewhere the hand Track Sword found the blade at (its 4th
        value) in the nearest frame it was sure of one hand, the right if it never was.
    """
    side = np.rint(np.asarray(side, dtype=float)).astype(int)
    apart = np.asarray(apart, dtype=bool)
    out = np.zeros(len(side), dtype=int)
    known = np.flatnonzero(np.asarray(sure, dtype=bool) & apart & (side != 0))
    if not len(known):
        out[apart] = 1
        return out
    t = np.arange(len(side))
    after = known[np.clip(np.searchsorted(known, t), 0, len(known) - 1)]
    before = known[np.clip(np.searchsorted(known, t) - 1, 0, len(known) - 1)]
    pick = np.where(np.abs(before - t) < np.abs(after - t), before, after)
    out[apart] = side[pick][apart]
    return out
