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
    return {"frames": pts.reshape(T, 120).tolist(), "times": times.tolist(), "vis": np.ones((T, 40)).tolist(),
            "meta": meta, "hands": {s: h.tolist() for s, h in hands.items()}, "spine": spine, "fps": fps,
            "width": int(np.asarray(data["width"]).reshape(-1)[0]),
            "height": int(np.asarray(data["height"]).reshape(-1)[0])}
