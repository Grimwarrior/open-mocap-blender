"""
    Automatic retargeting of the Open Mocap pose skeleton (40 landmark empties) onto a
    Mixamo armature.

    Every mapped limb bone copies the *direction* of a mocap segment instead of its rotation:
    a hidden helper empty sits on landmark A and aims at landmark B, and a child of it lies
    FAR units further along that line. The rig bone gets a Damped Track to that far point, so
    it points along A->B wherever the rig is standing, whatever its proportions and bone rolls.
    Hips, chest and head also get a Locked Track to the hip / shoulder / ear line for facing,
    and the hips a Copy Location from the mocap hip centre, scaled to the rig's leg length.
    Thighs and shins get a Locked Track that turns the knee toward the mocap leg's bend, and the
    feet copy the whole orientation of the mocap heel / toe / ankle triangle (see FOOT_MAP).

    All constraints are named "OM ..." so "Toggle Constraints" and "Bake Motion" work on them.
"""

import math
import bpy
import numpy as np
from mathutils import Matrix, Vector
from . import globalVariables as gvar


PREFIX = "OM "
HELPER_COLLECTION = "OpenMocapRetarget"
STASH_PROP = "open_mocap_stashed_action"
FAR = 10000.0

# MediaPipe landmarks (subject's left/right) + the 7 centroids the add-on appends.
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, L_TOE, R_TOE = 23, 24, 25, 26, 27, 28, 31, 32
NOSE, L_EAR, R_EAR = 0, 7, 8
HIP_C, SHOULDER_C, R_HAND_C, L_HAND_C, EAR_C = 33, 34, 35, 36, 37

# Mixamo bone (without "mixamorig:" prefix) -> Damped Track along landmark segment (a, b)
AIM_MAP = {
    "Hips": (HIP_C, SHOULDER_C),
    "Spine": (HIP_C, SHOULDER_C),
    "Spine1": (HIP_C, SHOULDER_C),
    "Spine2": (HIP_C, SHOULDER_C),
    "Neck": (SHOULDER_C, EAR_C),
    "LeftArm": (L_SHOULDER, L_ELBOW),
    "LeftForeArm": (L_ELBOW, L_WRIST),
    "RightArm": (R_SHOULDER, R_ELBOW),
    "RightForeArm": (R_ELBOW, R_WRIST),
    "LeftUpLeg": (L_HIP, L_KNEE),
    "LeftLeg": (L_KNEE, L_ANKLE),
    "RightUpLeg": (R_HIP, R_KNEE),
    "RightLeg": (R_KNEE, R_ANKLE),
}

L_PINKY, R_PINKY, L_INDEX, R_INDEX = 17, 18, 19, 20
L_HEEL, R_HEEL = 29, 30

# Feet take their whole orientation from the mocap heel -> toe line (forward) and the ankle (up).
# Aiming the Mixamo foot bone along ankle -> toe would be wrong: at rest that bone runs steeply
# down from the ankle to the ball of the foot, so matching it to the much flatter ankle -> toe tip
# line tips the foot up by ~40 degrees. Mixamo foot: (heel, toe, ankle)
FOOT_MAP = {
    "LeftFoot": (L_HEEL, L_TOE, L_ANKLE),
    "RightFoot": (R_HEEL, R_TOE, R_ANKLE),
}

# Knee direction: thigh and shin are turned about their own axis (Locked Track, bone Y locked) so the
# knee points where the mocap leg bends. Seen along the thigh the ankle lies behind the knee, and seen
# along the shin so does the hip, so the rig's backward axis tracks that far end of the leg. The target
# is moved KNEE_TOE_HINT foot lengths back along ankle -> toe, which keeps the knee over the toes when
# the leg is (nearly) straight and the bend gives no direction.
# Mixamo bone: (start of the bone, far end of the leg, ankle, toe)
KNEE_MAP = {
    "LeftUpLeg": (L_HIP, L_ANKLE, L_ANKLE, L_TOE),
    "LeftLeg": (L_KNEE, L_HIP, L_ANKLE, L_TOE),
    "RightUpLeg": (R_HIP, R_ANKLE, R_ANKLE, R_TOE),
    "RightLeg": (R_KNEE, R_HIP, R_ANKLE, R_TOE),
}
KNEE_TOE_HINT = 0.5

# Twist around the bone (Locked Track, bone Y locked): the bone is turned so that the axis lying
# along the rig's rest line ref_from -> ref_to follows the mocap landmark line a -> b.
# Mixamo bone: (a, b, ref_from, ref_to)
# The twist between hips and shoulders is spread over the spine: each spine bone is turned part of
# the way from what it inherits toward the shoulder line (influence), the last one all the way.
# Mixamo bone: (a, b, ref_from, ref_to, influence)
FACING_MAP = {
    "Hips": (R_HIP, L_HIP, "RightUpLeg", "LeftUpLeg", 1.0),
    "Spine": (R_SHOULDER, L_SHOULDER, "RightUpLeg", "LeftUpLeg", 1.0 / 3.0),
    "Spine1": (R_SHOULDER, L_SHOULDER, "RightUpLeg", "LeftUpLeg", 0.5),
    "Spine2": (R_SHOULDER, L_SHOULDER, "RightUpLeg", "LeftUpLeg", 1.0),
}

# Arms and hands. MediaPipe's pose model gives only three hand points (index, pinky, thumb), close
# together and noisy: the index -> pinky line is often a fraction of a real palm's width, and its
# direction then flips from frame to frame. So the hand's direction and roll are worked out per frame
# in Python, smoothed over time (frames with a short, unreliable palm line count less), and put on
# keyed helper targets:
# - hand: Damped Track along wrist -> hand centre (bend from the forearm limited to WRIST_MAX_BEND),
#   Locked Track of the rig's index -> pinky axis toward the mocap index -> pinky line;
# - forearm: the same roll with influence FOREARM_TWIST_SHARE, so the twist is spread over forearm
#   and wrist instead of wrapping the wrist;
# - upper arm: turned about its own axis so the elbow bends where the mocap arm bends (the rig's
#   forward axis, which a T-pose elbow bends toward, follows the forearm's direction away from the
#   upper arm line), with a pull toward the body's forward direction for a straight arm.
# side: (shoulder, elbow, wrist, index, pinky)
ARM_MAP = {
    "Left": (L_SHOULDER, L_ELBOW, L_WRIST, L_INDEX, L_PINKY),
    "Right": (R_SHOULDER, R_ELBOW, R_WRIST, R_INDEX, R_PINKY),
}
WRIST_MAX_BEND = 70.0       # degrees
HAND_SMOOTH_S = 0.08        # smoothing of hand direction and roll (s)
PALM_RELIABLE = 0.25        # index -> pinky length / forearm length at which the roll is fully trusted
FOREARM_TWIST_SHARE = 0.5
ELBOW_FORWARD_HINT = 0.1    # weight of the body's forward direction for the elbow of a straight arm


def _gauss_np(x, sigma):
    """Gaussian smoothing over time (axis 0), edges held."""
    if sigma < 0.5 or len(x) < 3:
        return x
    r = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    p = np.pad(x, [(r, r)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    if x.ndim == 1:
        return np.convolve(p, k, mode="valid")
    return np.stack([np.convolve(p[:, c], k, mode="valid") for c in range(x.shape[1])], axis=1)


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > 1e-9, v / np.maximum(n, 1e-9), 0.0)


def _smooth_directions(v, w, sigma, fallback):
    """Weighted Gaussian average of directions; where few frames are reliable, a 4x wider window."""
    num, den = _gauss_np(v * w[:, None], sigma), _gauss_np(w, sigma)
    num2, den2 = _gauss_np(v * w[:, None], 4 * sigma), _gauss_np(w, 4 * sigma)
    a = np.clip(den / 0.3, 0.0, 1.0)[:, None]
    out = a * num / np.maximum(den, 1e-9)[:, None] + (1 - a) * num2 / np.maximum(den2, 1e-9)[:, None]
    out = _unit(out)
    bad = np.linalg.norm(out, axis=1) < 0.5
    out[bad] = fallback[bad]
    return out


def arm_directions(empties, frames, fps):
    """
        Per frame and side, in the landmark empties' local space: elbow bend direction for the upper
        arm, hand direction (wrist -> hand centre, bend limited) and palm side (index -> pinky, across
        the hand), all smoothed. :return: {side: (wrist, elbow, curl, hand, palm)} as (N, 3) arrays.
    """
    idx = sorted({i for v in ARM_MAP.values() for i in v} | {L_HIP, R_HIP, HIP_C, SHOULDER_C})
    P = {i: np.array([tuple(local_location(empties[i], f)) for f in frames]) for i in idx}
    sigma = HAND_SMOOTH_S * fps
    up = P[SHOULDER_C] - P[HIP_C]
    body_fwd = _unit(np.cross(P[L_HIP] - P[R_HIP], up))
    ones = np.ones(len(frames))
    out = {}
    for side, (s, e, w, i, p) in ARM_MAP.items():
        upper, fore = P[e] - P[s], P[w] - P[e]
        u_dir, f_dir = _unit(upper), _unit(fore)
        # elbow: the forearm's direction away from the upper arm line, plus the body's forward
        curl = fore - np.sum(fore * u_dir, axis=1, keepdims=True) * u_dir
        bend = np.linalg.norm(curl, axis=1) / np.maximum(np.linalg.norm(fore, axis=1), 1e-9)   # sin(bend)
        hint = _unit(body_fwd - np.sum(body_fwd * u_dir, axis=1, keepdims=True) * u_dir)
        curl = _unit(_unit(curl) * bend[:, None] + ELBOW_FORWARD_HINT * hint)
        curl = _smooth_directions(curl, ones, sigma, hint)
        # hand direction, bend from the forearm limited
        hand = _unit((P[i] + P[p]) / 2 - P[w])
        cos_b = np.clip(np.sum(hand * f_dir, axis=1), -1.0, 1.0)
        over = np.arccos(cos_b) > np.radians(WRIST_MAX_BEND)
        if over.any():
            lim = np.radians(WRIST_MAX_BEND)
            perp = _unit(hand - cos_b[:, None] * f_dir)
            hand[over] = (np.cos(lim) * f_dir + np.sin(lim) * perp)[over]
        hand = _smooth_directions(hand, ones, sigma, f_dir)
        # palm side: index -> pinky across the hand, weighted by how long (reliable) it is
        across = P[p] - P[i]
        across = across - np.sum(across * hand, axis=1, keepdims=True) * hand
        rel = np.linalg.norm(across, axis=1) / np.maximum(np.linalg.norm(fore, axis=1), 1e-9)
        weight = np.clip(rel / PALM_RELIABLE, 0.0, 1.0) ** 2
        palm = _smooth_directions(_unit(across), weight + 1e-3, sigma, curl)
        palm = _unit(palm - np.sum(palm * hand, axis=1, keepdims=True) * hand)
        out[side] = (P[w], P[e], curl, hand, palm)
    return out


# Head: the retarget used to turn the head only about its own axis (ear line), so it took its tilt
# from the neck (shoulder centre -> ear centre), which leans far more than a head does. The head's
# up axis now follows the face: eyes minus mouth, across the ear line, smoothed; frames whose face
# points are out of proportion (eye-mouth height / ear width far from the take's usual) count less.
# Its turn follows ear centre -> nose. Not the ear line: the combine's left/right fix can swap the
# ear (and eye) labels of the side view, which flips their depth and turns an ear line backwards;
# the nose has no left/right label.
HEAD_SMOOTH_S = 0.1
EYE_L, EYE_R, MOUTH_L, MOUTH_R = 2, 5, 9, 10


def head_directions(empties, frames, fps):
    """
        :return: (ear centre, smoothed head up, smoothed face forward across that up) as (N, 3)
                 arrays in the empties' local space.
    """
    idx = (EYE_L, EYE_R, MOUTH_L, MOUTH_R, L_EAR, R_EAR, SHOULDER_C, EAR_C, NOSE)
    P = {i: np.array([tuple(local_location(empties[i], f)) for f in frames]) for i in idx}
    across = P[L_EAR] - P[R_EAR]
    left = _unit(across)
    up = (P[EYE_L] + P[EYE_R]) / 2 - (P[MOUTH_L] + P[MOUTH_R]) / 2
    up = up - np.sum(up * left, axis=1, keepdims=True) * left
    ratio = np.linalg.norm(up, axis=1) / np.maximum(np.linalg.norm(across, axis=1), 1e-9)
    usual = float(np.median(ratio)) or 1.0
    weight = np.clip(1.0 - np.abs(ratio - usual) / usual, 0.0, 1.0) ** 2
    neck = _unit(P[EAR_C] - P[SHOULDER_C])
    up = _smooth_directions(_unit(up), weight + 1e-3, HEAD_SMOOTH_S * fps, neck)
    fwd = P[NOSE] - P[EAR_C]
    fwd = fwd - np.sum(fwd * up, axis=1, keepdims=True) * up
    # a nose (nearly) on the ear line gives no direction: such frames borrow from their neighbours
    reach = np.linalg.norm(fwd, axis=1) / np.maximum(np.linalg.norm(across, axis=1), 1e-9)
    w_fwd = np.clip(reach / (float(np.median(reach)) or 1.0), 0.0, 1.0) ** 2
    fwd = _smooth_directions(_unit(fwd), w_fwd + 1e-3, HEAD_SMOOTH_S * fps, _unit(np.cross(left, up)))
    fwd = _unit(fwd - np.sum(fwd * up, axis=1, keepdims=True) * up)
    return P[EAR_C], up, fwd


# Arm reach. Copying the direction of each arm segment puts the rig's hands wherever its own arm
# proportions take them: MediaPipe forearms come out short (wrist pulled in), Mixamo forearms are
# nearly as long as the upper arm, so the rig's hands overshot the mocap's by up to a third of a
# shoulder width and two hands on one sword grip drifted apart. "Hand Position" instead places each
# rig hand where the mocap hand is relative to the chest centre (scaled by torso length, mocap ->
# rig; the rig's own shoulder sits half its shoulder width out along the mocap shoulder line), so
# hands keep their place on the body and their distance to each other, and solves the elbow for the
# rig's own upper arm and forearm lengths, bending it in the plane of the mocap elbow.
ARM_IK_MAP = {"Left": (L_SHOULDER, L_ELBOW, L_WRIST), "Right": (R_SHOULDER, R_ELBOW, R_WRIST)}
REACH_SMOOTH_S = 0.04       # smoothing of the hand and elbow positions for the arm solve (s)


GRIP_SMOOTH_S = 0.25        # two-handed grip: smoothing of the direction from one hand to the other


def arm_reach_directions(empties, frames, fps, upper_len, fore_len, rig_torso, rig_shoulder_width,
                         shoulders=None, grip='NONE'):
    """
        Two-bone solve per frame. :param upper_len, fore_len, rig_torso, rig_shoulder_width: rig
        lengths (any one unit).
        :param shoulders: {side: (N, 3) upper arm head} from the collar bones; None: half the rig's
                          shoulder width out along the mocap shoulder line.
        :param grip: 'RIGHT' / 'LEFT': both hands hold one thing and that hand leads. The other
                     hand keeps the take's usual distance to it, in a smoothed direction, so tracker
                     jitter cannot pull the two hands apart or through each other.
        :return: {side: (shoulder, upper arm direction, forearm direction)} as (N, 3) arrays in the
                 empties' local space.
    """
    idx = {i for v in ARM_IK_MAP.values() for i in v} | {HIP_C, SHOULDER_C, L_HIP, R_HIP}
    P = {i: np.array([tuple(local_location(empties[i], f)) for f in frames]) for i in idx}
    mocap_torso = float(np.median(np.linalg.norm(P[SHOULDER_C] - P[HIP_C], axis=1)))
    r = mocap_torso / max(rig_torso, 1e-9)                  # mocap units per rig unit
    a, b = upper_len * r, fore_len * r
    half = 0.5 * rig_shoulder_width * r
    sigma = REACH_SMOOTH_S * fps
    body_fwd = _unit(np.cross(P[L_HIP] - P[R_HIP], P[SHOULDER_C] - P[HIP_C]))
    across = _unit(_gauss_np(P[L_SHOULDER] - P[R_SHOULDER], sigma))     # toward the actor's left
    wrists = {side: P[w] for side, (s, e, w) in ARM_IK_MAP.items()}
    if grip in ('RIGHT', 'LEFT'):
        lead, other = ("Right", "Left") if grip == 'RIGHT' else ("Left", "Right")
        rel = wrists[other] - wrists[lead]
        usual = float(np.median(np.linalg.norm(rel, axis=1)))
        wrists[other] = wrists[lead] + usual * _unit(_gauss_np(rel, GRIP_SMOOTH_S * fps))
    out = {}
    for side, (s, e, w) in ARM_IK_MAP.items():
        if shoulders is not None and side in shoulders:
            rig_shoulder = shoulders[side]
        else:
            rig_shoulder = P[SHOULDER_C] + (half if side == "Left" else -half) * across
        T = _gauss_np(wrists[side] - rig_shoulder, sigma)   # hand relative to the rig's shoulder
        E = _gauss_np(P[e] - rig_shoulder, sigma)
        d = np.linalg.norm(T, axis=1)
        t = _unit(T)
        d = np.clip(d, abs(a - b) * 1.02 + 1e-6, (a + b) * 0.999)
        pole = E - np.sum(E * t, axis=1, keepdims=True) * t
        hint = body_fwd - np.sum(body_fwd * t, axis=1, keepdims=True) * t
        p = _unit(pole + 0.05 * a * _unit(hint))             # elbow side; body forward if the arm is straight
        cos_a = np.clip((a * a + d * d - b * b) / (2 * a * d), -1.0, 1.0)
        upper = _unit(cos_a[:, None] * t + np.sqrt(1 - cos_a ** 2)[:, None] * p)
        fore = _unit(t * d[:, None] - a * upper)
        out[side] = (P[s], upper, fore)
    return out


# Body frames. The rig's rest pose and the mocap are compared in a frame of the body: lateral (toward
# the actor's left), up (hip centre -> shoulder centre, what the Hips and Spine bones aim along) and
# forward = lateral x up. A rig offset measured in the rig's rest frame is the same three numbers in
# the mocap's frame, times the mocap units per rig unit. Cross products need a right-handed body, so
# all of this runs in the "true" space, with a mirrored take's depth flipped back (see _true).
def _true(empties, indices, frames, mirror):
    """{landmark: (N, 3)} keyed local positions, depth flipped if the take is mirrored."""
    flip = np.array([1.0, -1.0 if mirror else 1.0, 1.0])
    return {i: np.array([tuple(local_location(empties[i], f)) for f in frames]) * flip for i in indices}


def _body_frame(line, up):
    """(lateral, up, forward) unit vectors per frame from a left-right line and an up direction."""
    up = _unit(up)
    lat = _unit(line - np.sum(line * up, axis=1, keepdims=True) * up)
    return lat, up, np.cross(lat, up)


def _rig_components(rig, offset, left_name, right_name):
    """A rig rest offset (world) as (lateral, up, forward) components of the rig's own body frame."""
    mw = rig.matrix_world
    line = (mw @ find_bone(rig, left_name).bone.head_local) - (mw @ find_bone(rig, right_name).bone.head_local)
    lat, up, fwd = _body_frame(np.array([tuple(line)]), np.array([[0.0, 0.0, 1.0]]))
    o = np.array(tuple(offset))
    return float(o @ lat[0]), float(o @ up[0]), float(o @ fwd[0])


def _from_components(c, frame):
    lat, up, fwd = frame
    return c[0] * lat + c[1] * up + c[2] * fwd


# Feet. Copying the direction of thigh and shin leaves the rig's feet wherever its own proportions
# and the tracker's jitter take them: a standing foot slides and bobs. "Foot Position" instead solves
# each leg for the rig's own thigh and shin lengths so that the ankle goes where the mocap ankle is,
# and "Planted" holds that ankle still while the foot is down. A foot counts as down while its ankle
# is low (within PLANT_HEIGHT leg lengths of where it stands) and slow (under PLANT_SPEED leg lengths
# per second); it is then held at the middle of where it was during that time, eased in and out over
# PLANT_BLEND_S.
# Size. The mocap is in picture units: an actor who steps toward the camera grows, legs, hip height
# and all. Putting the rig's ankles straight onto the mocap ankles would make it crouch wherever the
# actor looks small and stretch where he looks big. So the legs are solved free of size: each leg
# points from hip to ankle as the mocap's does and is as bent as the mocap's (hip-ankle distance over
# that leg's own length at that moment), and the hips stand as high above the floor, in leg lengths
# of that moment, as the mocap's. That gives the rig's own ankle path, which is what gets held.
# The floor. One camera also sees a foot that stands further away higher in the picture, so the
# floor under the actor moves in the picture: it is taken per moment from whichever foot is down,
# and a held foot is put on the rig's floor (plus however much higher than that foot's standing
# height it is held, e.g. on its toes).
# Where a held foot ends up further from the hip than the rig's leg is long (the mocap's bone
# lengths vary with the tracker's depth), the hips are pulled toward the foot, by at most
# PLANT_HIP_PULL leg lengths; beyond that the foot gives way.
LEG_IK_MAP = {"Left": (L_HIP, L_KNEE, L_ANKLE), "Right": (R_HIP, R_KNEE, R_ANKLE)}
PLANT_HEIGHT = 0.15
PLANT_SPEED = 0.6
PLANT_MIN_S = 0.15          # shorter contacts are not held
PLANT_GAP_S = 0.07          # breaks this short inside a contact are bridged
PLANT_BLEND_S = 0.1
PLANT_SMOOTH_S = 0.05       # smoothing of the ankle before its speed is measured
PLANT_HIP_PULL = 0.25
PLANT_MERGE_S = 0.35        # two contacts this close in time ...
PLANT_MERGE_DIST = 0.08     # ... and this close in place (leg lengths) are one
PLANT_RAISE_MIN = 0.06      # a foot held less than this (leg lengths) above its standing height stands
PLANT_PULL_SMOOTH_S = 0.08


def _runs(mask):
    """[(start, end)] inclusive index ranges where mask is True."""
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return [(int(a), int(b)) for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0] - 1)]


def foot_contacts(ankle, leg_length, fps):
    """:return: (N,) bool, True while the foot of this ankle (N, 3, z up) is down."""
    n = len(ankle)
    if n < 3 or leg_length <= 1e-9:
        return np.zeros(n, dtype=bool)
    a = _gauss_np(ankle, PLANT_SMOOTH_S * fps)
    speed = np.linalg.norm(np.gradient(a, axis=0), axis=1) * fps
    floor = np.percentile(a[:, 2], 10)
    down = (a[:, 2] < floor + PLANT_HEIGHT * leg_length) & (speed < PLANT_SPEED * leg_length)
    for s, e in _runs(~down):                               # bridge short breaks
        if s > 0 and e < n - 1 and e - s + 1 <= round(PLANT_GAP_S * fps):
            down[s:e + 1] = True
    for s, e in _runs(down):                                # drop short contacts
        if e - s + 1 < max(2, round(PLANT_MIN_S * fps)):
            down[s:e + 1] = False
    # one contact, not two, where the foot is found in the same place again shortly after
    runs = _runs(down)
    for (s0, e0), (s1, e1) in zip(runs, runs[1:]):
        apart = np.linalg.norm(np.median(a[s0:e0 + 1], axis=0) - np.median(a[s1:e1 + 1], axis=0))
        if s1 - e0 - 1 <= round(PLANT_MERGE_S * fps) and apart < PLANT_MERGE_DIST * leg_length:
            down[e0:s1 + 1] = True
    return down


FLOOR_SMOOTH_S = 0.3        # smoothing of the floor height and of the apparent leg length ...
SIZE_SMOOTH_S = 0.5


def mocap_floor(empties, frames, fps, mirror):
    """
        :return: (floor: the ankle height the mocap stands at over the take, {side: down mask},
                 floor per frame (N,), apparent leg length per frame (N,)), in the empties' local units.
    """
    P = _true(empties, {L_HIP, L_KNEE, L_ANKLE, R_HIP, R_KNEE, R_ANKLE}, frames, mirror)
    legs = [np.linalg.norm(P[h] - P[k], axis=1) + np.linalg.norm(P[k] - P[a], axis=1)
            for h, k, a in ((L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE))]
    size = (legs[0] + legs[1]) / 2
    leg = float(np.median(size))
    half = max(1, int(round(FLOOR_SMOOTH_S * fps / 2)))
    padded = np.pad(size, half, mode="edge")
    size = np.array([np.median(padded[i:i + 2 * half + 1]) for i in range(len(size))])
    size = np.maximum(_gauss_np(size, SIZE_SMOOTH_S * fps), 0.5 * leg)
    down = {side: foot_contacts(P[an], leg, fps) for side, an in (("Left", L_ANKLE), ("Right", R_ANKLE))}
    z = {side: _gauss_np(P[an][:, 2], PLANT_SMOOTH_S * fps) for side, an in (("Left", L_ANKLE), ("Right", R_ANKLE))}
    stacked = np.stack([np.where(down[side], z[side], np.inf) for side in ("Left", "Right")], axis=1)
    lowest = stacked.min(axis=1)
    seen = np.isfinite(lowest)
    if seen.any():
        idx = np.arange(len(lowest))
        per_frame = _gauss_np(np.interp(idx, idx[seen], lowest[seen]), FLOOR_SMOOTH_S * fps)
        floor = float(np.median(lowest[seen]))
    else:                                                   # no foot ever stands: the low end of both
        floor = float(np.percentile(np.minimum(z["Left"], z["Right"]), 10))
        per_frame = np.full(len(lowest), floor)
    return floor, down, per_frame, size


def plant_foot(ankle, down, fps, floor=None, leg_length=0.0):
    """
        Ankle positions with every contact held at its median position, eased in and out.
        :param floor: the floor's height; a held foot is put on it (see "The floor" above).
    """
    out = ankle.copy()
    n = len(ankle)
    ramp = max(1, int(round(PLANT_BLEND_S * fps)))
    best = np.zeros(n)
    level = float(np.percentile(ankle[down, 2], 10)) if down.any() else 0.0
    for s, e in _runs(down):
        lock = np.median(ankle[s:e + 1], axis=0)
        if floor is not None:
            lock[2] = floor + max(0.0, lock[2] - level - PLANT_RAISE_MIN * leg_length)
        w = np.zeros(n)
        w[s:e + 1] = 1.0
        for k in range(1, ramp + 1):
            v = 1.0 - k / (ramp + 1)
            v = v * v * (3 - 2 * v)
            for j in (s - k, e + k):
                if 0 <= j < n and not down[j]:
                    w[j] = v
        use = w > best
        out[use] = ankle[use] + w[use, None] * (lock - ankle[use])
        best[use] = w[use]
    return out


def leg_directions(empties, frames, fps, mirror, ratio, thigh_len, shin_len, hip_offsets, hips_lift,
                   plant, floor_data=None):
    """
        Two-bone solve of both legs per frame.
        :param ratio: mocap units per rig unit. :param thigh_len, shin_len: rig lengths.
        :param hip_offsets: {side: rig hip joint relative to the Hips bone, body frame components}.
        :param hips_lift: how far (rig units) the Hips bone sits above the mocap hip centre.
        :param plant: hold the ankles still while the feet are down.
        :param floor_data: mocap_floor's result to stand the rig on the floor; None: the hips keep
                           the mocap hip centre's height.
        :return: ({side: (hip joint, thigh direction, knee, shin direction)} as (N, 3) arrays in the
                 empties' local space, hips shift (N, 3) in the same space or None,
                 {side: share of frames the foot is held})
    """
    idx = {i for v in LEG_IK_MAP.values() for i in v} | {HIP_C, SHOULDER_C}
    P = _true(empties, idx, frames, mirror)
    frame = _body_frame(P[L_HIP] - P[R_HIP], P[SHOULDER_C] - P[HIP_C])
    a, b = thigh_len * ratio, shin_len * ratio
    reach = (a + b) * 0.999
    flip = np.array([1.0, -1.0 if mirror else 1.0, 1.0])
    n = len(frames)
    # hips: as many leg lengths above the floor as the mocap's are at that moment
    shift = np.zeros((n, 3))
    floor = None
    if floor_data is not None:
        floor, down_masks, floor_now, size = floor_data
        shift[:, 2] = floor + (P[HIP_C][:, 2] - floor_now) * (a + b) / size - P[HIP_C][:, 2]
    else:
        down_masks = {"Left": foot_contacts(P[L_ANKLE], a + b, fps), "Right": foot_contacts(P[R_ANKLE], a + b, fps)}
    hips = P[HIP_C] + shift + np.array([0.0, 0.0, hips_lift * ratio])
    sigma = REACH_SMOOTH_S * fps
    joints, ankles, held, share = {}, {}, {}, {}
    for side, (h, k, an) in LEG_IK_MAP.items():
        joints[side] = hips + ratio * _from_components(hip_offsets[side], frame)
        # the rig's own ankle path: the mocap leg's direction and bend, at the rig's leg length
        T = _gauss_np(P[an] - P[h], sigma)
        own = _gauss_np(np.linalg.norm(P[k] - P[h], axis=1) + np.linalg.norm(P[an] - P[k], axis=1), sigma)
        bend = np.clip(np.linalg.norm(T, axis=1) / np.maximum(own, 1e-9), 0.3, 0.999)
        natural = joints[side] + (bend * (a + b))[:, None] * _unit(T)
        down = down_masks[side] if plant else np.zeros(n, dtype=bool)
        ankles[side] = plant_foot(natural, down, fps, floor, a + b) if down.any() else natural
        held[side] = down
        share[side] = float(down.mean())
    # hips pulled toward held feet that are out of reach
    if plant and any(d.any() for d in held.values()):
        pull = np.zeros((n, 3))
        for _ in range(4):
            for side in LEG_IK_MAP:
                T = ankles[side] - (joints[side] + pull)
                d = np.linalg.norm(T, axis=1)
                over = np.where(held[side], np.maximum(0.0, d - reach * 0.995), 0.0)
                pull += (over / np.maximum(d, 1e-9))[:, None] * T
        limit = PLANT_HIP_PULL * (a + b)
        size_ = np.linalg.norm(pull, axis=1, keepdims=True)
        pull = _gauss_np(pull * np.minimum(1.0, limit / np.maximum(size_, 1e-9)), PLANT_PULL_SMOOTH_S * fps)
        shift = shift + pull
        for side in LEG_IK_MAP:
            joints[side] = joints[side] + pull
    out = {}
    for side, (h, k, _an) in LEG_IK_MAP.items():
        joint = joints[side]
        T = ankles[side] - joint
        d = np.clip(np.linalg.norm(T, axis=1), abs(a - b) * 1.02 + 1e-6, reach)
        t = _unit(T)
        K = _gauss_np(P[k] - P[h], sigma)                    # knee side, as the mocap leg bends
        pole = K - np.sum(K * t, axis=1, keepdims=True) * t
        hint = frame[2] - np.sum(frame[2] * t, axis=1, keepdims=True) * t
        p = _unit(pole + 0.05 * a * _unit(hint))             # forward if the leg is straight
        cos_a = np.clip((a * a + d * d - b * b) / (2 * a * d), -1.0, 1.0)
        thigh = _unit(cos_a[:, None] * t + np.sqrt(1 - cos_a ** 2)[:, None] * p)
        knee = joint + a * thigh
        shin = _unit(joint + t * d[:, None] - knee)
        out[side] = (joint * flip, thigh * flip, knee * flip, shin * flip)
    moved = float(np.abs(shift).max()) > 1e-6
    return out, (shift * flip if moved else None), share


# Collar bones. The mocap shoulder points rise and fall with the collar bones (a shrug, a raised
# arm), which shows as the shoulder line tilting against the spine. The rig's chest only turns about
# the spine, so without this its shoulders stay level and a raised arm has to reach from too low.
# Each collar bone is lifted by the tilt of its half of the shoulder line, plus CLAVICLE_ARM_SHARE of
# the upper arm's angle above the horizontal (both arms up lifts both shoulders, which the line
# cannot show), within CLAVICLE_RANGE degrees.
CLAVICLE_ARM_SHARE = 0.2
CLAVICLE_RANGE = (-10.0, 35.0)
CLAVICLE_SMOOTH_S = 0.1


def clavicle_directions(empties, frames, fps, mirror, ratio, heads, rests):
    """
        :param heads: {side: collar bone head relative to the chest centre (midpoint of the upper arm
                      heads), body frame components, rig units}.
        :param rests: {side: collar bone head -> upper arm head at rest, body frame components}.
        :return: {side: (collar bone head, its direction, upper arm head)} as (N, 3) arrays in the
                 empties' local space.
    """
    idx = {HIP_C, SHOULDER_C, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW}
    P = _true(empties, idx, frames, mirror)
    frame = _body_frame(P[L_SHOULDER] - P[R_SHOULDER], P[SHOULDER_C] - P[HIP_C])
    up = frame[1]
    flip = np.array([1.0, -1.0 if mirror else 1.0, 1.0])
    out = {}
    for side, (s, e) in (("Left", (L_SHOULDER, L_ELBOW)), ("Right", (R_SHOULDER, R_ELBOW))):
        tilt = np.arcsin(np.clip(np.sum(_unit(P[s] - P[SHOULDER_C]) * up, axis=1), -1.0, 1.0))
        arm = np.arcsin(np.clip(np.sum(_unit(P[e] - P[s]) * up, axis=1), -1.0, 1.0))
        lift = tilt + CLAVICLE_ARM_SHARE * np.clip(arm, 0.0, None)
        lift = np.clip(_gauss_np(lift, CLAVICLE_SMOOTH_S * fps), *np.radians(CLAVICLE_RANGE))
        head = P[SHOULDER_C] + ratio * _from_components(heads[side], frame)
        rest = _from_components(rests[side], frame)                      # rig units, mocap body frame
        length = np.linalg.norm(rest, axis=1, keepdims=True)
        rest_dir = _unit(rest)
        toward = _unit(up - np.sum(up * rest_dir, axis=1, keepdims=True) * rest_dir)
        direction = np.cos(lift)[:, None] * rest_dir + np.sin(lift)[:, None] * toward
        out[side] = (head * flip, direction * flip, (head + ratio * length * direction) * flip)
    return out


def _keyed_far_point(coll, points, name, origins, dirs, frames):
    """Helper in the mocap space, keyed per frame FAR from `origins` along `dirs` (local space)."""
    return _keyed_point(coll, points.space, name, origins + FAR * dirs, frames)


def _keyed_point(coll, parent, name, locs, frames):
    """Helper empty under `parent` whose location is keyed per frame."""
    obj = _new_empty(coll, name, parent=parent)
    obj.animation_data_create()
    act = bpy.data.actions.new(PREFIX + name)
    obj.animation_data.action = act
    co = np.empty((len(frames), 2))
    co[:, 0] = frames
    for k in range(3):
        try:
            fc = act.fcurve_ensure_for_datablock(obj, "location", index=k)
        except AttributeError:
            fc = act.fcurves.new("location", index=k)
        co[:, 1] = locs[:, k]
        fc.keyframe_points.add(len(frames))
        fc.keyframe_points.foreach_set("co", co.ravel())
        for kp in fc.keyframe_points:
            kp.interpolation = 'LINEAR'
        fc.update()
    return obj


def _twist_axis(rig, pb, ref_dir):
    """Locked Track axis of the bone (X or Z, either sign) that lies closest to ref_dir at rest."""
    m = rig.matrix_world.to_3x3() @ pb.bone.matrix_local.to_3x3()
    axes = {'TRACK_X': m.col[0], 'TRACK_NEGATIVE_X': -m.col[0],
            'TRACK_Z': m.col[2], 'TRACK_NEGATIVE_Z': -m.col[2]}
    ref = ref_dir.normalized()
    return max(axes, key=lambda k: axes[k].normalized().dot(ref))


# -------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------
def find_bone(rig, short_name):
    """Find a Mixamo bone by name with any prefix ("mixamorig:", "mixamorig1:", none)."""
    for pb in rig.pose.bones:
        if pb.name == short_name or pb.name.split(":")[-1] == short_name:
            return pb
    return None


def is_mixamo_rig(rig):
    return rig is not None and rig.type == 'ARMATURE' and all(
        find_bone(rig, n) for n in ("Hips", "LeftArm", "RightArm", "LeftUpLeg", "RightUpLeg"))


# Mixamo control rig (the Mixamo add-on's "Ctrl_" bones): the mixamorig deform bones copy these
# controls, so the retarget drives the FK controls instead, with the arms and legs switched to FK.
# The control rig then moves the skeleton as designed and Bake Motion keys controls you can edit.
# The FK controls sit exactly on their deform bones at rest (the foot control is turned, which the
# foot constraint allows for: it works from the driven bone's own rest pose).
CONTROL_RIG_MAP = {
    "Hips": "Ctrl_Hips", "Spine": "Ctrl_Spine", "Spine1": "Ctrl_Spine1", "Spine2": "Ctrl_Spine2",
    "Neck": "Ctrl_Neck", "Head": "Ctrl_Head",
}
for _s in ("Left", "Right"):
    CONTROL_RIG_MAP.update({
        f"{_s}Arm": f"Ctrl_Arm_FK_{_s}", f"{_s}ForeArm": f"Ctrl_ForeArm_FK_{_s}", f"{_s}Hand": f"Ctrl_Hand_FK_{_s}",
        f"{_s}UpLeg": f"Ctrl_UpLeg_FK_{_s}", f"{_s}Leg": f"Ctrl_Leg_FK_{_s}", f"{_s}Foot": f"Ctrl_Foot_FK_{_s}",
    })
IKFK_SWITCHES = ("Ctrl_Hand_IK_Left", "Ctrl_Hand_IK_Right", "Ctrl_Foot_IK_Left", "Ctrl_Foot_IK_Right")
IKFK_PROP = "ik_fk_switch"          # 0 = IK, 1 = FK
IKFK_STASH_PROP = "open_mocap_ikfk"


def is_control_rig(rig):
    return all(n in rig.pose.bones for n in CONTROL_RIG_MAP.values())


def clavicle_bone(rig, side):
    """The collar bone to drive: the Mixamo bone, or on a control rig its control if there is one."""
    if is_control_rig(rig):
        for name in (f"Ctrl_Shoulder_{side}", f"Ctrl_Shoulder_FK_{side}"):
            if name in rig.pose.bones:
                return rig.pose.bones[name]
        return None
    return find_bone(rig, side + "Shoulder")


def drive_bone(rig, short_name):
    """The bone the retarget constrains: the FK control on a control rig, else the Mixamo bone."""
    if short_name in CONTROL_RIG_MAP and is_control_rig(rig):
        return rig.pose.bones[CONTROL_RIG_MAP[short_name]]
    return find_bone(rig, short_name)


def _set_fk(rig):
    """Switch a control rig's arms and legs to FK, remembering the previous switch values."""
    old = {}
    for n in IKFK_SWITCHES:
        pb = rig.pose.bones.get(n)
        if pb is not None and IKFK_PROP in pb:
            old[n] = float(pb[IKFK_PROP])
            pb[IKFK_PROP] = 1.0
    if old and IKFK_STASH_PROP not in rig:
        rig[IKFK_STASH_PROP] = old
    rig.update_tag()


def _restore_ikfk(rig):
    old = rig.get(IKFK_STASH_PROP)
    if old is None:
        return False
    for n, v in old.items():
        pb = rig.pose.bones.get(n)
        if pb is not None and IKFK_PROP in pb:
            pb[IKFK_PROP] = v
    del rig[IKFK_STASH_PROP]
    rig.update_tag()
    return True


def _is_keyed(objs):
    return bool(_location_fcurves(objs[0])) and bool(_location_fcurves(objs[HIP_C]))


# Landmark pairs a real skeleton keeps at a steady distance. When a mocap rig holds two empties for
# the same landmark (e.g. "14.001" and a duplicate "14.002" made while editing), the one whose
# distances to these neighbours stay steady over the take is used.
LANDMARK_LINKS = [
    (11, 13), (13, 15), (12, 14), (14, 16), (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
    (23, 25), (25, 27), (27, 29), (27, 31), (24, 26), (26, 28), (28, 30), (28, 32),
    (11, 12), (23, 24), (11, 23), (12, 24), (0, 7), (0, 8), (7, 8), (0, 2), (0, 5), (1, 2), (2, 3),
    (4, 5), (5, 6), (9, 10), (0, 9), (0, 10), (33, 23), (33, 24), (34, 11), (34, 12), (35, 18), (35, 20),
    (36, 17), (36, 19), (37, 7), (37, 8), (38, 9), (38, 10), (39, 1), (39, 4),
]


def _landmark_index(obj):
    base, _, suffix = obj.name.partition(".")
    if obj.type == 'EMPTY' and base.isdigit() and int(base) < 40 and (not suffix or suffix.isdigit()):
        return int(base)
    return None


def _landmark_children(root):
    """{landmark index: [empties]} among the children of a mocap rig root (e.g. "RigRoot.001")."""
    groups = {}
    for c in root.children:
        i = _landmark_index(c)
        if i is not None:
            groups.setdefault(i, []).append(c)
    return groups


def is_mocap_root(obj):
    return obj is not None and len(_landmark_children(obj)) == 40


def _pick_landmarks(groups, frames):
    """One empty per landmark; duplicates decided by how steady their distances to neighbours are."""
    chosen = {}
    for i, cands in groups.items():
        keyed = [c for c in cands if _location_fcurves(c)] or cands
        if len(keyed) == 1:
            chosen[i] = keyed[0]
    for i, cands in sorted(groups.items()):
        if i in chosen:
            continue
        cands = [c for c in cands if _location_fcurves(c)] or cands
        others = [b if a == i else a for a, b in LANDMARK_LINKS if i in (a, b)]
        others = [chosen[j] for j in others if j in chosen]

        def spread(c):
            cvs = []
            for o in others:
                d = [(local_location(c, f) - local_location(o, f)).length for f in frames]
                mean = sum(d) / len(d)
                if mean > 1e-9:
                    cvs.append(math.sqrt(sum((x - mean) ** 2 for x in d) / len(d)) / mean)
            return sum(cvs) / len(cvs) if cvs else 0.0

        chosen[i] = min(sorted(cands, key=lambda c: c.name), key=spread)
    return [chosen[i] for i in range(40)]


def find_landmark_empties(source=None, frame_start=None, frame_end=None):
    """
        :param source: a mocap rig root (the empty "RigRoot", "RigRoot.001", ... whose children are
                       the 40 landmark empties) to use; None picks one.
        :return: The 40 pose landmark empties, or None. Without a source: the set from this session,
                 otherwise the newest complete keyed set.
    """
    scene = bpy.context.scene
    fs = scene.frame_start if frame_start is None else frame_start
    fe = scene.frame_end if frame_end is None else frame_end
    frames = list(range(fs, fe + 1, max(1, (fe - fs) // 40)))
    if source is not None:
        groups = _landmark_children(source)
        return _pick_landmarks(groups, frames) if len(groups) == 40 else None
    objs = gvar.object_list
    try:
        if len(objs) == 40 and all(o.name in bpy.data.objects for o in objs) and _is_keyed(objs):
            return list(objs)
    except ReferenceError:
        pass
    roots = [o for o in bpy.data.objects if is_mocap_root(o)]
    for root in sorted(roots, key=lambda o: (o.name.partition(".")[2].zfill(6), o.name), reverse=True):
        found = _pick_landmarks(_landmark_children(root), frames)
        if _is_keyed(found):
            return found
    sets = {}
    for obj in bpy.data.objects:
        base, _, suffix = obj.name.partition(".")
        if obj.type == 'EMPTY' and base.isdigit() and int(base) < 40 and (not suffix or suffix.isdigit()):
            sets.setdefault(int(suffix or 0), {})[int(base)] = obj
    for suffix in sorted(sets, reverse=True):
        group = sets[suffix]
        if len(group) == 40:
            found = [group[i] for i in range(40)]
            if _is_keyed(found):
                return found
    return None


def _location_fcurves(obj):
    ad = obj.animation_data
    if not ad or not ad.action:
        return {}
    act = ad.action
    fcurves = None
    try:
        from bpy_extras.anim_utils import action_get_channelbag_for_slot
        bag = action_get_channelbag_for_slot(act, ad.action_slot)
        fcurves = bag.fcurves if bag else None
    except ImportError:
        pass
    if fcurves is None:
        fcurves = getattr(act, "fcurves", [])
    return {fc.array_index: fc for fc in fcurves if fc.data_path == "location"}


def local_location(obj, frame):
    """Keyed local location of a landmark empty at a frame, without changing the scene frame."""
    fcs = _location_fcurves(obj)
    return Vector([fcs[i].evaluate(frame) if i in fcs else obj.location[i] for i in range(3)])


def _median(values):
    values = sorted(values)
    return values[len(values) // 2] if values else 0.0


def mocap_leg_length(empties, frame_start, frame_end):
    """Median hip-knee-ankle length (both legs) in the empties' local space."""
    step = max(1, (frame_end - frame_start) // 60)
    lengths = []
    for f in range(frame_start, frame_end + 1, step):
        p = {i: local_location(empties[i], f) for i in (L_HIP, L_KNEE, L_ANKLE, R_HIP, R_KNEE, R_ANKLE)}
        left = (p[L_HIP] - p[L_KNEE]).length + (p[L_KNEE] - p[L_ANKLE]).length
        right = (p[R_HIP] - p[R_KNEE]).length + (p[R_KNEE] - p[R_ANKLE]).length
        lengths.append((left + right) / 2)
    return _median(lengths)


def depth_mirror_score(empties, frame_start, frame_end):
    """
        Is the mocap mirrored in depth? With Combine Data the depth comes from the side
        camera's x axis, whose sign depends on which side of the actor that camera stood.
        A real body is right-handed: forward = left x up, and the nose and toes point forward.
        :return: Average of nose/toe direction . forward over the take. Negative = mirrored.
    """
    step = max(1, (frame_end - frame_start) // 120)
    scores = []
    for f in range(frame_start, frame_end + 1, step):
        p = {i: local_location(empties[i], f) for i in (0, L_EAR, R_EAR, L_HIP, R_HIP, L_ANKLE, R_ANKLE,
                                                        L_TOE, R_TOE, HIP_C, SHOULDER_C)}
        up = p[SHOULDER_C] - p[HIP_C]
        left = p[L_HIP] - p[R_HIP]
        fwd = left.cross(up)
        if fwd.length < 1e-6:
            continue
        fwd.normalize()
        nose = p[0] - (p[L_EAR] + p[R_EAR]) / 2
        toes = (p[L_TOE] + p[R_TOE]) / 2 - (p[L_ANKLE] + p[R_ANKLE]) / 2
        for v in (nose, toes):
            if v.length > 1e-6:
                scores.append(v.normalized().dot(fwd))
    return sum(scores) / len(scores) if scores else 0.0


def mocap_foot_stats(empties, frame_start, frame_end, heel, toe, ankle):
    """
        :return: (median ankle -> toe length, pitch of heel -> toe in radians while the foot is down).
                 The pitch is the mocap's "flat foot" (MediaPipe's heel point sits a little above the
                 floor), measured over the frames where this ankle is in its lowest third.
    """
    step = max(1, (frame_end - frame_start) // 120)
    lengths, samples = [], []
    for f in range(frame_start, frame_end + 1, step):
        h, t, a = (local_location(empties[i], f) for i in (heel, toe, ankle))
        lengths.append((t - a).length)
        d = t - h
        if d.length > 1e-6:
            samples.append((a.z, math.asin(max(-1.0, min(1.0, d.z / d.length)))))
    if not samples:
        return _median(lengths), 0.0
    lo, hi = min(s[0] for s in samples), max(s[0] for s in samples)
    low = [p for z, p in samples if z <= lo + (hi - lo) / 3] or [p for _, p in samples]
    return _median(lengths), _median(low)


def rig_forward(rig):
    """The rig's forward direction at rest (world, horizontal): left x up, like depth_mirror_score."""
    mw = rig.matrix_world
    left = (mw @ find_bone(rig, "LeftUpLeg").bone.head_local) - (mw @ find_bone(rig, "RightUpLeg").bone.head_local)
    fwd = left.cross(Vector((0.0, 0.0, 1.0)))
    fwd.z = 0.0
    return fwd.normalized() if fwd.length > 1e-9 else Vector((0.0, -1.0, 0.0))


def rig_leg_length(rig):
    mw = rig.matrix_world
    up, knee, foot = (find_bone(rig, n) for n in ("LeftUpLeg", "LeftLeg", "LeftFoot"))
    if not (up and knee and foot):
        return 0.0
    h = [mw @ b.bone.head_local for b in (up, knee, foot)]
    return (h[0] - h[1]).length + (h[1] - h[2]).length


def _helper_collection(context):
    coll = bpy.data.collections.get(HELPER_COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(HELPER_COLLECTION)
        context.scene.collection.children.link(coll)
    return coll


def _new_empty(coll, name, parent=None):
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_type = 'PLAIN_AXES'
    obj.empty_display_size = 0.05
    coll.objects.link(obj)
    if parent:
        obj.parent = parent
    return obj


class _MocapSpace:
    """
        Proxy empties that follow the landmark empties, optionally mirrored in depth (Y),
        so the retarget can fix a mirrored take without touching the user's mocap.
    """

    def __init__(self, coll, empties, mirror):
        self.coll, self.empties, self.proxies = coll, empties, {}
        first = empties[0]
        base = (first.parent.matrix_world @ first.matrix_parent_inverse) if first.parent else Matrix.Identity(4)
        self.space = _new_empty(coll, "OM_mocap_space")
        self.space.matrix_world = base @ Matrix.Diagonal((1.0, -1.0 if mirror else 1.0, 1.0, 1.0))

    def __getitem__(self, i):
        if i not in self.proxies:
            proxy = _new_empty(self.coll, f"OM_landmark_{i}", parent=self.space)
            c = proxy.constraints.new('COPY_LOCATION')
            c.target = self.empties[i]
            c.target_space = 'LOCAL'
            c.owner_space = 'LOCAL'
            self.proxies[i] = proxy
        return self.proxies[i]


def _aim_point(coll, cache, points, a, b):
    """Helper empty FAR along landmark a -> b (shared by bones using the same segment)."""
    key = (a, b)
    if key not in cache:
        aim = _new_empty(coll, f"OM_aim_{a}_{b}")
        c = aim.constraints.new('COPY_LOCATION')
        c.target = points[a]
        c = aim.constraints.new('DAMPED_TRACK')
        c.target = points[b]
        c.track_axis = 'TRACK_Y'
        point = _new_empty(coll, f"OM_aimpoint_{a}_{b}", parent=aim)
        point.location = (0.0, FAR, 0.0)
        cache[key] = point
    return cache[key]


def _foot_frame(coll, points, heel, toe, ankle):
    """Helper empty oriented like the mocap foot: Y along heel -> toe, Z toward the ankle."""
    frame = _new_empty(coll, f"OM_foot_{heel}_{toe}_{ankle}")
    c = frame.constraints.new('COPY_LOCATION')
    c.target = points[heel]
    c = frame.constraints.new('DAMPED_TRACK')
    c.target = points[toe]
    c.track_axis = 'TRACK_Y'
    c = frame.constraints.new('LOCKED_TRACK')
    c.target = points[ankle]
    c.lock_axis = 'LOCK_Y'
    c.track_axis = 'TRACK_Z'
    return frame


def _knee_hint(coll, cache, points, near, far, ankle, toe, dist):
    """
        Helper point FAR away from the rig in the direction mocap landmark `near` -> (landmark `far`
        moved `dist` back along ankle -> toe). Like the aim points, only its direction matters, so
        the rig can stand anywhere and at any size.
    """
    aim = _aim_point(coll, cache, points, ankle, toe).parent      # sits on the ankle, Y along ankle -> toe
    hint = _new_empty(coll, f"OM_knee_{far}_{ankle}_{toe}")
    c = hint.constraints.new('COPY_LOCATION')
    c.target = points[far]
    c = hint.constraints.new('COPY_ROTATION')
    c.target = aim
    back = _new_empty(coll, f"OM_kneeback_{far}_{ankle}_{toe}", parent=hint)
    back.location = (0.0, -dist, 0.0)
    ray = _new_empty(coll, f"OM_kneeray_{near}_{far}")
    c = ray.constraints.new('COPY_LOCATION')
    c.target = points[near]
    c = ray.constraints.new('DAMPED_TRACK')
    c.target = back
    c.track_axis = 'TRACK_Y'
    point = _new_empty(coll, f"OM_kneepoint_{near}_{far}", parent=ray)
    point.location = (0.0, FAR, 0.0)
    return point


def clear_retarget(rig):
    """Remove retarget constraints from the rig and delete all helper objects."""
    if rig is not None and rig.type == 'ARMATURE':
        for pb in rig.pose.bones:
            for c in [c for c in pb.constraints if c.name.startswith(PREFIX)]:
                pb.constraints.remove(c)
    coll = bpy.data.collections.get(HELPER_COLLECTION)
    if coll:
        for obj in list(coll.objects):
            act = obj.animation_data.action if obj.animation_data else None
            bpy.data.objects.remove(obj, do_unlink=True)
            if act is not None and act.name.startswith(PREFIX) and act.users == 0:
                bpy.data.actions.remove(act)
        bpy.data.collections.remove(coll)


def _stash_action(rig):
    """Move the rig's current action aside (kept with a fake user) and reset it to rest."""
    ad = rig.animation_data
    if ad and ad.action:
        ad.action.use_fake_user = True
        rig[STASH_PROP] = ad.action.name
        ad.action = None
    for pb in rig.pose.bones:
        pb.location = (0.0, 0.0, 0.0)
        pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pb.rotation_euler = (0.0, 0.0, 0.0)
        pb.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)
        pb.scale = (1.0, 1.0, 1.0)


def _restore_action(rig):
    name = rig.get(STASH_PROP)
    action = bpy.data.actions.get(name) if name else None
    if action is None:
        return None
    if rig.animation_data is None:
        rig.animation_data_create()
    if rig.animation_data.action is None:
        rig.animation_data.action = action
        del rig[STASH_PROP]
        return action
    return None


LEGACY_MUTED_PROP = "open_mocap_muted"


def _mute_legacy_constraints(rig):
    """
        Copy constraints from the older manual setup (targets in the mocap skeleton, e.g.
        "hand_bone_12_14") fight the retarget, and stay on after its "OM" ones are muted for a bake.
        They are muted and remembered, so Remove Retarget can turn them back on.
    """
    muted = list(rig.get(LEGACY_MUTED_PROP, []))
    for pb in rig.pose.bones:
        for c in pb.constraints:
            target = getattr(c, "target", None)
            if c.name.startswith(PREFIX) or c.mute or target is None or target == rig:
                continue
            if c.type in {'COPY_ROTATION', 'COPY_LOCATION', 'COPY_TRANSFORMS', 'DAMPED_TRACK'} and (
                    target.name.startswith("hand_bone_") or _landmark_index(target) is not None):
                c.mute = True
                muted.append(f"{pb.name}|{c.name}")
    if muted:
        rig[LEGACY_MUTED_PROP] = muted
    return muted


def _restore_legacy_constraints(rig):
    for key in rig.get(LEGACY_MUTED_PROP, []):
        bone, _, name = key.partition("|")
        pb = rig.pose.bones.get(bone)
        c = pb.constraints.get(name) if pb else None
        if c is not None:
            c.mute = False
    if LEGACY_MUTED_PROP in rig:
        del rig[LEGACY_MUTED_PROP]


def setup_retarget(context, rig, empties, hips_motion=True, mirror_depth=False, arm_mode='POSITION',
                   leg_mode='PLANT', hip_height='FLOOR', shoulders=True, spine_twist=True, grip='NONE',
                   info=None):
    """
        Builds helpers and constraints. :return: (list of mapped bone names, missing names, scale).
        :param arm_mode: 'POSITION' hands placed relative to the shoulder, elbows solved for the rig's
                         arm lengths; 'DIRECTION' each arm segment copies the mocap segment direction.
        :param leg_mode: 'PLANT' ankles placed where the mocap ankles are and held while the foot is
                         down, 'POSITION' the same without holding, 'DIRECTION' thigh and shin copy
                         the mocap directions.
        :param hip_height: 'FLOOR' the rig's ankles stand at their rest height when the mocap's
                           stand on the floor; 'START' the hips start at their rest position.
        :param shoulders: drive the collar bones. :param spine_twist: spread the twist over the spine.
        :param grip: see arm_reach_directions. :param info: dict that receives numbers for the report.
    """
    info = {} if info is None else info
    clear_retarget(rig)
    _stash_action(rig)
    _mute_legacy_constraints(rig)
    coll = _helper_collection(context)
    scene = context.scene
    cache = {}
    mapped, missing = [], []
    points = _MocapSpace(coll, empties, mirror_depth)
    depth_sign = -1.0 if mirror_depth else 1.0

    mw = rig.matrix_world
    if is_control_rig(rig):
        _set_fk(rig)

    frames = np.arange(scene.frame_start, scene.frame_end + 1, dtype=float)
    fps = scene.render.fps / (scene.render.fps_base or 1.0)
    floor_data = mocap_floor(empties, frames, fps, mirror_depth) if len(frames) >= 3 else None
    floor = floor_data[0] if floor_data else None

    # --- Hips translation: mocap hip centre, scaled to the rig and anchored at its rest hips
    scale = 1.0
    hips_location = None
    follow = None
    if hips_motion:
        mocap_leg = mocap_leg_length(empties, scene.frame_start, scene.frame_end)
        rig_leg = rig_leg_length(rig)
        scale = rig_leg / mocap_leg if mocap_leg > 1e-6 else 1.0
        space = _new_empty(coll, "OM_hips_space")
        follow = _new_empty(coll, "OM_hips_follow", parent=space)
        c = follow.constraints.new('COPY_LOCATION')
        c.target = empties[HIP_C]
        c.target_space = 'LOCAL'
        c.owner_space = 'LOCAL'
        hips_rest = mw @ find_bone(rig, "Hips").bone.head_local
        space.scale = (scale, depth_sign * scale, scale)
        start = local_location(empties[HIP_C], scene.frame_start)
        space.location = hips_rest - Vector((scale * start.x, depth_sign * scale * start.y, scale * start.z))
        # Height from the floor instead of from the first frame (a take that starts crouched or in
        # the air would float or sink for its whole length): when the mocap ankles are on the
        # mocap's floor, the rig's hip joints are where its ankles at rest height need them.
        feet = [find_bone(rig, n) for n in ("LeftFoot", "RightFoot")]
        thighs = [find_bone(rig, n) for n in ("LeftUpLeg", "RightUpLeg")]
        if hip_height == 'FLOOR' and floor is not None and all(feet) and all(thighs):
            rig_ankle = sum((mw @ b.bone.head_local).z for b in feet) / 2
            hips_lift = hips_rest.z - sum((mw @ b.bone.head_local).z for b in thighs) / 2
            space.location.z = rig_ankle + hips_lift - scale * floor
            info["hips_lift"] = hips_lift
        hips = drive_bone(rig, "Hips")
        hips_location = hips.constraints.new('COPY_LOCATION')
        hips_location.name = PREFIX + "Hips Location"
        hips_location.target = follow

    # --- Directions
    reach_bones = {f"{s}{n}" for s in ("Left", "Right") for n in ("Arm", "ForeArm")}
    leg_bones = {n: find_bone(rig, n) for s in ("Left", "Right") for n in (s + "UpLeg", s + "Leg", s + "Foot")}
    solve_legs = (leg_mode in ('PLANT', 'POSITION') and len(frames) >= 3 and all(leg_bones.values())
                  and find_bone(rig, "Hips") is not None)
    stand = "hips_lift" in info                             # the rig stands on the floor
    if (solve_legs or stand) and len(frames) >= 3 and all(leg_bones.values()):
        h = {n: mw @ b.bone.head_local for n, b in leg_bones.items()}
        hips_head = mw @ find_bone(rig, "Hips").bone.head_local
        rig_leg = rig_leg_length(rig)
        mocap_leg = mocap_leg_length(empties, scene.frame_start, scene.frame_end)
        ratio = mocap_leg / rig_leg if rig_leg > 1e-9 else 1.0
        offsets = {s: _rig_components(rig, h[s + "UpLeg"] - hips_head, "LeftUpLeg", "RightUpLeg")
                   for s in ("Left", "Right")}
        # Feet are held where the rig is moved by the mocap hips; a rig that stays in place has
        # nothing to hold them against
        plant = leg_mode == 'PLANT' and hips_location is not None
        legs, shift, down = leg_directions(
            empties, frames, fps, mirror_depth, ratio, (h["LeftLeg"] - h["LeftUpLeg"]).length,
            (h["LeftFoot"] - h["LeftLeg"]).length, offsets, info.get("hips_lift", 0.0), plant,
            floor_data if stand else None)
        info["feet_down"] = down
        if not solve_legs:
            legs = {}                                       # only the hips' height is used
        if shift is not None and hips_location is not None:
            hips_location.target = _keyed_point(coll, follow, "OM_hips_shift", shift, frames)
            info["hips_pull"] = float(np.linalg.norm(shift, axis=1).max() / ratio)
        for side, (joint, thigh_dir, knee, shin_dir) in legs.items():
            for short, origin, dirs, nm in ((side + "UpLeg", joint, thigh_dir, "thigh"),
                                            (side + "Leg", knee, shin_dir, "shin")):
                pb = drive_bone(rig, short)
                c = pb.constraints.new('DAMPED_TRACK')
                c.name = PREFIX + "Aim"
                c.target = _keyed_far_point(coll, points, f"OM_leg_{nm}_{side}", origin, dirs, frames)
                c.track_axis = 'TRACK_Y'
                mapped.append(pb.name)
    solved = set(reach_bones) if arm_mode == 'POSITION' else set()
    if solve_legs:
        solved |= {s + n for s in ("Left", "Right") for n in ("UpLeg", "Leg")}
    for short, (a, b) in AIM_MAP.items():
        if short in solved:
            continue                                        # see "Arm reach" and "Feet"
        pb = drive_bone(rig, short)
        if pb is None:
            missing.append(short)
            continue
        c = pb.constraints.new('DAMPED_TRACK')
        c.name = PREFIX + "Aim"
        c.target = _aim_point(coll, cache, points, a, b)
        c.track_axis = 'TRACK_Y'
        mapped.append(pb.name)

    # --- Knees and feet
    space_scale = abs(points.space.matrix_world.to_scale().x)
    stats = {}
    for heel, toe, ankle in FOOT_MAP.values():
        stats[ankle] = mocap_foot_stats(empties, scene.frame_start, scene.frame_end, heel, toe, ankle)
    fwd = rig_forward(rig)
    up = Vector((0.0, 0.0, 1.0))
    for short, (near, far, ankle, toe) in KNEE_MAP.items():
        pb = drive_bone(rig, short)
        if pb is None:
            continue
        c = pb.constraints.new('LOCKED_TRACK')
        c.name = PREFIX + "Knee"
        c.target = _knee_hint(coll, cache, points, near, far, ankle, toe, KNEE_TOE_HINT * stats[ankle][0] * space_scale)
        c.lock_axis = 'LOCK_Y'
        c.track_axis = _twist_axis(rig, pb, -fwd)
    for short, (heel, toe, ankle) in FOOT_MAP.items():
        pb = drive_bone(rig, short)
        if pb is None:
            missing.append(short)
            continue
        # The rig's foot frame that matches the mocap frame of a foot standing flat
        p = stats[ankle][1]
        y = math.cos(p) * fwd + math.sin(p) * up
        z = -math.sin(p) * fwd + math.cos(p) * up
        frame_rest = Matrix((y.cross(z), y, z)).transposed()
        bone_rest = (mw.to_3x3() @ pb.bone.matrix_local.to_3x3()).normalized()
        offset = _new_empty(coll, f"OM_footbone_{short}", parent=_foot_frame(coll, points, heel, toe, ankle))
        offset.rotation_mode = 'QUATERNION'
        offset.rotation_quaternion = (frame_rest.inverted() @ bone_rest).to_quaternion()
        c = pb.constraints.new('COPY_ROTATION')
        c.name = PREFIX + "Foot"
        c.target = offset
        mapped.append(pb.name)

    # --- Arms and hands (keyed, smoothed targets; see ARM_MAP)
    reach = {}
    ref = {n: find_bone(rig, n) for n in ("LeftArm", "LeftForeArm", "LeftHand", "RightArm", "LeftUpLeg", "RightUpLeg")}
    if len(frames) >= 2 and all(ref.values()):
        h = {n: mw @ b.bone.head_local for n, b in ref.items()}
        upper_len = (h["LeftForeArm"] - h["LeftArm"]).length
        fore_len = (h["LeftHand"] - h["LeftForeArm"]).length
        chest = (h["LeftArm"] + h["RightArm"]) / 2
        rig_torso = (chest - (h["LeftUpLeg"] + h["RightUpLeg"]) / 2).length
        rig_width = (h["LeftArm"] - h["RightArm"]).length
        # Collar bones (on a control rig only if it has controls for them)
        arm_heads = None
        collars = {s: clavicle_bone(rig, s) for s in ("Left", "Right")} if shoulders else {}
        if collars and all(collars.values()):
            mocap_torso = _median([(local_location(empties[SHOULDER_C], f) - local_location(empties[HIP_C], f)).length
                                   for f in range(scene.frame_start, scene.frame_end + 1,
                                                  max(1, (scene.frame_end - scene.frame_start) // 60))])
            ratio_torso = mocap_torso / max(rig_torso, 1e-9)
            heads, rests = {}, {}
            for s, pb in collars.items():
                head = mw @ pb.bone.head_local
                heads[s] = _rig_components(rig, head - chest, "LeftArm", "RightArm")
                rests[s] = _rig_components(rig, h[s + "Arm"] - head, "LeftArm", "RightArm")
            arm_heads = {}
            for s, (head, direction, arm_head) in clavicle_directions(
                    empties, frames, fps, mirror_depth, ratio_torso, heads, rests).items():
                c = collars[s].constraints.new('DAMPED_TRACK')
                c.name = PREFIX + "Aim"
                c.target = _keyed_far_point(coll, points, f"OM_collar_{s}", head, direction, frames)
                c.track_axis = 'TRACK_Y'
                mapped.append(collars[s].name)
                arm_heads[s] = arm_head
        if arm_mode == 'POSITION':
            reach = arm_reach_directions(empties, frames, fps, upper_len, fore_len, rig_torso, rig_width,
                                         shoulders=arm_heads, grip=grip)
    if len(frames) >= 2:
        for side, (wrist, elbow, curl, hand_dir, palm) in arm_directions(empties, frames, fps).items():
            arm, fore, hand = (drive_bone(rig, side + n) for n in ("Arm", "ForeArm", "Hand"))
            index1, pinky1 = find_bone(rig, side + "HandIndex1"), find_bone(rig, side + "HandPinky1")
            if side in reach:
                shoulder, upper_dir, fore_dir = reach[side]
                for pb, origin, dirs, nm in ((arm, shoulder, upper_dir, "upper"), (fore, elbow, fore_dir, "fore")):
                    if pb is None:
                        missing.append(side + nm)
                        continue
                    c = pb.constraints.new('DAMPED_TRACK')
                    c.name = PREFIX + "Aim"
                    c.target = _keyed_far_point(coll, points, f"OM_reach_{nm}_{side}", origin, dirs, frames)
                    c.track_axis = 'TRACK_Y'
                    mapped.append(pb.name)
            if arm:
                c = arm.constraints.new('LOCKED_TRACK')
                c.name = PREFIX + "Elbow"
                c.target = _keyed_far_point(coll, points, f"OM_elbow_{side}", elbow, curl, frames)
                c.lock_axis = 'LOCK_Y'
                c.track_axis = _twist_axis(rig, arm, fwd)
            if hand is None:
                missing.append(side + "Hand")
                continue
            c = hand.constraints.new('DAMPED_TRACK')
            c.name = PREFIX + "Aim"
            c.target = _keyed_far_point(coll, points, f"OM_hand_{side}", wrist, hand_dir, frames)
            c.track_axis = 'TRACK_Y'
            mapped.append(hand.name)
            if index1 is None or pinky1 is None:
                missing.append(f"{side}Hand twist")
                continue
            ref_dir = (mw @ pinky1.bone.head_local) - (mw @ index1.bone.head_local)
            palm_target = _keyed_far_point(coll, points, f"OM_palm_{side}", wrist, palm, frames)
            for pb, influence in ((fore, FOREARM_TWIST_SHARE), (hand, 1.0)):
                if pb is None:
                    continue
                c = pb.constraints.new('LOCKED_TRACK')
                c.name = PREFIX + "Roll"
                c.target = palm_target
                c.lock_axis = 'LOCK_Y'
                c.track_axis = _twist_axis(rig, pb, ref_dir)
                c.influence = influence

    # --- Head: up axis from the face, turn toward the nose (keyed, smoothed; see head_directions)
    head = drive_bone(rig, "Head")
    if head is not None and len(frames) >= 2:
        ear_c, head_up, head_fwd = head_directions(empties, frames, fps)
        c = head.constraints.new('DAMPED_TRACK')
        c.name = PREFIX + "Aim"
        c.target = _keyed_far_point(coll, points, "OM_head_up", ear_c, head_up, frames)
        c.track_axis = 'TRACK_Y'
        c = head.constraints.new('LOCKED_TRACK')
        c.name = PREFIX + "Facing"
        c.target = _keyed_far_point(coll, points, "OM_head_fwd", ear_c, head_fwd, frames)
        c.lock_axis = 'LOCK_Y'
        c.track_axis = _twist_axis(rig, head, fwd)
        mapped.append(head.name)

    # --- Facing / roll (twist around the bone)
    for short, (a, b, ref_from, ref_to, influence) in FACING_MAP.items():
        if not spine_twist and short in ("Spine", "Spine1"):
            continue
        pb = drive_bone(rig, short)
        rf, rt = find_bone(rig, ref_from), find_bone(rig, ref_to)
        if pb is None or rf is None or rt is None:
            missing.append(f"{short} twist")
            continue
        ref_dir = (mw @ rt.bone.head_local) - (mw @ rf.bone.head_local)
        c = pb.constraints.new('LOCKED_TRACK')
        c.name = PREFIX + ("Roll" if short.endswith("Hand") else "Facing")
        c.target = _aim_point(coll, cache, points, a, b)
        c.lock_axis = 'LOCK_Y'
        c.track_axis = _twist_axis(rig, pb, ref_dir)
        c.influence = influence
        if pb.name not in mapped:
            mapped.append(pb.name)

    for obj in coll.objects:
        obj.hide_set(True)
    return mapped, missing, scale



# -------------------------------------------------------------
# OPERATORS
# -------------------------------------------------------------
class RetargetToMixamo(bpy.types.Operator):
    bl_idname = "object.open_mocap_retarget_mixamo"
    bl_label = "Retarget to Mixamo"
    bl_description = ("Drive the selected Mixamo rig with the tracked pose skeleton using constraints "
                      "(on a Mixamo control rig: its FK controls, with arms and legs switched to FK). "
                      "The rig's current action is kept aside with a fake user. Use Bake Motion afterwards")
    bl_options = {'REGISTER', 'UNDO'}

    target_rig: bpy.props.StringProperty(name="Target Rig")
    source: bpy.props.StringProperty(
        name="Mocap", default="",
        description="Mocap rig root (e.g. RigRoot.001) to retarget from. Empty: the newest take")
    hips_motion: bpy.props.BoolProperty(
        name="Move Hips", default=True,
        description="Copy the hip translation (scaled to the rig's leg length), not only rotations")
    mirror_depth: bpy.props.EnumProperty(
        name="Mirror Depth",
        items=[
            ('AUTO', "Auto", "Mirror front/back if the take is mirrored (nose and toes point backwards)"),
            ('ON', "Always", "Mirror the mocap front/back"),
            ('OFF', "Never", "Use the mocap depth as tracked"),
        ],
        default='AUTO',
        description="Combined front + side takes can come out mirrored front/back, depending on which "
                    "side of the actor the side camera stood. A mirrored take makes the rig lean the wrong way")
    arm_mode: bpy.props.EnumProperty(
        name="Arms",
        items=[
            ('POSITION', "Hand Position", "Hands go where the mocap hands are relative to the shoulder "
                                          "(scaled to the rig); elbows bend to fit the rig's arm lengths"),
            ('DIRECTION', "Bone Direction", "Upper arm and forearm copy the mocap directions; with other "
                                            "arm proportions the hands land elsewhere"),
        ],
        default='POSITION')
    leg_mode: bpy.props.EnumProperty(name="Legs", items=[
            ('PLANT', "Foot Position, Planted", "Ankles go where the mocap ankles are (knees bend to fit the "
                                                "rig's leg lengths) and are held still while the foot is down"),
            ('POSITION', "Foot Position", "Ankles go where the mocap ankles are, without holding them"),
            ('DIRECTION', "Bone Direction", "Thigh and shin copy the mocap directions; standing feet can slide"),
        ], default='PLANT')
    hip_height: bpy.props.EnumProperty(name="Hip Height", items=[
            ('FLOOR', "From Floor", "The rig's feet stand on its floor when the mocap's feet stand on theirs"),
            ('START', "From First Frame", "The hips start at the rig's rest position (use this if the "
                                          "feet are not in the video)"),
        ], default='FLOOR')
    shoulders: bpy.props.BoolProperty(
        name="Collar Bones", default=True,
        description="Lift the collar bones with the mocap shoulders (shrugs, raised arms)")
    spine_twist: bpy.props.BoolProperty(
        name="Spread Spine Twist", default=True,
        description="Spread the twist between hips and shoulders over the spine bones instead of "
                    "turning only the top one")
    grip: bpy.props.EnumProperty(name="Two-Handed Grip", items=[
            ('NONE', "Off", "Each hand follows its own mocap hand"),
            ('RIGHT', "Right Hand Leads", "Both hands hold one thing: the left hand keeps a steady distance to the right"),
            ('LEFT', "Left Hand Leads", "Both hands hold one thing: the right hand keeps a steady distance to the left"),
        ], default='NONE')

    def execute(self, context):
        rig = bpy.data.objects.get(self.target_rig)
        if not is_mixamo_rig(rig):
            self.report({'ERROR'}, "Select a Mixamo armature (bones like mixamorig:Hips, mixamorig:LeftArm)")
            return {'CANCELLED'}
        source = bpy.data.objects.get(self.source) if self.source else None
        if self.source and not is_mocap_root(source):
            self.report({'ERROR'}, f"{self.source} has no 40 landmark empties (0..39) as children")
            return {'CANCELLED'}
        empties = find_landmark_empties(source)
        if empties is None:
            self.report({'ERROR'}, "No pose skeleton found. Track a video and press Animate first")
            return {'CANCELLED'}

        scene = context.scene
        score = depth_mirror_score(empties, scene.frame_start, scene.frame_end)
        mirror = score < 0 if self.mirror_depth == 'AUTO' else self.mirror_depth == 'ON'
        info = {}
        mapped, missing, scale = setup_retarget(
            context, rig, empties, self.hips_motion, mirror, self.arm_mode, leg_mode=self.leg_mode,
            hip_height=self.hip_height, shoulders=self.shoulders, spine_twist=self.spine_twist,
            grip=self.grip, info=info)
        mapped = list(dict.fromkeys(mapped))
        scene.bake_rig = rig
        scene.bake_frame_start = scene.frame_start
        scene.bake_frame_end = scene.frame_end
        target = "FK controls of control rig " if is_control_rig(rig) else ""
        root = empties[0].parent.name if empties[0].parent else "landmarks"
        msg = (f"Retargeted {root} to {len(mapped)} {target}bones of {rig.name} (hips scale {scale:.3f}, "
               f"depth {'mirrored' if mirror else 'as tracked'}, facing check {score:+.2f}).")
        down = info.get("feet_down")
        if down and self.leg_mode == 'PLANT' and self.hips_motion:
            msg += f" Feet held: left {down['Left'] * 100:.0f}%, right {down['Right'] * 100:.0f}% of the take."
        if missing:
            msg += f" Not found: {', '.join(missing)}"
        self.report({'WARNING'} if missing else {'INFO'}, msg)
        return {'FINISHED'}


class ClearRetarget(bpy.types.Operator):
    bl_idname = "object.open_mocap_clear_retarget"
    bl_label = "Remove Retarget"
    bl_description = ("Remove the Open Mocap retarget constraints and helpers. "
                      "Restores the rig's previous action if it has none now")
    bl_options = {'REGISTER', 'UNDO'}

    target_rig: bpy.props.StringProperty(name="Target Rig")

    def execute(self, context):
        rig = bpy.data.objects.get(self.target_rig)
        clear_retarget(rig)
        restored = _restore_action(rig) if rig else None
        msg = "Retarget removed."
        if rig and LEGACY_MUTED_PROP in rig:
            _restore_legacy_constraints(rig)
            msg += " Earlier copy constraints turned back on."
        # A restored action was made for the rig's own IK/FK setting; a new bake plays in FK
        if restored and _restore_ikfk(rig):
            msg += " IK/FK switches restored."
        if restored:
            msg += f" Restored action {restored.name}."
        self.report({'INFO'}, msg)
        return {'FINISHED'}


classes = [RetargetToMixamo, ClearRetarget]
