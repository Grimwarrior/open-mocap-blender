"""
    Combine front + side pose tracking into one 3D take.

    The front video gives x (width) and y (height). The side video, filmed at ~90 degrees,
    sees the front camera's depth as its own x. The original Combine Data simply replaced
    each front point's depth with the side point's x, frame N with frame N. That assumes both
    videos start at the same moment, run at the same fps, are filmed from the same distance,
    and agree on which limb is left and which is right. The improved method fixes each of those:

    1. Sync      Both cameras see the same vertical motion. The side take is shifted in time
                 (with each video's own fps) until left/right-independent height signals
                 (hips, shoulders, nose, averaged wrists/elbows/knees/ankles) match best.
    2. Scale     Per frame, the ratio of the body's vertical extent in the two views gives the
                 side camera's scale relative to the front one (also follows perspective when
                 the actor walks), so side x becomes depth in front-view units.
    3. Left/right  A profile view often swaps the near and far limbs. For arms, legs and head
                 a Viterbi pass picks, per frame, the labelling whose heights match the front
                 view and whose motion is continuous.
    4. Facing    The depth sign depends on which side of the actor the side camera stood. It is
                 chosen so the body is right-handed (nose and toes point forward).
    5. Fallback  Where the side view cannot see a joint (low visibility) or does not cover the
                 frame, depth comes from the front view's own MediaPipe depth, calibrated
                 against the side depth.
    6. Smoothing Gaussian smoothing over time (depth more than width/height).
    7. Bones     Depth is corrected so bone lengths stay constant: with the front x/y kept,
                 |depth step| = sqrt(L^2 - dx^2 - dy^2), sign taken from the measurement.

    Data layout (as stored by utils.PoseTracking): per frame a flat list of 40 * [x, y, z],
    x = (px - w/2) / 100, y = (py - h) / 100, z = mediapipe z * w / 400.
    The combined output stores depth in the z slot, as before.
"""

import math
import numpy as np

N_PTS = 40
NOSE, L_EAR, R_EAR = 0, 7, 8
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP, L_ANKLE, R_ANKLE, L_TOE, R_TOE = 23, 24, 27, 28, 31, 32
HIP_C, SHOULDER_C, EAR_C = 33, 34, 37

CENTROIDS = {33: (23, 24), 34: (11, 12), 35: (18, 20), 36: (17, 19), 37: (7, 8), 38: (9, 10), 39: (1, 4)}

# Left/right label groups that swap together in a profile view
LR_GROUPS = {
    "arms": ([11, 13, 15, 17, 19, 21], [12, 14, 16, 18, 20, 22]),
    "legs": ([23, 25, 27, 29, 31], [24, 26, 28, 30, 32]),
    "head": ([1, 2, 3, 7, 9], [4, 5, 6, 8, 10]),
}

# Left/right independent height signals for syncing: lists of landmarks that are averaged
SYNC_SIGNALS = [[33], [34], [0], [15, 16], [13, 14], [25, 26], [27, 28]]

# Bone tree from the hip centre outwards (parent, child); followers move with an anchor
SEGMENTS = [(33, 23), (33, 24), (23, 25), (25, 27), (27, 29), (27, 31), (24, 26), (26, 28), (28, 30), (28, 32),
            (33, 34), (34, 11), (34, 12), (11, 13), (13, 15), (12, 14), (14, 16), (34, 37)]
FOLLOWERS = {15: [17, 19, 21], 16: [18, 20, 22], 37: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
LIMB_SEGMENTS = [(11, 13), (13, 15), (12, 14), (14, 16), (23, 25), (25, 27), (24, 26), (26, 28), (11, 12), (23, 24), (33, 34)]

# Tuning (chosen on the bundled Front/Side test pairs)
SIDE_VIS_LOW = 0.02       # side visibility below which the front view's depth takes over
SIDE_VIS_RANGE = 0.1      # ... blending to full side depth over this range. MediaPipe's own
                          # front depth is worse than even a low-visibility side measurement.
SCALE_SMOOTH_S = 1.0      # per-frame side scale smoothed over this many seconds (follows the actor
                          # walking toward/away from a camera); None: one scale for the whole take
PERSPECTIVE_SMOOTH_S = 1.0  # smoothing of the front view's perspective scale used for bone lengths
LR_SWITCH_PENALTY = 0.2   # extra cost for changing the left/right labelling, per joint
MIN_SYNC_MATCH = 0.3      # below this, the found offset is not trusted and 0 is used


# -------------------------------------------------------------
# SMALL HELPERS
# -------------------------------------------------------------
def _as_array(frames):
    return np.asarray(frames, dtype=np.float64).reshape(len(frames), N_PTS, 3)


def _gauss(a, sigma):
    """Gaussian smoothing along axis 0 (time), edges reflected."""
    if sigma <= 0 or len(a) < 3:
        return a
    radius = max(1, int(3 * sigma + 0.5))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    pad = [(radius, radius)] + [(0, 0)] * (a.ndim - 1)
    p = np.pad(a, pad, mode="reflect") if len(a) > radius else np.pad(a, pad, mode="edge")
    out = np.empty_like(a)
    flat_in = p.reshape(p.shape[0], -1)
    flat_out = out.reshape(a.shape[0], -1)
    for c in range(flat_in.shape[1]):
        flat_out[:, c] = np.convolve(flat_in[:, c], k, mode="valid")
    return out


def _interp(tq, ts, values):
    """Linear interpolation of values (Ns, ...) at times tq. :return: (Nq, ...)."""
    flat = values.reshape(len(ts), -1)
    out = np.empty((len(tq), flat.shape[1]))
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(tq, ts, flat[:, c])
    return out.reshape((len(tq),) + values.shape[1:])


def _zscore(v):
    s = v.std()
    return (v - v.mean()) / s if s > 1e-9 else v * 0.0


def _signals(P):
    """(N, G) left/right independent height signals."""
    return np.stack([P[:, idx, 1].mean(axis=1) for idx in SYNC_SIGNALS], axis=1)


def _fill_nan(v):
    ok = np.isfinite(v)
    if not ok.any():
        return np.ones_like(v)
    idx = np.arange(len(v))
    return np.interp(idx, idx[ok], v[ok])


def _median_filter(v, window):
    half = window // 2
    p = np.pad(v, half, mode="edge")
    return np.array([np.median(p[i:i + window]) for i in range(len(v))])


def to_blender(P_xy, depth):
    """Blender coordinates of front x/y + depth: (x, depth, -y), like utils.cv2blender."""
    return np.stack([P_xy[..., 0], depth, -P_xy[..., 1]], axis=-1)


def facing_score(B):
    """
        Right-handedness of a skeleton in Blender coordinates (N, 40, 3): mean of nose and
        toe direction . forward, with forward = left x up. Negative = mirrored front/back.
    """
    up = B[:, SHOULDER_C] - B[:, HIP_C]
    left = B[:, L_HIP] - B[:, R_HIP]
    fwd = np.cross(left, up)
    n = np.linalg.norm(fwd, axis=1, keepdims=True)
    ok = n[:, 0] > 1e-9
    fwd = fwd[ok] / n[ok]
    scores = []
    for v in (B[ok, NOSE] - (B[ok, L_EAR] + B[ok, R_EAR]) / 2,
              (B[ok, L_TOE] + B[ok, R_TOE]) / 2 - (B[ok, L_ANKLE] + B[ok, R_ANKLE]) / 2):
        vn = np.linalg.norm(v, axis=1)
        good = vn > 1e-9
        scores.append(np.sum(v[good] * fwd[good], axis=1) / vn[good])
    s = np.concatenate(scores) if scores else np.array([])
    return float(s.mean()) if len(s) else 0.0


def bone_length_cv(B, segments=LIMB_SEGMENTS, persp=None):
    """
        Mean coefficient of variation (%) of 3D limb lengths over the take. Real bones: 0.
        :param persp: optional per-frame perspective scale to divide out (actor walking to the camera).
    """
    cvs = []
    for a, b in segments:
        L = np.linalg.norm(B[:, b] - B[:, a], axis=1)
        if persp is not None:
            L = L / persp
        if L.mean() > 1e-9:
            cvs.append(L.std() / L.mean() * 100)
    return float(np.mean(cvs)) if cvs else 0.0


# -------------------------------------------------------------
# ORIGINAL METHOD
# -------------------------------------------------------------
def combine_simple(front_list, side_list, num_points=40):
    """Original Combine Data: frame N of the side replaces the depth of frame N of the front."""
    combined = []
    for f in range(min(len(front_list), len(side_list))):
        frame = list(front_list[f])
        for i in range(num_points):
            frame[3 * i + 2] = -side_list[f][3 * i + 0]
        combined.append(frame)
    return combined


# -------------------------------------------------------------
# IMPROVED METHOD
# -------------------------------------------------------------
def find_time_offset(tF, F, tS, S):
    """
        :return: (offset, correlation): side time = front time + offset (seconds), and the mean
                 correlation of the height signals at that offset.
    """
    sigF, sigS = _signals(F), _signals(S)
    durF, durS = tF[-1] - tF[0], tS[-1] - tS[0]
    min_overlap = 0.5 * min(durF, durS)
    lo = tS[0] - tF[-1] + min_overlap
    hi = tS[-1] - tF[0] - min_overlap
    fps = max(1.0 / max(np.median(np.diff(tF)), 1e-6), 1.0 / max(np.median(np.diff(tS)), 1e-6))
    step = 0.5 / fps
    if hi < lo:
        lo = hi = 0.0
    offsets = np.arange(lo, hi + step, step)
    scores = np.full(len(offsets), -np.inf)
    corrs = np.zeros(len(offsets))
    for n, d in enumerate(offsets):
        tq = tF + d
        valid = (tq >= tS[0]) & (tq <= tS[-1])
        if valid.sum() < 8:
            continue
        sS = _interp(tq[valid], tS, sigS)
        sF = sigF[valid]
        c = np.mean([np.mean(_zscore(sF[:, g]) * _zscore(sS[:, g])) for g in range(sF.shape[1])])
        corrs[n] = c
        overlap = (tq[valid][-1] - tq[valid][0]) / max(min(durF, durS), 1e-6)
        scores[n] = c - 0.1 * (1.0 - min(overlap, 1.0))
    best = int(np.argmax(scores))
    d = offsets[best]
    if 0 < best < len(offsets) - 1 and np.isfinite(scores[best - 1]) and np.isfinite(scores[best + 1]):
        y0, y1, y2 = scores[best - 1], scores[best], scores[best + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-12:
            d += step * 0.5 * (y0 - y2) / den
    return float(d), float(corrs[best])


def _scale_per_frame(F, S, valid, fps=30.0):
    """Side -> front scale per frame from the ratio of vertical body extents."""
    pairs = [([0], [27, 28]), ([34], [27, 28]), ([33], [34]), ([0], [33])]
    ratios = []
    for a, b in pairs:
        eF = np.abs(F[:, b, 1].mean(axis=1) - F[:, a, 1].mean(axis=1))
        eS = np.abs(S[:, b, 1].mean(axis=1) - S[:, a, 1].mean(axis=1))
        r = np.where((eS > 0.2) & (eF > 0.2) & valid, eF / np.maximum(eS, 1e-9), np.nan)
        ratios.append(r)
    stacked = np.stack(ratios, axis=1)
    r = np.full(len(F), np.nan)
    has = np.isfinite(stacked).any(axis=1)
    r[has] = np.nanmedian(stacked[has], axis=1)
    r = _fill_nan(r)
    r = _median_filter(r, 9)
    if SCALE_SMOOTH_S is None:
        return np.full(len(r), np.median(r))
    return _gauss(r, max(1.0, SCALE_SMOOTH_S * fps))


def _fix_left_right(F, S, visS, scale, valid):
    """Viterbi over (keep, swap) per label group. :return: swapped S, visS and swap counts."""
    S, visS = S.copy(), visS.copy()
    counts = {}
    yF = F[:, :, 1] - F[:, [HIP_C], 1]
    yS = (S[:, :, 1] - S[:, [HIP_C], 1]) * scale[:, None]
    XS = S[:, :, :2] * scale[:, None, None]
    n = len(F)
    for name, (L, R) in LR_GROUPS.items():
        L, R = np.array(L), np.array(R)
        joints = np.concatenate([L, R])
        assign = [joints, np.concatenate([R, L])]          # state 0: keep, state 1: swap
        emit = np.zeros((n, 2))
        for s in (0, 1):
            src = assign[s]
            emit[:, s] = np.sum(np.abs(yS[:, src] - yF[:, joints]) * visS[:, src], axis=1)
        emit[~valid] = 0.0
        cost = emit[0].copy()
        back = np.zeros((n, 2), dtype=int)
        for k in range(1, n):
            new = np.empty(2)
            for s in (0, 1):
                trans = [(np.sum(np.linalg.norm(XS[k, assign[s]] - XS[k - 1, assign[p]], axis=1))
                          if valid[k] and valid[k - 1] else 0.0)
                         + (LR_SWITCH_PENALTY * len(joints) * scale[k] if s != p else 0.0) for p in (0, 1)]
                options = cost + np.array(trans)
                back[k, s] = int(np.argmin(options))
                new[s] = options[back[k, s]] + emit[k, s]
            cost = new
        states = np.zeros(n, dtype=int)
        states[-1] = int(np.argmin(cost))
        for k in range(n - 1, 0, -1):
            states[k - 1] = back[k, states[k]]
        swap = states == 1
        if swap.any():
            S[np.ix_(swap, L)], S[np.ix_(swap, R)] = S[np.ix_(swap, R)].copy(), S[np.ix_(swap, L)].copy()
            visS[np.ix_(swap, L)], visS[np.ix_(swap, R)] = visS[np.ix_(swap, R)].copy(), visS[np.ix_(swap, L)].copy()
        counts[name] = int(np.sum(np.diff(np.concatenate([[0], states])) != 0))
    for c, (a, b) in CENTROIDS.items():
        S[:, c] = (S[:, a] + S[:, b]) / 2
        visS[:, c] = np.minimum(visS[:, a], visS[:, b])
    return S, visS, counts


def _side_weight(visS):
    """How much to trust the side depth of each joint (0 = use the front view's depth)."""
    return np.clip((visS - SIDE_VIS_LOW) / SIDE_VIS_RANGE, 0.0, 1.0)


def front_perspective_scale(F, D, fps):
    """
        How big the actor appears in the front video per frame, relative to the take's median
        (walking toward the camera makes every bone longer in image units). Taken from the 3D
        lengths of all limbs together, smoothed over about a second.
    """
    ratios = []
    for p, c in LIMB_SEGMENTS:
        L3 = np.sqrt(np.sum((F[:, c, :2] - F[:, p, :2]) ** 2, axis=1) + (D[:, c] - D[:, p]) ** 2)
        med = np.median(L3)
        if med > 1e-9:
            ratios.append(L3 / med)
    if not ratios:
        return np.ones(len(F))
    s = np.median(np.stack(ratios, axis=1), axis=1)
    s = _gauss(_median_filter(s, 9), max(1.0, PERSPECTIVE_SMOOTH_S * fps))
    return s / np.median(s)


def _enforce_bone_lengths(F, D, weight, strength, fps=30.0):
    """Correct depth so segment lengths stay constant (front x/y unchanged), allowing for perspective."""
    if strength <= 0:
        return D
    Dn = D.copy()
    n = len(F)
    persp = front_perspective_scale(F, D, fps)
    for p, c in SEGMENTS:
        dxy = np.linalg.norm(F[:, c, :2] - F[:, p, :2], axis=1)
        rel = D[:, c] - D[:, p]
        both = (weight[:, p] > 0.5) & (weight[:, c] > 0.5)
        L3 = np.sqrt(dxy ** 2 + rel ** 2)
        L = max(np.percentile(dxy / persp, 90), np.median(L3[both] / persp[both]) if both.sum() > 10 else 0.0)
        Lt = L * persp
        dz_len = np.sqrt(np.maximum(0.0, Lt * Lt - dxy ** 2))
        sign = 1.0 if (rel[0] if n else 0) >= 0 else -1.0
        for k in range(n):
            if abs(rel[k]) >= 0.1 * Lt[k]:
                sign = 1.0 if rel[k] >= 0 else -1.0
            dz = (1 - strength) * rel[k] + strength * sign * dz_len[k]
            Dn[k, c] = Dn[k, p] + dz
        for f in FOLLOWERS.get(c, []):
            Dn[:, f] = D[:, f] - D[:, c] + Dn[:, c]
    for c, (a, b) in CENTROIDS.items():
        Dn[:, c] = (Dn[:, a] + Dn[:, b]) / 2
    return Dn


def combine_improved(front_list, side_list, front_times, side_times, front_vis=None, side_vis=None,
                     smoothing=1.5, bone_strength=0.8, offset=None):
    """
        :param front_list/side_list: tracked frames (flat 40 * [x, y, z]).
        :param front_times/side_times: time of each tracked frame in its own video (seconds).
        :param front_vis/side_vis: 40 visibility scores per frame (optional).
        :param smoothing: Gaussian sigma in frames for depth (half of it for x/y). 0 = off.
        :param bone_strength: 0..1, how strongly depth is corrected to keep bone lengths.
        :param offset: force the side time offset (seconds) instead of finding it.
        :return: (combined frames as flat lists, report dict)
    """
    F = _as_array(front_list)
    S_raw = _as_array(side_list)
    tF = np.asarray(front_times, dtype=np.float64)
    tS = np.asarray(side_times, dtype=np.float64)
    visF = np.asarray(front_vis, dtype=np.float64) if front_vis is not None else np.ones((len(F), N_PTS))
    visS_raw = np.asarray(side_vis, dtype=np.float64) if side_vis is not None else np.ones((len(S_raw), N_PTS))
    report = {}

    # 1. Sync
    if offset is None:
        offset, sync_corr = find_time_offset(tF, F, tS, S_raw)
        report["sync_trusted"] = sync_corr >= MIN_SYNC_MATCH
        if not report["sync_trusted"]:
            offset = 0.0                      # no clear match: assume the videos start together
    else:
        sync_corr = float("nan")
        report["sync_trusted"] = True
    tq = tF + offset
    valid = (tq >= tS[0]) & (tq <= tS[-1])
    S = _interp(tq, tS, S_raw)
    visS = _interp(tq, tS, visS_raw) * valid[:, None]
    report.update(offset_s=offset, sync_corr=sync_corr, coverage=float(valid.mean()))

    # 2. Scale
    fps = 1.0 / max(float(np.median(np.diff(tF))), 1e-6) if len(tF) > 1 else 30.0
    scale = _scale_per_frame(F, S, valid, fps)
    report["scale"] = float(np.median(scale[valid])) if valid.any() else 1.0

    # 3. Left/right labels in the side view
    S, visS, swaps = _fix_left_right(F, S, visS, scale, valid)
    report["lr_switches"] = swaps

    # 4. Facing: side x -> depth with the sign that makes the body right-handed
    depth_side = S[:, :, 0] * scale[:, None]
    sign = -1.0
    if facing_score(to_blender(F, sign * depth_side)) < 0:
        sign = 1.0
    depth_side = sign * depth_side
    report["depth_sign"] = sign

    # 5. Fallback from the front view's own depth, calibrated against the side depth
    w = _side_weight(visS)
    root = depth_side[:, HIP_C].copy()
    if valid.any() and not valid.all():
        idx = np.arange(len(F))
        root = np.interp(idx, idx[valid], root[valid])
    zF_rel = F[:, :, 2] - F[:, [HIP_C], 2]
    side_rel = depth_side - root[:, None]
    good = (w > 0.8) & (visF > 0.5)
    k = 4.0
    if good.sum() > 50:
        x, y = zF_rel[good], side_rel[good]
        den = float(np.dot(x, x))
        if den > 1e-9:
            k_fit = float(np.dot(x, y) / den)
            corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-9 and y.std() > 1e-9 else 0.0
            k = float(np.clip(k_fit, -8, 8)) if abs(corr) > 0.2 else 4.0 * (1 if k_fit >= 0 else -1)
    fallback = root[:, None] + k * zF_rel
    D = w * depth_side + (1 - w) * fallback
    report["fallback_share"] = float(1 - w[:, :33].mean())

    # 6. Smoothing
    D = _gauss(D, smoothing)
    XY = _gauss(F[:, :, :2], smoothing / 2)
    F_out = F.copy()
    F_out[:, :, :2] = XY

    # 7. Constant bone lengths
    persp = front_perspective_scale(F_out, D, fps)
    report["perspective_range"] = (float(persp.min()), float(persp.max()))
    report["bone_cv_before"] = bone_length_cv(to_blender(F_out, D), persp=persp)
    D = _enforce_bone_lengths(F_out, D, w, bone_strength, fps)
    report["bone_cv_after"] = bone_length_cv(to_blender(F_out, D), persp=persp)
    report["facing"] = facing_score(to_blender(F_out, D))

    F_out[:, :, 2] = D
    return F_out.reshape(len(F_out), -1).tolist(), report


def format_report(r):
    sw = r.get("lr_switches", {})
    match = "manual" if r["sync_corr"] != r["sync_corr"] else f"match {r['sync_corr']:.2f}"
    return (f"side offset {r['offset_s']:+.2f}s ({match}), side scale {r['scale']:.2f}, "
            f"L/R fixes arms {sw.get('arms', 0)} legs {sw.get('legs', 0)}, "
            f"depth from front view {r['fallback_share'] * 100:.0f}%, "
            f"bone length variation {r['bone_cv_before']:.0f}% -> {r['bone_cv_after']:.0f}%")
