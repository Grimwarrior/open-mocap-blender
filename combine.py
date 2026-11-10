"""
    Combine front + side pose tracking into one 3D take.

    The front video gives x (width) and y (height). The side video, filmed at ~90 degrees,
    sees the front camera's depth as its own x. The original Combine Data simply replaced
    each front point's depth with the side point's x, frame N with frame N. That assumes both
    videos start at the same moment, run at the same fps, are filmed from the same distance,
    at exactly 90 degrees, and agree on which limb is left and which is right. The improved
    method fixes each of those:

    1. Sync      Both cameras see the same vertical motion. The side take is shifted in time
                 (with each video's own fps) until left/right-independent height signals
                 (hips, shoulders, nose, averaged wrists/elbows/knees/ankles) match best.
                 A shift is only used when it matches clearly better than none (MIN_SYNC_GAIN):
                 with two separate performances the match is weak and a few frames of shift can
                 follow the noise. Time Warp (for two separate performances of the same move, e.g.
                 one camera, actor turned 90 degrees for the second take) then lets the side
                 timing drift by up to WARP_BAND_S around that offset (dynamic time warping).
    2. Scale     Per frame, the ratio of the body's vertical extent in the two views gives the
                 side camera's scale relative to the front one (also follows perspective when
                 the actor walks), so side x becomes depth in front-view units.
    3. Left/right  A profile view often swaps the near and far limbs. For arms and legs a
                 Viterbi pass picks, per frame, the labelling whose heights match the front view and
                 whose motion is continuous; a swap must be clearly better (LR_SWAP_BIAS), since in
                 separate takes heights differ anyway. The head is left alone: left and right ear
                 heights are nearly equal, so a swap there was a coin toss that turned the face.
    4. Angle     A side camera at angle a sees x_side = cos(a) * x_front - sin(a) * depth. Two views
                 alone cannot tell a from depth; the angle whose depth keeps the bones at a
                 constant length while the actor moves is used (90 unless clearly better).
       Facing    The depth sign depends on which side of the actor the side camera stood. It is
                 chosen so the body is right-handed (nose and toes point forward).
    5. Gaps      Where the side view loses a joint for a moment (low visibility), its depth is
                 interpolated from neighbouring frames that saw it; only longer gaps or frames
                 the side video does not cover use the front view's own MediaPipe depth,
                 calibrated against the side depth.
    6. Smoothing Gaussian smoothing over time (depth more than width/height).
    7. Bones     Bones that come out too long are shortened: with the front x/y kept, the
                 depth step is limited to sqrt(L^2 - dx^2 - dy^2). L (left and right limbs share
                 one) follows the actor's distance to the front camera, measured from 2D limb
                 lengths only. Depth is never lengthened to reach L: setting every bone to exactly
                 L made limbs pop front to back whenever the step changed sign, and stretched them
                 in depth wherever L came out a little long.

    Data layout (as stored by utils.PoseTracking): per frame a flat list of 40 * [x, y, z],
    x = (px - w/2) / 100, y = (py - h) / 100, z = mediapipe world depth in pixels / 100.
    The combined output stores depth in the z slot, as before.
"""

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
PERSPECTIVE_HOLD_S = 0.5  # ... its upper envelope of limb lengths is held over this time
LR_SWITCH_PENALTY = 0.2   # extra cost for changing the left/right labelling, per joint
LR_FIX_GROUPS = ("arms", "legs")  # label groups the left/right fix may swap (not the head)
LR_SWAP_BIAS = 0.005      # extra cost of a swapped frame per joint, in body heights
MIN_SYNC_GAIN = 0.075     # a time shift must raise the sync match by this much over no shift
MIN_SYNC_MATCH = 0.3      # below this, the found offset is not trusted and 0 is used
WARP_BAND_S = 1.0         # time warp: how far (s) the side timing may drift from the constant offset
WARP_SLOPE_PENALTY = 0.3  # time warp: cost of speeding up / slowing down the side take for a frame
WARP_SMOOTH_S = 0.15      # time warp: smoothing of the found timing (s)
WARP_VEL_WEIGHT = 0.7     # time warp: weight of signal velocities next to the heights
ANGLE_RANGE = (40.0, 140.0)  # view angles searched (degrees)
ANGLE_MIN_GAIN = 0.03     # an angle other than 90 must lower bone length variation by this much
PEAK_SIGMA = 1.6          # peak-keeping smoothing: window width relative to the plain Gaussian's
SMOOTH_PEAKS = True       # final smoothing keeps peaks (False: plain Gaussian, as before)
FUSE_HEIGHTS = True       # heights of joints hidden in the front view come from the side view
FUSE_MAX_HEIGHT_ERR = 2.5  # ... only if the views' heights agree this well (% of body height; two
                          # cameras on one performance: about 1, separate takes: 5 and more)
FUSE_FRONT_VIS = 0.5      # ... front visibility below which that starts (and side visibility above which)
FUSE_VIS_RANGE = 0.3      # ... and the range over which it blends in
GAP_FILL_S = 0.1          # occluded joints: side depth borrowed from frames within about this time


# -------------------------------------------------------------
# SMALL HELPERS
# -------------------------------------------------------------
def _as_array(frames):
    return np.asarray(frames, dtype=np.float64).reshape(len(frames), N_PTS, 3)


def _gauss(a, sigma, trend=False, peaks=False):
    """
        Gaussian smoothing along axis 0 (time), edges reflected. trend=True reflects oddly
        (point-symmetric about the end value), which keeps a steady rise or fall at the ends instead
        of bending it toward the mirrored values (scale of an actor walking toward a camera, times).
        peaks=True fits a parabola under the same Gaussian window instead of a constant (a
        Savitzky-Golay filter with Gaussian weights): a plain average flattens every peak and
        rounds every fast move, the parabola follows them, at PEAK_SIGMA times the width for
        about the same noise reduction.
    """
    if sigma <= 0 or len(a) < 3:
        return a
    if peaks:
        sigma = sigma * PEAK_SIGMA
    radius = max(1, int(3 * sigma + 0.5))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    if peaks and radius >= 2:
        m0, m2, m4 = k.sum(), (k * x ** 2).sum(), (k * x ** 4).sum()
        k = k * (m4 - m2 * x ** 2) / (m0 * m4 - m2 * m2)
    k /= k.sum()
    pad = [(radius, radius)] + [(0, 0)] * (a.ndim - 1)
    if len(a) > radius:
        p = np.pad(a, pad, mode="reflect", reflect_type="odd" if trend else "even")
    else:
        p = np.pad(a, pad, mode="edge")
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
def _fps(t):
    return 1.0 / max(float(np.median(np.diff(t))), 1e-6) if len(t) > 1 else 30.0


def _offset_score(tF, sigF, tS, sigS, d, durF, durS):
    """(score, correlation) of the height signals with side time = front time + d."""
    tq = tF + d
    valid = (tq >= tS[0]) & (tq <= tS[-1])
    if valid.sum() < 8:
        return -np.inf, 0.0
    sS = _interp(tq[valid], tS, sigS)
    sF = sigF[valid]
    c = float(np.mean([np.mean(_zscore(sF[:, g]) * _zscore(sS[:, g])) for g in range(sF.shape[1])]))
    overlap = (tq[valid][-1] - tq[valid][0]) / max(min(durF, durS), 1e-6)
    return c - 0.1 * (1.0 - min(overlap, 1.0)), c


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
    step = 1.0 / max(_fps(tF), _fps(tS))
    if hi < lo:
        lo = hi = 0.0
    # Coarse search on a whole-frame grid through 0, then a fine search around the best offset.
    # (A grid that does not contain 0 plus parabolic refinement was biased by up to half a frame.)
    offsets = np.arange(np.floor(lo / step), np.ceil(hi / step) + 1) * step
    scores = np.array([_offset_score(tF, sigF, tS, sigS, d, durF, durS)[0] for d in offsets])
    best = offsets[int(np.argmax(scores))]
    fine = best + np.linspace(-step, step, 21)
    res = [_offset_score(tF, sigF, tS, sigS, d, durF, durS) for d in fine]
    k = int(np.argmax([r[0] for r in res]))
    # Keep the videos as they start unless a shift matches clearly better: with two separate
    # performances the match is weak and a few frames of shift can just follow the noise.
    _, c0 = _offset_score(tF, sigF, tS, sigS, 0.0, durF, durS)
    if res[k][1] - c0 < MIN_SYNC_GAIN:
        return 0.0, float(c0)
    return float(fine[k]), float(res[k][1])


def _warp_features(P):
    """Per-frame features for time warping: z-scored height signals and their velocities."""
    sig = _signals(P)
    sig = (sig - sig.mean(axis=0)) / np.maximum(sig.std(axis=0), 1e-9)
    vel = np.gradient(_gauss(sig, 1.5), axis=0)
    vel = vel / np.maximum(vel.std(axis=0), 1e-9)
    return np.concatenate([sig, WARP_VEL_WEIGHT * vel], axis=1)


def time_warp(tF, F, tS, S, offset):
    """
        Dynamic time warping for two separate performances of the same move, whose timing drifts.
        Searches within WARP_BAND_S of the constant offset, slopes between 1/2 and 2.
        :return: side time for every front frame (seconds).
    """
    fpsF = _fps(tF)
    band = max(WARP_BAND_S, 0.0)
    # Side signals resampled on the front frame rate, padded by the band at both ends
    # (padded columns are "no side video" and cost a neutral amount).
    tg = np.arange(tS[0] - band, tS[-1] + band + 0.5 / fpsF, 1.0 / fpsF)
    inside = (tg >= tS[0]) & (tg <= tS[-1])
    fF = _warp_features(F)
    fS = _warp_features(S)
    fG = _interp(np.clip(tg, tS[0], tS[-1]), tS, fS)
    n, m = len(tF), len(tg)
    # Only a band of columns around the constant offset is stored per row: row i covers the
    # columns base[i] .. base[i] + W - 1 (memory n * W instead of n * m for long takes).
    half = int(np.ceil(band * fpsF))
    W = 2 * half + 1
    base = np.round((tF + offset - tg[0]) * fpsF).astype(int) - half
    cols = base[:, None] + np.arange(W)[None, :]                       # (n, W) column index
    ok = (cols >= 0) & (cols < m)
    cc = np.clip(cols, 0, m - 1)
    cost = np.mean((fG[cc] - fF[:, None, :]) ** 2, axis=2)
    neutral = float(np.median(cost[ok & inside[cc]])) if (ok & inside[cc]).any() else 1.0
    cost = np.where(inside[cc], cost, neutral)                         # padding: no side video
    cost[~ok] = np.inf

    def prev(Drow, b_prev, b_cur, delta):
        """Values of a previous row's band at columns (current band columns - delta)."""
        idx = np.arange(W) + (b_cur - b_prev) - delta
        out = np.full(W, np.inf)
        good = (idx >= 0) & (idx < W)
        out[good] = Drow[idx[good]]
        return out

    # DP with steps (1,1), (1,2), (2,1); off-diagonal steps pay a slope penalty
    pen = WARP_SLOPE_PENALTY
    D = np.full((n, W), np.inf)
    move = np.zeros((n, W), dtype=np.int8)
    D[0] = cost[0]                                                     # free start column
    for i in range(1, n):
        opts = np.stack([prev(D[i - 1], base[i - 1], base[i], 1),
                         prev(D[i - 1], base[i - 1], base[i], 2) + pen,
                         prev(D[i - 2], base[i - 2], base[i], 1) + pen if i >= 2 else np.full(W, np.inf)])
        k = np.argmin(opts, axis=0)
        D[i] = cost[i] + opts[k, np.arange(W)]
        move[i] = k
    # Backtrack from the cheapest end column (free end)
    b = int(np.argmin(D[-1]))
    if not np.isfinite(D[-1, b]):
        return tF + offset                                             # no path: keep the offset
    col = np.full(n, np.nan)
    i, j = n - 1, base[-1] + b
    col[i] = j
    while i > 0:
        k = move[i, j - base[i]]
        if k == 0:
            i, j = i - 1, j - 1
        elif k == 1:
            i, j = i - 1, j - 2
        else:
            col[i - 1] = j - 0.5                                        # row skipped: halfway
            i, j = i - 2, j - 1
        if i >= 0:
            col[i] = j
    col = _fill_nan(col)
    ts = tg[0] + col / fpsF
    ts = _gauss(ts, WARP_SMOOTH_S * fpsF, trend=True)
    return np.maximum.accumulate(ts)


def height_agreement(F, S, scale, visS, valid):
    """
        How well the side view's joint heights match the front view's (both relative to the hips),
        as mean error in % of the body height. Tells how well the two takes are synced.
    """
    yF = F[:, :33, 1] - F[:, [HIP_C], 1]
    yS = (S[:, :33, 1] - S[:, [HIP_C], 1]) * scale[:, None]
    body = np.abs(F[:, NOSE, 1] - (F[:, L_ANKLE, 1] + F[:, R_ANKLE, 1]) / 2)
    w = visS[:, :33] * valid[:, None]
    if w.sum() < 1e-9:
        return float("nan")
    err = np.abs(yF - yS) / np.maximum(body, 1e-9)[:, None]
    return float(np.sum(err * w) / np.sum(w) * 100)


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
    return _gauss(r, max(1.0, SCALE_SMOOTH_S * fps), trend=True)


def _fix_left_right(F, S, visS, scale, valid):
    """Viterbi over (keep, swap) per label group. :return: swapped S, visS and swap counts."""
    S, visS = S.copy(), visS.copy()
    counts = {}
    yF = F[:, :, 1] - F[:, [HIP_C], 1]
    yS = (S[:, :, 1] - S[:, [HIP_C], 1]) * scale[:, None]
    XS = S[:, :, :2] * scale[:, None, None]
    n = len(F)
    body = np.abs(F[:, NOSE, 1] - (F[:, L_ANKLE, 1] + F[:, R_ANKLE, 1]) / 2)
    for name, (L, R) in LR_GROUPS.items():
        if name not in LR_FIX_GROUPS:
            counts[name] = 0
            continue
        L, R = np.array(L), np.array(R)
        joints = np.concatenate([L, R])
        assign = [joints, np.concatenate([R, L])]          # state 0: keep, state 1: swap
        emit = np.zeros((n, 2))
        for s in (0, 1):
            src = assign[s]
            emit[:, s] = np.sum(np.abs(yS[:, src] - yF[:, joints]) * visS[:, src], axis=1)
        emit[~valid] = 0.0
        emit[:, 1] += LR_SWAP_BIAS * len(joints) * body     # a swap must be clearly better
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


def front_perspective_scale(F, fps):
    """
        How big the actor appears in the front video per frame, relative to the take's median
        (walking toward the camera makes every bone longer in image units). From the front video's
        2D limb lengths only, so wrong depth cannot feed back into it: per frame the limbs closest to
        the image plane (upper envelope of 2D length / longest 2D length), held over
        PERSPECTIVE_HOLD_S and smoothed over PERSPECTIVE_SMOOTH_S.
    """
    L2 = np.stack([np.linalg.norm(F[:, c, :2] - F[:, p, :2], axis=1) for p, c in LIMB_SEGMENTS], axis=1)
    persp = np.ones(len(F))
    hold = max(1, int(round(PERSPECTIVE_HOLD_S * fps)))
    for _ in range(2):
        ref = np.percentile(L2 / persp[:, None], 95, axis=0)
        r = L2 / np.maximum(ref, 1e-9)
        s = np.percentile(r, 80, axis=1)
        p = np.pad(s, hold, mode="edge")
        s = np.array([p[k:k + 2 * hold + 1].max() for k in range(len(s))])
        s = _gauss(s, max(1.0, PERSPECTIVE_SMOOTH_S * fps), trend=True)
        persp = s / np.median(s)
    return persp


# Left/right segments that share one length
MIRRORED_SEGMENTS = {(23, 25): (24, 26), (25, 27): (26, 28), (11, 13): (12, 14), (13, 15): (14, 16),
                     (33, 23): (33, 24), (34, 11): (34, 12), (27, 29): (28, 30), (27, 31): (28, 32)}


def _segment_lengths(F, D, weight, persp):
    """Length of each segment at perspective scale 1; mirrored left/right segments share one."""
    out = {}
    for p, c in SEGMENTS:
        dxy = np.linalg.norm(F[:, c, :2] - F[:, p, :2], axis=1)
        both = (weight[:, p] > 0.5) & (weight[:, c] > 0.5)
        L3 = np.sqrt(dxy ** 2 + (D[:, c] - D[:, p]) ** 2)
        l2 = np.percentile(dxy / persp, 90)
        out[(p, c)] = max(l2, np.median(L3[both] / persp[both]) if both.sum() > 10 else 0.0)
    for a, b in MIRRORED_SEGMENTS.items():
        if a in out and b in out:
            out[a] = out[b] = 0.5 * (out[a] + out[b])
    return out


def _enforce_bone_lengths(F, D, weight, strength, fps=30.0):
    """Shorten the depth step of segments that come out too long (front x/y unchanged)."""
    if strength <= 0:
        return D
    Dn = D.copy()
    persp = front_perspective_scale(F, fps)
    lengths = _segment_lengths(F, D, weight, persp)
    for p, c in SEGMENTS:
        dxy = np.linalg.norm(F[:, c, :2] - F[:, p, :2], axis=1)
        rel = D[:, c] - D[:, p]
        Lt = lengths[(p, c)] * persp
        dz_max = np.sqrt(np.maximum(0.0, Lt * Lt - dxy ** 2))
        dz = np.sign(rel) * np.minimum(np.abs(rel), dz_max)
        Dn[:, c] = Dn[:, p] + (1 - strength) * rel + strength * dz
        for f in FOLLOWERS.get(c, []):
            Dn[:, f] = D[:, f] - D[:, c] + Dn[:, c]
    for c, (a, b) in CENTROIDS.items():
        Dn[:, c] = (Dn[:, a] + Dn[:, b]) / 2
    return Dn


def estimate_view_angle(F, U, w, valid, fps):
    """
        Angle between the two views (degrees, 90 = side camera at a right angle). Two orthographic
        views cannot tell the angle apart from depth on their own; the right angle is the one whose
        depth keeps the bones at a constant length while the actor moves.
        :param U: side x in front units (N, 40), w: side weight per joint.
        :return: (angle, bone variation at that angle, bone variation at 90 degrees)
    """
    m = valid & (np.mean(w[:, 11:29] > 0.5, axis=1) > 0.7)
    if m.sum() < 20:
        return 90.0, float("nan"), float("nan")
    Fm, Um = F[m], U[m]
    Xr, Ur = Fm[:, :, 0] - Fm[:, [HIP_C], 0], Um - Um[:, [HIP_C]]
    persp = front_perspective_scale(F, fps)[m]

    def cv(phi):
        p = np.radians(phi)
        D = _gauss((Ur - np.cos(p) * Xr) / np.sin(p), 1.5)
        return bone_length_cv(to_blender(Fm, D), persp=persp)

    grid = np.arange(ANGLE_RANGE[0], ANGLE_RANGE[1] + 0.1, 2.0)
    vals = np.array([cv(a) for a in grid])
    k = int(np.argmin(vals))
    best, best_cv = float(grid[k]), float(vals[k])
    if 0 < k < len(grid) - 1:
        y0, y1, y2 = vals[k - 1], vals[k], vals[k + 1]
        den = y0 - 2 * y1 + y2
        if den > 1e-12:
            best = float(grid[k] + 2.0 * 0.5 * (y0 - y2) / den)
            best_cv = cv(best)
    cv90 = cv(90.0)
    if best_cv > cv90 * (1.0 - ANGLE_MIN_GAIN):
        return 90.0, cv90, cv90           # no clear evidence: keep the right angle
    return best, best_cv, cv90


def _fill_gaps(rel, w, fps):
    """
        Side depth (relative to the hips) for joints the side view lost for a moment: interpolated
        from neighbouring frames where the joint was seen. :return: (filled, confidence 0..1)
    """
    sigma = max(1.0, GAP_FILL_S * fps)
    num = _gauss(rel * w, sigma)
    den = _gauss(w, sigma)
    filled = num / np.maximum(den, 1e-6)
    conf = np.clip(den / 0.5, 0.0, 1.0)
    return filled, conf


def combine_improved(front_list, side_list, front_times, side_times, front_vis=None, side_vis=None,
                     smoothing=1.5, bone_strength=0.8, offset=None, sync="OFFSET", side_angle=None):
    """
        :param front_list/side_list: tracked frames (flat 40 * [x, y, z]).
        :param front_times/side_times: time of each tracked frame in its own video (seconds).
        :param front_vis/side_vis: 40 visibility scores per frame (optional).
        :param smoothing: Gaussian sigma in frames for depth (half of it for x/y). 0 = off.
        :param bone_strength: 0..1, how strongly depth is corrected to keep bone lengths.
        :param offset: force the side time offset (seconds) instead of finding it.
        :param sync: 'OFFSET' one time offset (both videos filmed at the same time), 'WARP' time warp
                     on top of the offset (two separate performances of the same move).
        :param side_angle: angle between the views in degrees; None = estimate it.
        :return: (combined frames as flat lists, report dict)
    """
    F = _as_array(front_list)
    S_raw = _as_array(side_list)
    tF = np.asarray(front_times, dtype=np.float64)
    tS = np.asarray(side_times, dtype=np.float64)
    visF = np.asarray(front_vis, dtype=np.float64) if front_vis is not None else np.ones((len(F), N_PTS))
    visS_raw = np.asarray(side_vis, dtype=np.float64) if side_vis is not None else np.ones((len(S_raw), N_PTS))
    fps = _fps(tF)
    report = {}

    # 1. Sync: one offset, then optionally a time warp for drifting timing
    if offset is None:
        offset, sync_corr = find_time_offset(tF, F, tS, S_raw)
        report["sync_trusted"] = sync_corr >= MIN_SYNC_MATCH
        if not report["sync_trusted"]:
            offset = 0.0                      # no clear match: assume the videos start together
    else:
        sync_corr = float("nan")
        report["sync_trusted"] = True
    tq = tF + offset
    report["sync_mode"] = "offset"
    if sync == "WARP" and report["sync_trusted"] and len(tF) >= 20 and len(tS) >= 20:
        tq, report["sync_mode"] = time_warp(tF, F, tS, S_raw, offset), "warp"
    valid = (tq >= tS[0]) & (tq <= tS[-1])
    S = _interp(tq, tS, S_raw)
    visS = _interp(tq, tS, visS_raw) * valid[:, None]
    drift = tq - tF - offset
    report.update(offset_s=offset, sync_corr=sync_corr, coverage=float(valid.mean()),
                  warp_range=(float(drift.min()), float(drift.max())))

    # 2. Scale
    scale = _scale_per_frame(F, S, valid, fps)
    report["scale"] = float(np.median(scale[valid])) if valid.any() else 1.0

    # 3. Left/right labels in the side view
    S, visS, swaps = _fix_left_right(F, S, visS, scale, valid)
    report["lr_switches"] = swaps
    report["height_err"] = height_agreement(F, S, scale, visS, valid)

    # 3b. Heights of joints the front view cannot see (an arm behind the body) taken from the side
    #     view, which measures height too. Only for one performance filmed by two cameras, which
    #     shows in how well the two views' heights agree (FUSE_MAX_HEIGHT_ERR): in separate takes
    #     the heights differ by more than the front view's guess is off.
    report["heights_from_side"] = 0.0
    if FUSE_HEIGHTS and sync != "WARP" and report["height_err"] < FUSE_MAX_HEIGHT_ERR:
        a = (np.clip((FUSE_FRONT_VIS - visF[:, :33]) / FUSE_VIS_RANGE, 0.0, 1.0)
             * np.clip((visS[:, :33] - FUSE_FRONT_VIS) / FUSE_VIS_RANGE, 0.0, 1.0))
        y_side = F[:, [HIP_C], 1] + (S[:, :33, 1] - S[:, [HIP_C], 1]) * scale[:, None]
        F = F.copy()
        F[:, :33, 1] = (1 - a) * F[:, :33, 1] + a * y_side
        for c, (i, j) in CENTROIDS.items():
            F[:, c, 1] = (F[:, i, 1] + F[:, j, 1]) / 2
        report["heights_from_side"] = float(a.mean())

    # 4. View angle, then side x -> depth with the sign that makes the body right-handed
    U = S[:, :, 0] * scale[:, None]
    w = _side_weight(visS)
    if side_angle is None:
        angle, _, _ = estimate_view_angle(F, U, w, valid, fps)
    else:
        angle = float(np.clip(side_angle, ANGLE_RANGE[0], ANGLE_RANGE[1]))
    report["side_angle"] = angle
    p = np.radians(angle)
    depth_side = (U - np.cos(p) * F[:, :, 0]) / np.sin(p)
    sign = -1.0
    if facing_score(to_blender(F, sign * depth_side)) < 0:
        sign = 1.0
    depth_side = sign * depth_side
    report["depth_sign"] = sign

    # 5. Short side occlusions filled from neighbouring frames where the joint was seen,
    #    the front view's own depth (calibrated against the side depth) only where neither helps
    root = depth_side[:, HIP_C].copy()
    if valid.any() and not valid.all():
        idx = np.arange(len(F))
        root = np.interp(idx, idx[valid], root[valid])
    side_rel = depth_side - root[:, None]
    if GAP_FILL_S > 0:
        filled, conf = _fill_gaps(side_rel, w, fps)
    else:
        filled, conf = side_rel, np.zeros_like(w)
    zF_rel = F[:, :, 2] - F[:, [HIP_C], 2]
    good = (w > 0.8) & (visF > 0.5)
    k = 4.0
    if good.sum() > 50:
        x, y = zF_rel[good], side_rel[good]
        den = float(np.dot(x, x))
        if den > 1e-9:
            k_fit = float(np.dot(x, y) / den)
            corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-9 and y.std() > 1e-9 else 0.0
            k = float(np.clip(k_fit, -8, 8)) if abs(corr) > 0.2 else 4.0 * (1 if k_fit >= 0 else -1)
    fallback = k * zF_rel
    rest = conf * filled + (1 - conf) * fallback
    D = root[:, None] + w * side_rel + (1 - w) * rest
    report["fallback_share"] = float(np.mean((1 - w[:, :33]) * (1 - conf[:, :33])))
    report["filled_share"] = float(np.mean((1 - w[:, :33]) * conf[:, :33]))
    w_eff = w + (1 - w) * conf

    # 6. Smoothing
    D = _gauss(D, smoothing, peaks=SMOOTH_PEAKS)
    XY = _gauss(F[:, :, :2], smoothing / 2, peaks=SMOOTH_PEAKS)
    F_out = F.copy()
    F_out[:, :, :2] = XY

    # 7. Constant bone lengths
    persp = front_perspective_scale(F_out, fps)
    report["perspective_range"] = (float(persp.min()), float(persp.max()))
    report["bone_cv_before"] = bone_length_cv(to_blender(F_out, D), persp=persp)
    D = _enforce_bone_lengths(F_out, D, w_eff, bone_strength, fps)
    report["bone_cv_after"] = bone_length_cv(to_blender(F_out, D), persp=persp)
    report["facing"] = facing_score(to_blender(F_out, D))

    F_out[:, :, 2] = D
    return F_out.reshape(len(F_out), -1).tolist(), report


def format_report(r):
    sw = r.get("lr_switches", {})
    match = "manual" if r["sync_corr"] != r["sync_corr"] else f"match {r['sync_corr']:.2f}"
    sync = f"side offset {r['offset_s']:+.2f}s ({match})"
    if r.get("sync_mode") == "warp":
        lo, hi = r["warp_range"]
        sync += f", time warp {lo:+.2f}..{hi:+.2f}s"
    return (f"{sync}, heights agree to {r['height_err']:.1f}% of body height, "
            f"view angle {r['side_angle']:.0f} deg, side scale {r['scale']:.2f}, "
            f"L/R fixes arms {sw.get('arms', 0)} legs {sw.get('legs', 0)}, "
            f"depth filled from neighbours {r.get('filled_share', 0) * 100:.0f}% / from front view {r['fallback_share'] * 100:.0f}%, "
            f"bone length variation {r['bone_cv_before']:.0f}% -> {r['bone_cv_after']:.0f}%")
