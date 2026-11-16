"""
    Depth Lifter: a better depth for a tracked take from its 2D motion (MotionAGFormer).

    A pose tracker guesses each frame's depth from that one picture. A 2D-to-3D lifter reads the
    whole movement instead (243 frames at a time) and gives the 3D pose that best explains it, so the
    depth is far steadier and less often mistaken about which limb is nearer the camera.

    Model: MotionAGFormer-B (TaatiTeam/MotionAGFormer, Apache-2.0, 11.7M parameters) trained on
    Human3.6M, exported to ONNX (47.5 MB). It runs once over a take when its video has been tracked,
    on the CPU (DirectML refuses the graph), about 1.5 s per 243 frames with the mirrored pass.

    How a take is fed to it:
    - the 17 Human3.6M joints are built from the add-on's 40-point layout (H36M_FROM_POSE), with the
      tracker's visibility as the third input channel, as the model was trained;
    - the take is scaled so the actor's usual height is TARGET_HEIGHT (in the model's units, where
      the picture is 2 wide): the training footage shows people about 0.3-0.45 of the picture
      width, test takes 0.16-0.28. Without this the Whirlpool take's depth came out 2.8x too large
      and unrelated to the side cameras (agreement 0.19);
    - a take of up to 243 frames is stretched to 243; a longer one is read in overlapping windows,
      blended where they overlap;
    - the model also sees the mirrored take (left and right swapped) and the two answers are
      averaged, as in the authors' demo.

    The lifted depth replaces nothing outright: the 17 joints get the lifter's depth mixed with the
    tracker's at LIFTER_SHARE (where the tracker had no depth, the lifter's alone); every other point
    (face, hand knuckles, heels and toes) moves in depth with the joint it hangs on, and the pair
    centres (points 33-39) are recomputed.

    Measured 2026-11-14 with view pairs about 90 degrees apart (agreement of the depth with the
    other camera's x / how often it tells which shoulder, hip or wrist is nearer / depth jitter),
    RTMW 2D:
                            katana                  Whirlpool
        tracker depth       0.769  88.7%  0.071     0.733  91.8%  0.076
        lifter only         0.786  93.5%  0.014     0.769  93.8%  0.019
        lifter share 0.5    0.839  90.1%  0.038     0.812  94.4%  0.041

    numpy only (the model call is passed in), so the unit tests can run without Blender.
"""
import numpy as np

WINDOW = 243
TARGET_HEIGHT = 0.66            # actor's usual height in model units (0.33 of the picture width)
LIFTER_SHARE = 0.5              # default share of the lifted depth in the mix with the tracker's
# Human3.6M joint -> add-on point (MediaPipe numbering); pairs are averaged.
H36M_FROM_POSE = ((23, 24), (24,), (26,), (28,), (23,), (25,), (27,), None, (11, 12),
                  (0,), (2, 5), (11,), (13,), (15,), (12,), (14,), (16,))
SPINE = 7                       # midway between pelvis (0) and thorax (8)
# joints the model gives depth for, by add-on point (H36M index 0, 7, 8, 10 are not single points)
LIFTED_POINTS = {0: 9, 11: 11, 13: 12, 15: 13, 12: 14, 14: 15, 16: 16,
                 23: 4, 25: 5, 27: 6, 24: 1, 26: 2, 28: 3}
# the other points move in depth with the lifted point they hang on
FOLLOWERS = {**{i: 0 for i in range(1, 11)},
             17: 15, 19: 15, 21: 15, 18: 16, 20: 16, 22: 16,
             29: 27, 31: 27, 30: 28, 32: 28}
# pair centres stored after the 33 landmarks (utils.PoseTracking.detect_pose, same order)
CENTROID_PAIRS = ((23, 24), (11, 12), (18, 20), (17, 19), (7, 8), (9, 10), (1, 4))
FLIP_A, FLIP_B = [1, 2, 3, 14, 15, 16], [4, 5, 6, 11, 12, 13]


def to_h36m(points, vis):
    """:param points: (T, 40, 2). :param vis: (T, 40). :return: (T, 17, 3) x, y, visibility."""
    points, vis = np.asarray(points, dtype=float), np.asarray(vis, dtype=float)
    out = np.zeros(points.shape[:1] + (17, 3))
    for j, src in enumerate(H36M_FROM_POSE):
        if src is None:
            continue
        out[:, j, :2] = points[:, list(src)].mean(axis=1)
        out[:, j, 2] = vis[:, list(src)].min(axis=1)
    out[:, SPINE] = (out[:, 0] + out[:, 8]) / 2
    out[:, SPINE, 2] = np.minimum(out[:, 0, 2], out[:, 8, 2])
    return out


def normalize(h36):
    """
        Centre the take on the actor's usual hip position and scale it to TARGET_HEIGHT.
        :return: (normalized copy, units per model unit) - multiply model output by the second
                 to get the take's own units back.
    """
    h36 = np.array(h36, dtype=float)
    height = float(np.median(np.ptp(h36[:, :, 1], axis=1)))
    if not np.isfinite(height) or height <= 1e-9:
        height = 1.0
    unit = height / TARGET_HEIGHT
    centre = np.median(h36[:, 0, :2], axis=0)
    h36[:, :, :2] = (h36[:, :, :2] - centre) / unit
    return h36, unit


def windows(count, size=WINDOW):
    """
        :return: list of (frame indices (size,), weights (size,)): a take of up to `size` frames is
                 stretched to `size`; a longer one is covered by windows half overlapping, the
                 last one ending on the last frame. Weights fall off toward a window's ends so
                 overlaps blend smoothly; stretched frames read more than once share their weight.
    """
    if count <= size:
        return [(np.linspace(0, count, size, endpoint=False).astype(int), np.ones(size))]
    starts = list(range(0, count - size, size // 2)) + [count - size]
    taper = np.hanning(size + 2)[1:-1]
    return [(np.arange(s, s + size), taper) for s in sorted(set(starts))]


def flip(x):
    """Mirror a (…, 17, C) pose: x negated, left and right joints exchanged."""
    x = np.array(x, dtype=np.float32)
    x[..., 0] *= -1
    x[..., FLIP_A + FLIP_B, :] = x[..., FLIP_B + FLIP_A, :]
    return x


def lift(points, vis, model):
    """
        :param points: (T, 40, 2) x/y of a take (any unit). :param vis: (T, 40) visibility.
        :param model: callable, (1, 243, 17, 3) float32 -> (1, 243, 17, 3).
        :return: (T, 17) depth of the Human3.6M joints about the pelvis, in the take's x/y units.
    """
    h36, unit = normalize(to_h36m(points, vis))
    count = len(h36)
    total, weight = np.zeros((count, 17)), np.zeros(count)
    for idx, w in windows(count):
        x = h36[idx][None].astype(np.float32)
        out = (np.asarray(model(x)) + flip(model(flip(x)))) / 2.0
        z = out[0, :, :, 2] - out[0, :, :1, 2]
        np.add.at(total, idx, z * w[:, None])
        np.add.at(weight, idx, w)
    return total / np.maximum(weight, 1e-9)[:, None] * unit


def apply(frames, vis, meta, depth, share=LIFTER_SHARE):
    """
        Mix the lifted depth into a take.
        :param frames: list of T flat 120-value frames (40 points x, y, z), changed in place.
        :param vis: (T, 40) visibility. :param meta: list of T metadata dicts, changed in place.
        :param depth: (T, 17) from lift(). :param share: the lifted depth's share, 0..1.
    """
    share = float(np.clip(share, 0.0, 1.0))
    vis = np.asarray(vis, dtype=float)
    for t, frame in enumerate(frames):
        z = np.asarray(frame[2::3], dtype=float)
        old = z.copy()
        m = meta[t] if t < len(meta) else {}
        conf = np.asarray(m.get('depth_confidence', vis[t]), dtype=float)
        new_conf = conf.copy()
        hips = (old[23] + old[24]) / 2.0
        for point, joint in LIFTED_POINTS.items():
            keep = (1.0 - share) * (conf[point] > 0)          # a tracker depth of 0 confidence is no depth
            mix = share + keep
            if mix <= 0:
                continue
            z[point] = (share * (hips + depth[t, joint]) + keep * old[point]) / mix
            new_conf[point] = (share * vis[t, point] + keep * conf[point]) / mix
        for point, parent in FOLLOWERS.items():
            z[point] = old[point] + (z[parent] - old[parent])
            new_conf[point] = max(conf[point], min(new_conf[parent], vis[t, point]))
        for i, (a, b) in enumerate(CENTROID_PAIRS):
            z[33 + i] = (z[a] + z[b]) / 2.0
            new_conf[33 + i] = min(new_conf[a], new_conf[b])
        frame[2::3] = z.tolist()
        if t < len(meta):
            m['depth_confidence'] = new_conf.tolist()
            m['depth_source'] = f"{m.get('depth_source', 'missing')}+lifter"
            m['depth_state'] = ['observed' if c > 0 else 'missing' for c in new_conf]
