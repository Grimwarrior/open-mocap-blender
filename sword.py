"""
    Sword tracking: the 2D direction of a sword blade in each tracked video, per frame.

    A pose tracker sees three points per hand (wrist, index and pinky knuckles). With both hands
    closed on one handle those points crowd together, so the hand's roll is unseen most of the time
    (katana take: index -> pinky line too short to trust in 56-71% of the front video's frames).
    The blade is a long straight object that the hands hold, and it is easy to see: for a two-handed
    sword the index -> pinky line of each hand runs along the handle, so the blade's direction gives
    the hands' roll, and where the two hands sit on the handle.

    Per video:
    1. Background  The camera is assumed to stand still: the per-pixel median of frames spread over
                   the take is the empty stage (the actor moves through it).
    2. Evidence    Per frame, around the grip (the tracked hand points), what differs from the stage
                   and is thin: the stage difference minus its morphological opening, at two widths
                   (a blade is a few pixels wide, a motion-blurred one wider; arms and legs are
                   wider still and drop out), with the tracked body drawn as a mask over it. Thin
                   structures on the body (the white piping of a sports shirt) fall under the mask.
    3. Rays        From the grip, the mean evidence along each of N_ANGLES directions.
    4. Track       A Viterbi pass over the take picks one direction per frame: high evidence, and
                   little turning from frame to frame. Frames where the blade is blurred away or
                   hidden take their direction from their neighbours, with low confidence.

    Measured on the katana Front and rightToleft videos against blade directions set by eye on 25
    frames: 23 within 20 degrees (median error 3 degrees); the two misses are strongly blurred.

    Picture coordinates as in the captures: x = (px - width/2) / 100, y = (py - height) / 100, so
    a direction (dx, dy) has x to the right and y downward.
    The image work is numpy only; reading the video needs OpenCV (installed with the trackers).
"""

import numpy as np

N_ANGLES = 180                # directions searched (2 degrees apart)
WORK_SCALE = 0.5              # frames are searched at half size
BG_SAMPLES = 40               # frames for the background median
FG_THRESHOLD = 25.0           # difference from the stage (0..255) where foreground starts ...
FG_RANGE = 30.0               # ... and the range over which it reaches full strength
THIN_WIDTH = 0.035            # opening width for "thin" (share of the body's height in the picture)
MID_WIDTH = 0.07              # ... and for "medium" (a blurred blade)
MID_WEIGHT = 0.7              # weights of the medium-width and of the plain foreground evidence
FG_WEIGHT = 0.2
RAY_RANGE = (0.06, 0.40)      # searched along each ray, from the grip (share of the body's height)
RAY_SAMPLES = 48
TURN_COST = 0.003             # Viterbi: cost of turning by a degree between two frames at 30 fps
CONF_SCALE = 0.6              # evidence along the chosen ray that counts as full confidence (katana takes:
                              # median 0.45-0.5; the two wrong frames of 25 labelled ones 0.25 and 0.34)
GRIP_POINTS = (15, 16, 17, 18, 19, 20)          # wrists, pinky and index knuckles
LIMBS = ((11, 13), (13, 15), (12, 14), (14, 16), (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32),
         (11, 12), (23, 24))
TORSO = (11, 12, 24, 23)
ANGLES = np.arange(N_ANGLES) * (2.0 * np.pi / N_ANGLES)


# -------------------------------------------------------------
# Image helpers (numpy)
# -------------------------------------------------------------
def downscale(img, factor=WORK_SCALE):
    """Area downscale by an integer step (1 / factor)."""
    step = max(1, int(round(1.0 / factor)))
    h, w = img.shape[0] // step * step, img.shape[1] // step * step
    out = sum(img[i:h:step, j:w:step].astype(np.float32) for i in range(step) for j in range(step))
    return out / float(step * step)


def _filter1d(a, k, axis, op):
    """Running min/max over k pixels along an axis, edges padded with the edge value."""
    if k <= 1:
        return a
    lo, hi = k // 2, k - 1 - k // 2
    pad = [(0, 0)] * a.ndim
    pad[axis] = (lo, hi)
    win = np.lib.stride_tricks.sliding_window_view(np.pad(a, pad, mode="edge"), k, axis=axis)
    return op(win, axis=-1)


def opening(a, k):
    """Morphological opening with a k x k square: removes bright structures narrower than k."""
    eroded = _filter1d(_filter1d(a, k, 0, np.min), k, 1, np.min)
    return _filter1d(_filter1d(eroded, k, 0, np.max), k, 1, np.max)


def blur(a, sigma=1.0):
    """Small separable Gaussian blur."""
    r = max(1, int(3 * sigma + 0.5))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    p = np.pad(a, r, mode="edge")
    p = sum(k[i] * p[i:i + a.shape[0], :] for i in range(2 * r + 1))
    return sum(k[i] * p[:, i:i + a.shape[1]] for i in range(2 * r + 1))


def sample(img, x, y):
    """Bilinear samples of a 2D array at float pixel positions; 0 outside."""
    h, w = img.shape
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    out = np.zeros(x.shape, dtype=np.float64)
    for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + dx, y0 + dy
        inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        out += np.where(inside, img[np.clip(yi, 0, h - 1), np.clip(xi, 0, w - 1)], 0.0) * wt
    return out


def body_mask(shape, origin, points, size):
    """
        1 where the tracked body covers the picture: torso, limbs as thick lines, head, hands.
        :param origin: (x, y) of the array's top-left pixel in the picture. :param points: (40, 2) pixels.
    """
    h, w = shape
    X, Y = np.meshgrid(np.arange(w) + origin[0], np.arange(h) + origin[1])
    m = np.zeros(shape, dtype=bool)
    quad = points[list(TORSO)]
    cross = [(quad[(i + 1) % 4, 0] - quad[i, 0]) * (Y - quad[i, 1]) - (quad[(i + 1) % 4, 1] - quad[i, 1]) * (X - quad[i, 0])
             for i in range(4)]
    m |= np.all(np.stack(cross) >= 0, axis=0) | np.all(np.stack(cross) <= 0, axis=0)
    half = max(1.0, 0.045 * size)
    for a, b in LIMBS:
        pa, pb = points[a], points[b]
        d = pb - pa
        t = np.clip(((X - pa[0]) * d[0] + (Y - pa[1]) * d[1]) / max(float(d @ d), 1e-9), 0.0, 1.0)
        m |= np.hypot(X - pa[0] - t * d[0], Y - pa[1] - t * d[1]) <= half
    m |= np.hypot(X - points[0, 0], Y - points[0, 1]) <= 0.11 * size
    for j in GRIP_POINTS:
        m |= np.hypot(X - points[j, 0], Y - points[j, 1]) <= 0.045 * size
    return m


# -------------------------------------------------------------
# Per frame and per take
# -------------------------------------------------------------
def frame_scores(frame, background, points, grip, size):
    """
        Mean blade evidence along each ray from the grip. All in work-scale pixels.
        :param frame, background: (H, W, 3) float arrays. :param points: (40, 2) landmarks.
        :return: (N_ANGLES,) scores.
    """
    reach = RAY_RANGE[1] * size + 4.0
    x0, y0 = int(max(0, np.floor(grip[0] - reach))), int(max(0, np.floor(grip[1] - reach)))
    x1 = int(min(frame.shape[1], np.ceil(grip[0] + reach) + 1))
    y1 = int(min(frame.shape[0], np.ceil(grip[1] + reach) + 1))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return np.zeros(N_ANGLES)
    f, bg = frame[y0:y1, x0:x1], background[y0:y1, x0:x1]
    diff = np.abs(f - bg).max(axis=2)
    bright = f.mean(axis=2) - bg.mean(axis=2)
    fg = blur(np.clip((np.maximum(diff, 1.5 * bright) - FG_THRESHOLD) / FG_RANGE, 0.0, 1.0))
    k1 = max(3, int(THIN_WIDTH * size) | 1)
    k2 = max(5, int(MID_WIDTH * size) | 1)
    thin = np.clip(fg - opening(fg, k1), 0.0, 1.0)
    mid = np.clip(fg - opening(fg, k2), 0.0, 1.0)
    body = body_mask(fg.shape, (x0, y0), points, size)
    body_small = body_mask(fg.shape, (x0, y0), points, 0.6 * size)
    evidence = thin * ~body_small + (MID_WEIGHT * mid + FG_WEIGHT * fg) * ~body
    r = np.linspace(RAY_RANGE[0], RAY_RANGE[1], RAY_SAMPLES) * size
    X = grip[0] - x0 + r[None, :] * np.cos(ANGLES)[:, None]
    Y = grip[1] - y0 + r[None, :] * np.sin(ANGLES)[:, None]
    return sample(evidence, X, Y).mean(axis=1)


def track_angles(scores, fps=30.0):
    """
        Viterbi over directions: per frame the most evident one, with a cost for turning.
        :param scores: (N, N_ANGLES). :return: (angle index per frame, its score).
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return np.zeros(0, dtype=int), np.zeros(0)
    step = np.abs(np.arange(N_ANGLES)[:, None] - np.arange(N_ANGLES)[None, :])
    turn = np.minimum(step, N_ANGLES - step) * (360.0 / N_ANGLES)
    cost = TURN_COST * (fps / 30.0) * turn
    acc = scores[0].copy()
    back = np.zeros((n, N_ANGLES), dtype=int)
    for i in range(1, n):
        cand = acc[None, :] - cost                 # [to, from]
        back[i] = np.argmax(cand, axis=1)
        acc = cand[np.arange(N_ANGLES), back[i]] + scores[i]
    path = np.zeros(n, dtype=int)
    path[-1] = int(np.argmax(acc))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path, scores[np.arange(n), path]


def picture_points(frames, width, height, scale=WORK_SCALE):
    """(N, 40, 2) landmark pixels at work scale from capture frames (flat 40 * [x, y, z])."""
    P = np.asarray(frames, dtype=np.float64).reshape(len(frames), -1, 3)
    return np.stack([P[:, :, 0] * 100.0 + width / 2.0, P[:, :, 1] * 100.0 + height], axis=2) * scale


def grips_and_sizes(points, vis):
    """Grip centre (visibility-weighted hand points) and the body's height in the picture, per frame."""
    vis = np.asarray(vis, dtype=np.float64)
    w = np.clip(vis[:, list(GRIP_POINTS)], 0.05, 1.0)
    grip = np.sum(points[:, list(GRIP_POINTS)] * w[:, :, None], axis=1) / w.sum(axis=1)[:, None]
    body = float(np.median(np.abs(points[:, 0, 1] - points[:, [27, 28], 1].mean(axis=1))))
    torso = np.linalg.norm(points[:, 33] - points[:, 34], axis=1)
    rel = torso / max(float(np.median(torso)), 1e-9)
    if len(rel) >= 3:
        half = min(7, (len(rel) - 1) // 2)
        rel = np.convolve(np.pad(rel, half, mode="edge"), np.ones(2 * half + 1) / (2 * half + 1), mode="valid")
    return grip, np.maximum(body * rel, 8.0)


def track_frames(read_frames, frames, vis, times, video_fps, width, height, progress=None):
    """
        Sword direction per tracked frame of one video.
        :param read_frames: callable that yields (index, BGR image) for every video frame in order
                            (called twice: background, then the search).
        :param frames, vis, times: the view's capture (flat landmarks, visibility, seconds).
        :param progress: optional callable(share 0..1).
        :return: list of [dx, dy, confidence] per tracked frame, (dx, dy) a unit vector in picture
                 coordinates (x right, y down).
    """
    if len(frames) == 0:
        return []
    scores = view_scores(read_frames, frames, vis, times, video_fps, width, height, progress)
    path, best = track_angles(scores, video_fps or 30.0)
    conf = np.clip(best / CONF_SCALE, 0.0, 1.0)
    return [[float(np.cos(ANGLES[p])), float(np.sin(ANGLES[p])), float(c)] for p, c in zip(path, conf)]


def view_scores(read_frames, frames, vis, times, video_fps, width, height, progress=None):
    """(N, N_ANGLES) ray scores of every tracked frame of one video (see track_frames)."""
    n = len(frames)
    wanted = np.rint(np.asarray(times, dtype=np.float64) * video_fps).astype(int)
    points = picture_points(frames, width, height)
    grip, size = grips_and_sizes(points, vis)
    last = int(wanted.max())
    every = max(1, (last + 1) // BG_SAMPLES)
    samples = [downscale(img).astype(np.uint8) for i, img in read_frames() if i % every == 0 and i <= last]
    scores = np.zeros((n, N_ANGLES))
    if not samples:
        return scores
    background = np.median(np.stack(samples), axis=0).astype(np.float32)
    del samples
    rows = {}
    for k, idx in enumerate(wanted):
        rows.setdefault(int(idx), []).append(k)
    done = 0
    for i, img in read_frames():
        if i > last:
            break
        if i not in rows:
            continue
        work = downscale(img)
        for k in rows[i]:
            scores[k] = frame_scores(work, background, points[k], grip[k], size[k])
            done += 1
        if progress:
            progress(done / n)
    return scores


def video_reader(path):
    """(read_frames, fps, width, height) for a video file, read with OpenCV."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    def read_frames():
        c = cv2.VideoCapture(path)
        try:
            i = 0
            while True:
                ok, img = c.read()
                if not ok:
                    return
                yield i, img
                i += 1
        finally:
            c.release()

    return read_frames, fps, width, height


def track_video(path, frames, vis, times, progress=None):
    """track_frames for a video file."""
    read_frames, fps, width, height = video_reader(path)
    return track_frames(read_frames, frames, vis, times, fps, width, height, progress)
