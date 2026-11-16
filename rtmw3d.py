"""
    RTMW3D (experimental pose tracker): turning the model's output into the add-on's landmarks.

    RTMW3D (OpenMMLab, rtmlib's RTMPose3d) gives each of the 133 whole-body points an x/y in the
    picture and a depth in metres from the hip centre (camera z: larger is farther, like MediaPipe).
    The add-on wants depth in the picture's own pixels, so the metres are scaled by the actor's
    pixels per metre, fitted per frame from average adult bone lengths and smoothed over time.

    Two things rtmlib 0.0.16 gets wrong, handled here and in utils.RTMW3DDetector:
    - The depth axis has its own size (288 for the 288x384 model: 576 SimCC bins), but rtmlib
      decodes it with the picture height (384), which shrinks and shifts every depth. decode_depth
      uses the size of the model's own depth output.
    - The model was trained on RGB pictures; rtmlib feeds OpenCV's BGR. RGB jitters about a third
      less in depth.

    Measured on four views of one performance (Whirlpool take, 2026-11-12): the front depth agrees
    with a 90 degree camera's x at correlation 0.12..0.55, against 0.69..0.81 for MediaPipe's world
    depth, and jitters about twice as much. Hence "experimental", and the reduced DEPTH_TRUST.

    "RTMW3D + MediaPipe depth" runs MediaPipe alongside and mixes the two depths (blend_depth),
    RTMW3D's at DEPTH_SHARE. Measured with view pairs about 90 degrees apart (agreement of the depth
    with the other camera's x, and how often it tells which shoulder / hip / wrist is nearer):
        RTMW3D share            0       0.1     0.2     0.3     0.5     1.0
        katana  agreement       0.772   0.781   0.786   0.783   0.742   0.373
                nearer side     90.4%   90.5%   89.1%   87.6%   81.1%   42.9%
        Whirlpool agreement     0.732   0.730   0.720   0.698   0.603   0.233
                nearer side     92.7%   92.9%   93.1%   91.8%   86.6%   57.8%
    0.1 helps the katana take and leaves Whirlpool as it was (depth jitter about 5% lower).
    Weighting by MediaPipe's visibility instead was no better (katana 0.784, Whirlpool 0.721).

    numpy only, so the unit tests can run without Blender.
"""
import numpy as np

Z_RANGE = 2.1744869             # metres covered by half the depth axis (mmpose SimCC3DLabel default)
SIMCC_SPLIT = 2.0               # SimCC bins per input pixel
# Scores are the smaller of the x and y SimCC peaks (about 0.5..0.9). Matched to MediaPipe's
# visibility on the four Whirlpool views: 0.50 -> 0.4, 0.60 -> 0.56, 0.70 -> 0.87, 0.73+ -> ~1.
SCORE_LOW, SCORE_RANGE = 0.40, 0.33
DEPTH_TRUST = 0.5               # depth confidence = visibility x this (MediaPipe world depth: 1.0)
DEPTH_SHARE = 0.1               # RTMW3D's share of the depth next to MediaPipe's (see above)
SCALE_SMOOTH = 0.2              # share of the new frame's pixels-per-metre in the running value
# Bones in the add-on's 33-point (MediaPipe) layout with average adult lengths in metres.
# RTMW3D's metric depth assumes a person of about this size, so these are the right yardstick.
TEMPLATE_BONES = ((11, 13, 0.29), (12, 14, 0.29), (13, 15, 0.26), (14, 16, 0.26),
                  (23, 25, 0.43), (24, 26, 0.43), (25, 27, 0.42), (26, 28, 0.42),
                  (11, 12, 0.36), (23, 24, 0.25), (11, 23, 0.50), (12, 24, 0.50))
HIPS = (23, 24)


def depth_bins(output_shapes, default=288.0):
    """
        :param output_shapes: shapes of the model's three outputs (x, y, z SimCC).
        :return: size of the depth axis in input pixels (288 for RTMW3D-x).
    """
    try:
        size = output_shapes[2][-1]
        return float(size) / SIMCC_SPLIT if isinstance(size, (int, np.integer)) and size > 0 else default
    except (IndexError, TypeError):
        return default


def decode_depth(simcc_z, bins):
    """:param simcc_z: depth peaks in input pixels (rtmlib's keypoints_simcc[..., 2]). :return: metres."""
    return (np.asarray(simcc_z, dtype=float) / (bins / 2.0) - 1.0) * Z_RANGE


def visibility(scores):
    """:return: RTMW3D scores as 0..1 visibility, on MediaPipe's scale."""
    return np.clip((np.asarray(scores, dtype=float) - SCORE_LOW) / SCORE_RANGE, 0.0, 1.0)


def pixels_per_metre(xy, z, vis=None):
    """
        The actor's size in this frame: for each template bone of length L whose ends lie dz apart
        in depth, its length in the picture is s * sqrt(L^2 - dz^2). Least squares in s^2.
        :param xy: (33, 2) pixel points. :param z: (33,) depth in metres. :param vis: (33,) weights.
        :return: pixels per metre, or None if too few bones are seen.
    """
    xy, z = np.asarray(xy, dtype=float), np.asarray(z, dtype=float)
    vis = np.ones(len(xy)) if vis is None else np.asarray(vis, dtype=float)
    a, b, length = (np.array(c) for c in zip(*TEMPLATE_BONES))
    weight = np.minimum(vis[a], vis[b])
    seen = weight > 0.3
    if seen.sum() < 4:
        return None
    image2 = np.sum((xy[a] - xy[b]) ** 2, axis=1)
    full2 = length ** 2
    # a bone can't be longer in depth than it is: keep at least half its length across the picture
    across2 = np.maximum(full2 - (z[a] - z[b]) ** 2, 0.25 * full2)
    w = weight * seen
    den = float(np.sum(w * across2 * across2))
    if den <= 1e-12:
        return None
    s2 = float(np.sum(w * image2 * across2)) / den
    return float(np.sqrt(s2)) if s2 > 0 and np.isfinite(s2) else None


def depth_pixels(xy, z, vis, previous_scale=None):
    """
        :param xy: (33, 2) pixel points. :param z: (33,) depth in metres. :param vis: (33,) 0..1.
        :param previous_scale: the running pixels per metre of the earlier frames, or None.
        :return: ((33,) depth in pixels about the hip centre, new running pixels per metre),
                 or (None, previous_scale) if the actor's size can't be told yet.
    """
    z = np.asarray(z, dtype=float)
    scale = pixels_per_metre(xy, z, vis)
    if scale is not None and previous_scale:
        scale = (1.0 - SCALE_SMOOTH) * previous_scale + SCALE_SMOOTH * scale
    elif scale is None:
        scale = previous_scale
    if not scale:
        return None, previous_scale
    hip = (z[HIPS[0]] + z[HIPS[1]]) / 2.0
    return (z - hip) * scale, scale


def blend_depth(depth, vis, mp_depth=None, mp_confidence=None, share=DEPTH_SHARE):
    """
        Mix RTMW3D's depth with MediaPipe's, both in pixels about the hip centre.
        :param depth: (33,) RTMW3D depth or None. :param vis: (33,) RTMW3D visibility.
        :param mp_depth: (33,) MediaPipe depth, or None when MediaPipe missed this frame or found
                         another actor. :param mp_confidence: (33,) MediaPipe depth confidence.
        :param share: RTMW3D's share, 0 (MediaPipe only) .. 1 (RTMW3D only).
        :return: ((33,) depth, (33,) depth confidence, source name), or (None, zeros, 'missing').
    """
    vis = np.asarray(vis, dtype=float)
    own_conf = vis * DEPTH_TRUST
    if mp_depth is None:
        if depth is None:
            return None, np.zeros_like(vis), 'missing'
        return np.asarray(depth, dtype=float), own_conf, 'rtmw3d'
    mp_depth = np.asarray(mp_depth, dtype=float)
    mp_conf = np.minimum(vis, np.asarray(mp_confidence, dtype=float)) if mp_confidence is not None else vis
    if depth is None:
        return mp_depth, mp_conf, 'mediapipe'
    a = float(np.clip(share, 0.0, 1.0))
    return ((1.0 - a) * mp_depth + a * np.asarray(depth, dtype=float),
            (1.0 - a) * mp_conf + a * own_conf, 'rtmw3d+mediapipe')
