"""Capture timing and metadata, independent of Blender and detector libraries."""

import math
import time


class CaptureClock:
    """Seconds at acquisition, before inference; tolerate missing/broken decoder PTS."""

    def __init__(self, fps=30.0, realtime=False):
        self.step = 1.0 / (fps if fps and math.isfinite(fps) and fps > 0 else 30.0)
        self.realtime = realtime
        self.start = None
        self.last = None

    def next(self, pts_ms=None, now=None):
        if self.realtime:
            now = time.monotonic() if now is None else now
            if self.start is None:
                self.start = now
            value = now - self.start
        else:
            value = pts_ms / 1000.0 if pts_ms is not None else float('nan')
        if not math.isfinite(value) or value < 0 or (self.last is not None and value <= self.last):
            value = 0.0 if self.last is None else self.last + self.step
        self.last = value
        return value


def depth_confidence(metadata, visibility):
    """Legacy captures have only visibility; preserve that interpretation on load."""
    if len(metadata) != len(visibility):
        return [list(row) for row in visibility]
    return [list(meta.get('depth_confidence', vis)) for meta, vis in zip(metadata, visibility)]


def validate_pose_capture(data):
    """Validate before replacing the in-memory take. Version 1 metadata stays optional."""
    version = data.get('version', 1)
    if version not in (1, 2):
        raise ValueError(f'Unsupported capture version: {version}')
    raw = data.get('positionList_raw')
    if raw is not None:
        _rows(raw, 120, 'positionList_raw')
    for frames_key, times_key, vis_key, meta_key, missing_key in (
            ('positionList', 'position_times', 'position_vis', 'position_meta', 'position_missing_times'),
            ('zlist', 'zlist_times', 'zlist_vis', 'zlist_meta', 'zlist_missing_times')):
        frames = data.get(frames_key, [])
        _rows(frames, 120, frames_key)
        times = data.get(times_key, [])
        _times(times, times_key)
        # Original Combine could truncate only the result, leaving the raw front timeline intact.
        counts = {len(frames)}
        if frames_key == 'positionList' and raw is not None and len(raw) >= len(frames):
            counts.add(len(raw))
        if times and len(times) not in counts:
            raise ValueError(f'{times_key} does not match the frame count')
        vis = data.get(vis_key, [])
        _rows(vis, 40, vis_key, confidence=True)
        if vis and len(vis) not in counts:
            raise ValueError(f'{vis_key} does not match the frame count')
        _metadata(data.get(meta_key, []), len(frames), meta_key)
        _times(data.get(missing_key, []), missing_key)
    if raw is not None:
        if data.get('position_times') and len(raw) != len(data['position_times']):
            raise ValueError('Raw pose frames do not match the timestamps')
    _metadata(data.get('position_meta_raw') or [], len(raw or []), 'position_meta_raw')
    for key in ('fps', 'front_fps', 'side_fps', 'hand_fps', 'hand_front_fps', 'hand_side_fps'):
        if key in data and (not _finite(data[key]) or data[key] < 0):
            raise ValueError(f'Invalid {key}')


def validate_extra_views(views):
    """Further views of a capture file (absent in older files). :return: the views, checked."""
    if not isinstance(views, list):
        raise ValueError('Invalid extra_views')
    checked = []
    for view in views:
        if not isinstance(view, dict):
            raise ValueError('Invalid extra view')
        frames, times, vis = view.get('frames', []), view.get('times', []), view.get('vis', [])
        _rows(frames, 120, 'extra view frames')
        _times(times, 'extra view times')
        _rows(vis, 40, 'extra view visibility', confidence=True)
        if not (len(frames) == len(times) == len(vis)):
            raise ValueError('An extra view has mismatched frame, time and visibility counts')
        meta = view.get('meta', [])
        _metadata(meta if isinstance(meta, list) else None, len(frames), 'extra view metadata')
        missing = view.get('missing_times', [])
        _times(missing, 'extra view missing times')
        fps = view.get('fps', 0)
        if not _finite(fps) or fps < 0:
            raise ValueError('Invalid extra view fps')
        checked.append({'path': str(view.get('path', '')), 'name': str(view.get('name', '') or 'view'),
                        'frames': frames, 'times': times, 'vis': vis, 'meta': meta,
                        'missing_times': missing, 'fps': fps})
    return checked


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _rows(rows, width, name, confidence=False):
    if not isinstance(rows, list) or any(
            not isinstance(row, list) or len(row) != width or any(
                not _finite(v) or (confidence and not 0 <= v <= 1) for v in row) for row in rows):
        raise ValueError(f'Invalid {name}')


def _times(values, name):
    if (not isinstance(values, list) or any(not _finite(t) or t < 0 for t in values)
            or any(b <= a for a, b in zip(values, values[1:]))):
        raise ValueError(f'{name} must contain increasing, finite timestamps')


def _metadata(values, count, name):
    if not isinstance(values, list) or (values and len(values) != count):
        raise ValueError(f'Invalid {name} frame count')
    for meta in values:
        if not isinstance(meta, dict):
            raise ValueError(f'Invalid {name}')
        _rows([meta.get('depth_confidence')], 40, name, confidence=True)
