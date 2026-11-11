"""Numerical and mocked Blender regressions. Run: python -B -m unittest discover -s tests -v

No Blender scene is opened or modified. Blender integration still needs an editor check.
"""
import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import capture_data
import combine
from synthetic_combine import make_take


def load_addon_modules():
    package = types.ModuleType('mocap_batch_test')
    package.__path__ = [str(ROOT)]
    bpy = types.ModuleType('bpy')
    bpy.types = types.SimpleNamespace(Operator=type('Operator', (), {}))
    bpy.props = MagicMock()
    extras = types.ModuleType('bpy_extras.io_utils')
    extras.ExportHelper = type('ExportHelper', (), {})
    extras.ImportHelper = type('ImportHelper', (), {})
    modules = {'mocap_batch_test': package, 'bpy': bpy, 'mathutils': MagicMock(),
               'mocap_batch_test.PackageInstaller': MagicMock(), 'bpy_extras': types.ModuleType('bpy_extras'),
               'bpy_extras.io_utils': extras}
    with patch.dict(sys.modules, modules):
        retarget = importlib.import_module('mocap_batch_test.retarget')
        utils = importlib.import_module('mocap_batch_test.utils')
        operators = importlib.import_module('mocap_batch_test.operators')
    return retarget, utils, operators


retarget, utils, operators = load_addon_modules()


class TimingTests(unittest.TestCase):
    def test_live_uses_elapsed_acquisition_time(self):
        clock = capture_data.CaptureClock(30, realtime=True)
        np.testing.assert_allclose([clock.next(now=t) for t in (100, 100.1, 100.5)], [0, .1, .5])

    def test_video_preserves_variable_pts(self):
        clock = capture_data.CaptureClock(30)
        np.testing.assert_allclose([clock.next(t) for t in (0, 33, 90, 140)], [0, .033, .09, .14])

    def test_broken_decoder_pts_falls_back_monotonically(self):
        clock = capture_data.CaptureClock(30)
        np.testing.assert_allclose([clock.next(t) for t in (0, 0, float('nan'), -1)], np.arange(4) / 30)

    def test_detector_receives_increasing_milliseconds(self):
        clock = utils._TimestampClock(30)
        self.assertEqual([clock.next(t) for t in (0, .0001, .1, .5)], [0, 1, 100, 500])

    def test_close_samples_keep_subframe_timing(self):
        np.testing.assert_allclose(operators.frames_from_times([0, .01, .25], 3, 30), [1, 1.3, 8.5])


class CaptureTests(unittest.TestCase):
    def payload(self):
        return dict(version=2, positionList=[[0.] * 120] * 2, position_times=[0., .1],
                    position_vis=[[.8] * 40] * 2,
                    position_meta=[{'depth_confidence': [.2] * 40}] * 2,
                    position_missing_times=[.033, .066])

    def test_metadata_survives_json(self):
        data = json.loads(json.dumps(self.payload()))
        capture_data.validate_pose_capture(data)
        self.assertEqual(capture_data.depth_confidence(data['position_meta'], data['position_vis'])[0][0], .2)

    def test_legacy_visibility_is_supported(self):
        data = self.payload()
        data['version'] = 1
        del data['position_meta']
        capture_data.validate_pose_capture(data)
        self.assertEqual(capture_data.depth_confidence([], data['position_vis']), data['position_vis'])

    def test_legacy_truncated_combine_and_untimed_raw_capture(self):
        data = self.payload()
        data.pop('position_meta')
        data['version'] = 1
        data['positionList_raw'] = list(data['positionList'])
        data['positionList'] = data['positionList'][:1]
        capture_data.validate_pose_capture(data)
        data['position_times'] = []
        capture_data.validate_pose_capture(data)

    def test_invalid_load_preserves_current_capture(self):
        op = operators.LoadCapture()
        op.report = MagicMock()
        data = self.payload()
        data['format'] = operators.CAPTURE_FORMAT
        data['position_times'] = [0, 0]
        previous = operators.gvar.positionList
        with tempfile.TemporaryDirectory() as tmp:
            op.filepath = str(Path(tmp) / 'capture.json')
            Path(op.filepath).write_text(json.dumps(data), encoding='utf-8')
            self.assertEqual(op.execute(None), {'CANCELLED'})
        self.assertIs(operators.gvar.positionList, previous)

    def test_save_load_round_trip_and_legacy_defaults(self):
        g = operators.gvar
        keys = operators.CAPTURE_LISTS + operators.CAPTURE_VALUES + ('positionList_raw', 'position_meta_raw')
        snapshot = {k: getattr(g, k) for k in keys}
        try:
            for k in operators.CAPTURE_LISTS:
                setattr(g, k, [])
            for k in operators.CAPTURE_VALUES:
                setattr(g, k, '' if k.endswith('_path') else 0)
            data = self.payload()
            for k, v in data.items():
                if k != 'version':
                    setattr(g, k, v)
            g.positionList_raw = list(g.positionList)
            g.position_meta_raw = list(g.position_meta)
            with tempfile.TemporaryDirectory() as tmp:
                save, load = operators.SaveCapture(), operators.LoadCapture()
                save.filepath = load.filepath = str(Path(tmp) / 'capture.json')
                save.report = load.report = MagicMock()
                self.assertEqual(save.execute(None), {'FINISHED'})
                g.position_meta = []
                self.assertEqual(load.execute(None), {'FINISHED'})
                self.assertEqual(g.position_meta, data['position_meta'])
                self.assertEqual(g.position_meta_raw, data['position_meta'])
                legacy = dict(format=operators.CAPTURE_FORMAT, version=1, positionList=[[0.] * 120])
                Path(load.filepath).write_text(json.dumps(legacy), encoding='utf-8')
                self.assertEqual(load.execute(None), {'FINISHED'})
                self.assertEqual(g.position_meta, [])
                self.assertIsNone(g.position_meta_raw)
                self.assertEqual(g.hand_front_fps, 0)
        finally:
            for k, v in snapshot.items():
                setattr(g, k, v)

    def test_invalid_metadata_is_rejected(self):
        for key, bad in [('position_times', [0., 0.]), ('position_meta', [{'depth_confidence': [1.]}] * 2),
                         ('position_missing_times', [float('nan')]), ('version', 99)]:
            data = self.payload()
            data[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                capture_data.validate_pose_capture(data)

    def test_source_confidence_is_zero_inside_long_gap(self):
        data = dict(frames=[1, 31], fps=30, visibility=[[1.] * 40] * 2,
                    metadata=[{'depth_confidence': [.4] * 40}] * 2)
        root = {'open_mocap_metadata': json.dumps(data)}
        empties = {0: types.SimpleNamespace(parent=root)}
        np.testing.assert_allclose(retarget.landmark_confidence(empties, [1, 16, 31], [15]), [.4, 0, .4])


class TrackingTests(unittest.TestCase):
    def detector(self, mp_shift=0, missing=False):
        detector = utils.RTMWDetector.__new__(utils.RTMWDetector)
        points = np.zeros((1, 133, 2))
        for i, xy in [(5, (40, 20)), (6, (60, 20)), (11, (40, 80)), (12, (60, 80))]:
            points[0, i] = xy
        mp_points = [[float(x + mp_shift), float(y), 10.] for x, y in points[0][utils._WB_TO_MP]]
        detector.depth = types.SimpleNamespace(
            find_pose=MagicMock(return_value=(None, [] if missing else mp_points)),
            visibility=[] if missing else [.9] * 33,
            depth_confidence=[] if missing else [.8] * 33, depth_source='mediapipe_world')
        detector.model = MagicMock(return_value=(points, np.ones((1, 133)) * 6))
        detector.previous = None
        return detector

    def test_matched_models_keep_separate_depth_confidence(self):
        detector = self.detector()
        _, points = detector.find_pose(None, draw=False, timestamp=.2)
        self.assertEqual(detector.depth_confidence, [.8] * 33)
        self.assertEqual(detector.visibility, [1.] * 33)
        self.assertEqual(points[0][2], 10.)
        detector.depth.find_pose.assert_called_once_with(None, draw=False, timestamp=.2)

    def test_other_actor_depth_is_rejected(self):
        detector = self.detector(mp_shift=1000)
        _, points = detector.find_pose(None, draw=False)
        self.assertEqual(detector.depth_confidence, [0.] * 33)
        self.assertEqual(detector.depth_source, 'missing')
        self.assertTrue(all(p[2] == 0 for p in points))

    def test_missing_depth_does_not_inherit_rtmw_confidence(self):
        detector = self.detector(missing=True)
        detector.find_pose(None, draw=False)
        self.assertEqual(detector.depth_confidence, [0.] * 33)
        self.assertEqual(detector.visibility, [1.] * 33)

    def test_failed_pose_frame_keeps_its_timestamp(self):
        tracker = utils.PoseTracking.__new__(utils.PoseTracking)
        tracker.mode, tracker.frame_index, tracker.video_fps = 'front_pose', -1, 30
        tracker.cap = types.SimpleNamespace(get=MagicMock(side_effect=[0, 33, 100]))
        img = np.zeros((100, 100, 3))
        points = [[50., 50., 0.]] * 33
        tracker.detector = types.SimpleNamespace(
            find_pose=MagicMock(side_effect=[(img, points), (img, []), (img, points)]),
            visibility=[.9] * 33, depth_confidence=[.2] * 33, xy_source='rtmw', depth_source='mediapipe_world', hands={})
        g = utils.gvar
        names = ('positionList', 'position_times', 'position_vis', 'position_meta', 'position_missing_times')
        previous = {k: getattr(g, k) for k in names}
        try:
            for k in names:
                setattr(g, k, [])
            with patch.object(utils, 'cv2', types.SimpleNamespace(CAP_PROP_POS_MSEC=0)), patch.object(utils, 'mp_vision', object()):
                for _ in range(3):
                    tracker.detect_pose(img)
            np.testing.assert_allclose(g.position_times, [0, .1])
            np.testing.assert_allclose(g.position_missing_times, [.033])
            self.assertEqual(g.position_meta[0]['depth_confidence'], [.2] * 40)
        finally:
            for k, v in previous.items():
                setattr(g, k, v)


class CombineTests(unittest.TestCase):
    def test_long_gap_has_no_interpolated_confidence(self):
        np.testing.assert_allclose(combine.interpolation_support(np.array([0., .5, 1.]), np.array([0., 1.])), [1, 0, 1])

    def test_short_gap_confidence_decays(self):
        times = np.array([0, 1, 2, 5, 6, 7]) / 30
        support = combine.interpolation_support(np.array([2, 3.5, 5]) / 30, times)
        self.assertEqual(support[0], 1)
        self.assertGreater(support[1], 0)
        self.assertLess(support[1], .5)
        self.assertEqual(support[2], 1)

    def test_untrusted_sync_preserves_front(self):
        take = make_take()
        front = [take['front'][0]] * len(take['front'])
        side = [take['side'][0]] * len(take['side'])
        result, report = combine.combine_improved(front, side, take['tF'], take['tS'])
        self.assertFalse(report['combined'])
        self.assertEqual(result, front)
        self.assertIn('manual offset', combine.format_report(report))

    def test_manual_offset_overrides_automatic_rejection(self):
        take = make_take()
        front = [take['front'][0]] * len(take['front'])
        side = [take['side'][0]] * len(take['side'])
        _, report = combine.combine_improved(front, side, take['tF'], take['tS'], offset=0)
        self.assertTrue(report['combined'])

    def test_manual_offset_without_overlap_is_rejected(self):
        take = make_take()
        result, report = combine.combine_improved(take['front'], take['side'], take['tF'], take['tS'], offset=100)
        self.assertFalse(report['combined'])
        self.assertEqual(result, take['front'])

    def test_side_dropout_stays_untrusted_after_gap_fill(self):
        take = make_take(swap_bursts=0)
        keep = (take['tS'] < 2) | (take['tS'] > 3)
        _, report = combine.combine_improved(
            take['front'], np.asarray(take['side'])[keep], take['tF'], take['tS'][keep],
            take['visF'], take['visS'][keep], offset=take['offset'], front_depth_confidence=take['visF'])
        index = int(np.argmin(np.abs(take['tF'] - 2.9)))
        self.assertEqual(report['metadata'][index]['depth_confidence'], [0.] * 40)
        self.assertEqual(report['metadata'][index]['depth_state'], ['missing'] * 40)

    def test_filter_does_not_join_disconnected_segments(self):
        times = np.r_[np.arange(10) / 30, 2 + np.arange(10) / 30]
        values = np.r_[np.zeros(10), np.ones(10) * 100]
        np.testing.assert_allclose(combine._gauss_timed(values, times, 2), values, atol=1e-10)


class LimbTests(unittest.TestCase):
    def test_arm_solve_uses_each_sides_lengths(self):
        P = {33: (0, 0, 0), 34: (0, 0, 1), 23: (.1, 0, 0), 24: (-.1, 0, 0),
             11: (.2, 0, 1), 12: (-.2, 0, 1), 13: (.4, -.1, .8), 14: (-.4, -.1, .8),
             15: (.3, -.2, .5), 16: (-.3, -.2, .5)}
        a, b = {'Left': .4, 'Right': .65}, {'Left': .35, 'Right': .5}
        with patch.object(retarget, 'local_location', side_effect=lambda obj, f: P[obj]):
            out = retarget.arm_reach_directions({i: i for i in P}, np.arange(6), 30, a, b, 1, .4)
        for side, wrist in [('Left', 15), ('Right', 16)]:
            shoulder, upper, fore = out[side]
            end = shoulder + a[side] * upper + b[side] * fore
            np.testing.assert_allclose(end, np.tile(P[wrist], (6, 1)), atol=1e-8)

    def test_leg_solve_uses_each_sides_lengths(self):
        n = 6
        points = {33: (0, 0, 1), 34: (0, 0, 2), 23: (.1, 0, 1), 24: (-.1, 0, 1),
                  25: (.1, -.2, .6), 26: (-.1, -.2, .6), 27: (.1, 0, .2), 28: (-.1, 0, .2)}
        P = {i: np.tile(p, (n, 1)) for i, p in points.items()}
        a, b = {'Left': .4, 'Right': .7}, {'Left': .4, 'Right': .6}
        offsets = {'Left': (.1, 0, 0), 'Right': (-.1, 0, 0)}
        with patch.object(retarget, '_true', return_value=P), patch.object(retarget, 'landmark_confidence', return_value=np.ones(n)):
            out, _, _ = retarget.leg_directions({}, np.arange(n), 30, False, 1, a, b, offsets, 0, False)
        for side in a:
            joint, thigh, knee, shin = out[side]
            np.testing.assert_allclose(np.linalg.norm(knee - joint, axis=1), a[side])
            end = knee + b[side] * shin
            # Both observed legs have the same relative bend, but the target limbs differ in length.
            natural_reach = .8 / (2 * np.sqrt(.2 ** 2 + .4 ** 2)) * (a[side] + b[side])
            np.testing.assert_allclose(np.linalg.norm(end - joint, axis=1), natural_reach, atol=1e-8)

    def test_unreliable_ankle_cannot_be_planted(self):
        points = np.zeros((30, 3))
        self.assertFalse(retarget.foot_contacts(points, 1, 30, np.zeros(30)).any())
        self.assertTrue(retarget.foot_contacts(points, 1, 30, np.ones(30)).all())


class FakeID(dict):
    __hash__ = object.__hash__
    __eq__ = object.__eq__
    __ne__ = object.__ne__

    def __init__(self, name, kind='EMPTY', target=None):
        super().__init__()
        self.name, self.type, self.parent = name, kind, None
        self.animation_data = None
        self.constraints = [] if target is None else [types.SimpleNamespace(name='OM Aim', target=target)]
        self.pose = types.SimpleNamespace(bones=[types.SimpleNamespace(constraints=self.constraints)])


class OwnershipTests(unittest.TestCase):
    def run_cleanup(self, legacy=False, shared=False):
        target_a, target_b = FakeID('a'), FakeID('b')
        rig_a = FakeID('Rig A', 'ARMATURE', target_a)
        rig_b = FakeID('Renamed Rig B', 'ARMATURE', target_a if shared else target_b)
        ca, cb = FakeID('OpenMocapRetarget' if legacy else 'collection a'), FakeID('collection b')
        ca.objects, cb.objects = [target_a], [target_b]
        ca.children, cb.children = [], []
        if legacy:
            ca.objects.append(target_b)
            collections = [ca]
        else:
            ca[retarget.HELPER_OWNER], cb[retarget.HELPER_OWNER] = rig_a, rig_b
            collections = [ca, cb]
        objects = [rig_a, rig_b, target_a, target_b]

        class Objects(list):
            def remove(self, obj, do_unlink=False):
                super().remove(obj)
                for coll in collections:
                    if obj in coll.objects:
                        coll.objects.remove(obj)

        data = types.SimpleNamespace(collections=collections, objects=Objects(objects), actions=MagicMock())
        with patch.object(retarget.bpy, 'data', data, create=True):
            retarget.clear_retarget(rig_a)
        self.assertEqual(rig_a.constraints, [])
        self.assertEqual(len(rig_b.constraints), 1)
        self.assertIn(target_a if shared else target_b, data.objects)
        if not shared:
            self.assertNotIn(target_a, data.objects)

    def test_owned_helpers_are_isolated(self):
        self.run_cleanup()

    def test_legacy_shared_collection_is_not_deleted_wholesale(self):
        self.run_cleanup(legacy=True)

    def test_duplicated_rig_keeps_shared_targets(self):
        self.run_cleanup(shared=True)


if __name__ == '__main__':
    unittest.main()
