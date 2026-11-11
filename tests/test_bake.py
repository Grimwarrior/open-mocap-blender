"""Bake failure reporting, source sampling and constraint handoff regressions."""
import types
import unittest
from unittest.mock import MagicMock, patch

from test_capture_batch import operators


class BakeTests(unittest.TestCase):
    def fixture(self, selected=False):
        bones = []
        for name in ('Hips', 'Hand'):
            bones.append(types.SimpleNamespace(
                name=name, path_from_id=lambda prop, n=name: f'pose.bones["{n}"].{prop}',
                constraints=[types.SimpleNamespace(name='OM Aim', mute=False),
                             types.SimpleNamespace(name='Rig Control', mute=False)]))
        previous = types.SimpleNamespace(name='Source', use_fake_user=False)
        ad = types.SimpleNamespace(action=previous, action_slot=None, use_nla=True)
        rig = types.SimpleNamespace(pose=types.SimpleNamespace(bones=bones), animation_data=ad)
        scene = types.SimpleNamespace(frame_current=3, frame_subframe=0.0, frame_set=MagicMock())
        context = types.SimpleNamespace(scene=scene, selected_pose_bones=[bones[1]])
        op = operators.BakeMotion()
        op.frame_start, op.frame_end = 1, 10
        op.only_selected_bones, op.action_name, op.in_place = selected, 'Baked', False
        op.report = MagicMock()
        return op, context, rig, previous

    def test_new_action_samples_existing_animation_then_mutes_only_mocap(self):
        op, context, rig, previous = self.fixture()

        def bake(**kwargs):
            self.assertIs(rig.animation_data.action, previous)
            self.assertFalse(kwargs['use_current_action'])
            self.assertFalse(kwargs['only_selected'])
            curves = [types.SimpleNamespace(data_path=b.path_from_id('location'), keyframe_points=[1])
                      for b in rig.pose.bones]
            rig.animation_data.action = types.SimpleNamespace(name='Action', use_fake_user=False, fcurves=curves)
            return {'FINISHED'}

        with patch.object(operators.bpy, 'ops', types.SimpleNamespace(nla=types.SimpleNamespace(bake=bake)), create=True):
            self.assertEqual(op._bake(context, rig), {'FINISHED'})
        self.assertEqual(rig.animation_data.action.name, 'Baked')
        self.assertTrue(previous.use_fake_user)
        self.assertFalse(rig.animation_data.use_nla)
        for bone in rig.pose.bones:
            self.assertTrue(bone.constraints[0].mute)
            self.assertFalse(bone.constraints[1].mute)

    def test_cancelled_bake_restores_source_and_does_not_mute(self):
        op, context, rig, previous = self.fixture()
        bake = MagicMock(return_value={'CANCELLED'})
        with patch.object(operators.bpy, 'ops', types.SimpleNamespace(nla=types.SimpleNamespace(bake=bake)), create=True):
            self.assertEqual(op._bake(context, rig), {'CANCELLED'})
        self.assertIs(rig.animation_data.action, previous)
        self.assertFalse(previous.use_fake_user)
        self.assertFalse(rig.pose.bones[0].constraints[0].mute)
        self.assertEqual(op.report.call_args.args[0], {'ERROR'})

    def test_selected_mode_without_selection_does_not_bake_entire_rig(self):
        op, context, rig, _ = self.fixture(selected=True)
        context.selected_pose_bones = []
        self.assertEqual(op._bake(context, rig), {'CANCELLED'})

    def test_invalid_frame_range_is_rejected(self):
        op, context, rig, _ = self.fixture()
        op.frame_end = 0
        self.assertEqual(op._bake(context, rig), {'CANCELLED'})

    def test_success_without_keys_is_failure(self):
        op, context, rig, previous = self.fixture()

        def bake(**kwargs):
            rig.animation_data.action = types.SimpleNamespace(name='Empty', fcurves=[])
            return {'FINISHED'}

        with patch.object(operators.bpy, 'ops', types.SimpleNamespace(nla=types.SimpleNamespace(bake=bake)), create=True):
            self.assertEqual(op._bake(context, rig), {'CANCELLED'})
        self.assertIs(rig.animation_data.action, previous)
        self.assertFalse(rig.pose.bones[0].constraints[0].mute)

    def test_partial_bake_mutes_only_baked_bone(self):
        op, context, rig, _ = self.fixture(selected=True)

        def bake(**kwargs):
            self.assertTrue(kwargs['only_selected'])
            curve = types.SimpleNamespace(data_path=rig.pose.bones[1].path_from_id('location'), keyframe_points=[1])
            rig.animation_data.action = types.SimpleNamespace(name='Partial', fcurves=[curve], use_fake_user=False)
            return {'FINISHED'}

        with patch.object(operators.bpy, 'ops', types.SimpleNamespace(nla=types.SimpleNamespace(bake=bake)), create=True):
            self.assertEqual(op._bake(context, rig), {'FINISHED'})
        self.assertFalse(rig.pose.bones[0].constraints[0].mute)
        self.assertTrue(rig.pose.bones[1].constraints[0].mute)
        self.assertTrue(rig.animation_data.use_nla)


if __name__ == '__main__':
    unittest.main()
