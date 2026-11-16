"""Retarget and Bake Motion on rigs with animation layers (NLA tracks), against stand-in NLA data."""
import types
import unittest
from unittest.mock import MagicMock, patch

from test_capture_batch import operators

layers = operators.layers


class Rig(dict):
    """An object: ID properties (dict) plus animation_data."""

    def __init__(self, ad):
        super().__init__()
        self.animation_data = ad


class Track:
    def __init__(self, name, mute=False):
        self.name, self.mute, self.select, self.strips = name, mute, False, []


class Tracks(list):
    def new(self, prev=None):
        t = Track("NlaTrack")
        self.insert(self.index(prev) + 1 if prev is not None else len(self), t)
        t.strips = Strips()
        return t


class Strips(list):
    def new(self, name, start, action):
        s = types.SimpleNamespace(name=name, action=action, action_slot=None, blend_type='COMBINE',
                                  use_auto_blend=True, blend_in=1.0, blend_out=1.0, extrapolation='HOLD_FORWARD')
        self.append(s)
        return s


def animdata(*names, tweak=False, action=None):
    tracks = Tracks(Track(n) for n in names)
    return types.SimpleNamespace(nla_tracks=tracks, use_nla=True, use_tweak_mode=tweak, action=action,
                                 action_slot="slot")


class LayersOff(unittest.TestCase):
    def test_retarget_leaves_tweak_mode_and_switches_layers_off_until_removed(self):
        rig = Rig(animdata("Base Layer", "Layer 1", tweak=True))
        with patch.object(layers, "layers_core", return_value=None):
            self.assertEqual(layers.layers_off(None, rig), 2)
        self.assertFalse(rig.animation_data.use_tweak_mode)
        self.assertFalse(rig.animation_data.use_nla)
        self.assertTrue(layers.layers_back(rig))
        self.assertTrue(rig.animation_data.use_nla)
        self.assertNotIn(layers.NLA_PROP, rig)

    def test_a_second_retarget_keeps_what_the_layers_did_before_the_first(self):
        rig = Rig(animdata("Base Layer"))
        with patch.object(layers, "layers_core", return_value=None):
            layers.layers_off(None, rig)
            layers.layers_off(None, rig)            # NLA is off now; True must be kept
        layers.layers_back(rig)
        self.assertTrue(rig.animation_data.use_nla)

    def test_rig_without_layers_is_left_alone(self):
        rig = Rig(animdata())
        rig.animation_data.use_nla = True
        self.assertEqual(layers.layers_off(None, rig), 0)
        self.assertTrue(rig.animation_data.use_nla)
        self.assertFalse(layers.layers_back(rig))


class AddToLayers(unittest.TestCase):
    def add(self, ad, mode, full=True):
        rig = Rig(ad)
        rig[layers.NLA_PROP] = True
        baked = types.SimpleNamespace(name="Baked", frame_range=(1.0, 75.0))
        ad.action = baked
        with patch.object(layers, "layers_core", return_value=None):
            track, muted = layers.add_to_layers(None, rig, baked, mode, "Mocap", full)
        return rig, track, muted, baked

    def test_new_base_goes_above_the_base_and_mutes_it(self):
        ad = animdata("Base Layer", "Layer 1")
        rig, track, muted, baked = self.add(ad, 'BASE')
        self.assertEqual([t.name for t in ad.nla_tracks], ["Base Layer", "Mocap", "Layer 1"])
        self.assertIs(muted, ad.nla_tracks[0])
        self.assertTrue(ad.nla_tracks[0].mute)
        self.assertFalse(ad.nla_tracks[2].mute)
        strip = track.strips[0]
        self.assertIs(strip.action, baked)
        self.assertEqual((strip.blend_type, strip.extrapolation, strip.action_slot), ('REPLACE', 'HOLD', "slot"))
        self.assertIsNone(ad.action)
        self.assertTrue(ad.use_nla)
        self.assertNotIn(layers.NLA_PROP, rig)

    def test_next_base_goes_above_the_lowest_playing_layer(self):
        ad = animdata("Base Layer", "Mocap", "Layer 1")
        ad.nla_tracks[0].mute = True
        _, track, muted, _ = self.add(ad, 'BASE')
        self.assertEqual([t.name for t in ad.nla_tracks], ["Base Layer", "Mocap", "Mocap 2", "Layer 1"])
        self.assertEqual(muted.name, "Mocap")

    def test_partial_bake_keeps_the_base_playing(self):
        ad = animdata("Base Layer", "Layer 1")
        _, _, muted, _ = self.add(ad, 'BASE', full=False)
        self.assertIsNone(muted)
        self.assertFalse(ad.nla_tracks[0].mute)

    def test_top_layer_covers_the_others_and_mutes_nothing(self):
        ad = animdata("Base Layer", "Layer 1")
        _, track, muted, _ = self.add(ad, 'TOP')
        self.assertEqual([t.name for t in ad.nla_tracks], ["Base Layer", "Layer 1", "Mocap"])
        self.assertIsNone(muted)
        self.assertFalse(any(t.mute for t in ad.nla_tracks))


class BakeIntoLayers(unittest.TestCase):
    def test_bake_samples_with_layers_off_into_a_new_action_then_makes_it_a_layer(self):
        bone = types.SimpleNamespace(name="Hips", path_from_id=lambda prop: f'pose.bones["Hips"].{prop}',
                                     constraints=[types.SimpleNamespace(name="OM Aim", mute=False)])
        ad = animdata("Base Layer", "Layer 1", tweak=True)
        rig = Rig(ad)
        rig.pose = types.SimpleNamespace(bones=[bone])
        rig.name = "Armature"
        rig[layers.NLA_PROP] = True
        scene = types.SimpleNamespace(frame_current=3, frame_subframe=0.0, frame_set=MagicMock())
        context = types.SimpleNamespace(scene=scene, selected_pose_bones=[])
        op = operators.BakeMotion()
        op.frame_start, op.frame_end = 1, 10
        op.only_selected_bones, op.action_name, op.in_place, op.layers = False, '', False, 'BASE'
        op.report = MagicMock()

        def bake(**kwargs):
            self.assertFalse(ad.use_tweak_mode)
            self.assertFalse(ad.use_nla)                            # the layers do not leak into the bake
            self.assertFalse(kwargs['use_current_action'])          # never into a layer's action
            ad.action = types.SimpleNamespace(name='Action', use_fake_user=False, frame_range=(1.0, 10.0),
                                              fcurves=[types.SimpleNamespace(data_path='pose.bones["Hips"].location',
                                                                             keyframe_points=[1])])
            return {'FINISHED'}

        with patch.object(layers, "layers_core", return_value=None), \
                patch.object(operators.bpy, 'ops', types.SimpleNamespace(nla=types.SimpleNamespace(bake=bake)), create=True):
            self.assertEqual(op._bake(context, rig), {'FINISHED'})
        self.assertEqual([t.name for t in ad.nla_tracks], ["Base Layer", "Mocap", "Layer 1"])
        self.assertEqual(ad.nla_tracks[1].strips[0].action.name, "Armature_Mocap")
        self.assertTrue(ad.use_nla)
        self.assertIsNone(ad.action)
        self.assertTrue(bone.constraints[0].mute)
        self.assertIn("as layer 'Mocap'", op.report.call_args.args[1])


if __name__ == "__main__":
    unittest.main()
