"""Run in Blender 5.1 with -b --factory-startup --python-exit-code 1 -P tests/blender_foot_lock.py."""
import math
from pathlib import Path
import sys

import bpy
import numpy as np

# Load the checkout under test even if the installed add-on is a different copy.
import importlib.util

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('foot_lock_test_addon', root/'__init__.py',
                                              submodule_search_locations=[str(root)])
addon = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = addon
spec.loader.exec_module(addon)
addon.register()

scene = bpy.context.scene
arm = bpy.data.armatures.new("Skeleton")
rig = bpy.data.objects.new("Pelvis and legs", arm)
scene.collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig
rig.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')


def bone(name, head, tail, parent=None):
    b = arm.edit_bones.new(name)
    b.head, b.tail, b.parent = head, tail, parent
    return b


hip = bone("Hips", (0, 0, .8), (0, 0, 1))
for side, x in [('Left', .12), ('Right', -.12)]:
    points = [(x, 0, .8), (x, -.08, .45), (x, 0, .11), (x, -.12, .035), (x, -.20, .035)]
    parent = hip
    for i, suffix in enumerate(("UpLeg", "Leg", "Foot", "ToeBase")):
        parent = bone(side+suffix, points[i], points[i+1], parent)
bpy.ops.object.mode_set(mode='OBJECT')
rig.rotation_euler.z = math.radians(52)
rig.location.z = .30
boots = []
for side, x in [('Left', .12), ('Right', -.12)]:
    mesh = bpy.data.meshes.new(side)
    coords = [(x+u, v, z) for z in [0, .10] for v in [-.20, .065] for u in [-.05, .05]]
    mesh.from_pydata(coords, [], [(0, 1, 3, 2), (4, 5, 7, 6)])
    obj = bpy.data.objects.new(side+" boot", mesh)
    scene.collection.objects.link(obj)
    obj.parent = rig
    obj.vertex_groups.new(name=side+'Foot').add(list(range(8)), 1., 'REPLACE')
    mod = obj.modifiers.new("Skin", 'ARMATURE')
    mod.object = rig
    boots.append(obj)

hip = rig.pose.bones["Hips"]
for f in range(1, 14):
    hip.location.x = .001*(f-1)
    hip.keyframe_insert(data_path='location', frame=f)
source = rig.animation_data.action
source.name = "Slow drift"
scene.frame_set(4, subframe=.5)
assert bpy.ops.object.open_mocap_lock_feet(target_rig=rig.name, floor_z=.30) == {'FINISHED'}
locked = rig.animation_data.action
assert locked != source and locked.use_fake_user and source.use_fake_user
assert locked['open_mocap_samples_per_frame'] == 4
assert (scene.frame_current, scene.frame_subframe) == (4, .5)

samples = []
for f in np.linspace(1, 13, 97):
    scene.frame_set(int(f), subframe=f%1)
    row = []
    for obj in boots:
        ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        row.append([tuple(ev.matrix_world @ v.co) for v in ev.data.vertices[:4]])
    samples.append(row)
vertices = np.array(samples)
assert vertices[..., 2].min() >= .30, vertices[..., 2].min()
assert vertices[..., 2].max() < .3015, vertices[..., 2].max()
assert np.ptp(vertices[..., :2], axis=0).max() < .0005

# A failed run must leave the current action, slot, time and action count intact.
slot = rig.animation_data.action_slot
count = len(bpy.data.actions)
try:
    bpy.ops.object.open_mocap_lock_feet(target_rig=rig.name, floor_z=5.)
except RuntimeError as error:
    assert 'No sustained contact' in str(error)
else:
    raise AssertionError('Expected a contact detection failure at the wrong floor height')
assert rig.animation_data.action == locked and rig.animation_data.action_slot == slot
assert len(bpy.data.actions) == count and scene.frame_current == 13
print("Foot lock passed rotated, raised, multi-mesh rig, subframe contact, and rollback checks")
