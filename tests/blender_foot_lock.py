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
# Reachable feet: the hips are left alone.
assert not any(fc.data_path.endswith('.location') and any(k.co.x % 1 for k in fc.keyframe_points)
               for fc in addon.foot_lock.curve_bag(rig)), "hips were keyed though both feet reach"
print("Foot lock passed rotated, raised, multi-mesh rig, subframe contact, and rollback checks")

fl = addon.foot_lock


def make_rig(name, x0, knee_y=-.08, lift_left=0.):
    """A pelvis with two legs and boots; the left leg (and boot) can stand `lift_left` higher."""
    arm = bpy.data.armatures.new(name)
    obj = bpy.data.objects.new(name, arm)
    scene.collection.objects.link(obj)
    obj.location.x = x0
    for other in scene.objects:
        other.select_set(False)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    hip = arm.edit_bones.new("Hips")
    hip.head, hip.tail = (0, 0, .8), (0, 0, 1)
    for side, x in [('Left', .12), ('Right', -.12)]:
        up = lift_left if side == 'Left' else 0.
        pts = [(x, 0, .8), (x, knee_y, .455 + up/2), (x, 0, .11 + up), (x, -.12, .035 + up), (x, -.20, .035 + up)]
        parent = hip
        for i, suffix in enumerate(("UpLeg", "Leg", "Foot", "ToeBase")):
            b = arm.edit_bones.new(side+suffix)
            b.head, b.tail, b.parent = pts[i], pts[i+1], parent
            parent = b
    bpy.ops.object.mode_set(mode='OBJECT')
    meshes = []
    for side, x in [('Left', .12), ('Right', -.12)]:
        up = lift_left if side == 'Left' else 0.
        mesh = bpy.data.meshes.new(name+side)
        mesh.from_pydata([(x+u, v, z+up) for z in [0, .10] for v in [-.20, .065] for u in [-.05, .05]],
                         [], [(0, 1, 3, 2), (4, 5, 7, 6)])
        boot = bpy.data.objects.new(name+side+" boot", mesh)
        scene.collection.objects.link(boot)
        boot.parent = obj
        boot.vertex_groups.new(name=side+'Foot').add(list(range(8)), 1., 'REPLACE')
        boot.modifiers.new("Skin", 'ARMATURE').object = obj
        meshes.append(boot)
    return obj, meshes


def soles(meshes, start=1, end=13):
    out = []
    for f in np.linspace(start, end, int((end-start)*8)+1):
        scene.frame_set(int(f), subframe=f % 1)
        graph = bpy.context.evaluated_depsgraph_get()
        out.append([[tuple(ev.matrix_world @ v.co) for v in ev.data.vertices[:4]]
                    for ev in (m.evaluated_get(graph) for m in meshes)])
    return np.array(out)


def animate(obj, lift=0.):
    hips = obj.pose.bones["Hips"]
    for f in range(1, 14):
        hips.location = (.001*(f-1), lift, 0)     # bone Y is world up
        hips.keyframe_insert(data_path='location', frame=f)
    return obj.animation_data.action


def lock(obj, **kw):
    return bpy.ops.object.open_mocap_lock_feet(target_rig=obj.name, use_markers=kw.pop('use_markers', False), **kw)


# Hips 2 cm too high for straight legs: the planted soles only reach the floor if the hips go down.
tall, tall_boots = make_rig("Straight legs", 3., knee_y=-.005)
tall_source = animate(tall, lift=.02)
assert lock(tall, floor_z=0., lower_hips=False) == {'FINISHED'}
assert soles(tall_boots)[..., 2].min() > .015, "a straight leg cannot reach 2 cm further"
tall.animation_data.action = tall_source
assert lock(tall, floor_z=0., lower_hips=True) == {'FINISHED'}
v = soles(tall_boots)
assert v[..., 2].min() >= -1e-4 and v[..., 2].max() < .0015, (v[..., 2].min(), v[..., 2].max())
assert np.ptp(v[..., :2], axis=0).max() < .0005
scene.frame_set(7)
assert .015 < .82 - (tall.matrix_world @ tall.pose.bones["Hips"].head).z < .025

# Multi level: the left foot rests on a 15 cm step. Flat ground leaves it sliding; multi level pins both.
step, step_boots = make_rig("On a step", 6., lift_left=.15)
step_source = animate(step)
assert lock(step, floor_z=0., mode='FLAT') == {'FINISHED'}
v = soles(step_boots)
assert np.ptp(v[:, 0, :, :2], axis=0).max() > .008 and np.ptp(v[:, 1, :, :2], axis=0).max() < .0005
step.animation_data.action = step_source
assert lock(step, mode='MULTI') == {'FINISHED'}
v = soles(step_boots)
assert np.ptp(v[..., :2], axis=0).max() < .0005 and np.ptp(v[..., 2], axis=0).max() < .0015
assert abs(v[:, 0, :, 2].min() - .15) < .002 and abs(v[:, 1, :, 2].min()) < .002

# Markers: the left foot is pinned only from frame 3 to 7 and slides elsewhere; the right stays automatic.
mark, mark_boots = make_rig("Marked", 9.)
animate(mark)
scene.timeline_markers.new("L+", frame=3)
scene.timeline_markers.new("l-", frame=7)
assert lock(mark, floor_z=0., use_markers=True) == {'FINISHED'}
v = soles(mark_boots)
held = soles(mark_boots, 3, 7)
assert np.ptp(held[:, 0, :, :2], axis=0).max() < .0005
assert np.ptp(v[:, 0, :, :2], axis=0).max() > .006 and np.ptp(v[:, 1, :, :2], axis=0).max() < .0005
scene.timeline_markers.clear()

# Presets: Firm pins fully, Gentle pins at 60% so 40% of the 12 mm drift is kept.
mark_source = bpy.data.actions["MarkedAction"]
mark.animation_data.action = mark_source
assert lock(mark, floor_z=0., preset='FIRM') == {'FINISHED'}
assert np.ptp(soles(mark_boots)[..., :2], axis=0).max() < .0005
mark.animation_data.action = mark_source
assert lock(mark, floor_z=0., preset='GENTLE') == {'FINISHED'}
assert .002 < np.ptp(soles(mark_boots)[..., :2], axis=0).max() < .008

# Pure helpers.
fps, span = 24., .7
jump = np.r_[np.zeros(30), np.linspace(0, .4, 8), np.full(6, .4), np.linspace(.4, 0, 8), np.zeros(30)]
flight = fl.airborne(jump, fps, span)
assert flight[38:44].all() and not flight[:30].any() and not flight[-25:].any()
assert not fl.airborne(np.linspace(0, .4, 300), fps, span).any(), "a slow rise with no landing is not a jump"
mask = np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1], bool)
points = np.zeros((13, 3))
assert fl.bridge(mask, points, fps, .08, .01)[3:5].all() and not fl.bridge(mask, points, fps, .08, .01)[7:12].any()
points[5:] = .05
assert not fl.bridge(mask, points, fps, .08, .01)[3:5].any(), "a foot that moved is not one stance"
masks = fl.marker_masks([("R+", 2), ("R-", 4), ("r+", 8), ("Hit", 5)], np.arange(1., 11.))
assert list(masks) == [1] and masks[1].tolist() == [0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
need = fl.hip_drop(np.array([[[0, 0, 1.]]]), np.array([[[0, 0, 0.]]]), [.9], np.ones((1, 1)), .3)
assert abs(need[0] - .1009) < 1e-3
print("Foot lock passed hip drop, multi level, marker and helper checks")
