"""Run in Blender 5.1 with -b --factory-startup --python-exit-code 1 -P tests/blender_hand_lock.py."""
import importlib.util
import math
from pathlib import Path
import sys

import bpy
import numpy as np

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('hand_lock_test_addon', root / '__init__.py',
                                              submodule_search_locations=[str(root)])
addon = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = addon
spec.loader.exec_module(addon)
addon.register()
hl = addon.hand_lock
scene = bpy.context.scene


def make_rig(name):
    """A crouched body (hips 0.5 up) leaning over both palms, which rest on the floor in front."""
    arm = bpy.data.armatures.new(name)
    rig = bpy.data.objects.new(name, arm)
    scene.collection.objects.link(rig)
    rig.rotation_euler.z = math.radians(30)
    for other in scene.objects:
        other.select_set(False)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')

    def bone(n, head, tail, parent=None):
        b = arm.edit_bones.new(n)
        b.head, b.tail, b.parent = head, tail, parent
        return b

    hips = bone("Hips", (0, .25, .5), (0, .2, .6))
    spine = bone("Spine", (0, .2, .6), (0, .05, .65), hips)
    for side, x in (("Left", .2), ("Right", -.2)):
        bone(side + "UpLeg", (x / 2, .25, .5), (x / 2, .1, .25), hips)
        upper = bone(side + "Arm", (x, 0, .5), (x, -.2, .35), spine)          # bent: 5 cm short of full reach
        fore = bone(side + "ForeArm", (x, -.2, .35), (x, -.25, .06), upper)
        bone(side + "Hand", (x, -.25, .06), (x, -.35, .05), fore)
    bpy.ops.object.mode_set(mode='OBJECT')
    hands = []
    for side, x in (("Left", .2), ("Right", -.2)):
        mesh = bpy.data.meshes.new(name + side)
        mesh.from_pydata([(x + u, v, z) for z in (0.0, .05) for v in (-.24, -.37) for u in (-.04, .04)],
                         [], [(0, 1, 3, 2), (4, 5, 7, 6)])
        obj = bpy.data.objects.new(name + side + " hand", mesh)
        scene.collection.objects.link(obj)
        obj.parent = rig
        obj.vertex_groups.new(name=side + "Hand").add(list(range(8)), 1.0, 'REPLACE')
        obj.modifiers.new("Skin", 'ARMATURE').object = rig
        hands.append(obj)
    return rig, hands


def sway(rig, frames=21):
    """The body rocks 4 cm toward the palms and dips 1 cm while the palms should stay planted."""
    from mathutils import Vector
    hips = rig.pose.bones["Hips"]
    to_bone = (rig.matrix_world.to_3x3() @ hips.bone.matrix_local.to_3x3()).inverted()
    for f in range(1, frames + 1):
        s = (f - 1) / (frames - 1)
        hips.location = to_bone @ (rig.matrix_world.to_3x3() @ Vector((0.0, -.04 * s, -.01 * math.sin(math.pi * s))))
        hips.keyframe_insert(data_path='location', frame=f)
    return rig.animation_data.action


def palms(hands, start=1, end=21):
    out = []
    for f in np.linspace(start, end, int(end - start) * 4 + 1):
        scene.frame_set(int(f), subframe=f % 1)
        graph = bpy.context.evaluated_depsgraph_get()
        out.append([[tuple(ev.matrix_world @ v.co) for v in ev.data.vertices[:4]]
                    for ev in (h.evaluated_get(graph) for h in hands)])
    return np.array(out)


def lock(rig, **kw):
    kw.setdefault('use_markers', False)
    return bpy.ops.object.open_mocap_lock_hands(target_rig=rig.name, **kw)


rig, hands = make_rig("Crouch")
source = sway(rig)
before = palms(hands)
assert np.ptp(before[..., :2], axis=0).max() > .02, "the test body must make the palms slide"
scene.frame_set(11)
hand_turn = [(rig.matrix_world @ rig.pose.bones[s + "Hand"].matrix).to_quaternion() for s in ("Left", "Right")]
hips_keys = [tuple(k.co) for fc in addon.foot_lock.curve_bag(rig) for k in fc.keyframe_points]

assert lock(rig, floor_z=0.0) == {'FINISHED'}
locked = rig.animation_data.action
assert locked != source and locked.name == source.name + " Hand Lock" and source.use_fake_user
after = palms(hands)
assert np.ptp(after[..., :2], axis=0).max() < .0006, np.ptp(after[..., :2], axis=0).max()
assert abs(after[..., 2].min()) < .0015 and after[..., 2].max() < .0015, (after[..., 2].min(), after[..., 2].max())
scene.frame_set(11)
for s, q in zip(("Left", "Right"), hand_turn):
    now = (rig.matrix_world @ rig.pose.bones[s + "Hand"].matrix).to_quaternion()
    assert math.degrees(q.rotation_difference(now).angle) < .05, "the hand keeps its baked turn"
hips_path = rig.pose.bones["Hips"].path_from_id("location")
assert all(fc.data_path != hips_path or len(fc.keyframe_points) == 21 for fc in addon.foot_lock.curve_bag(rig))
print("Palm lock passed sway, floor, hand turn and torso checks")

# Constraints: one that only places the arm (a control rig's Copy Location to its shoulder) may stay on;
# one that turns it must be baked first, and the message names it.
arm_r = rig.pose.bones["RightArm"]
place = arm_r.constraints.new('COPY_LOCATION')
place.name, place.target, place.subtarget = "Copy Location", rig, "RightUpLeg"   # unkeyed: holds the arm in place
place.target_space = place.owner_space = 'LOCAL'
rig.animation_data.action = source
assert lock(rig, floor_z=0.0) == {'FINISHED'}
assert np.ptp(palms(hands)[..., :2], axis=0).max() < .0006
rig.animation_data.action = source
aim = arm_r.constraints.new('DAMPED_TRACK')
aim.name, aim.target, aim.subtarget = "Aim", rig, "Hips"
try:
    lock(rig, floor_z=0.0)
except RuntimeError as error:
    assert "RightArm: Aim" in str(error) and "Copy Location" not in str(error), str(error)
else:
    raise AssertionError("a constraint turning the arm was let through")
arm_r.constraints.remove(aim)
arm_r.constraints.remove(place)
print("Palm lock passed location-only and turning constraint checks")

# Floor rule: no contact at a floor half a metre up; Multi Level finds them without a floor.
rig.animation_data.action = source
count = len(bpy.data.actions)
try:
    lock(rig, floor_z=.5)
except RuntimeError as error:
    assert "No palm contact" in str(error)
else:
    raise AssertionError("palms 50 cm under the floor height were taken as contacts")
assert rig.animation_data.action == source and len(bpy.data.actions) == count
assert lock(rig, floor_z=.5, mode='MULTI') == {'FINISHED'}
assert np.ptp(palms(hands)[..., :2], axis=0).max() < .0006

# Markers: the right palm pinned only from frame 5 to 12; the left automatic.
rig.animation_data.action = source
scene.timeline_markers.new("RH+", frame=5)
scene.timeline_markers.new("rh-", frame=12)
assert lock(rig, floor_z=0.0, use_markers=True) == {'FINISHED'}
v = palms(hands)
assert np.ptp(palms(hands, 5, 12)[:, 1, :, :2], axis=0).max() < .0006
assert np.ptp(v[:, 1, :, :2], axis=0).max() > .01 and np.ptp(v[:, 0, :, :2], axis=0).max() < .0006
scene.timeline_markers.clear()

# Plan rules on their own: a still hand at the hip or overhead is no support.
n, fps, length = 40, 30.0, .6
still = np.tile([.3, -.4, .02], (n, 1))
shoulder = np.tile([.2, 0.0, 1.4], (n, 1))
hip = np.tile([.1, 0.0, .9], (n, 1))
preset = hl.PRESETS['NORMAL']
assert hl.plan(still, still[:, 2], shoulder, hip, length, fps, 0.0, preset)[2] == 1
at_hip = np.tile([.25, 0.0, .85], (n, 1))
assert hl.plan(at_hip, at_hip[:, 2], shoulder, hip, length, fps, .85, preset, multi=True)[2] == 0
overhead = np.tile([.2, -.2, 1.9], (n, 1))
assert hl.plan(overhead, overhead[:, 2], shoulder, hip, length, fps, 0.0, preset, multi=True)[2] == 0
moving = still + np.linspace(0, 1.0, n)[:, None] * [1.0, 0, 0]            # 0.75 m/s: not still
assert hl.plan(moving, moving[:, 2], shoulder, hip, length, fps, 0.0, preset)[2] == 0
print("Palm lock passed floor rule, multi level, marker and plan rule checks")
