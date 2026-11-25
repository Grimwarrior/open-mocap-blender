"""
    Post-bake palm contacts on a Mixamo rig: a palm that rests on a support (the floor in a push-up,
    a wall, a box being vaulted) is held still by turning the arm alone. The torso, legs and the
    hand's own turn are left as baked. After RGB MoCap's palm contacts (mocap/hand_lock.py,
    github.com/toyxyz/toyxyz_RGBMocap, MIT): a contact needs the palm still, below the shoulder and
    away from the hips (a hand resting at the actor's side is not a support), each contact is pinned
    at one place (the median of its frames), and manual timing replaces the detection.

    One addition: by default (Flat Ground) the palm must also be near the floor. A sword or staff
    held still in guard is otherwise a "contact", and pinning it would bend the arms against the
    body's motion. Multi Level lifts that rule for walls and vaults.
"""
import math

import bpy
import numpy as np
from mathutils import Vector

from . import retarget
from .foot_lock import (MODE_ITEMS, blocking_constraints, bridge, continuous_keys, curve_bag, marker_masks, runs, set_time,
                        world_vertices, write_rotation)

SIDES = ("Left", "Right")
CHAIN = ("Arm", "ForeArm", "Hand")
MARKERS = ("LH", "RH")          # timeline markers "LH+" / "LH-" and "RH+" / "RH-"

# In arm lengths (upper arm + forearm) and seconds; NORMAL as RGB MoCap's (0.18 m/s for a 0.55 m arm)
PRESETS = {
    'GENTLE': dict(speed=.18, gap=0.0, minimum=.30),
    'NORMAL': dict(speed=.33, gap=0.10, minimum=.25),
    'FIRM': dict(speed=.45, gap=0.15, minimum=.20),
}
PRESET_ITEMS = [
    ('GENTLE', "Gentle", "Only palms held very still, for at least 0.3 s"),
    ('NORMAL', "Normal", "Balanced palm contacts"),
    ('FIRM', "Firm", "Looser contacts; bridges dropouts up to 150 ms where the palm stayed in place"),
]
BELOW_SHOULDER = .22    # palm at least this far under its shoulder joint
AWAY_FROM_HIPS = .55    # and this far from the hip joint on its side
NEAR_FLOOR = .27        # Flat Ground: palm surface within this of Floor Z
GAP_LIMIT = .045        # a bridged dropout moved less than this
EASE_S = .1             # the pin fades in and out over this
PATCH_SHARE = .2        # the palm surface: hand vertices this far out along the palm's normal (share of its depth)


class Palm:
    """The skinned palm surface of one hand, and the arm controls that move it."""

    def __init__(self, context, rig, side):
        self.side = side
        self.bones = [retarget.find_bone(rig, side + n) for n in CHAIN]
        if not all(self.bones):
            raise ValueError(f"{side} arm needs upper arm, forearm and hand deform bones")
        self.hip = retarget.find_bone(rig, side + "UpLeg")
        mw = rig.matrix_world
        rest = [mw @ b.bone.matrix_local for b in self.bones]
        self.length = sum((rest[i + 1].translation - rest[i].translation).length for i in range(2))
        if self.length < 1e-7:
            raise ValueError(f"{side} arm has no length")
        normal = self.palm_normal(rig, rest[2])
        depsgraph = context.evaluated_depsgraph_get()
        self.meshes, points = [], []
        for obj in context.view_layer.objects:
            if obj.type != 'MESH' or not any(m.type == 'ARMATURE' and m.object == rig and m.show_viewport
                                              for m in obj.modifiers):
                continue
            group = obj.vertex_groups.get(self.bones[2].name)
            if group is None:
                continue
            mesh = obj.evaluated_get(depsgraph)
            ids = [v.index for v in mesh.data.vertices if any(g.group == group.index and g.weight > .64 for g in v.groups)]
            if ids:
                self.meshes.append((obj, np.asarray(ids), len(mesh.data.vertices)))
                points.append(world_vertices(mesh, ids))
        if not points:
            raise ValueError(f"No skinned {side.lower()} hand vertices were found on {rig.name}")
        out = np.concatenate(points) @ np.array(normal)
        self.patch = np.flatnonzero(out >= out.max() - PATCH_SHARE * np.ptp(out))
        self.controls = [retarget.drive_bone(rig, side + n) for n in CHAIN]
        self.offsets = [b.bone.matrix_local.to_quaternion().inverted() @ c.bone.matrix_local.to_quaternion()
                        for b, c in zip(self.bones, self.controls)]

    def palm_normal(self, rig, hand):
        """The palm's outward direction at rest (world): across the hand, on the side the thumb sits,
        or (no fingers to tell) the side facing down, as palms do in a T-pose."""
        along = hand.col[1].xyz.normalized()
        mw = rig.matrix_world
        heads = {n: retarget.find_bone(rig, self.side + "Hand" + n) for n in ("Index1", "Pinky1", "Thumb2")}
        normal = None
        if heads["Index1"] and heads["Pinky1"]:
            lateral = (mw @ heads["Index1"].bone.head_local) - (mw @ heads["Pinky1"].bone.head_local)
            normal = along.cross(lateral)
        if normal is None or normal.length < 1e-9:
            normal = Vector((0.0, 0.0, -1.0)) - along * along.z
        normal.normalize()
        down = -normal.z
        thumb = 0.0
        if heads["Thumb2"]:
            thumb = ((mw @ heads["Thumb2"].bone.head_local) - hand.translation).dot(normal)
        if abs(down) >= .3 or abs(thumb) < 1e-4 * self.length:
            return normal if down >= 0 else -normal
        return normal if thumb > 0 else -normal

    def measure(self, graph):
        """(palm surface centre (3), its lowest height)."""
        points = []
        for obj, ids, count in self.meshes:
            mesh = obj.evaluated_get(graph)
            if len(mesh.data.vertices) != count:
                raise ValueError(f"{obj.name} changes topology during playback")
            points.append(world_vertices(mesh, ids))
        patch = np.concatenate(points)[self.patch]
        return patch.mean(axis=0), float(patch[:, 2].min())


def plan(points, lowest, shoulders, hips, length, fps, floor, preset, multi=False, manual=None):
    """
        Palm contacts of one hand. :param points: (N, 3) palm centre per sample; lowest (N,) its lowest
        height; shoulders, hips (N, 3) that side's shoulder and hip joints; fps samples per second.
        :return: (pin weight (N,), goal palm centre (N, 3), contact count)
    """
    n = len(points)
    smooth = np.stack([np.convolve(np.pad(points[:, c], 2, mode='edge'), np.ones(5) / 5, 'valid')
                       for c in range(3)], axis=1) if n >= 5 else points
    speed = np.linalg.norm(np.gradient(smooth, axis=0), axis=1) * fps / length if n > 1 else np.zeros(n)
    if manual is not None:
        active = manual.copy()
        minimum = 1
    else:
        active = (speed < preset['speed']) & (points[:, 2] < shoulders[:, 2] - BELOW_SHOULDER * length)
        if hips is not None:
            active &= np.linalg.norm(points - hips, axis=1) > AWAY_FROM_HIPS * length
        if not multi:
            active &= np.abs(lowest - floor) < NEAR_FLOOR * length
        active = bridge(active, points, fps, preset['gap'], GAP_LIMIT * length)
        minimum = max(2, math.ceil(preset['minimum'] * fps))
    weight = np.zeros(n)
    goal = points.copy()
    count = 0
    index = np.arange(n)
    ease = max(1.0, EASE_S * fps)
    for start, end in runs(active):
        if end - start < minimum:
            continue
        count += 1
        anchor = np.median(points[start:end], axis=0)
        if not multi:
            # the palm surface rests on the floor
            anchor[2] += floor + .001 * length - np.median(lowest[start:end])
        t = np.clip(1 + np.minimum(index - start, end - 1 - index) / ease, 0, 1)
        influence = t * t * (3 - 2 * t)
        use = influence > weight
        goal[use] = anchor
        weight[use] = influence[use]
    return weight, goal, count


def pose_arm(context, rig, palm, i, wrist_goal, previous, frame):
    """Two-bone arm reaching the wrist to `wrist_goal`, elbow on its own side, hand keeping its world turn."""
    mats = palm.pose[i]
    shoulder, elbow, wrist = [m.translation for m in mats]
    upper, lower = (elbow - shoulder).length, (wrist - elbow).length
    ray = wrist_goal - shoulder
    axis = ray.normalized()
    distance = max(abs(upper - lower) + 1e-8, min(ray.length, upper + lower - 1e-8))
    mid = (distance ** 2 + upper ** 2 - lower ** 2) / (2 * distance)
    pole = elbow - shoulder - axis * (elbow - shoulder).dot(axis)
    if pole.length < 1e-8:
        pole = axis.orthogonal()
    bent = shoulder + axis * mid + pole.normalized() * math.sqrt(max(0.0, upper ** 2 - mid ** 2))
    target = shoulder + axis * distance
    rotations = ((elbow - shoulder).rotation_difference(bent - shoulder) @ mats[0].to_quaternion(),
                 (wrist - elbow).rotation_difference(target - bent) @ mats[1].to_quaternion(),
                 mats[2].to_quaternion())
    for bone, rotation, offset in zip(palm.controls, rotations, palm.offsets):
        write_rotation(rig, bone, rotation @ offset, previous, frame)
        context.view_layer.update()


def key_contact(context, rig, palm, i, frame, previous, graph):
    """Key sample i: the palm moved `weight` of the way to its goal (measured on the mesh, refined)."""
    goal = palm.points[i] + palm.weight[i] * (palm.goal[i] - palm.points[i])
    wrist = palm.pose[i][2].translation.copy()
    for _ in range(3):
        pose_arm(context, rig, palm, i, wrist, previous, None)
        delta = Vector(tuple(goal - palm.measure(graph)[0]))
        wrist += delta
        if delta.length < .00002 * palm.length:
            break
    pose_arm(context, rig, palm, i, wrist, previous, frame)
    return float(np.linalg.norm(goal - palm.measure(graph)[0]))


def apply_pass(context, rig, palms, frames, rate, floor, preset, multi, markers=()):
    """Plan and key both palms. :return: (contacts per palm, largest miss (world units), sides from markers)."""
    scene = context.scene
    graph = context.evaluated_depsgraph_get()
    shoulders, hips = {p.side: [] for p in palms}, {p.side: [] for p in palms}
    for palm in palms:
        palm.pose, palm.points, palm.lowest = [], [], []
    for k, f in enumerate(frames):
        set_time(scene, f)
        for palm in palms:
            palm.pose.append([(rig.matrix_world @ b.matrix).copy() for b in palm.bones])
            centre, low = palm.measure(graph)
            palm.points.append(centre)
            palm.lowest.append(low)
            shoulders[palm.side].append(tuple(palm.pose[-1][0].translation))
            if palm.hip is not None:
                hips[palm.side].append(tuple(rig.matrix_world @ palm.hip.head))
        context.window_manager.progress_update(k)
    manual = marker_masks(markers, frames, MARKERS)
    counts = []
    for side, palm in zip(SIDES, palms):
        palm.points = np.array(palm.points)
        palm.weight, palm.goal, count = plan(
            palm.points, np.array(palm.lowest), np.array(shoulders[palm.side]),
            np.array(hips[palm.side]) if palm.hip is not None else None, palm.length, rate, floor, preset,
            multi, manual.get(SIDES.index(side)))
        counts.append(count)
    if not any(counts):
        raise ValueError("No palm contact found" + ("" if multi else " near this floor height"))
    switches = [rig.pose.bones.get('Ctrl_Hand_IK_' + s) for s in SIDES] if retarget.is_control_rig(rig) else []
    previous, miss = {}, 0.0
    for k, f in enumerate(frames):
        set_time(scene, f)
        for switch in switches:
            if switch and retarget.IKFK_PROP in switch:
                switch[retarget.IKFK_PROP] = 1.0
                switch.keyframe_insert(data_path='["ik_fk_switch"]', frame=f, group=switch.name)
        if switches:
            context.view_layer.update()
        for palm in palms:
            error = key_contact(context, rig, palm, k, f, previous, graph)
            if palm.weight[k] > .999:
                miss = max(miss, error)
        context.window_manager.progress_update(len(frames) + k)
    paths = {b.path_from_id(prop) for palm in palms for b in palm.controls
             for prop in ("rotation_euler", "rotation_axis_angle", "rotation_quaternion")}
    for fc in curve_bag(rig):
        if fc.data_path in paths:
            for key in fc.keyframe_points:
                key.interpolation = 'LINEAR'
            fc.update()
    return counts, miss, sorted(manual)


class LockHands(bpy.types.Operator):
    bl_idname = "object.open_mocap_lock_hands"
    bl_label = "Lock Palms on Baked Action"
    bl_description = ("Create a new action where palms resting on a support stay put, by turning the arms alone. "
                      "Torso, legs and hand turn stay as baked")
    bl_options = {'REGISTER', 'UNDO'}

    target_rig: bpy.props.StringProperty(options={'HIDDEN'})
    floor_z: bpy.props.FloatProperty(name="Floor Z", default=0., subtype='DISTANCE')
    mode: bpy.props.EnumProperty(name="Ground", items=MODE_ITEMS, default='FLAT')
    preset: bpy.props.EnumProperty(name="Contacts", items=PRESET_ITEMS, default='NORMAL')
    use_markers: bpy.props.BoolProperty(
        name="Contact Markers", default=True,
        description="Timeline markers named LH+ / LH- and RH+ / RH- start and end a palm contact by hand")

    def execute(self, context):
        rig = bpy.data.objects.get(self.target_rig) if self.target_rig else context.active_object
        ad = rig.animation_data if rig and rig.type == 'ARMATURE' else None
        if not ad or not ad.action:
            self.report({'ERROR'}, "Choose an armature with an active baked action")
            return {'CANCELLED'}
        if ad.use_tweak_mode or (ad.use_nla and any(not t.mute and any(not s.mute for s in t.strips)
                                                    for t in ad.nla_tracks)):
            self.report({'ERROR'}, "Bake the full NLA stack into an active action before locking palms")
            return {'CANCELLED'}
        scene = context.scene
        saved_frame = scene.frame_current + scene.frame_subframe
        old = ad.action
        old_slot = getattr(ad, 'action_slot', None)
        keep = old.use_fake_user
        copy = None
        original_basis, original_switches = {}, {}
        progress_started = False
        try:
            start, end = map(float, old.frame_range)
            if end - start < 1:
                raise ValueError("The active action needs at least two frames")
            samples = max(1, int(old.get("open_mocap_samples_per_frame", 1)))
            frames = np.linspace(start, end, math.ceil(end - start) * samples + 1)
            rest = rig.data.pose_position
            try:
                rig.data.pose_position = 'REST'
                context.view_layer.update()
                palms = [Palm(context, rig, side) for side in SIDES]
            finally:
                rig.data.pose_position = rest
                context.view_layer.update()
            if rest != 'POSE':
                raise ValueError("Switch the armature to Pose Position before locking palms")
            blocking = blocking_constraints([b for palm in palms for b in palm.controls])
            if blocking:
                raise ValueError("Bake and disable the constraints that turn the arm controls first ("
                                 + ", ".join(blocking) + ")")
            markers = [(m.name, m.frame) for m in scene.timeline_markers] if self.use_markers else []
            old.use_fake_user = True
            copy = old.copy()
            copy.name = old.name + " Hand Lock"
            ad.action = copy
            if old_slot:
                ad.action_slot = next(s for s in copy.slots if s.identifier == old_slot.identifier)
            affected = {b for palm in palms for b in palm.controls}
            original_basis = {b: b.matrix_basis.copy() for b in affected}
            for side in SIDES:
                switch = rig.pose.bones.get("Ctrl_Hand_IK_" + side)
                if switch and retarget.IKFK_PROP in switch:
                    original_switches[switch] = switch[retarget.IKFK_PROP]
            continuous_keys(rig, affected | {p for b in affected for p in b.parent_recursive})
            context.window_manager.progress_begin(0, 2 * len(frames))
            progress_started = True
            fps = scene.render.fps / scene.render.fps_base
            counts, miss, manual = apply_pass(context, rig, palms, frames, fps * samples, self.floor_z,
                                              PRESETS[self.preset], self.mode == 'MULTI', markers)
            if hasattr(copy, 'mr_o3de_name'):
                copy.mr_o3de_name = ""
            copy.use_fake_user = True
        except Exception as exc:
            if copy is not None:
                ad.action = old
                if old_slot:
                    ad.action_slot = old_slot
                for b, basis in original_basis.items():
                    b.matrix_basis = basis
                for b, value in original_switches.items():
                    b[retarget.IKFK_PROP] = value
                bpy.data.actions.remove(copy)
            old.use_fake_user = keep
            self.report({'ERROR'}, f"Palm lock: {exc}")
            return {'CANCELLED'}
        finally:
            set_time(scene, saved_frame)
            if progress_started:
                context.window_manager.progress_end()
        units = scene.unit_settings.scale_length * 1000
        msg = f"{copy.name}: {counts[0]} left and {counts[1]} right palm contacts."
        if manual:
            msg += " " + " and ".join(SIDES[s].lower() for s in manual) + " from markers."
        msg += f" Largest miss while pinned {miss * units:.1f} mm"
        if miss > .02 * palms[0].length:
            msg += " (out of the arm's reach: the torso is not moved)"
        self.report({'WARNING'} if miss > .02 * palms[0].length else {'INFO'}, msg)
        return {'FINISHED'}
