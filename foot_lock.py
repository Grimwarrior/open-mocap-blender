"""Post-bake sole contact on a Mixamo rig. Positions and the ground plane are in world space."""
import json
import math

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

from . import retarget


SIDES = ("Left", "Right")
CHAIN = ("UpLeg", "Leg", "Foot", "ToeBase")
SUBSTEPS = 4
CONTACTS = "open_mocap_foot_contacts"


def set_time(scene, frame):
    whole = math.floor(frame)
    scene.frame_set(whole, subframe=float(frame - whole))


def world_vertices(obj, indices):
    coords = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
    obj.data.vertices.foreach_get("co", coords)
    world = np.asarray(obj.matrix_world)
    return coords.reshape(-1, 3)[indices] @ world[:3, :3].T + world[:3, 3]


class Boot:
    """Track the lowest surface and fixed material patches at either end of the sole."""

    def __init__(self, context, rig, side):
        self.bones = [retarget.find_bone(rig, side + n) for n in CHAIN]
        if not all(self.bones):
            raise ValueError(f"{side} leg needs thigh, shin, foot and toe deform bones")
        rest = [rig.matrix_world @ b.bone.matrix_local for b in self.bones]
        self.length = sum((rest[i+1].translation - rest[i].translation).length for i in range(2))
        heading = rest[3].translation - rest[2].translation
        heading.z = 0
        if heading.length < 1e-7 or self.length < 1e-7:
            raise ValueError(f"{side} foot has a degenerate rest direction")
        heading.normalize()
        self.samples = []
        self.meshes = []
        points = []
        depsgraph = context.evaluated_depsgraph_get()
        for obj in context.view_layer.objects:
            if obj.type != 'MESH' or not any(m.type == 'ARMATURE' and m.object == rig and m.show_viewport
                                              for m in obj.modifiers):
                continue
            groups = {g.index for g in obj.vertex_groups if g.name in {b.name for b in self.bones[2:]}}
            if not groups:
                continue
            mesh = obj.evaluated_get(depsgraph)
            ids = [v.index for v in mesh.data.vertices if sum(g.weight for g in v.groups if g.group in groups) > .64]
            if ids:
                self.meshes.append((obj, np.asarray(ids), len(mesh.data.vertices)))
                points.append(world_vertices(mesh, ids))
        if not points:
            raise ValueError(f"No skinned {side.lower()} boot vertices were found on {rig.name}")
        pts = np.concatenate(points)
        low = np.flatnonzero(pts[:, 2] <= pts[:, 2].min() + .006 * self.length)
        along = pts[low] @ np.array(heading)
        if np.ptp(along) < .055 * self.length:
            raise ValueError(f"Cannot distinguish heel and forefoot on the {side.lower()} rest sole")
        self.toe = low[along >= np.percentile(along, 75)]
        self.heel = low[along <= np.percentile(along, 25)]
        self.controls = [retarget.drive_bone(rig, side+n) for n in CHAIN]
        if retarget.is_control_rig(rig):
            self.controls[3] = rig.pose.bones.get("Ctrl_Toe_FK_" + side) or self.controls[3]
        self.offsets = [b.bone.matrix_local.to_quaternion().inverted() @ c.bone.matrix_local.to_quaternion()
                        for b, c in zip(self.bones, self.controls)]

    def measure(self, graph):
        points = []
        for obj, ids, count in self.meshes:
            mesh = obj.evaluated_get(graph)
            if len(mesh.data.vertices) != count:
                raise ValueError(f"{obj.name} changes topology during playback")
            points.append(world_vertices(mesh, ids))
        cloud = np.concatenate(points)
        return np.r_[cloud[:, 2].min(), cloud[self.toe].mean(0), cloud[self.heel].mean(0),
                     cloud[self.toe, 2].min(), cloud[self.heel, 2].min()]


def intervals(mask, minimum):
    indices = np.flatnonzero(mask)
    return [span for span in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
            if len(span) >= minimum]


def anchors(frames, mask, points, fps, minimum):
    """Use a constant anchor within contact, ease towards the nearest anchor outside it."""
    weights = np.zeros(len(frames))
    goals = points.copy()
    spans = intervals(mask, minimum)
    for span in spans:
        start, end = frames[::SUBSTEPS][span[[0, -1]]]
        ease = max(1., .075 * fps)
        t = np.clip(1 + np.minimum(frames-start, end-frames)/ease, 0, 1)
        influence = t*t*(3-2*t)
        use = influence > weights
        goals[use] = np.median(points[(frames >= start) & (frames <= end)], axis=0)
        weights[use] = influence[use]
    return weights, goals, len(spans)


def plan(boot, frames, fps, floor, hint):
    data = np.array(boot.samples)
    keys = data[::SUBSTEPS]
    span = boot.length
    heading = np.unwrap(np.array([math.atan2((m[3].translation-m[2].translation).y,
                                             (m[3].translation-m[2].translation).x) for m in boot.pose]))
    toe_speed = np.linalg.norm(np.gradient(keys[:, 1:3], axis=0), axis=1) * fps
    heel_speed = np.linalg.norm(np.gradient(keys[:, 4:6], axis=0), axis=1) * fps
    plant = (np.abs(keys[:, 7]-floor) < .045*span) & (toe_speed < .48*span)
    flat = (keys[:, 8] < floor + .018*span) & (heel_speed < .46*span)
    flat &= np.abs(np.gradient(heading[::SUBSTEPS])) * fps < 1.7
    if hint is not None:
        ankle, ball = hint.T
        plant &= ball > .46
        flat &= ankle > .50
    duration = max(2, math.ceil(.085*fps))
    weight, toe, count = anchors(frames, plant, data[:, 1:3], fps, duration)
    blend, yaw, flats = anchors(frames, plant & flat, heading, fps, duration)
    # Keep the sampled lift during a swing. A floor guard also covers landing between frames.
    clearance = .001*span
    ground = np.maximum(floor+clearance, np.interp(frames, frames[::SUBSTEPS], keys[:, 0]))
    boot.height = (1-weight)*ground + weight*(floor+clearance)
    boot.anchor = data[:, 1:3]*(1-weight[:, None]) + toe*weight[:, None]
    boot.flat, boot.yaw = blend, yaw
    return count, flats


def curve_bag(rig):
    ad = rig.animation_data
    if hasattr(ad.action, "slots"):
        from bpy_extras.anim_utils import action_get_channelbag_for_slot
        bag = action_get_channelbag_for_slot(ad.action, ad.action_slot)
        return bag.fcurves if bag else ()
    return ad.action.fcurves


def continuous_keys(rig, bones):
    """Repair antipodal baked quaternion keys on the working copy before subframe sampling."""
    curves = curve_bag(rig)
    for b in bones:
        path = b.path_from_id("rotation_quaternion")
        fc = sorted((c for c in curves if c.data_path == path), key=lambda c: c.array_index)
        if len(fc) != 4 or len({len(c.keyframe_points) for c in fc}) != 1:
            continue
        prev = None
        for keys in zip(*(c.keyframe_points for c in fc)):
            if len({k.co.x for k in keys}) != 1:
                break
            q = np.array([k.co.y for k in keys])
            if prev is not None and q @ prev < 0:
                q *= -1
                for k in keys:
                    k.co.y *= -1
                    k.handle_left.y *= -1
                    k.handle_right.y *= -1
            prev = q
        for c in fc:
            c.update()


def write_rotation(rig, bone, world, previous, frame):
    # Convert to the bone's basis without touching its location, scale or rotation mode.
    m = bone.matrix.copy()
    loc, _, scale = m.decompose()
    pose = Matrix.LocRotScale(loc, rig.matrix_world.to_quaternion().inverted() @ world, scale)
    q = rig.convert_space(pose_bone=bone, matrix=pose, from_space='POSE', to_space='LOCAL').to_quaternion()
    mode = bone.rotation_mode
    if mode == 'QUATERNION':
        q.make_compatible(previous.get(bone.name, bone.rotation_quaternion))
        bone.rotation_quaternion = q
        prop = 'rotation_quaternion'
    elif mode == 'AXIS_ANGLE':
        q.make_compatible(previous.get(bone.name, q))
        axis, angle = q.to_axis_angle()
        bone.rotation_axis_angle = (angle, *axis)
        prop = 'rotation_axis_angle'
    else:
        rot = q.to_euler(mode, previous.get(bone.name, bone.rotation_euler))
        bone.rotation_euler = rot
        prop = 'rotation_euler'
    if frame is not None:
        previous[bone.name] = rot.copy() if mode not in {'QUATERNION', 'AXIS_ANGLE'} else q.copy()
        bone.keyframe_insert(data_path=prop, frame=frame, group=bone.name)


def pose_leg(context, rig, boot, index, goal, foot, toe, previous, frame):
    mats = boot.pose[index]
    hip, knee, ankle = [m.translation for m in mats[:3]]
    upper, lower = (knee-hip).length, (ankle-knee).length
    ray = goal-hip
    axis = ray.normalized()
    distance = max(abs(upper-lower)+1e-8, min(ray.length, upper+lower-1e-8))
    mid = (distance**2 + upper**2 - lower**2)/(2*distance)
    pole = knee-hip-axis*(knee-hip).dot(axis)
    if pole.length < 1e-8:
        pole = axis.orthogonal()
    bent_knee = hip + axis*mid + pole.normalized()*math.sqrt(max(0., upper**2-mid**2))
    target = hip+axis*distance
    rotations = ((knee-hip).rotation_difference(bent_knee-hip) @ mats[0].to_quaternion(),
                 (ankle-knee).rotation_difference(target-bent_knee) @ mats[1].to_quaternion(), foot, toe)
    for bone, rotation, offset in zip(boot.controls, rotations, boot.offsets):
        write_rotation(rig, bone, rotation @ offset, previous, frame)
        context.view_layer.update()


def key_contact(context, rig, boot, i, f, previous, graph):
    mats = boot.pose[i]
    rest = [rig.matrix_world @ b.bone.matrix_local for b in boot.bones[2:]]
    v = rest[1].translation-rest[0].translation
    flat = Quaternion((0, 0, 1), boot.yaw[i]-math.atan2(v.y, v.x))
    foot, toe = [m.to_quaternion().slerp(flat @ r.to_quaternion(), float(boot.flat[i]))
                 for m, r in zip(mats[2:], rest)]
    goal = mats[2].translation.copy()
    for _ in range(3):
        pose_leg(context, rig, boot, i, goal, foot, toe, previous, None)
        current = boot.measure(graph)
        delta = Vector((*tuple(boot.anchor[i]-current[1:3]), boot.height[i]-current[0]))
        goal += delta
        if delta.length < .00002*boot.length:
            break
    pose_leg(context, rig, boot, i, goal, foot, toe, previous, f)


def check_floor(scene, boots, frames, graph, floor):
    # FK interpolation follows arcs even when two consecutive keys are level.
    # Project a measured deficit back to its two surrounding IK targets.
    extra = np.zeros((len(boots), len(frames)))
    minimum = math.inf
    for sample in range(2*len(frames)-1):
        i, half = divmod(sample, 2)
        j = i+half
        set_time(scene, .5*(frames[i]+frames[j]))
        for side, boot in enumerate(boots):
            height = boot.measure(graph)[0] - floor
            minimum = min(minimum, height)
            deficit = .00075*boot.length - height
            if deficit > 0:
                extra[side, i:j+1] = np.maximum(extra[side, i:j+1], deficit)
    return minimum, extra


def apply_pass(context, rig, boots, frames, floor, confidence):
    scene = context.scene
    graph = context.evaluated_depsgraph_get()
    fps = scene.render.fps/scene.render.fps_base
    for boot in boots:
        boot.pose = []
    for i, f in enumerate(frames):
        set_time(scene, f)
        for boot in boots:
            boot.pose.append([(rig.matrix_world @ b.matrix).copy() for b in boot.bones])
            boot.samples.append(boot.measure(graph))
        context.window_manager.progress_update(i)
    counts = [plan(b, frames, fps, floor, hint) for b, hint in zip(boots, confidence)]
    if not any(n for n, _ in counts):
        raise ValueError("No sustained contact at this floor height")
    previous = {}
    # FK controls on the copied action can reproduce the visual leg pose from either IK or FK.
    switches = [rig.pose.bones.get('Ctrl_Foot_IK_'+s) for s in SIDES] if retarget.is_control_rig(rig) else []
    for i, f in enumerate(frames):
        set_time(scene, f)
        for switch in switches:
            if switch and retarget.IKFK_PROP in switch:
                switch[retarget.IKFK_PROP] = 1.
                switch.keyframe_insert(data_path='["ik_fk_switch"]', frame=f, group=switch.name)
        if switches:
            context.view_layer.update()
        for boot in boots:
            key_contact(context, rig, boot, i, f, previous, graph)
        context.window_manager.progress_update(len(frames)+i)
    paths = {b.path_from_id(prop) for boot in boots for b in boot.controls
             for prop in ("rotation_euler", "rotation_axis_angle", "rotation_quaternion")}
    for fc in curve_bag(rig):
        if fc.data_path in paths:
            for k in fc.keyframe_points:
                k.interpolation = 'LINEAR'
            fc.update()
    minimum, extra = check_floor(scene, boots, frames, graph, floor)
    if extra.any():
        for side, boot in enumerate(boots):
            boot.height += extra[side]
            for i in np.flatnonzero(extra[side]):
                set_time(scene, frames[i])
                key_contact(context, rig, boot, i, frames[i], {}, graph)
        minimum, _ = check_floor(scene, boots, frames, graph, floor)
    return counts, minimum


def source_hints(action, frame_keys):
    raw = action.get(CONTACTS)
    if raw:
        try:
            source = json.loads(raw)
            times = np.asarray(source["frames"], dtype=float)
            values = np.asarray(source["confidence"], dtype=float)
        except (TypeError, ValueError, KeyError):
            return None, None
        if (times.ndim == 1 and times.size and values.shape == (times.size, 4)
                and np.isfinite(values).all() and np.isfinite(times).all() and np.all(np.diff(times) > 0)):
            cols = [np.interp(frame_keys, times, col, left=1., right=1.) for col in values.T]
            return np.array(cols[:2]).T, np.array(cols[2:]).T
    return None, None


class LockFeet(bpy.types.Operator):
    bl_idname = "object.open_mocap_lock_feet"
    bl_label = "Lock Feet on Baked Action"
    bl_description = "Create a new action with world-space sole contacts. Swings and toe pivots retain their motion"
    bl_options = {'REGISTER', 'UNDO'}

    target_rig: bpy.props.StringProperty(options={'HIDDEN'})
    floor_z: bpy.props.FloatProperty(name="Floor Z", default=0., subtype='DISTANCE',
                                     description="Height of the horizontal ground plane in world space")

    def execute(self, context):
        rig = bpy.data.objects.get(self.target_rig) if self.target_rig else context.active_object
        ad = rig.animation_data if rig and rig.type == 'ARMATURE' else None
        if not ad or not ad.action:
            self.report({'ERROR'}, "Choose an armature with an active baked action")
            return {'CANCELLED'}
        if ad.use_tweak_mode or (ad.use_nla and any(not t.mute and any(not s.mute for s in t.strips)
                                                    for t in ad.nla_tracks)):
            self.report({'ERROR'}, "Bake the full NLA stack into an active action before locking feet")
            return {'CANCELLED'}
        scene = context.scene
        saved_frame = scene.frame_current + scene.frame_subframe
        old = ad.action
        old_slot = getattr(ad, 'action_slot', None)
        keep = old.use_fake_user
        copy = None
        original_basis = {}
        original_switches = {}
        progress_started = False
        try:
            start, end = map(float, old.frame_range)
            if end-start < 1:
                raise ValueError("The active action needs at least two frames")
            frames = np.linspace(start, end, math.ceil(end-start)*SUBSTEPS+1)
            # Sample the rest mesh with its modifiers and its actual armature transform.
            rest = rig.data.pose_position
            try:
                rig.data.pose_position = 'REST'
                context.view_layer.update()
                boots = [Boot(context, rig, side) for side in SIDES]
            finally:
                rig.data.pose_position = rest
                context.view_layer.update()
            if rest != 'POSE':
                raise ValueError("Switch the armature to Pose Position before locking feet")
            for boot in boots:
                if any(not c.mute and c.influence > 0 for b in boot.controls for c in b.constraints):
                    raise ValueError("Bake and disable the constraints driving the leg controls first")
            hints = source_hints(old, frames[::SUBSTEPS])
            old.use_fake_user = True
            copy = old.copy()
            copy.name = old.name + " Foot Lock"
            ad.action = copy
            if old_slot:
                ad.action_slot = next(s for s in copy.slots if s.identifier == old_slot.identifier)
            affected = {b for boot in boots for b in boot.controls}
            original_basis = {b: b.matrix_basis.copy() for b in affected}
            for side in SIDES:
                switch = rig.pose.bones.get("Ctrl_Foot_IK_" + side)
                if switch and retarget.IKFK_PROP in switch:
                    original_switches[switch] = switch[retarget.IKFK_PROP]
            continuous_keys(rig, affected | {p for b in affected for p in b.parent_recursive})
            context.window_manager.progress_begin(0, 2 * len(frames))
            progress_started = True
            counts, minimum = apply_pass(context, rig, boots, frames, self.floor_z, hints)
            copy["open_mocap_samples_per_frame"] = SUBSTEPS
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
            self.report({'ERROR'}, f"Foot lock: {exc}")
            return {'CANCELLED'}
        finally:
            set_time(scene, saved_frame)
            if progress_started:
                context.window_manager.progress_end()
        units = scene.unit_settings.scale_length * 1000
        self.report({'WARNING'} if minimum < 0 else {'INFO'},
                    f"{copy.name}: {sum(n for n, _ in counts)} toe contacts, "
                    f"{sum(n for _, n in counts)} flat plants. Minimum sole clearance {minimum*units:.2f} mm")
        return {'FINISHED'}
