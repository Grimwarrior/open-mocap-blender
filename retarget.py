"""
    Automatic retargeting of the Open Mocap pose skeleton (40 landmark empties) onto a
    Mixamo armature.

    Every mapped rig bone copies the *direction* of a mocap segment instead of its rotation:
    a hidden helper empty sits on landmark A and aims at landmark B, and a child of it lies
    FAR units further along that line. The rig bone gets a Damped Track to that far point, so
    it points along A->B wherever the rig is standing, whatever its proportions and bone rolls.
    Hips, chest and head also get a Locked Track to the hip / shoulder / ear line for facing,
    and the hips a Copy Location from the mocap hip centre, scaled to the rig's leg length.

    All constraints are named "OM ..." so "Toggle Constraints" and "Bake Motion" work on them.
"""

import bpy
from mathutils import Matrix, Vector
from . import globalVariables as gvar


PREFIX = "OM "
HELPER_COLLECTION = "OpenMocapRetarget"
STASH_PROP = "open_mocap_stashed_action"
FAR = 10000.0

# MediaPipe landmarks (subject's left/right) + the 7 centroids the add-on appends.
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, L_TOE, R_TOE = 23, 24, 25, 26, 27, 28, 31, 32
L_EAR, R_EAR = 7, 8
HIP_C, SHOULDER_C, R_HAND_C, L_HAND_C, EAR_C = 33, 34, 35, 36, 37

# Mixamo bone (without "mixamorig:" prefix) -> Damped Track along landmark segment (a, b)
AIM_MAP = {
    "Hips": (HIP_C, SHOULDER_C),
    "Spine": (HIP_C, SHOULDER_C),
    "Spine1": (HIP_C, SHOULDER_C),
    "Spine2": (HIP_C, SHOULDER_C),
    "Neck": (SHOULDER_C, EAR_C),
    "LeftArm": (L_SHOULDER, L_ELBOW),
    "LeftForeArm": (L_ELBOW, L_WRIST),
    "LeftHand": (L_WRIST, L_HAND_C),
    "RightArm": (R_SHOULDER, R_ELBOW),
    "RightForeArm": (R_ELBOW, R_WRIST),
    "RightHand": (R_WRIST, R_HAND_C),
    "LeftUpLeg": (L_HIP, L_KNEE),
    "LeftLeg": (L_KNEE, L_ANKLE),
    "LeftFoot": (L_ANKLE, L_TOE),
    "RightUpLeg": (R_HIP, R_KNEE),
    "RightLeg": (R_KNEE, R_ANKLE),
    "RightFoot": (R_ANKLE, R_TOE),
}

L_PINKY, R_PINKY, L_INDEX, R_INDEX = 17, 18, 19, 20

# Twist around the bone (Locked Track, bone Y locked): the bone is turned so that the axis lying
# along the rig's rest line ref_from -> ref_to follows the mocap landmark line a -> b.
# Mixamo bone: (a, b, ref_from, ref_to)
FACING_MAP = {
    "Hips": (R_HIP, L_HIP, "RightUpLeg", "LeftUpLeg"),
    "Spine2": (R_SHOULDER, L_SHOULDER, "RightUpLeg", "LeftUpLeg"),
    "Head": (R_EAR, L_EAR, "RightUpLeg", "LeftUpLeg"),
    # hand roll: the palm's index -> pinky knuckle line
    "LeftHand": (L_INDEX, L_PINKY, "LeftHandIndex1", "LeftHandPinky1"),
    "RightHand": (R_INDEX, R_PINKY, "RightHandIndex1", "RightHandPinky1"),
}


def _twist_axis(rig, pb, ref_dir):
    """Locked Track axis of the bone (X or Z, either sign) that lies closest to ref_dir at rest."""
    m = rig.matrix_world.to_3x3() @ pb.bone.matrix_local.to_3x3()
    axes = {'TRACK_X': m.col[0], 'TRACK_NEGATIVE_X': -m.col[0],
            'TRACK_Z': m.col[2], 'TRACK_NEGATIVE_Z': -m.col[2]}
    ref = ref_dir.normalized()
    return max(axes, key=lambda k: axes[k].normalized().dot(ref))


# -------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------
def find_bone(rig, short_name):
    """Find a Mixamo bone by name with any prefix ("mixamorig:", "mixamorig1:", none)."""
    for pb in rig.pose.bones:
        if pb.name == short_name or pb.name.split(":")[-1] == short_name:
            return pb
    return None


def is_mixamo_rig(rig):
    return rig is not None and rig.type == 'ARMATURE' and all(
        find_bone(rig, n) for n in ("Hips", "LeftArm", "RightArm", "LeftUpLeg", "RightUpLeg"))


def _is_keyed(objs):
    return bool(_location_fcurves(objs[0])) and bool(_location_fcurves(objs[HIP_C]))


def find_landmark_empties():
    """
        :return: The 40 pose landmark empties created by Animate, or None. Prefers the set from
                 this session; otherwise the newest complete set ("0".."39", "0.001".."39.001", ...)
                 that actually has keyframes.
    """
    objs = gvar.object_list
    try:
        if len(objs) == 40 and all(o.name in bpy.data.objects for o in objs) and _is_keyed(objs):
            return list(objs)
    except ReferenceError:
        pass
    sets = {}
    for obj in bpy.data.objects:
        base, _, suffix = obj.name.partition(".")
        if obj.type == 'EMPTY' and base.isdigit() and int(base) < 40 and (not suffix or suffix.isdigit()):
            sets.setdefault(int(suffix or 0), {})[int(base)] = obj
    for suffix in sorted(sets, reverse=True):
        group = sets[suffix]
        if len(group) == 40:
            found = [group[i] for i in range(40)]
            if _is_keyed(found):
                return found
    return None


def _location_fcurves(obj):
    ad = obj.animation_data
    if not ad or not ad.action:
        return {}
    act = ad.action
    fcurves = None
    try:
        from bpy_extras.anim_utils import action_get_channelbag_for_slot
        bag = action_get_channelbag_for_slot(act, ad.action_slot)
        fcurves = bag.fcurves if bag else None
    except ImportError:
        pass
    if fcurves is None:
        fcurves = getattr(act, "fcurves", [])
    return {fc.array_index: fc for fc in fcurves if fc.data_path == "location"}


def local_location(obj, frame):
    """Keyed local location of a landmark empty at a frame, without changing the scene frame."""
    fcs = _location_fcurves(obj)
    return Vector([fcs[i].evaluate(frame) if i in fcs else obj.location[i] for i in range(3)])


def _median(values):
    values = sorted(values)
    return values[len(values) // 2] if values else 0.0


def mocap_leg_length(empties, frame_start, frame_end):
    """Median hip-knee-ankle length (both legs) in the empties' local space."""
    step = max(1, (frame_end - frame_start) // 60)
    lengths = []
    for f in range(frame_start, frame_end + 1, step):
        p = {i: local_location(empties[i], f) for i in (L_HIP, L_KNEE, L_ANKLE, R_HIP, R_KNEE, R_ANKLE)}
        left = (p[L_HIP] - p[L_KNEE]).length + (p[L_KNEE] - p[L_ANKLE]).length
        right = (p[R_HIP] - p[R_KNEE]).length + (p[R_KNEE] - p[R_ANKLE]).length
        lengths.append((left + right) / 2)
    return _median(lengths)


def depth_mirror_score(empties, frame_start, frame_end):
    """
        Is the mocap mirrored in depth? With Combine Data the depth comes from the side
        camera's x axis, whose sign depends on which side of the actor that camera stood.
        A real body is right-handed: forward = left x up, and the nose and toes point forward.
        :return: Average of nose/toe direction . forward over the take. Negative = mirrored.
    """
    step = max(1, (frame_end - frame_start) // 120)
    scores = []
    for f in range(frame_start, frame_end + 1, step):
        p = {i: local_location(empties[i], f) for i in (0, L_EAR, R_EAR, L_HIP, R_HIP, L_ANKLE, R_ANKLE,
                                                        L_TOE, R_TOE, HIP_C, SHOULDER_C)}
        up = p[SHOULDER_C] - p[HIP_C]
        left = p[L_HIP] - p[R_HIP]
        fwd = left.cross(up)
        if fwd.length < 1e-6:
            continue
        fwd.normalize()
        nose = p[0] - (p[L_EAR] + p[R_EAR]) / 2
        toes = (p[L_TOE] + p[R_TOE]) / 2 - (p[L_ANKLE] + p[R_ANKLE]) / 2
        for v in (nose, toes):
            if v.length > 1e-6:
                scores.append(v.normalized().dot(fwd))
    return sum(scores) / len(scores) if scores else 0.0


def rig_leg_length(rig):
    mw = rig.matrix_world
    up, knee, foot = (find_bone(rig, n) for n in ("LeftUpLeg", "LeftLeg", "LeftFoot"))
    if not (up and knee and foot):
        return 0.0
    h = [mw @ b.bone.head_local for b in (up, knee, foot)]
    return (h[0] - h[1]).length + (h[1] - h[2]).length


def _helper_collection(context):
    coll = bpy.data.collections.get(HELPER_COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(HELPER_COLLECTION)
        context.scene.collection.children.link(coll)
    return coll


def _new_empty(coll, name, parent=None):
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_type = 'PLAIN_AXES'
    obj.empty_display_size = 0.05
    coll.objects.link(obj)
    if parent:
        obj.parent = parent
    return obj


class _MocapSpace:
    """
        Proxy empties that follow the landmark empties, optionally mirrored in depth (Y),
        so the retarget can fix a mirrored take without touching the user's mocap.
    """

    def __init__(self, coll, empties, mirror):
        self.coll, self.empties, self.proxies = coll, empties, {}
        first = empties[0]
        base = (first.parent.matrix_world @ first.matrix_parent_inverse) if first.parent else Matrix.Identity(4)
        self.space = _new_empty(coll, "OM_mocap_space")
        self.space.matrix_world = base @ Matrix.Diagonal((1.0, -1.0 if mirror else 1.0, 1.0, 1.0))

    def __getitem__(self, i):
        if i not in self.proxies:
            proxy = _new_empty(self.coll, f"OM_landmark_{i}", parent=self.space)
            c = proxy.constraints.new('COPY_LOCATION')
            c.target = self.empties[i]
            c.target_space = 'LOCAL'
            c.owner_space = 'LOCAL'
            self.proxies[i] = proxy
        return self.proxies[i]


def _aim_point(coll, cache, points, a, b):
    """Helper empty FAR along landmark a -> b (shared by bones using the same segment)."""
    key = (a, b)
    if key not in cache:
        aim = _new_empty(coll, f"OM_aim_{a}_{b}")
        c = aim.constraints.new('COPY_LOCATION')
        c.target = points[a]
        c = aim.constraints.new('DAMPED_TRACK')
        c.target = points[b]
        c.track_axis = 'TRACK_Y'
        point = _new_empty(coll, f"OM_aimpoint_{a}_{b}", parent=aim)
        point.location = (0.0, FAR, 0.0)
        cache[key] = point
    return cache[key]


def clear_retarget(rig):
    """Remove retarget constraints from the rig and delete all helper objects."""
    if rig is not None and rig.type == 'ARMATURE':
        for pb in rig.pose.bones:
            for c in [c for c in pb.constraints if c.name.startswith(PREFIX)]:
                pb.constraints.remove(c)
    coll = bpy.data.collections.get(HELPER_COLLECTION)
    if coll:
        for obj in list(coll.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(coll)


def _stash_action(rig):
    """Move the rig's current action aside (kept with a fake user) and reset it to rest."""
    ad = rig.animation_data
    if ad and ad.action:
        ad.action.use_fake_user = True
        rig[STASH_PROP] = ad.action.name
        ad.action = None
    for pb in rig.pose.bones:
        pb.location = (0.0, 0.0, 0.0)
        pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pb.rotation_euler = (0.0, 0.0, 0.0)
        pb.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)
        pb.scale = (1.0, 1.0, 1.0)


def _restore_action(rig):
    name = rig.get(STASH_PROP)
    action = bpy.data.actions.get(name) if name else None
    if action is None:
        return None
    if rig.animation_data is None:
        rig.animation_data_create()
    if rig.animation_data.action is None:
        rig.animation_data.action = action
        del rig[STASH_PROP]
        return action
    return None


def setup_retarget(context, rig, empties, hips_motion=True, mirror_depth=False):
    """
        Builds helpers and constraints. :return: (list of mapped bone names, missing names, scale).
    """
    clear_retarget(rig)
    _stash_action(rig)
    coll = _helper_collection(context)
    scene = context.scene
    cache = {}
    mapped, missing = [], []
    points = _MocapSpace(coll, empties, mirror_depth)
    depth_sign = -1.0 if mirror_depth else 1.0

    mw = rig.matrix_world

    # --- Hips translation: mocap hip centre, scaled to the rig and anchored at its rest hips
    scale = 1.0
    if hips_motion:
        mocap_leg = mocap_leg_length(empties, scene.frame_start, scene.frame_end)
        rig_leg = rig_leg_length(rig)
        scale = rig_leg / mocap_leg if mocap_leg > 1e-6 else 1.0
        space = _new_empty(coll, "OM_hips_space")
        follow = _new_empty(coll, "OM_hips_follow", parent=space)
        c = follow.constraints.new('COPY_LOCATION')
        c.target = empties[HIP_C]
        c.target_space = 'LOCAL'
        c.owner_space = 'LOCAL'
        hips_rest = mw @ find_bone(rig, "Hips").bone.head_local
        space.scale = (scale, depth_sign * scale, scale)
        start = local_location(empties[HIP_C], scene.frame_start)
        space.location = hips_rest - Vector((scale * start.x, depth_sign * scale * start.y, scale * start.z))
        hips = find_bone(rig, "Hips")
        c = hips.constraints.new('COPY_LOCATION')
        c.name = PREFIX + "Hips Location"
        c.target = follow

    # --- Directions
    for short, (a, b) in AIM_MAP.items():
        pb = find_bone(rig, short)
        if pb is None:
            missing.append(short)
            continue
        c = pb.constraints.new('DAMPED_TRACK')
        c.name = PREFIX + "Aim"
        c.target = _aim_point(coll, cache, points, a, b)
        c.track_axis = 'TRACK_Y'
        mapped.append(pb.name)

    # --- Facing / roll (twist around the bone)
    for short, (a, b, ref_from, ref_to) in FACING_MAP.items():
        pb = find_bone(rig, short)
        rf, rt = find_bone(rig, ref_from), find_bone(rig, ref_to)
        if pb is None or rf is None or rt is None:
            missing.append(f"{short} twist")
            continue
        ref_dir = (mw @ rt.bone.head_local) - (mw @ rf.bone.head_local)
        c = pb.constraints.new('LOCKED_TRACK')
        c.name = PREFIX + ("Roll" if short.endswith("Hand") else "Facing")
        c.target = _aim_point(coll, cache, points, a, b)
        c.lock_axis = 'LOCK_Y'
        c.track_axis = _twist_axis(rig, pb, ref_dir)
        if pb.name not in mapped:
            mapped.append(pb.name)

    for obj in coll.objects:
        obj.hide_set(True)
    return mapped, missing, scale



# -------------------------------------------------------------
# OPERATORS
# -------------------------------------------------------------
class RetargetToMixamo(bpy.types.Operator):
    bl_idname = "object.open_mocap_retarget_mixamo"
    bl_label = "Retarget to Mixamo"
    bl_description = ("Drive the selected Mixamo rig with the tracked pose skeleton using constraints. "
                      "The rig's current action is kept aside with a fake user. Use Bake Motion afterwards")
    bl_options = {'REGISTER', 'UNDO'}

    target_rig: bpy.props.StringProperty(name="Target Rig")
    hips_motion: bpy.props.BoolProperty(
        name="Move Hips", default=True,
        description="Copy the hip translation (scaled to the rig's leg length), not only rotations")
    mirror_depth: bpy.props.EnumProperty(
        name="Mirror Depth",
        items=[
            ('AUTO', "Auto", "Mirror front/back if the take is mirrored (nose and toes point backwards)"),
            ('ON', "Always", "Mirror the mocap front/back"),
            ('OFF', "Never", "Use the mocap depth as tracked"),
        ],
        default='AUTO',
        description="Combined front + side takes can come out mirrored front/back, depending on which "
                    "side of the actor the side camera stood. A mirrored take makes the rig lean the wrong way")

    def execute(self, context):
        rig = bpy.data.objects.get(self.target_rig)
        if not is_mixamo_rig(rig):
            self.report({'ERROR'}, "Select a Mixamo armature (bones like mixamorig:Hips, mixamorig:LeftArm)")
            return {'CANCELLED'}
        empties = find_landmark_empties()
        if empties is None:
            self.report({'ERROR'}, "No pose skeleton found. Track a video and press Animate first")
            return {'CANCELLED'}

        scene = context.scene
        score = depth_mirror_score(empties, scene.frame_start, scene.frame_end)
        mirror = score < 0 if self.mirror_depth == 'AUTO' else self.mirror_depth == 'ON'
        mapped, missing, scale = setup_retarget(context, rig, empties, self.hips_motion, mirror)
        scene.bake_rig = rig
        scene.bake_frame_start = scene.frame_start
        scene.bake_frame_end = scene.frame_end
        msg = (f"Retargeted {len(mapped)} bones to {rig.name} (hips scale {scale:.3f}, "
               f"depth {'mirrored' if mirror else 'as tracked'}, facing check {score:+.2f}).")
        if missing:
            msg += f" Not found: {', '.join(missing)}"
        self.report({'WARNING'} if missing else {'INFO'}, msg)
        return {'FINISHED'}


class ClearRetarget(bpy.types.Operator):
    bl_idname = "object.open_mocap_clear_retarget"
    bl_label = "Remove Retarget"
    bl_description = ("Remove the Open Mocap retarget constraints and helpers. "
                      "Restores the rig's previous action if it has none now")
    bl_options = {'REGISTER', 'UNDO'}

    target_rig: bpy.props.StringProperty(name="Target Rig")

    def execute(self, context):
        rig = bpy.data.objects.get(self.target_rig)
        clear_retarget(rig)
        restored = _restore_action(rig) if rig else None
        msg = "Retarget removed."
        if restored:
            msg += f" Restored action {restored.name}."
        self.report({'INFO'}, msg)
        return {'FINISHED'}


classes = [RetargetToMixamo, ClearRetarget]
