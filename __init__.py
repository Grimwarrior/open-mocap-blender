bl_info = {
    "name": "Open Mocap",
    "author": "Larenju Rai",
    "version": (1, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Tools",
    "doc_url": "https://github.com/Larenju-Rai/open-mocap-blender",
    "category": "Animation",
}



import bpy
from . import PackageInstaller, operators, retarget, panels

classes = PackageInstaller.classes + operators.classes + retarget.classes + panels.classes



def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.num_hands = bpy.props.IntProperty(
        name="Number of Hands",
        description="Set number of hands to track (1 or 2)",
        default=2,
        min=1,
        max=2
    )
    bpy.types.Scene.cam_index = bpy.props.IntProperty(
        name="Camera Index",
        description="Choose a camera. Index 0 means primary webcam.",
        default=0,
        min=0,
        max=5
    )
    bpy.types.Scene.bake_rig = bpy.props.PointerProperty(
        name="Rig",
        type=bpy.types.Object,
        description="Choose the armature to bake",
        poll=lambda self, obj: obj.type == 'ARMATURE'
    )
    bpy.types.Scene.bake_frame_start = bpy.props.IntProperty(name="Start Frame", default=1, min = 1)
    bpy.types.Scene.bake_frame_end = bpy.props.IntProperty(name="End Frame", default=250, min = 1)
    bpy.types.Scene.bake_only_selected = bpy.props.BoolProperty(name="Only Selected Bones", default=False)
    bpy.types.Scene.bake_action_name = bpy.props.StringProperty(
        name="Action Name", default="",
        description="Bake into a new action with this name (kept with a fake user, so each named take "
                    "stays in the file). If the name is taken, Blender adds .001. "
                    "Empty: bake into the rig's current action")
    bpy.types.Scene.multi_view_tracking = bpy.props.BoolProperty(name="Multi View Tracking", default=False)
    bpy.types.Scene.realtime_tracking = bpy.props.BoolProperty(name="Realtime Tracking", default=False)
    bpy.types.Scene.combine_method = bpy.props.EnumProperty(
        name="Method",
        items=[
            ('IMPROVED', "Improved", "Sync the videos, match the side camera's scale, fix left/right swaps, "
                                     "fill hidden joints from the front view, smooth and keep bone lengths"),
            ('SIMPLE', "Original", "Side frame N replaces the depth of front frame N"),
        ],
        default='IMPROVED')
    bpy.types.Scene.combine_smoothing = bpy.props.FloatProperty(
        name="Smoothing", default=1.5, min=0.0, max=10.0,
        description="Smoothing over time in frames (depth; half of it for width/height). 0 = off")
    bpy.types.Scene.combine_auto_offset = bpy.props.BoolProperty(
        name="Auto Sync", default=True,
        description="Find the time offset between the front and side videos automatically")
    bpy.types.Scene.combine_offset = bpy.props.FloatProperty(
        name="Side Offset", default=0.0, min=-30.0, max=30.0, step=1, precision=2, unit='TIME_ABSOLUTE',
        description="Side video time minus front video time for the same moment (seconds). "
                    "Combine Data reports the offset it found; adjust it here if the sync is off")
    bpy.types.Scene.combine_sync = bpy.props.EnumProperty(
        name="Timing",
        items=[
            ('OFFSET', "Filmed Together", "Both videos show the same performance (two cameras at once): "
                                          "the side video is only shifted in time"),
            ('WARP', "Separate Takes", "The side video is a second performance of the same move (e.g. one "
                                       "camera, actor turned 90 degrees): its timing may drift by up to a "
                                       "second and is matched frame by frame (time warp)"),
        ],
        default='OFFSET')
    bpy.types.Scene.combine_auto_angle = bpy.props.BoolProperty(
        name="Auto Angle", default=True,
        description="Find the angle between the front and side views from constant bone lengths "
                    "(stays at 90 unless another angle fits clearly better)")
    bpy.types.Scene.combine_side_angle = bpy.props.FloatProperty(
        name="View Angle", default=90.0, min=40.0, max=140.0, precision=0,
        description="Angle between the front and side camera directions in degrees (90 = right angle). "
                    "Combine Data reports the angle it found")
    bpy.types.Scene.combine_bone_strength = bpy.props.FloatProperty(
        name="Keep Bone Lengths", default=0.8, min=0.0, max=1.0, subtype='FACTOR',
        description="How strongly bones that come out too long in depth are shortened to their length "
                    "(bones are never stretched in depth to reach it)")
    bpy.types.Scene.retarget_source = bpy.props.PointerProperty(
        name="Mocap", type=bpy.types.Object,
        description="Mocap rig to retarget from: the empty whose children are the 40 landmark empties "
                    "(RigRoot, RigRoot.001, ...). Empty: the newest take",
        poll=lambda self, obj: obj.type == 'EMPTY' and retarget.is_mocap_root(obj))
    bpy.types.Scene.retarget_arm_mode = bpy.props.EnumProperty(
        name="Arms",
        items=[
            ('POSITION', "Hand Position", "Hands go where the mocap hands are relative to the shoulder "
                                          "(scaled to the rig); elbows bend to fit the rig's arm lengths"),
            ('DIRECTION', "Bone Direction", "Upper arm and forearm copy the mocap directions; with other "
                                            "arm proportions the hands land elsewhere"),
        ],
        default='POSITION')
    bpy.types.Scene.retarget_leg_mode = bpy.props.EnumProperty(name="Legs", items=[
            ('PLANT', "Foot Position, Planted", "Ankles go where the mocap ankles are (knees bend to fit the "
                                                "rig's leg lengths) and are held still while the foot is down"),
            ('POSITION', "Foot Position", "Ankles go where the mocap ankles are, without holding them"),
            ('DIRECTION', "Bone Direction", "Thigh and shin copy the mocap directions; standing feet can slide"),
        ], default='PLANT')
    bpy.types.Scene.retarget_hip_height = bpy.props.EnumProperty(name="Hip Height", items=[
            ('FLOOR', "From Floor", "The rig's feet stand on its floor when the mocap's feet stand on theirs"),
            ('START', "From First Frame", "The hips start at the rig's rest position (use this if the "
                                          "feet are not in the video)"),
        ], default='FLOOR')
    bpy.types.Scene.retarget_shoulders = bpy.props.BoolProperty(
        name="Collar Bones", default=True,
        description="Lift the collar bones with the mocap shoulders (shrugs, raised arms)")
    bpy.types.Scene.retarget_spine_twist = bpy.props.BoolProperty(
        name="Spread Spine Twist", default=True,
        description="Spread the twist between hips and shoulders over the spine bones instead of "
                    "turning only the top one")
    bpy.types.Scene.retarget_grip = bpy.props.EnumProperty(name="Two-Handed Grip", items=[
            ('NONE', "Off", "Each hand follows its own mocap hand"),
            ('RIGHT', "Right Hand Leads", "Both hands hold one thing: the left hand keeps a steady distance to the right"),
            ('LEFT', "Left Hand Leads", "Both hands hold one thing: the right hand keeps a steady distance to the left"),
        ], default='NONE')
    bpy.types.Scene.bake_in_place = bpy.props.BoolProperty(
        name="In Place", default=False,
        description="After baking, remove the hips' travel over the ground (their height is kept), "
                    "for engines that move the character themselves")
    bpy.types.Scene.retarget_move_hips = bpy.props.BoolProperty(
        name="Move Hips", default=True,
        description="Copy the hip translation (scaled to the rig's leg length), not only rotations")
    bpy.types.Scene.retarget_mirror_depth = bpy.props.EnumProperty(
        name="Mirror Depth",
        items=[
            ('AUTO', "Auto", "Mirror front/back if the take is mirrored (nose and toes point backwards)"),
            ('ON', "Always", "Mirror the mocap front/back"),
            ('OFF', "Never", "Use the mocap depth as tracked"),
        ],
        default='AUTO',
        description="Combined front + side takes can come out mirrored front/back. "
                    "A mirrored take makes the rig lean the wrong way")



def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.num_hands
    del bpy.types.Scene.cam_index
    del bpy.types.Scene.bake_rig
    del bpy.types.Scene.bake_frame_start
    del bpy.types.Scene.bake_frame_end
    del bpy.types.Scene.bake_only_selected
    del bpy.types.Scene.bake_action_name
    del bpy.types.Scene.multi_view_tracking
    del bpy.types.Scene.realtime_tracking
    del bpy.types.Scene.combine_method
    del bpy.types.Scene.combine_smoothing
    del bpy.types.Scene.combine_bone_strength
    del bpy.types.Scene.combine_auto_offset
    del bpy.types.Scene.combine_offset
    del bpy.types.Scene.combine_sync
    del bpy.types.Scene.combine_auto_angle
    del bpy.types.Scene.combine_side_angle
    del bpy.types.Scene.retarget_move_hips
    del bpy.types.Scene.retarget_mirror_depth
    del bpy.types.Scene.retarget_source
    del bpy.types.Scene.retarget_arm_mode
    del bpy.types.Scene.retarget_leg_mode
    del bpy.types.Scene.retarget_hip_height
    del bpy.types.Scene.retarget_shoulders
    del bpy.types.Scene.retarget_spine_twist
    del bpy.types.Scene.retarget_grip
    del bpy.types.Scene.bake_in_place



if __name__ == "__main__":
    register()

