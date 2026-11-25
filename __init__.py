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
from . import PackageInstaller, operators, retarget, panels, layers, foot_lock, hand_lock

classes = (PackageInstaller.classes + operators.classes + retarget.classes + [foot_lock.LockFeet, hand_lock.LockHands]
           + panels.classes)



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
    bpy.types.Scene.hand_live_smoothing = bpy.props.FloatProperty(
        name="Live Smoothing", default=50.0, min=0.0, max=100.0, subtype='PERCENTAGE',
        description="Steadies the live hands (One Euro filter): still hands strongly, fast moves little. "
                    "Higher is steadier but lags more in fast moves; 0 is off. Can be changed while tracking")
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
    bpy.types.Scene.foot_lock_floor_z = bpy.props.FloatProperty(
        name="Floor Z", subtype='DISTANCE', default=0.,
        description="World height of the ground for Lock Feet on Baked Action")
    bpy.types.Scene.foot_lock_mode = bpy.props.EnumProperty(
        name="Ground", items=foot_lock.MODE_ITEMS, default='FLAT')
    bpy.types.Scene.foot_lock_preset = bpy.props.EnumProperty(
        name="Contacts", items=foot_lock.PRESET_ITEMS, default='NORMAL')
    bpy.types.Scene.hand_lock_preset = bpy.props.EnumProperty(
        name="Palm Contacts", items=hand_lock.PRESET_ITEMS, default='NORMAL')
    bpy.types.Scene.foot_lock_lower_hips = bpy.props.BoolProperty(
        name="Lower Hips", default=True,
        description="Key the hips down where a planted foot cannot reach its pin with the leg straight")
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
    bpy.types.Scene.combine_smoothing_method = bpy.props.EnumProperty(
        name="Smoothing Method", default='PEAKS',
        items=[('PEAKS', "Peak-Keeping Gaussian",
                "Every joint's depth on its own, with a window that follows peaks and fast moves (best measured)"),
               ('ONE_EURO', "One Euro on Bones",
                "Depth from the skeleton smoothed as a body: hip path, each bone's turn and length, by a "
                "speed-adaptive One Euro filter run both ways. Steadier holds in theory; measured a little "
                "jitterier than the Gaussian on the synthetic test")])
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
    bpy.types.Scene.combine_perspective = bpy.props.BoolProperty(
        name="Correct Perspective", default=True,
        description="A camera shows what is nearer bigger and, below its height, lower (the nearer foot looks "
                    "lower than the farther one). With the depth known this is undone, so both feet stand on "
                    "the same floor and a limb reaching toward the camera keeps its width")
    bpy.types.Scene.combine_fov = bpy.props.FloatProperty(
        name="Field of View", default=60.0, min=10.0, max=120.0, precision=0,   # plain degrees: subtype ANGLE is radians
        description="Horizontal field of view of the front camera in degrees (phone main camera about 65-70, "
                    "a 35 mm lens on a full-frame camera about 54, zoomed in 30). Too wide overcorrects")
    bpy.types.Scene.gemx_fov = bpy.props.FloatProperty(
        name="GEM-X Field of View", default=0.0, min=0.0, max=120.0, precision=0,
        description="Horizontal field of view of the camera in degrees, for GEM-X's distances. 0: GEM-X's own "
                    "guess (about 47 degrees on a 16:9 video)")
    bpy.types.Scene.combine_auto_fov = bpy.props.BoolProperty(
        name="Find FOV", default=True,
        description="Find the field of view that keeps the actor one size while walking toward or away from the "
                    "front camera, and set FOV to it. Needs the actor to change distance; otherwise FOV is used "
                    "as set. It also makes up for depth the other views get too long")
    bpy.types.Scene.combine_auto_horizon = bpy.props.BoolProperty(
        name="Find Horizon", default=True,
        description="Find the camera's horizon (its height and tilt) from the frames where both feet stand: the "
                    "one that puts them on one floor. Off: a level camera at the actor's hip height (the horizon "
                    "at the middle of the picture)")
    bpy.types.Scene.combine_grip = bpy.props.BoolProperty(
        name="Hands on One Handle", default=False,
        description="Both hands hold one handle (a two-handed sword): a hand the videos lose is put back on the "
                    "handle next to the other, along the tracked blade, and its elbow solved again")
    bpy.types.Scene.combine_whole_take = bpy.props.BoolProperty(
        name="Whole Take", default=True,
        description="Scale every frame to one distance from the camera, so an actor stepping toward or away "
                    "from it keeps one size and the floor stays at one height. Off: each frame is corrected "
                    "relative to its own hips, which levels the feet against each other but leaves the body "
                    "as big and as high as the camera saw it")
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
    bpy.types.Scene.retarget_fingers = bpy.props.BoolProperty(
        name="Fingers", default=True,
        description="Curl the fingers as the hands in the body video do (needs a take tracked with the "
                    "RTMW pose tracker and animated with this version)")
    bpy.types.Scene.retarget_spine_twist = bpy.props.BoolProperty(
        name="Spread Spine Twist", default=True,
        description="Spread the twist between hips and shoulders over the spine bones instead of "
                    "turning only the top one")
    bpy.types.Scene.retarget_grip = bpy.props.EnumProperty(name="Two-Handed Grip", items=[
            ('NONE', "Off", "Each hand follows its own mocap hand"),
            ('RIGHT', "Right Hand Leads", "Both hands hold one thing: the left hand keeps a steady distance to the right"),
            ('LEFT', "Left Hand Leads", "Both hands hold one thing: the right hand keeps a steady distance to the left"),
        ], default='NONE')
    bpy.types.Scene.retarget_sword = bpy.props.BoolProperty(
        name="Sword", default=True,
        description="Turn the hands to the tracked sword (Track Sword, then Combine Data and Animate): the "
                    "hands hold its handle. With a two-handed grip the other hand also sits along the blade")
    bpy.types.Scene.bake_layers = bpy.props.EnumProperty(
        name="Layers", items=layers.LAYER_MODES, default='BASE',
        description="Where Bake Motion puts the mocap on a rig with animation layers (NLA tracks)")
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
    del bpy.types.Scene.hand_live_smoothing
    del bpy.types.Scene.bake_rig
    del bpy.types.Scene.bake_frame_start
    del bpy.types.Scene.bake_frame_end
    del bpy.types.Scene.bake_only_selected
    del bpy.types.Scene.foot_lock_floor_z
    del bpy.types.Scene.foot_lock_mode
    del bpy.types.Scene.foot_lock_preset
    del bpy.types.Scene.foot_lock_lower_hips
    del bpy.types.Scene.hand_lock_preset
    del bpy.types.Scene.bake_action_name
    del bpy.types.Scene.multi_view_tracking
    del bpy.types.Scene.realtime_tracking
    del bpy.types.Scene.combine_method
    del bpy.types.Scene.combine_smoothing
    del bpy.types.Scene.combine_smoothing_method
    del bpy.types.Scene.combine_bone_strength
    del bpy.types.Scene.combine_perspective
    del bpy.types.Scene.combine_fov
    del bpy.types.Scene.combine_auto_fov
    del bpy.types.Scene.gemx_fov
    del bpy.types.Scene.combine_auto_horizon
    del bpy.types.Scene.combine_whole_take
    del bpy.types.Scene.combine_grip
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
    del bpy.types.Scene.retarget_fingers
    del bpy.types.Scene.retarget_grip
    del bpy.types.Scene.retarget_sword
    del bpy.types.Scene.bake_in_place
    del bpy.types.Scene.bake_layers



if __name__ == "__main__":
    register()

