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
                    "Combine Data reports the offset it found; adjust it here for separate performances "
                    "whose timing drifts")
    bpy.types.Scene.combine_bone_strength = bpy.props.FloatProperty(
        name="Keep Bone Lengths", default=0.8, min=0.0, max=1.0, subtype='FACTOR',
        description="How strongly depth is corrected so bones keep a constant length")
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
    del bpy.types.Scene.multi_view_tracking
    del bpy.types.Scene.realtime_tracking
    del bpy.types.Scene.combine_method
    del bpy.types.Scene.combine_smoothing
    del bpy.types.Scene.combine_bone_strength
    del bpy.types.Scene.combine_auto_offset
    del bpy.types.Scene.combine_offset
    del bpy.types.Scene.retarget_move_hips
    del bpy.types.Scene.retarget_mirror_depth



if __name__ == "__main__":
    register()

