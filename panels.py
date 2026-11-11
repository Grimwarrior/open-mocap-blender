import bpy
from . import globalVariables as gvar
from . import PackageInstaller, retarget
from .PackageInstaller import check_required_packages



# -------------------------------------------------------------
# UI PANEL FOR INSTALLING DEPENDENCY PACKAGES
# -------------------------------------------------------------
class OpenMocapAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    mediapipe_channel: bpy.props.EnumProperty(
        name="MediaPipe",
        items=[
            ('STABLE', f"Stable ({PackageInstaller.STABLE_VERSION})",
             "Tested with Blender 5.1. Sends no usage statistics"),
            ('LATEST', "Latest",
             "Newest MediaPipe release from PyPI. Sends usage statistics to Google every time "
             "tracking stops; this cannot be turned off"),
        ],
        default='STABLE')

    pose_tracker: bpy.props.EnumProperty(
        name="Pose Tracker",
        items=[
            ('MEDIAPIPE', "MediaPipe", "MediaPipe pose tracking (default)"),
            ('RTMW', "RTMW + MediaPipe depth",
             "OpenMMLab RTMW whole-body tracker for the joint positions (steadier, real hand and foot "
             "points), MediaPipe alongside for depth. Downloads about 330 MB of models"),
        ],
        default='MEDIAPIPE')
    pose_model: bpy.props.EnumProperty(
        name="Pose Model",
        items=[
            ('FULL', "Full", "MediaPipe's standard pose model (default)"),
            ('HEAVY', "Heavy", "MediaPipe's largest pose model: more accurate, slower to track. "
                               "Downloads about 30 MB"),
        ],
        default='FULL')
    rtmw_device: bpy.props.EnumProperty(
        name="RTMW Device",
        items=[
            ('GPU', "GPU (DirectML)", "Any DirectX 12 graphics card, about 10x faster"),
            ('CPU', "CPU", "Slower (about 3 frames per second on 1080p video)"),
        ],
        default='GPU')

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.label(text="Dependency Management", icon="PREFERENCES")

        status = PackageInstaller.status
        if status["busy"]:
            col.label(text=status["message"] or "Working...", icon="SORTTIME")
            return

        col.prop(self, "mediapipe_channel")
        if self.mediapipe_channel == 'LATEST':
            col.label(text="Newer MediaPipe versions send usage statistics to Google when tracking stops.",
                      icon="ERROR")
        installed = PackageInstaller.installed_mediapipe_version()
        col.label(text=f"Installed: MediaPipe {installed}" if installed else "Installed: no MediaPipe",
                  icon="PACKAGE")
        col.separator()
        col.prop(self, "pose_tracker")
        if self.pose_tracker == 'RTMW':
            col.prop(self, "rtmw_device")
        col.prop(self, "pose_model")
        col.separator()

        missing = check_required_packages(self.mediapipe_channel, self.pose_tracker, self.pose_model)

        if missing:
            col.label(text=f"Missing: {', '.join(missing)}", icon="ERROR")
            col.operator("addon.install_dependencies", text="Install Dependencies", icon="CONSOLE")
            if PackageInstaller.mediapipe_loaded() and installed:
                col.label(text="Switching versions needs a Blender restart once tracking has run.", icon="INFO")
        else:
            col.label(text="All dependencies are installed.", icon="CHECKMARK")
            if self.mediapipe_channel == 'LATEST':
                col.operator("addon.install_dependencies", text="Check for MediaPipe Update",
                             icon="FILE_REFRESH").update = True
            col.label(text=f"Location: {PackageInstaller.data_dir()}", icon="FILE_FOLDER")
        if status["message"]:
            col.label(text=status["message"], icon="INFO")
            #col.operator("addon.uninstall_dependencies", text="Uninstall Dependencies", icon="TRASH")



# -------------------------------------------------------------
#UI PANEL FOR POSE TRACKING
# -------------------------------------------------------------
class PoseTrackingPanel(bpy.types.Panel):
    bl_label = "Pose Tracking"
    bl_idname = "OPENMOCAP_PT_pose_tracking"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Open Mocap"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        scene = context.scene
        layout = self.layout
        if PackageInstaller.pose_tracker() == 'RTMW':
            ready = not PackageInstaller.rtmw_missing()
            device = "GPU" if PackageInstaller.rtmw_device() == 'dml' else "CPU"
            layout.label(text=f"Tracker: RTMW ({device}) + MediaPipe depth" if ready
                         else "Tracker: RTMW not installed, using MediaPipe",
                         icon="CHECKMARK" if ready else "ERROR")
        else:
            layout.label(text="Tracker: MediaPipe", icon="ARMATURE_DATA")
        layout.prop(scene, "multi_view_tracking")
        #layout.label(text="Realtime Pose-tracking")
        #layout.operator("object.start_tracking", text="Start RealTime Pose Tracking", icon='PLAY').track = "rt_pose"
        #layout.operator("object.stop_tracking", text="Stop Pose Tracking", icon='PAUSE').mode = "realtime"
        #layout.separator()
        #layout.label(text="Offline Pose-tracking")

        #Front Video
        if scene.multi_view_tracking:
            layout.label(text="Front Video")
        if gvar.front_video_path:
            layout.label(text = gvar.front_video_path, icon = 'FILE_MOVIE')
        if scene.multi_view_tracking:
            layout.operator("object.open_mocap_filebrowser", text="Select Front Video", icon='FILEBROWSER').video = "front_video_pose"
        else:
            layout.operator("object.open_mocap_filebrowser", text="Select Video", icon='FILEBROWSER').video = "front_video_pose"
        layout.operator("object.start_tracking", text="Start Pose Tracking", icon='PLAY').track = "front_pose"
        layout.operator("object.clear_video_path", text="Delete Video", icon='X').video = "front_video_pose"

        #Side Video
        if scene.multi_view_tracking:
            layout.separator()
            layout.label(text="Side Video")
            if gvar.side_video_path:
                layout.label(text = gvar.side_video_path, icon = 'FILE_MOVIE')
            layout.operator("object.open_mocap_filebrowser", text="Select Side Video", icon='FILEBROWSER').video = "side_video_pose"
            layout.operator("object.start_tracking", text="Start Pose Tracking", icon='PLAY').track = "side_pose"
            layout.operator("object.clear_video_path", text="Delete Video", icon='X').video = "side_video_pose"

        #Settings
        layout.separator()
        layout.label(text="Settings")
        layout.operator("object.stop_tracking", text="Stop Pose Tracking", icon='PAUSE')
        if scene.multi_view_tracking:
            box = layout.box()
            box.prop(scene, "combine_method")
            if scene.combine_method == 'IMPROVED':
                box.prop(scene, "combine_sync")
                row = box.row(align=True)
                row.prop(scene, "combine_auto_offset", toggle=True)
                sub = row.row(align=True)
                sub.enabled = not scene.combine_auto_offset
                sub.prop(scene, "combine_offset", text="Offset")
                row = box.row(align=True)
                row.prop(scene, "combine_auto_angle", toggle=True)
                sub = row.row(align=True)
                sub.enabled = not scene.combine_auto_angle
                sub.prop(scene, "combine_side_angle", text="Angle")
                box.prop(scene, "combine_smoothing")
                box.prop(scene, "combine_bone_strength")
            box.operator("object.combine_data", text="Combine Data", icon='PLUS').mode = "pose"
        layout.operator("object.animate_obj", text="Animate", icon='RENDER_ANIMATION')
        layout.operator("object.hide_skeleton", text="Hide/Unhide Empties", icon='HIDE_OFF').collection_name = "PoseEmpties"
        layout.operator("object.hide_skeleton", text="Hide/Unhide Skeleton", icon='HIDE_OFF').collection_name = "PoseSkeleton"
        layout.operator("object.clear_mocap_cache", text="Clear Capture Cache", icon='TRASH').cache = "pose"
        row = layout.row(align=True)
        row.operator("object.open_mocap_save_capture", text="Save Capture", icon='EXPORT')
        row.operator("object.open_mocap_load_capture", text="Load Capture", icon='IMPORT')
        layout.separator()
        layout.label(text="Animation Baking")
        layout.prop(scene, "bake_rig", text="Select Rig")
        if scene.bake_rig:
            if retarget.is_mixamo_rig(scene.bake_rig):
                box = layout.box()
                box.prop(scene, "retarget_source", icon="EMPTY_AXIS")
                box.prop(scene, "retarget_arm_mode")
                if scene.retarget_arm_mode == 'POSITION':
                    box.prop(scene, "retarget_grip")
                box.prop(scene, "retarget_leg_mode")
                box.prop(scene, "retarget_move_hips")
                if scene.retarget_move_hips:
                    box.prop(scene, "retarget_hip_height")
                row = box.row(align=True)
                row.prop(scene, "retarget_shoulders")
                row.prop(scene, "retarget_spine_twist")
                box.prop(scene, "retarget_fingers")
                box.prop(scene, "retarget_mirror_depth")
                row = box.row(align=True)
                op = row.operator("object.open_mocap_retarget_mixamo", text="Retarget to Mixamo", icon="ARMATURE_DATA")
                op.target_rig = scene.bake_rig.name
                op.source = scene.retarget_source.name if scene.retarget_source else ""
                op.hips_motion = scene.retarget_move_hips
                op.mirror_depth = scene.retarget_mirror_depth
                op.arm_mode = scene.retarget_arm_mode
                op.leg_mode = scene.retarget_leg_mode
                op.hip_height = scene.retarget_hip_height
                op.shoulders = scene.retarget_shoulders
                op.spine_twist = scene.retarget_spine_twist
                op.fingers = scene.retarget_fingers
                op.grip = scene.retarget_grip
                op = row.operator("object.open_mocap_clear_retarget", text="", icon="X")
                op.target_rig = scene.bake_rig.name
            op = layout.operator("pose.toggle_copy_rotation", text="Toggle Constraints", icon="CONSTRAINT")
            op.target_rig = scene.bake_rig.name
        layout.prop(scene, "bake_frame_start")
        layout.prop(scene, "bake_frame_end")
        layout.prop(scene, "bake_only_selected")
        if scene.bake_only_selected and scene.bake_rig:
            count = sum(bool(getattr(pb, 'select', getattr(pb.bone, 'select', False)))
                        for pb in scene.bake_rig.pose.bones)
            layout.label(text=f"Partial bake: {count} selected bones", icon='ERROR')
        layout.prop(scene, "bake_in_place")
        layout.prop(scene, "bake_action_name", text="Action", icon="ACTION")

        op = layout.operator("object.bake_motion", text="Bake Motion", icon="ACTION_TWEAK")
        if scene.bake_rig:
            op.target_rig = scene.bake_rig.name
        op.frame_start = scene.bake_frame_start
        op.frame_end = scene.bake_frame_end
        op.only_selected_bones = scene.bake_only_selected
        op.action_name = scene.bake_action_name
        op.in_place = scene.bake_in_place



# -------------------------------------------------------------
#UI PANEL FOR HAND TRACKING
# -------------------------------------------------------------
class HandTrackingPanel(bpy.types.Panel):
    bl_label = "Hand Tracking"
    bl_idname = "OPENMOCAP_PT_hand_tracking"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Open Mocap"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        scene = context.scene
        layout = self.layout
        layout.prop(scene, "realtime_tracking")
        if scene.realtime_tracking:
            layout.label(text="Realtime Hand-tracking")
            layout.prop(scene, "cam_index", text="Choose a camera")
            layout.prop(scene, "num_hands", text="Number of Hands")
            layout.operator("object.animatehand", text="Start RealTime Hand Tracking", icon='PLAY').mode = "realtime"
            layout.operator("object.stop_tracking", text="Stop Hand Tracking", icon='PAUSE').mode = "realtime"
        layout.separator()
        layout.label(text="Offline Hand-tracking")
        layout.prop(scene, "multi_view_tracking")

        #Front Video
        if scene.multi_view_tracking:
            layout.label(text="Front Video")
        if gvar.hand_front_video_path:
            layout.label(text = gvar.hand_front_video_path, icon = 'FILE_MOVIE')
        if scene.multi_view_tracking:
            layout.operator("object.open_mocap_filebrowser", text="Select Front Video", icon='FILEBROWSER').video = "front_video_hand"
        else:
            layout.operator("object.open_mocap_filebrowser", text="Select Video", icon='FILEBROWSER').video = "front_video_hand"
        
        layout.operator("object.start_tracking", text="Start Hand Tracking", icon='PLAY').track = "front_hand"
        layout.operator("object.clear_video_path", text="Delete Video", icon='X').video = "front_video_hand"

        #Side Video
        if scene.multi_view_tracking:
            layout.separator()
            layout.label(text="Side Video")
            if gvar.hand_side_video_path:
                layout.label(text = gvar.hand_side_video_path, icon = 'FILE_MOVIE')
            layout.operator("object.open_mocap_filebrowser", text="Select Side Video", icon='FILEBROWSER').video = "side_video_hand"
            layout.operator("object.start_tracking", text="Start Hand Tracking", icon='PLAY').track = "side_hand"
            layout.operator("object.clear_video_path", text="Delete Video", icon='X').video = "side_video_hand"

        #Settings
        layout.separator()
        layout.label(text="Settings")
        layout.operator("object.stop_tracking", text="Stop Hand Tracking", icon='PAUSE')
        if scene.multi_view_tracking:
            layout.operator("object.combine_data", text="Combine Data", icon='PLUS').mode = "hand"
        layout.operator("object.animatehand", text="Animate", icon='RENDER_ANIMATION').mode = "offline"
        layout.operator("object.clear_mocap_cache", text="Clear Capture Cache", icon='TRASH').cache = "hand"
        row = layout.row(align=True)
        row.operator("object.open_mocap_save_capture", text="Save Capture", icon='EXPORT')
        row.operator("object.open_mocap_load_capture", text="Load Capture", icon='IMPORT')
        layout.operator("object.hide_skeleton", text="Hide/Unhide R_Empties", icon='HIDE_OFF').collection_name = "RightHandEmpties"
        layout.operator("object.hide_skeleton", text="Hide/Unhide R_Skeleton", icon='HIDE_OFF').collection_name = "RightHandSkeleton"
        layout.operator("object.hide_skeleton", text="Hide/Unhide L_Empties", icon='HIDE_OFF').collection_name = "LeftHandEmpties"
        layout.operator("object.hide_skeleton", text="Hide/Unhide L_Skeleton", icon='HIDE_OFF').collection_name = "LeftHandSkeleton"
        layout.separator()
        layout.label(text="Animation Baking")
        layout.prop(scene, "bake_rig", text="Select Rig")
        if scene.bake_rig:
            op = layout.operator("pose.toggle_copy_rotation", text="Toggle Constraints", icon="CONSTRAINT")
            op.target_rig = scene.bake_rig.name
        else:
            layout.label(text="Select a rig to toggle constraints", icon="INFO")
        layout.prop(scene, "bake_frame_start")
        layout.prop(scene, "bake_frame_end")
        op = layout.operator("object.bake_motion", text="Bake Motion", icon="ACTION_TWEAK")
        layout.prop(scene, "bake_only_selected")
        if scene.bake_only_selected and scene.bake_rig:
            count = sum(bool(getattr(pb, 'select', getattr(pb.bone, 'select', False)))
                        for pb in scene.bake_rig.pose.bones)
            layout.label(text=f"Partial bake: {count} selected bones", icon='ERROR')

        if scene.bake_rig:
            op.target_rig = scene.bake_rig.name
        op.frame_start = scene.bake_frame_start
        op.frame_end = scene.bake_frame_end
        op.only_selected_bones = scene.bake_only_selected



classes = [OpenMocapAddonPreferences, PoseTrackingPanel, HandTrackingPanel]
