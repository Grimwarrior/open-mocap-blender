"""
    All the necessary operators for the addon are here.
    The operators in order are as follows:
    1) Main operators:
    StartTracking, StopTracking, CombineMotionData, AnimatePose, AnimateHand, BakeMotion
    2) Helper operators:
    ClearMoCapCache, VideoSelect,  ClearVideoPath, ToggleCopyRotationConstraints, HideSkeleton
"""

import bpy
import json
import math
import os
from bpy_extras.io_utils import ExportHelper, ImportHelper
from . import utils
from . import combine
from . import capture_data
from mathutils import Vector
from . import globalVariables as gvar

# -------------------------------------------------------------
# FRAME TIMING
# -------------------------------------------------------------
def set_scene_fps(scene, fps):
    """Scene frame rate = video frame rate, also for rates like 29.97 (fps 30 / base 1.001)."""
    fps = float(fps) if fps and fps > 0 else 30.0
    whole = max(1, int(round(fps)))
    scene.render.fps = whole
    scene.render.fps_base = whole / fps if abs(whole - fps) > 1e-3 else 1.0


def start_playback(context):
    """Play the animation (not in background mode, where there is no screen to play it on)."""
    if not bpy.app.background and context.screen is not None:
        bpy.ops.screen.animation_play()


def frames_from_times(times, count, fps):
    """
        Scene frame of each of `count` tracked frames: its time in the video (video frame 0 = scene
        frame 1). Without usable times (data from an older version): 1, 2, 3, ...
    """
    if count and len(times) >= count and fps and fps > 0:
        frames = [float(t * fps) + 1 for t in times[:count]]
        if all(math.isfinite(f) and f >= 1 for f in frames) and all(b > a for a, b in zip(frames, frames[1:])):
            return frames
    return list(range(1, count + 1))


def match_by_time(times, other_times, fps):
    """
        For every entry of `times` the index of the entry of `other_times` closest in time, or None if
        that one is more than three quarters of a frame away (the other view lost the hand there).
    """
    out = []
    j = 0
    limit = 0.75 / fps if fps and fps > 0 else 0.025
    for t in times:
        while j + 1 < len(other_times) and abs(other_times[j + 1] - t) <= abs(other_times[j] - t):
            j += 1
        out.append(j if other_times and abs(other_times[j] - t) <= limit else None)
    return out


def hand_capture():
    """
        The hand data to animate: from a hand video if there is any, else the whole hands the RTMW
        pose tracker found in the body video.
        :return: (from_body, fps, {"R"/"L": (frames, times)}) or None.
    """
    if gvar.R_hand_positionList or gvar.L_hand_positionList:
        return False, gvar.hand_front_fps or gvar.hand_fps or gvar.fps, {
            "R": (gvar.R_hand_positionList, gvar.R_hand_times),
            "L": (gvar.L_hand_positionList, gvar.L_hand_times)}
    if gvar.R_body_hand_positionList or gvar.L_body_hand_positionList:
        return True, gvar.front_fps or gvar.fps, {
            "R": (gvar.R_body_hand_positionList, gvar.R_body_hand_times),
            "L": (gvar.L_body_hand_positionList, gvar.L_body_hand_times)}
    return None


POSE_WRIST = {"R": 16, "L": 15}


def body_hand_frames(side, frames, times, fps):
    """
        Body-video hands have no depth of their own: every point of a hand gets the depth of the
        pose wrist at that moment (after Combine Data: the combined depth), so the hand sits on the
        pose skeleton's arm. :return: frames with the depth filled in.
    """
    z_index = 3 * POSE_WRIST[side] + 2
    out, depth = [], 0.0
    for frame, j in zip(frames, match_by_time(times, gvar.position_times[:len(gvar.positionList)], fps)):
        if j is not None:
            depth = gvar.positionList[j][z_index]
        frame = list(frame)
        frame[2::3] = [depth] * 21
        out.append(frame)
    return out


def keep_in_place(rig):
    """
        Remove the travel over the ground from the baked hips (the rig's root motion): every key keeps
        its height, and the horizontal position of the first key. :return: True if keys were changed.
    """
    from . import retarget
    hips = retarget.drive_bone(rig, "Hips")
    ad = rig.animation_data
    if hips is None or not ad or not ad.action:
        return False
    fcurves = None
    try:
        from bpy_extras.anim_utils import action_get_channelbag_for_slot
        bag = action_get_channelbag_for_slot(ad.action, ad.action_slot)
        fcurves = bag.fcurves if bag else None
    except ImportError:
        pass
    if fcurves is None:
        fcurves = getattr(ad.action, "fcurves", [])
    path = hips.path_from_id("location")
    fcs = {fc.array_index: fc for fc in fcurves if fc.data_path == path}
    if len(fcs) != 3 or len({len(fc.keyframe_points) for fc in fcs.values()}) != 1:
        return False
    # bone location -> world directions (the bone's rest axes, turned by its parent's rest and the object)
    to_world = rig.matrix_world.to_3x3() @ hips.bone.matrix_local.to_3x3()
    back = to_world.inverted()
    first = None
    for k in range(len(fcs[0].keyframe_points)):
        points = [fcs[i].keyframe_points[k] for i in range(3)]
        world = to_world @ Vector([p.co[1] for p in points])
        if first is None:
            first = world.copy()
        local = back @ Vector((first.x, first.y, world.z))
        for i, p in enumerate(points):
            delta = local[i] - p.co[1]
            p.co[1] = local[i]
            p.handle_left[1] += delta
            p.handle_right[1] += delta
    for fc in fcs.values():
        fc.update()
    return True


# -------------------------------------------------------------
# MAIN OPERATORS
# -------------------------------------------------------------
class StartTracking(bpy.types.Operator):
    bl_idname = "object.start_tracking"
    bl_label = "Start Motion Tracking"
    bl_description = "Starts motion tracking."

    track: bpy.props.StringProperty()
    """
        It starts tracking the video from a file or realtime video stream from camera.
        :param track: Helps in deciding whether tracking pose or hand, 
                        whether tracking offline or realtime 
                        and whether tracking from front view or from side view
    """


    def execute(self, context):
        if self.track == "front_pose":
            gvar.pose_tracker = utils.PoseTracking("front_pose")

        elif self.track == "side_pose":
            gvar.pose_tracker = utils.PoseTracking("side_pose")

        elif self.track == "rt_pose":
            #realtime tracking
            gvar.pose_tracker = utils.PoseTracking("rt_pose")

        elif self.track == "front_hand":
            num_hands = context.scene.num_hands
            gvar.hand_tracker = utils.HandTracking(num_hands=num_hands, mode="front_hand")

        elif self.track == "side_hand":
            num_hands = context.scene.num_hands
            gvar.hand_tracker = utils.HandTracking(num_hands=num_hands, mode="side_hand")

        else:
            #realtime tracking
            num_hands = context.scene.num_hands
            gvar.hand_tracker = utils.HandTracking(num_hands=num_hands, mode="rt_hand")

        return {'FINISHED'}



class StopTracking(bpy.types.Operator):
    bl_idname = "object.stop_tracking"
    bl_label = "Stop Motion Tracking"
    bl_description = "Stops motion tracking."

    mode: bpy.props.StringProperty()
    """
        It stops motion tracking (all types).
        :param mode: Tells if it is a realtime tracker.
                    If so, then Video plane and its textures must be deleted from Blender scene.
    """

    def execute(self, context):
        if self.mode == "realtime":
            utils.VideoPlaneManager.remove_plane()

        #stop playing animation    
        if bpy.context.screen.is_animation_playing:
            bpy.ops.screen.animation_play()

        #stop pose tracker if running
        if gvar.pose_tracker is not None:
            gvar.pose_tracker.stop()
            gvar.pose_tracker = None

        #stop hand tracker if running
        elif gvar.hand_tracker is not None:
            gvar.hand_tracker.stop()
            gvar.hand_tracker = None

        else:
            self.report({'WARNING'}, "No tracking running")

        return {'FINISHED'}
    


class CombineMotionData(bpy.types.Operator):
    bl_idname = "object.combine_data"
    bl_label = "Combine Motion Data"
    bl_description = "Synchronously combines the mocap data from two different cameras views."

    mode: bpy.props.StringProperty()
    """
        It replaces the inaccurate z coordinate of a tracked point from front view
          with accurate x coordinate of a tracked point from side view. 
          This reduces inaccuracy of motion tracking.
        :param mode: Tells whether pose data is combined or hand tracking data is combined.
    """

    def combine_lists(self, front_list, side_list, num_points):
        """
            Combine front and side lists into new list with updated Z values.
            :param front_list: Motion data from front view.
            :param side_list: Motion data from side view.
            :param num_points: If tracking pose, each video frame will have 40 points.
                                If tracking hand, each frame will have 21 points per hand.
            :return combined: The new combined motion data with improved tracking accuracy.
        """
        combined = []
        min_frames = min(len(front_list), len(side_list))

        for f in range(min_frames):
            frame_front = front_list[f].copy()  # copy to avoid modifying original
            frame_side = side_list[f]

            for i in range(num_points):
                frame_front[3 * i + 2] = -frame_side[3 * i + 0]  # z = -x (side)
            combined.append(frame_front)

        return combined
    
    def combine_pose(self, context):
        scene = context.scene
        # Always combine from the raw front data, so Combine Data can be re-run with other settings
        front = gvar.positionList_raw if gvar.positionList_raw is not None else gvar.positionList
        metadata = gvar.position_meta_raw if gvar.position_meta_raw is not None else gvar.position_meta
        side = gvar.zlist
        if not front or not side:
            self.report({'ERROR'}, "Track both the front and the side video first")
            return {'CANCELLED'}

        have_meta = (len(gvar.position_times) == len(front) and len(gvar.position_vis) == len(front)
                     and len(gvar.zlist_times) == len(side) and len(gvar.zlist_vis) == len(side))
        method = scene.combine_method
        if method == 'IMPROVED' and have_meta and len(front) >= 10 and len(side) >= 10:
            try:
                combined, rep = combine.combine_improved(
                    front, side, gvar.position_times, gvar.zlist_times, gvar.position_vis, gvar.zlist_vis,
                    smoothing=scene.combine_smoothing, bone_strength=scene.combine_bone_strength,
                    offset=None if scene.combine_auto_offset else scene.combine_offset,
                    sync=scene.combine_sync,
                    side_angle=None if scene.combine_auto_angle else scene.combine_side_angle,
                    front_depth_confidence=capture_data.depth_confidence(metadata, gvar.position_vis) if metadata else None)
            except ValueError as error:
                self.report({'ERROR'}, str(error))
                return {'CANCELLED'}
            if not rep.get('combined', True):
                self.report({'WARNING'}, 'Combine skipped: ' + combine.format_report(rep))
                return {'CANCELLED'}
            combined_meta = rep['metadata']
            if scene.combine_auto_offset:
                scene.combine_offset = rep["offset_s"]   # start manual tweaking from what was found
            if scene.combine_auto_angle:
                scene.combine_side_angle = rep["side_angle"]
            msg = "Combined: " + combine.format_report(rep)
            level = {'WARNING'} if rep["sync_corr"] < 0.4 else {'INFO'}
            if rep["sync_corr"] < 0.4:
                msg += ". Low sync match: check that both videos show the same performance"
        else:
            if method == 'IMPROVED':
                self.report({'WARNING'}, 'Improved combine needs at least 10 samples per view with timestamps '
                            'and confidence. Re-track or explicitly select the original method.')
                return {'CANCELLED'}
            combined = combine.combine_simple(front, side, 40)
            combined.extend([list(f) for f in front[len(combined):]])
            combined_meta = []  # original combine has no reliable confidence propagation
            msg = f"Combined {len(combined)} frames (original method)."
            level = {'INFO'}
        if gvar.positionList_raw is None:
            gvar.positionList_raw = list(front)
            gvar.position_meta_raw = list(metadata)
        gvar.positionList = combined
        gvar.position_meta = combined_meta
        if gvar.front_fps:
            gvar.fps = gvar.front_fps   # the combined take runs at the front video's frame rate
        print("[Open Mocap]", msg)
        self.report(level, msg)
        return {'FINISHED'}

    def execute(self, context):

        if self.mode == "pose":
            return self.combine_pose(context)

        elif self.mode == "hand":
            for side in ("R", "L"):
                front = getattr(gvar, f"{side}_hand_positionList")
                depth = getattr(gvar, f"{side}_hand_zlist")
                times = getattr(gvar, f"{side}_hand_times")
                ztimes = getattr(gvar, f"{side}_hand_ztimes")
                if not front or not depth:
                    continue
                if len(times) == len(front) and len(ztimes) == len(depth):
                    # Match the two views by time: either one can lose the hand for some frames,
                    # after which frame N of one list is no longer frame N of the other. A front
                    # frame the side view has no match for keeps its own depth.
                    combined = []
                    for frame, j in zip(front, match_by_time(times, ztimes, gvar.hand_side_fps or gvar.hand_fps or gvar.fps)):
                        frame = list(frame)
                        if j is not None:
                            for i in range(21):
                                frame[3 * i + 2] = -depth[j][3 * i + 0]
                        combined.append(frame)
                else:
                    combined = self.combine_lists(front, depth, 21)
                    del times[len(combined):]
                setattr(gvar, f"{side}_hand_positionList", combined)

        else:
            self.report({'WARNING'}, f"Unknown mode: {self.mode}")

        return {'FINISHED'}



class AnimatePose(bpy.types.Operator):
    bl_idname = "object.animate_obj"
    bl_label = "Animate Pose"
    bl_description = "Animates a skeleton with the tracked pose motion data."

    """
        Creates 40 empty objects to copy motion data of 40 pose landmarks.
        Creates a root empty and parent all 40 empties to this.
        Creates a skeleton with these 40 empties and adds keyframes for 40 empties with the motion data.
        This animates the skeleton automatically.
    """

    def execute(self, context):
        if not gvar.positionList:
            self.report({'ERROR'}, "No tracked pose data. Run Start Pose Tracking first "
                                   "(the capture cache is emptied when Blender restarts or the add-on reloads)")
            return {'CANCELLED'}
        gvar.bones_list.clear()
        gvar.object_list.clear()
        # --- Get or create collection ---
        pose_empties_collection = utils.BlenderUtility.create_collection(context,"PoseEmpties")
        root_collection = utils.BlenderUtility.create_collection(context,"RootEmpties")

        # --- Create root empty ---
        root = bpy.data.objects.new("RigRoot", None)
        root.empty_display_type = 'SPHERE'
        root.empty_display_size = 1
        root_collection.objects.link(root)
        gvar.root = root

        # --- Create empties ---
        for i in range(40):
            empty = bpy.data.objects.new(name=str(i), object_data=None)
            empty.empty_display_size = 0.1
            empty.empty_display_type = 'SPHERE'
            pose_empties_collection.objects.link(empty)
            empty.parent = gvar.root
            gvar.object_list.append(empty)

        # --- Define bone pairs ---
        bone_pairs = [
            (7,8), (11,12), (23,24), (16,35), (15,36),
            (11,13), (13,15), (33,34), (38,39),
            (12,14), (14,16), (34,37),
            (24,26), (26,28), (28,32),
            (23,25), (25,27), (27,31)
        ]

        # --- Create skeleton (with default Y tracking) ---
        bones_created = utils.Skeleton.create_skeleton(context, bone_pairs, "pose")
        # --- Apply configuration-specific adjustments ---
        # --- These adjustments help in smooth motion retargeting for specific bones---
        track_x_pairs = {(7,8), (11,12), (23,24)}
        rotate_y_pairs = {(12,14),(11,13),(13,15),(14,16), (16,35), (15,36),}
        for a, b, obj in bones_created:
            #Adjust tracking axis for specific pairs
            if (a, b) in track_x_pairs:
                for constraint in obj.constraints:
                    if constraint.type == 'DAMPED_TRACK':
                        constraint.track_axis = 'TRACK_NEGATIVE_X'
                        break
            # Rotate if needed
            if (a, b) in rotate_y_pairs:
                obj.rotation_euler.rotate_axis('Y', math.radians(180))

        # --- Keyframe insertion ---
        # Each tracked frame is keyed at its own time in the video: a frame where nobody was found
        # has no entry, and keying the entries one after another would play the rest too early.
        fps = gvar.front_fps or gvar.fps          # gvar.fps is the last tracked video's, maybe the side one
        set_scene_fps(context.scene, fps)
        frame_numbers = frames_from_times(gvar.position_times, len(gvar.positionList), fps)
        context.scene.frame_start = 1
        context.scene.frame_end = math.ceil(frame_numbers[-1])
        # Persist confidence with the animated source, so cache changes/reloads cannot change it.
        root['open_mocap_metadata'] = json.dumps({
            'frames': frame_numbers, 'fps': fps, 'visibility': gvar.position_vis[:len(frame_numbers)],
            'metadata': gvar.position_meta, 'missing_times': gvar.position_missing_times})

        # Hand points the RTMW tracker found in the body video (x/y only), for the retarget's fingers
        hands = {}
        for side, key in (("Left", "L"), ("Right", "R")):
            hand_frames = getattr(gvar, f"{key}_body_hand_positionList")
            hand_times = getattr(gvar, f"{key}_body_hand_times")
            if hand_frames and len(hand_frames) == len(hand_times):
                hands[side] = {'frames': [float(t * fps) + 1 for t in hand_times],
                               'points': [[v for i in range(21) for v in f[3 * i:3 * i + 2]] for f in hand_frames]}
        if hands:
            root['open_mocap_hands'] = json.dumps(hands)

        for frame_index, frame in zip(frame_numbers, gvar.positionList):
            for i in range(40):
                location = utils.MathUtility.cv2blender_coordinates([frame[0 + (i * 3)],frame[1 + (i * 3)],frame[2 + (i * 3)]])
                obj = gvar.object_list[i]
                local_pos = Vector(location) - gvar.root.location
                obj.location = local_pos
                obj.keyframe_insert(data_path="location", frame=frame_index)

        start_playback(context)
        return {'FINISHED'}



class AnimateHand(bpy.types.Operator):
    bl_idname = "object.animatehand"
    bl_label = "Animate Hand"
    bl_description = "Animates a skeleton with the tracked hand motion data."
    mode: bpy.props.StringProperty()
    """ Creates left/right hand(s) and animates them (realtime/offline)"""

    def create_hand(self, context, prefix):
        """
            Creates 21 empty objects to copy motion data of 21 hand landmarks.
            Creates a root empty and parent all 21 empties to this.
            Creates a skeleton with these 21 empties.
            :param prefix: R for right hand and L for left hand
        """
        if prefix == "R":
            hand_name = "Right"
            gvar.R_hand_bones_list = []
            gvar.R_hand_object_list = []
            object_list = gvar.R_hand_object_list
        else:
            hand_name = "Left"
            gvar.L_hand_bones_list = []
            gvar.L_hand_object_list = []
            gvar.L_hand_object_list.clear()
            object_list = gvar.L_hand_object_list

        root_collection = utils.BlenderUtility.create_collection(context,"RootEmpties")
        hand_empties_collection = utils.BlenderUtility.create_collection(context, f"{hand_name}HandEmpties")

        bone_pairs = [
            (0,1),(1,2),(2,3),(3,4),
            (5,6),(6,7),(7,8),
            (9,10),(10,11),(11,12),
            (13,14),(14,15),(15,16),
            (17,18),(18,19),(19,20),
            (0,5),(0,9),(0,13),(0,17),
            (5,9),(9,13),(13,17),(1,5)
        ]

        # ---------------Create Empty Root------------------
        root = bpy.data.objects.new(f"{hand_name}HandRoot", None)
        root.empty_display_type = 'SPHERE'
        root.empty_display_size = 0.5
        root_collection.objects.link(root)
        if prefix == "R":
            gvar.R_hand_root = root
        else:
            gvar.L_hand_root = root
            
        # ---------------Create 21 Empties------------------
        for i in range(21):
            empty = bpy.data.objects.new(name=f"{prefix}_h{i}", object_data=None)
            empty.empty_display_size = 0.1
            empty.empty_display_type = 'SPHERE'
            hand_empties_collection.objects.link(empty)
            empty.parent = root
            object_list.append(empty)

        if prefix == "R":
            utils.Skeleton.create_skeleton(context, bone_pairs, "R_hand")
        else:
            utils.Skeleton.create_skeleton(context, bone_pairs, "L_hand")


    def hand_exists(self, prefix):
        """
        Check if the hand already exists by checking:
        - 21 empties exist (R_h0 ... R_h20 OR L_h0 ... L_h20)
        - root empty exists (RightHandRoot or LeftHandRoot)
        """
        hand_name = "Right" if prefix == "R" else "Left"
        root_name = f"{hand_name}HandRoot"

        # Check root
        if root_name not in bpy.data.objects:
            return False

        # Check 21 empties
        for i in range(21):
            obj_name = f"{prefix}_h{i}"
            if obj_name not in bpy.data.objects:
                return False

        return True


    def execute(self, context):
        if self.mode == "realtime":
            num_hands = context.scene.num_hands
            gvar.hand_tracker = utils.HandTracking(num_hands,"rt_hand")

            # --- Create Right Hand ---
            if not self.hand_exists("R"):
                self.create_hand(context, "R")

            # --- Create Left Hand ---
            if not self.hand_exists("L"):
                self.create_hand(bpy.context, "L")

            bpy.app.timers.register(self.animate_realtime, first_interval=0.0)
            start_playback(context)

        else:
            source = hand_capture()
            if source is None:
                self.report({'ERROR'}, "No tracked hand data. Run Start Hand Tracking first, or track a body "
                                       "video with the RTMW pose tracker (the capture cache is emptied when "
                                       "Blender restarts or the add-on reloads)")
                return {'CANCELLED'}
            from_body, fps, hands = source

            # --- Create Right Hand ---
            if not self.hand_exists("R"):
                self.create_hand(context, "R")

            # --- Create Left Hand ---
            if not self.hand_exists("L"):
                self.create_hand(bpy.context, "L")

            # Each tracked frame is keyed at its own time in the video (a hand that was not found
            # in a frame has no entry there)
            last = 1
            for side, (frames, times) in hands.items():
                objects = getattr(gvar, f"{side}_hand_object_list")
                root = getattr(gvar, f"{side}_hand_root")
                if not objects or not frames:
                    continue
                if from_body:
                    frames = body_hand_frames(side, frames, times, fps)
                numbers = frames_from_times(times, len(frames), fps)
                last = max(last, numbers[-1])
                for frame_index, frame in zip(numbers, frames):
                    for i in range(21):
                        loc = utils.MathUtility.cv2blender_coordinates([
                            frame[i*3], frame[i*3+1], frame[i*3+2]
                        ])
                        obj = objects[i]
                        obj.location = Vector(loc) - root.location
                        obj.keyframe_insert(data_path="location", frame=frame_index)
            context.scene.frame_start = 1
            context.scene.frame_end = math.ceil(last)
            set_scene_fps(context.scene, fps)

            # --- Playback ---
            tracked_R = bool(hands["R"][0])
            tracked_L = bool(hands["L"][0])
            where = " from the body video, at the pose wrists' depth" if from_body else ""
            self.report({'INFO'}, f"Hand animation complete{where}. (Right: {tracked_R}, Left: {tracked_L})")
            start_playback(context)
        return {'FINISHED'}
    
    def animate_realtime(self):
        if not gvar.hand_tracker or not gvar.hand_tracker.running:
            print("Stopping timer: tracker stopped.")
            return None
        success, img = gvar.hand_tracker.cap.read()
        if not success:
            print("Stopping timer: failed to read frame.")
            gvar.hand_tracker.stop()
            return None

        img = gvar.hand_tracker.detect_hand(img)


        # --- Animate Right Hand ---
        if gvar.R_hand_object_list and len(gvar.rt_hand_lmlist_R) > 0:
            for i, point in enumerate(gvar.rt_hand_lmlist_R):
                loc = utils.MathUtility.cv2blender_coordinates(point)
                obj = gvar.R_hand_object_list[i]
                obj.location = Vector(loc) - gvar.R_hand_root.location

        # --- Animate Left Hand ---
        if gvar.L_hand_object_list and len(gvar.rt_hand_lmlist_L) > 0:
            for i, point in enumerate(gvar.rt_hand_lmlist_L):
                loc = utils.MathUtility.cv2blender_coordinates(point)
                obj = gvar.L_hand_object_list[i]
                obj.location = Vector(loc) - gvar.L_hand_root.location
        utils.VideoPlaneManager.update_frame(img)
        return gvar.delay


class BakeMotion(bpy.types.Operator):
    bl_idname = "object.bake_motion"
    bl_label = "Bake Motion"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Bakes animation of selected armature or bones"

    frame_start: bpy.props.IntProperty(name="Start Frame", default=1)
    frame_end: bpy.props.IntProperty(name="End Frame", default=250)
    only_selected_bones: bpy.props.BoolProperty(
        name="Only Selected Bones",
        default=False,
        description="Bake only selected bones in Pose Mode"
    )
    target_rig: bpy.props.StringProperty(name="Target Rig")
    action_name: bpy.props.StringProperty(
        name="Action Name", default="",
        description="Bake into a new action with this name (fake user). Empty: the rig's current action")
    in_place: bpy.props.BoolProperty(
        name="In Place", default=False,
        description="After baking, remove the hips' travel over the ground (their height is kept)")

    """
        :param frame_start: Starting frame for baking animation.
        :param frame_end: Final frame for baking animation.
        :param only_selected_bones: Bake only selected bones in Pose Mode.
                                    Otherwise bakes animation for all the bones in the armature selected.
        :param target_rig: Armature, whose animation is to be baked.
        :param action_name: Name for a new action to bake into; empty keeps the rig's current action.
    """

    def execute(self, context):
        obj = bpy.data.objects.get(self.target_rig)

        if not obj or obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Please select a valid Armature object")
            return {'CANCELLED'}

        # nla.bake bakes every selected object (each armature in pose mode), so only the rig may be
        # selected; the user's selection comes back afterwards
        start_mode = context.mode
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        was_selected = [o for o in context.view_layer.objects if o.select_get()]
        was_active = context.view_layer.objects.active
        for o in was_selected:
            o.select_set(False)
        # A hidden rig cannot be selected or posed, and the bake would silently do nothing
        was_hidden, was_hidden_viewport = obj.hide_get(), obj.hide_viewport
        obj.hide_viewport = False
        obj.hide_set(False)
        if not obj.visible_get():
            obj.hide_viewport = was_hidden_viewport
            obj.hide_set(was_hidden)
            for o in was_selected:
                o.select_set(True)
            self.report({'ERROR'}, f"{obj.name} is in a hidden or excluded collection: show it to bake")
            return {'CANCELLED'}
        context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode='POSE')
        try:
            return self._bake(context, obj)
        finally:
            if context.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            obj.select_set(False)
            for o in was_selected:
                if o.name in context.view_layer.objects:
                    o.select_set(True)
            context.view_layer.objects.active = was_active if was_active and was_active.name in context.view_layer.objects else obj
            obj.hide_viewport = was_hidden_viewport
            obj.hide_set(was_hidden)
            active = context.view_layer.objects.active
            if start_mode == 'POSE' and active.type == 'ARMATURE' and active.visible_get():
                bpy.ops.object.mode_set(mode='POSE')

    def _bake(self, context, obj):
        if self.frame_end < self.frame_start:
            self.report({'ERROR'}, 'End frame must be at or after the start frame')
            return {'CANCELLED'}
        bones = list(context.selected_pose_bones or []) if self.only_selected_bones else list(obj.pose.bones)
        if not bones:
            self.report({'ERROR'}, 'No selected bones to bake. Select bones or turn off Only Selected Bones.')
            return {'CANCELLED'}
        # Leave the source action attached during sampling. Blender creates the new action
        # and its slot AFTER sampling when use_current_action=False. Detaching it first loses
        # any motion (fingers, controls, etc.) supplied by the existing action.
        name = self.action_name.strip()
        previous = obj.animation_data.action if obj.animation_data else None
        previous_slot = getattr(obj.animation_data, 'action_slot', None)
        previous_fake = previous.use_fake_user if previous else False
        if name and previous:
            previous.use_fake_user = True
        try:
            result = bpy.ops.nla.bake(
                frame_start=self.frame_start,
                frame_end=self.frame_end,
                only_selected=self.only_selected_bones,
                visual_keying=True,
                clear_constraints=False,
                use_current_action=not bool(name),
                bake_types={'POSE'}
            )
            if 'FINISHED' not in result:
                raise RuntimeError('Blender did not bake an action')
            baked = obj.animation_data.action if obj.animation_data else None
            if baked is None:
                raise RuntimeError('Blender returned no baked action')
            try:
                from bpy_extras.anim_utils import action_get_channelbag_for_slot
                bag = action_get_channelbag_for_slot(baked, obj.animation_data.action_slot)
                curves = bag.fcurves if bag else []
            except ImportError:
                curves = getattr(baked, 'fcurves', [])
            keyed_paths = {fc.data_path for fc in curves if len(fc.keyframe_points)}
            missing = [pb.name for pb in bones if pb.path_from_id('location') not in keyed_paths]
            if missing:
                raise RuntimeError(f'Bake produced no location keys for {len(missing)} requested bones')
        except Exception as error:
            if obj.animation_data and name:
                obj.animation_data.action = previous
                if previous_slot is not None:
                    obj.animation_data.action_slot = previous_slot
            if previous:
                previous.use_fake_user = previous_fake
            self.report({'ERROR'}, f'Bake failed: {error}')
            return {'CANCELLED'}

        if name and baked is not None:
            baked.name = name              # Blender adds .001 if the name is taken
            baked.use_fake_user = True
        # The baked pose replaces the mocap drivers. Keep helpers for re-enabling/rebaking,
        # and keep control-rig constraints intact; only the baked bones' OM constraints mute.
        muted = 0
        for pb in bones:
            for constraint in pb.constraints:
                if constraint.name.startswith('OM ') and not constraint.mute:
                    constraint.mute = True
                    muted += 1
        if not self.only_selected_bones:
            obj.animation_data.use_nla = False  # the visual bake already includes the evaluated stack
        if self.in_place and baked is not None:
            if not keep_in_place(obj):
                self.report({'WARNING'}, "In Place: no baked location keys on the hips, nothing changed")
        context.scene.frame_set(context.scene.frame_current, subframe=context.scene.frame_subframe)
        scope = 'Selected bones' if self.only_selected_bones else 'Full rig'
        self.report({'INFO'}, f"{scope}: baked {len(bones)} bones into '{baked.name}'; muted {muted} mocap constraints.")
        return {'FINISHED'}



# -------------------------------------------------------------
# HELPER OPERATORS
# -------------------------------------------------------------
class ClearMoCapCache(bpy.types.Operator):
    bl_idname = "object.clear_mocap_cache"
    bl_label = "Clear Capture Cache"
    bl_description = "Clears the captured motion data from the temporary cache."

    cache: bpy.props.StringProperty()

    """
        :param cache: If cache = "pose", then all motion data for pose tracked recently will be cleared.
                        Else all motion data for hand tracked recently will be cleared.
    """

    def execute(self,context):
        if self.cache == "pose":
            gvar.positionList.clear()
            gvar.zlist.clear()
            for lst in (gvar.position_times, gvar.position_vis, gvar.zlist_times, gvar.zlist_vis,
                        gvar.position_meta, gvar.zlist_meta, gvar.position_missing_times, gvar.zlist_missing_times):
                lst.clear()
            gvar.positionList_raw = None
            gvar.position_meta_raw = None
            gvar.object_list.clear()
            gvar.bones_list.clear()
            for lst in (gvar.R_body_hand_positionList, gvar.L_body_hand_positionList,
                        gvar.R_body_hand_times, gvar.L_body_hand_times):
                lst.clear()
            self.report({'INFO'}, "Body Motion Capture data cleared.")
        else:
            gvar.R_hand_positionList.clear()
            gvar.R_hand_zlist.clear()
            gvar.R_hand_object_list.clear()
            gvar.R_hand_bones_list.clear()
            gvar.L_hand_positionList.clear()
            gvar.L_hand_zlist.clear()
            gvar.L_hand_object_list.clear()
            gvar.L_hand_bones_list.clear()
            for lst in (gvar.R_hand_times, gvar.L_hand_times, gvar.R_hand_ztimes, gvar.L_hand_ztimes):
                lst.clear()
            self.report({'INFO'}, "Hand Motion Capture data cleared.")
        return {'FINISHED'}



# Tracked data lives in memory only and is gone when Blender closes or the add-on reloads. These two
# write it to a file and read it back, so a take can be combined and animated again without tracking.
CAPTURE_FORMAT = "open_mocap_capture"
CAPTURE_LISTS = ("positionList", "zlist", "position_times", "position_vis", "zlist_times", "zlist_vis",
                 "position_meta", "zlist_meta", "position_missing_times", "zlist_missing_times",
                 "R_hand_positionList", "L_hand_positionList", "R_hand_zlist", "L_hand_zlist",
                 "R_hand_times", "L_hand_times", "R_hand_ztimes", "L_hand_ztimes",
                 "R_body_hand_positionList", "L_body_hand_positionList", "R_body_hand_times", "L_body_hand_times")
CAPTURE_VALUES = ("fps", "front_fps", "side_fps", "hand_fps", "hand_front_fps", "hand_side_fps", "front_video_path", "side_video_path",
                  "hand_front_video_path", "hand_side_video_path")


class SaveCapture(bpy.types.Operator, ExportHelper):
    bl_idname = "object.open_mocap_save_capture"
    bl_label = "Save Capture"
    bl_description = ("Save the tracked pose and hand data (front and side views, before and after "
                      "Combine Data) to a file, to load it again in another Blender session")

    filename_ext = ".json"
    filter_glob: bpy.props.StringProperty(default="*.json", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return any(getattr(gvar, name) for name in CAPTURE_LISTS)

    def invoke(self, context, event):
        video = gvar.front_video_path or gvar.hand_front_video_path
        if video and not self.filepath:
            self.filepath = os.path.splitext(video)[0] + "_capture.json"
        return super().invoke(context, event)

    def execute(self, context):
        data = {"format": CAPTURE_FORMAT, "version": 2}
        for name in CAPTURE_LISTS + CAPTURE_VALUES:
            data[name] = getattr(gvar, name)
        data["positionList_raw"] = gvar.positionList_raw
        data["position_meta_raw"] = gvar.position_meta_raw
        try:
            capture_data.validate_pose_capture(data)
            encoded = json.dumps(data, allow_nan=False)
            with open(self.filepath, "w", encoding="utf-8") as f:
                f.write(encoded)
        except (OSError, ValueError) as e:
            self.report({'ERROR'}, f"Could not save the capture: {e}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Saved {len(gvar.positionList)} front and {len(gvar.zlist)} side pose frames "
                              f"to {self.filepath}")
        return {'FINISHED'}


class LoadCapture(bpy.types.Operator, ImportHelper):
    bl_idname = "object.open_mocap_load_capture"
    bl_label = "Load Capture"
    bl_description = "Load tracked data saved with Save Capture. Replaces the tracked data in memory"

    filename_ext = ".json"
    filter_glob: bpy.props.StringProperty(default="*.json", options={'HIDDEN'})

    def execute(self, context):
        try:
            with open(self.filepath, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            self.report({'ERROR'}, f"Could not read the capture: {e}")
            return {'CANCELLED'}
        if not isinstance(data, dict) or data.get("format") != CAPTURE_FORMAT:
            self.report({'ERROR'}, "This is not an Open Mocap capture file")
            return {'CANCELLED'}
        try:
            capture_data.validate_pose_capture(data)
        except ValueError as error:
            self.report({'ERROR'}, f'Invalid capture: {error}')
            return {'CANCELLED'}
        # The lists are filled in place: other parts of the add-on hold on to them
        for name in CAPTURE_LISTS:
            value = data.get(name)
            target = getattr(gvar, name)
            target.clear()
            if isinstance(value, list):
                target.extend(value)
        for name in CAPTURE_VALUES:
            setattr(gvar, name, data.get(name, '' if name.endswith('_path') else 0))
        raw = data.get("positionList_raw")
        gvar.positionList_raw = raw if isinstance(raw, list) else None
        raw_meta = data.get('position_meta_raw')
        gvar.position_meta_raw = raw_meta if isinstance(raw_meta, list) else None
        if gvar.fps:
            gvar.delay = 1.0 / gvar.fps
        self.report({'INFO'}, f"Loaded {len(gvar.positionList)} front and {len(gvar.zlist)} side pose frames, "
                              f"{len(gvar.R_hand_positionList)} right and {len(gvar.L_hand_positionList)} "
                              "left hand frames")
        return {'FINISHED'}



class VideoSelect(bpy.types.Operator):
    bl_idname = "object.open_mocap_filebrowser"
    bl_label = "Open Video File"
    bl_description = "Selects a video file from your disk."

    video: bpy.props.StringProperty()

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    
    filter_glob: bpy.props.StringProperty(
        default="*.mp4;*.avi;*.mov;*.mkv;*.flv;*.wmv",
        options={'HIDDEN'}
    )

    """
        :param video: Tells if video is selected for hand or pose tracking and whether it is front view or side view.
        :param filepath: Stores file path of the selected video.
        :param filter_glob: Allows only common video formats.
    """
    def execute(self, context):
        # Validate the selected file extension
        allowed_exts = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")
        if not self.filepath.lower().endswith(allowed_exts):
            self.report({'ERROR'}, "Please select a valid video file.")
            return {'CANCELLED'}

        if self.video == "front_video_pose":
            gvar.front_video_path = self.filepath
        elif self.video == "side_video_pose":
            gvar.side_video_path = self.filepath
        elif self.video == "front_video_hand":
            gvar.hand_front_video_path = self.filepath
        else:
            gvar.hand_side_video_path = self.filepath

        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}



class ClearVideoPath(bpy.types.Operator):
    bl_idname = "object.clear_video_path"
    bl_label = "Clear Video Path"
    bl_description = "Deletes video path."

    video: bpy.props.StringProperty()
    """
        :param video: Tells which video panel's video path to delete.
    """

    def execute(self, context):
        if self.video == "front_video_pose":
            gvar.front_video_path = ""
        elif self.video == "side_video_pose":
            gvar.side_video_path = ""
        elif self.video == "front_video_hand":
            gvar.hand_front_video_path = ""
        else:
            gvar.hand_side_video_path = ""
        return {'FINISHED'}



class ToggleCopyRotationConstraints(bpy.types.Operator):
    bl_idname = "pose.toggle_copy_rotation"
    bl_label = "Toggle Constraints"
    bl_description = "Enable or disable all bone constraints on the selected rig"

    target_rig: bpy.props.StringProperty(name="Target Rig")

    """
        :param target_rig: Toggles bone constraints of all the bones of the target_rig.
    """

    def execute(self, context):
        obj = bpy.data.objects.get(self.target_rig)

        if not obj or obj.type != 'ARMATURE':
            self.report({'WARNING'}, "Please select a valid armature rig.")
            return {'CANCELLED'}

        # The "OM ..." constraints made by Retarget to Mixamo. Only on rigs without them, any copy
        # rotation/location (the older manual setup); a control rig's own copy constraints must stay.
        has_om = any(c.name.startswith("OM ") for bone in obj.pose.bones for c in bone.constraints)

        def is_mocap_constraint(c):
            if has_om:
                return c.name.startswith("OM ")
            return c.type in {'COPY_ROTATION', 'COPY_LOCATION'}

        # Check if at least one constraint is currently active (unmuted)
        any_enabled = any(
            is_mocap_constraint(c) and not c.mute
            for bone in obj.pose.bones
            for c in bone.constraints
        )

        # Determine target state (True = enable, False = disable)
        enable = not any_enabled
        count = 0

        # Toggle constraints
        for bone in obj.pose.bones:
            for c in bone.constraints:
                if is_mocap_constraint(c):
                    c.mute = not enable
                    count += 1

        state = "enabled" if enable else "disabled"
        self.report({'INFO'}, f"{count} mocap constraints {state}.")
        return {'FINISHED'}
    


class HideSkeleton(bpy.types.Operator):
    bl_idname = "object.hide_skeleton"
    bl_label = "Hide or Unhide Skeleton"
    bl_description = "Hides or Unhides Bones or Empties of Skeleton"

    collection_name: bpy.props.StringProperty()
    #collection_name = "PoseEmpties", "PoseSkeleton", "RightHandEmpties", "RightHandSkeleton", "LeftHandEmpties", "LeftHandSkeleton" 

    def execute(self, context):
        collection = bpy.data.collections.get(self.collection_name)
        if collection:
            for obj in collection.objects:
                obj.hide_set(not obj.hide_get())
        return {'FINISHED'}



classes = [StartTracking, StopTracking, CombineMotionData, AnimatePose, AnimateHand, BakeMotion,
           ClearMoCapCache, SaveCapture, LoadCapture, VideoSelect,  ClearVideoPath,
           ToggleCopyRotationConstraints, HideSkeleton]
