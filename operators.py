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
import queue
import subprocess
import threading
from bpy_extras.io_utils import ExportHelper, ImportHelper
from . import utils
from . import combine
from . import capture_data
from . import gemx
from . import layers
from . import PackageInstaller
from . import smoothing
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

        elif self.track == "extra_pose":
            if not gvar.extra_video_path:
                self.report({'ERROR'}, "Select the video of the further view first")
                return {'CANCELLED'}
            gvar.pose_tracker = utils.PoseTracking("extra_pose")

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
    


class TrackSword(bpy.types.Operator):
    bl_idname = "object.open_mocap_track_sword"
    bl_label = "Track Sword"
    bl_description = ("Find the sword blade in every tracked video (front, side and further views). Combine Data "
                      "turns it into one 3D direction and the retarget turns the hands to it. Needs a camera "
                      "that stands still, and the pose tracked first")

    @classmethod
    def poll(cls, context):
        return bool(gvar.positionList or gvar.positionList_raw)

    def execute(self, context):
        if not utils.load_libraries():
            self.report({'ERROR'}, "OpenCV is not installed: install the dependencies in the add-on preferences")
            return {'CANCELLED'}
        from . import sword
        front = gvar.positionList_raw if gvar.positionList_raw is not None else gvar.positionList
        jobs = []
        if front and len(gvar.position_times) == len(front) and len(gvar.position_vis) == len(front):
            jobs.append(("front video", gvar.front_video_path, front, gvar.position_vis, gvar.position_times, "front"))
        if gvar.zlist and len(gvar.zlist_times) == len(gvar.zlist) and len(gvar.zlist_vis) == len(gvar.zlist):
            jobs.append(("side video", gvar.side_video_path, gvar.zlist, gvar.zlist_vis, gvar.zlist_times, "side"))
        for view in gvar.extra_views:
            if view.get("frames") and len(view["times"]) == len(view["frames"]) == len(view["vis"]):
                jobs.append((view.get("name", "view"), view.get("path", ""), view["frames"], view["vis"], view["times"], view))
        if not jobs:
            self.report({'ERROR'}, "Track the pose in at least one video first (with timestamps and confidence)")
            return {'CANCELLED'}
        wm = context.window_manager
        wm.progress_begin(0, 100 * len(jobs))
        lines, problems = [], []
        try:
            for k, (name, path, frames, vis, times, target) in enumerate(jobs):
                if not path or not os.path.isfile(path):
                    problems.append(f"{name}: video not found ({path or 'no path'})")
                    continue
                try:
                    rows = sword.track_video(path, frames, vis, times,
                                             progress=lambda share, k=k: wm.progress_update(int(100 * (k + share))))
                except (ValueError, OSError) as error:
                    problems.append(f"{name}: {error}")
                    continue
                if target == "front":
                    gvar.sword_front[:] = rows
                elif target == "side":
                    gvar.sword_side[:] = rows
                else:
                    target["sword"] = rows
                seen = sum(r[2] > 0.5 for r in rows) / max(len(rows), 1)
                lines.append(f"{name} {seen:.0%}")
                print(f"[Open Mocap]   sword in {name}: clearly seen in {seen:.0%} of {len(rows)} frames")
        finally:
            wm.progress_end()
        gvar.sword_axis.clear()           # until Combine Data fuses the new directions
        if not lines:
            self.report({'ERROR'}, "No sword tracked: " + "; ".join(problems))
            return {'CANCELLED'}
        msg = "Sword tracked, clearly seen in: " + ", ".join(lines)
        if problems:
            msg += ". Skipped: " + "; ".join(problems)
        msg += ". Run Combine Data to give it depth" if (gvar.zlist or gvar.extra_views) else ""
        self.report({'WARNING'} if problems else {'INFO'}, msg)
        return {'FINISHED'}


def sword_rows_for_animate(count):
    """[x, y, depth, confidence] per front frame: Combine Data's 3D blade, else the front picture's."""
    if len(gvar.sword_axis) == count:
        return [list(r) for r in gvar.sword_axis]
    if len(gvar.sword_front) == count:
        return [[r[0], r[1], 0.0, 0.5 * r[2]] for r in gvar.sword_front]
    return []


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
        # Further views: the side video (if there is one) first, then every other tracked view
        more = [v for v in gvar.extra_views
                if len(v["frames"]) >= 10 and len(v["times"]) == len(v["frames"]) and len(v["vis"]) == len(v["frames"])]
        if not front or not (side or more):
            self.report({'ERROR'}, "Track the front video and at least one other view first")
            return {'CANCELLED'}

        have_meta = (len(gvar.position_times) == len(front) and len(gvar.position_vis) == len(front)
                     and len(gvar.zlist_times) == len(side) and len(gvar.zlist_vis) == len(side))
        method = scene.combine_method
        if method != 'IMPROVED' and not side:
            self.report({'ERROR'}, "The original method needs the side video; further views need the improved method")
            return {'CANCELLED'}
        if method == 'IMPROVED' and have_meta and len(front) >= 10 and (len(side) >= 10 or (not side and more)):
            views = ([{"frames": side, "times": gvar.zlist_times, "vis": gvar.zlist_vis, "name": "Side video",
                       "sword": gvar.sword_side}]
                     if side else []) + more
            try:
                combined, rep = combine.combine_multi(
                    front, views, gvar.position_times, gvar.position_vis,
                    smoothing=scene.combine_smoothing, bone_strength=scene.combine_bone_strength,
                    smoothing_method=scene.combine_smoothing_method,
                    offset=None if (scene.combine_auto_offset or not side) else scene.combine_offset,
                    sync=scene.combine_sync,
                    side_angle=None if (scene.combine_auto_angle or not side) else scene.combine_side_angle,
                    front_depth_confidence=capture_data.depth_confidence(metadata, gvar.position_vis) if metadata else None)
            except ValueError as error:
                self.report({'ERROR'}, str(error))
                return {'CANCELLED'}
            view_lines = combine.format_views(rep) if len(views) > 1 else []
            for line in view_lines:
                print("[Open Mocap]  ", line)
            if not rep.get('combined', True):
                self.report({'WARNING'}, 'Combine skipped: ' + combine.format_report(rep)
                            + ("  " + "; ".join(view_lines) if view_lines else ""))
                return {'CANCELLED'}
            combined_meta = rep['metadata']
            sword_line = ""
            gvar.sword_axis.clear()
            if len(gvar.sword_front) == len(front) and front:
                axis, srep = combine.sword_axis(front, gvar.position_times, gvar.sword_front, views, combined, rep,
                                                gvar.position_vis)
                gvar.sword_axis.extend(axis)
                depth_from = ", ".join(v["name"] for v in srep["views"]) or "no further view (blade kept in the front picture's plane)"
                sword_line = (f". Sword: seen in {srep['seen']:.0%} of front frames, depth from {depth_from} "
                              f"({srep['with_depth']:.0%} of frames)")
            if getattr(scene, "combine_grip", False):
                vis = gvar.position_vis[:len(combined)]
                reliability = [[min(v, c) for v, c in zip(vr, m['depth_confidence'])]
                               for vr, m in zip(vis, combined_meta)]
                combined, grep = combine.two_hand_grip(combined, reliability, gvar.sword_axis or None,
                                                       gvar.front_fps or gvar.fps or 30.0)
                sword_line += (f". Hands on one handle {grep['spacing']:.2f} shoulder widths apart (tracked: median "
                               f"{grep['gap_before']:.2f}), moved in {grep['moved']:.0%} of frames")
            if scene.combine_perspective:
                # A camera of unknown make: a horizontal field of view stands in for its focal length
                width, height = gvar.front_width or 1920, gvar.front_height or 1080
                focal = combine.focal_from_fov(width, scene.combine_fov)
                fps = gvar.front_fps or gvar.fps or 30.0
                fov_line = f"{scene.combine_fov:.0f} deg field of view"
                if getattr(scene, "combine_auto_fov", False):
                    lo, hi = combine.FOV_RANGE
                    limits = (combine.focal_from_fov(width, hi), combine.focal_from_fov(width, lo))
                    focal, frep = combine.fit_focal(combined, focal, fps, limits)
                    if frep["focal_found"]:
                        scene.combine_fov = combine.fov_from_focal(width, focal)   # start manual tweaking from it
                        fov_line = (f"{scene.combine_fov:.0f} deg field of view found from the actor's size "
                                    f"(it changes {frep['focal_size_change']:.0%} with distance)")
                    else:
                        fov_line += f" (not found: {frep['focal_reason']})"
                horizon, hrep = -height / 200.0, {}
                whole = scene.combine_whole_take
                if scene.combine_auto_horizon:
                    horizon, hrep = combine.fit_horizon(combined, focal, fps, default=horizon, whole_take=whole)
                combined, prep = combine.correct_perspective(
                    combined, focal, horizon=horizon, confidence=[m['depth_confidence'] for m in combined_meta],
                    bone_strength=scene.combine_bone_strength, fps=fps, whole_take=whole)
                lo, hi = prep["perspective_factor"]
                # the horizon as a share of the picture's height above its bottom edge
                samples = "standing feet" if whole else "frames with both feet standing"
                where = (f"horizon {-horizon * 100.0 / height:.0%} up the picture, found from "
                         f"{hrep['horizon_samples']} {samples}" if hrep.get("horizon_found")
                         else "horizon at the middle of the picture"
                         + (f" ({hrep['horizon_reason']})" if hrep.get("horizon_reason") else ""))
                print(f"[Open Mocap]   perspective corrected ({fov_line}, {where}, "
                      f"{'whole take at one distance, ' if whole else ''}size factor {lo:.2f}..{hi:.2f})")
            if scene.combine_auto_offset and side:
                scene.combine_offset = rep["offset_s"]   # start manual tweaking from what was found
            if scene.combine_auto_angle and side:
                scene.combine_side_angle = rep["side_angle"]
            msg = "Combined: " + combine.format_report(rep)
            if view_lines:
                msg += f". {rep.get('views_for_depth', 0)} of {len(views)} views measure depth: " + "; ".join(view_lines)
            msg += sword_line
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
            gvar.sword_axis.clear()       # the sword is fused by the improved method only
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
        # --- A collection of its own for this take (root and empties, skeleton) ---
        video = os.path.splitext(os.path.basename(gvar.front_video_path or ""))[0]
        _take, pose_empties_collection, skeleton_collection = utils.BlenderUtility.create_take_collections(
            context, f"Mocap {video}" if video else "Mocap")

        # --- Create root empty ---
        root = bpy.data.objects.new("RigRoot", None)
        root.empty_display_type = 'SPHERE'
        root.empty_display_size = 1
        pose_empties_collection.objects.link(root)
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
        bones_created = utils.Skeleton.create_skeleton(context, bone_pairs, "pose", skeleton_collection)
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
                dims = 3 if gvar.body_hands_3d else 2          # GEM-X hands have real depth
                hands[side] = {'frames': [float(t * fps) + 1 for t in hand_times], 'dims': dims,
                               'points': [[v for i in range(21) for v in f[3 * i:3 * i + dims]] for f in hand_frames]}
        if hands:
            root['open_mocap_hands'] = json.dumps(hands)
        # GEM-X's spine chain (see gemx.SPINE_CHAIN), in the empties' coordinates, for the retarget
        spine = getattr(gvar, "body_spine", None)
        if spine and len(spine["points"]) == len(frame_numbers):
            pts = [[c for k in range(len(row) // 3)
                    for c in utils.MathUtility.cv2blender_coordinates(row[3 * k:3 * k + 3])] for row in spine["points"]]
            root['open_mocap_spine'] = json.dumps({'frames': frame_numbers, 'names': spine["names"], 'points': pts,
                                                   'rest': spine["rest"],
                                                   'rest_up': spine["rest_up"]})

        for frame_index, frame in zip(frame_numbers, gvar.positionList):
            for i in range(40):
                location = utils.MathUtility.cv2blender_coordinates([frame[0 + (i * 3)],frame[1 + (i * 3)],frame[2 + (i * 3)]])
                obj = gvar.object_list[i]
                local_pos = Vector(location) - gvar.root.location
                obj.location = local_pos
                obj.keyframe_insert(data_path="location", frame=frame_index)

        # Sword (Track Sword): its direction for the retarget, and an arrow from the grip to see it
        sword_rows = sword_rows_for_animate(len(frame_numbers))
        if sword_rows:
            dirs = [utils.MathUtility.cv2blender_coordinates(r[:3]) for r in sword_rows]
            root['open_mocap_sword'] = json.dumps({'frames': frame_numbers, 'dirs': dirs,
                                                   'confidence': [r[3] for r in sword_rows]})
            add_sword_arrow(root, pose_empties_collection, frame_numbers, gvar.positionList, dirs)

        start_playback(context)
        return {'FINISHED'}


def add_sword_arrow(root, collection, frame_numbers, frames, dirs):
    """An arrow empty along the blade, from the centre of the hand points, keyed per frame."""
    pts = [[utils.MathUtility.cv2blender_coordinates(f[3 * i:3 * i + 3]) for i in (0, 27, 28, 15, 16, 17, 18, 19, 20)]
           for f in frames]
    heights = sorted(p[0][2] - (p[1][2] + p[2][2]) / 2 for p in pts)
    arrow = bpy.data.objects.new("Sword", None)
    arrow.empty_display_type = 'SINGLE_ARROW'
    arrow.empty_display_size = max(0.4 * abs(heights[len(heights) // 2]), 0.1)   # a blade is ~0.4 body heights
    arrow.rotation_mode = 'QUATERNION'
    collection.objects.link(arrow)
    arrow.parent = root
    previous = None
    for frame_index, p, d in zip(frame_numbers, pts, dirs):
        grip = Vector([sum(q[k] for q in p[3:]) / 6.0 for k in range(3)])
        quat = Vector((0.0, 0.0, 1.0)).rotation_difference(Vector(d))
        if previous is not None:
            quat.make_compatible(previous)
        previous = quat
        arrow.location = grip
        arrow.rotation_quaternion = quat
        arrow.keyframe_insert(data_path="location", frame=frame_index)
        arrow.keyframe_insert(data_path="rotation_quaternion", frame=frame_index)



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
            gvar.live_hands = smoothing.LiveHands(context.scene.hand_live_smoothing)

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

        # Each hand's latest points, steadied by the live filter (smoothing.py) once per new detection
        live = gvar.live_hands
        if live is not None:
            live.set_strength(bpy.context.scene.hand_live_smoothing)
        for side, points, times, objects, root in (
                ("R", gvar.rt_hand_lmlist_R, gvar.R_hand_times, gvar.R_hand_object_list, gvar.R_hand_root),
                ("L", gvar.rt_hand_lmlist_L, gvar.L_hand_times, gvar.L_hand_object_list, gvar.L_hand_root)):
            if not objects or not points:
                continue
            if live is not None and times:
                points = live.points(side, points, times[-1])
            for obj, point in zip(objects, points):
                obj.location = Vector(utils.MathUtility.cv2blender_coordinates(point)) - root.location
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
    layers: bpy.props.EnumProperty(
        name="Layers", items=layers.LAYER_MODES, default='BASE',
        description="Where the bake goes on a rig with animation layers (NLA tracks)")

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
        # A layer being edited (NLA tweak mode) would take the bake into its own action
        layers.leave_tweak(context, obj)
        # With animation layers the mocap is baked on its own (layers off) into a new action that
        # then becomes a layer (layers.py)
        layered = self.layers != 'ACTION' and layers.has_layers(obj)
        layer_name = self.action_name.strip() or "Mocap"
        # Leave the source action attached during sampling. Blender creates the new action
        # and its slot AFTER sampling when use_current_action=False. Detaching it first loses
        # any motion (fingers, controls, etc.) supplied by the existing action.
        name = self.action_name.strip()
        if layered and not name:
            name = f"{obj.name}_{layers.unique_name(obj.animation_data, layer_name)}"
        previous = obj.animation_data.action if obj.animation_data else None
        previous_slot = getattr(obj.animation_data, 'action_slot', None)
        previous_fake = previous.use_fake_user if previous else False
        nla_was = obj.animation_data.use_nla if layered else None
        if name and previous:
            previous.use_fake_user = True
        try:
            if layered:
                obj.animation_data.use_nla = False
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
            if obj.animation_data and nla_was is not None:
                obj.animation_data.use_nla = nla_was
            if previous:
                previous.use_fake_user = previous_fake
            self.report({'ERROR'}, f'Bake failed: {error}')
            return {'CANCELLED'}

        if name and baked is not None:
            baked.name = name              # Blender adds .001 if the name is taken
            baked.use_fake_user = True
        # Bind tracker evidence to the take while the retarget is driving it. Unrelated
        # actions and imported FBXs use the boot mesh for contact detection.
        if not self.only_selected_bones and any(
                "Leg" in b.name and c.name.startswith("OM ") and not c.mute for b in bones for c in b.constraints):
            if obj.get("open_mocap_foot_contacts"):
                baked["open_mocap_foot_contacts"] = obj["open_mocap_foot_contacts"]
            elif "open_mocap_foot_contacts" in baked:
                del baked["open_mocap_foot_contacts"]
        # The baked pose replaces the mocap drivers. Keep helpers for re-enabling/rebaking,
        # and keep control-rig constraints intact; only the baked bones' OM constraints mute.
        muted = 0
        for pb in bones:
            for constraint in pb.constraints:
                if constraint.name.startswith('OM ') and not constraint.mute:
                    constraint.mute = True
                    muted += 1
        if not self.only_selected_bones and not layered:
            # The visual bake already includes the evaluated NLA stack, so its tracks go off. A rig
            # without tracks keeps NLA on: turned off, layers made from the bake later (NLA Anim
            # Layers) played nothing, and the edited layer froze the bones
            if layers.has_layers(obj):
                obj.animation_data.use_nla = False
            layers.forget(obj)
        if self.in_place and baked is not None:
            if not keep_in_place(obj):
                self.report({'WARNING'}, "In Place: no baked location keys on the hips, nothing changed")
        where = ""
        if layered:
            track, old_base = layers.add_to_layers(context, obj, baked, self.layers, layer_name,
                                                   full=not self.only_selected_bones)
            where = f" as layer '{track.name}'"
            if old_base is not None:
                where += f" (old base '{old_base.name}' muted)"
            if previous is not None and previous != baked:
                where += f"; the earlier active action '{previous.name}' is kept with a fake user"
        context.scene.frame_set(context.scene.frame_current, subframe=context.scene.frame_subframe)
        scope = 'Selected bones' if self.only_selected_bones else 'Full rig'
        self.report({'INFO'}, f"{scope}: baked {len(bones)} bones into '{baked.name}'{where}; "
                              f"muted {muted} mocap constraints.")
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
            gvar.extra_views.clear()
            gvar.object_list.clear()
            gvar.bones_list.clear()
            for lst in (gvar.R_body_hand_positionList, gvar.L_body_hand_positionList,
                        gvar.R_body_hand_times, gvar.L_body_hand_times,
                        gvar.sword_front, gvar.sword_side, gvar.sword_axis):
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
                 "R_body_hand_positionList", "L_body_hand_positionList", "R_body_hand_times", "L_body_hand_times",
                 "sword_front", "sword_side", "sword_axis")
CAPTURE_VALUES = ("fps", "front_fps", "side_fps", "front_width", "front_height", "hand_fps", "hand_front_fps", "hand_side_fps", "front_video_path", "side_video_path",
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
        data["extra_views"] = gvar.extra_views
        data["extra_video_path"] = gvar.extra_video_path
        try:
            capture_data.validate_pose_capture(data)
            encoded = json.dumps(data, allow_nan=False)
            with open(self.filepath, "w", encoding="utf-8") as f:
                f.write(encoded)
        except (OSError, ValueError) as e:
            self.report({'ERROR'}, f"Could not save the capture: {e}")
            return {'CANCELLED'}
        more = f", {len(gvar.extra_views)} further views" if gvar.extra_views else ""
        self.report({'INFO'}, f"Saved {len(gvar.positionList)} front and {len(gvar.zlist)} side pose frames{more} "
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
            views = capture_data.validate_extra_views(data.get("extra_views", []))
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
        gvar.extra_views[:] = views
        gvar.extra_video_path = data.get("extra_video_path", "") if isinstance(data.get("extra_video_path"), str) else ""
        if gvar.fps:
            gvar.delay = 1.0 / gvar.fps
        self.report({'INFO'}, f"Loaded {len(gvar.positionList)} front and {len(gvar.zlist)} side pose frames, "
                              f"{len(gvar.extra_views)} further views, "
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
        elif self.video == "extra_video_pose":
            gvar.extra_video_path = self.filepath
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
        elif self.video == "extra_video_pose":
            gvar.extra_video_path = ""
        elif self.video == "front_video_hand":
            gvar.hand_front_video_path = ""
        else:
            gvar.hand_side_video_path = ""
        return {'FINISHED'}



class RemoveExtraView(bpy.types.Operator):
    bl_idname = "object.open_mocap_remove_view"
    bl_label = "Remove View"
    bl_description = "Remove this further view's tracked data (the video file is not touched)"

    index: bpy.props.IntProperty()

    def execute(self, context):
        if not 0 <= self.index < len(gvar.extra_views):
            return {'CANCELLED'}
        view = gvar.extra_views.pop(self.index)
        self.report({'INFO'}, f"Removed the view {view.get('name', '')}. Run Combine Data again to leave it out.")
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
        # The pose buttons act on the newest take's own collections (see create_take_collections);
        # scenes animated before takes had collections of their own still have the shared ones
        role = {"PoseEmpties": "Empties", "PoseSkeleton": "Skeleton"}.get(self.collection_name)
        collection = utils.BlenderUtility.newest_take_part(role) if role else None
        if collection is None:
            collection = bpy.data.collections.get(self.collection_name)
        if collection:
            for obj in collection.objects:
                obj.hide_set(not obj.hide_get())
        return {'FINISHED'}



class TrackGEMX(bpy.types.Operator):
    bl_idname = "object.open_mocap_track_gemx"
    bl_label = "Track with GEM-X"
    bl_description = ("Track the selected video with NVIDIA GEM-X in its own environment (add-on preferences): "
                      "whole-body 3D motion from this one video, fingers included, for a fixed camera. The "
                      "result replaces the tracked pose like Start Pose Tracking + Combine Data; press "
                      "Animate next. Esc cancels")

    _timer = None
    _proc = None
    _lines = None
    _out = ""

    @classmethod
    def poll(cls, context):
        return bool(gvar.front_video_path) and not gvar.gemx_status.endswith("...")

    def execute(self, context):
        missing = PackageInstaller.gemx_missing()
        if missing:
            self.report({'ERROR'}, "GEM-X is not set up (add-on preferences): missing " + ", ".join(missing))
            return {'CANCELLED'}
        video = bpy.path.abspath(gvar.front_video_path)
        if not os.path.isfile(video):
            self.report({'ERROR'}, f"Video not found: {video}")
            return {'CANCELLED'}
        folder = os.path.join(PackageInstaller.data_dir(), "gemx")
        os.makedirs(folder, exist_ok=True)
        self._out = os.path.join(folder, os.path.splitext(os.path.basename(video))[0] + "_gemx.npz")
        runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemx_runner.py")
        cmd = [PackageInstaller.gemx_python(), "-u", runner, "--repo", PackageInstaller.gemx_repo(),
               "--video", video, "--out", self._out]
        if context.scene.gemx_fov > 0:
            cmd += ["--fov", f"{context.scene.gemx_fov:.3f}"]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                      encoding="utf-8", errors="replace", creationflags=flags)
        self._lines = queue.Queue()
        threading.Thread(target=self._read, args=(self._proc, self._lines), daemon=True).start()
        gvar.gemx_status = "GEM-X: starting..."
        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    @staticmethod
    def _read(proc, lines):
        for line in proc.stdout:
            lines.put(line.rstrip())
        lines.put(None)

    def _status(self, context, text):
        gvar.gemx_status = text
        for area in context.screen.areas if context.screen else []:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

    def modal(self, context, event):
        if event.type == 'ESC':
            self._proc.kill()
            return self._finish(context, {'WARNING'}, "GEM-X cancelled")
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        result = None
        while True:
            try:
                line = self._lines.get_nowait()
            except queue.Empty:
                break
            if line is None:
                result = result or "ERROR GEM-X stopped without a result"
                break
            print("[Open Mocap] GEM-X: " + line)
            if line.startswith("STAGE "):
                self._status(context, "GEM-X: " + line[6:] + "...")
            elif line.startswith(("DONE ", "ERROR ")):
                result = line
        if result is None:
            return {'PASS_THROUGH'}
        if result.startswith("ERROR"):
            return self._finish(context, {'ERROR'}, result[6:])
        try:
            frames = load_gemx_capture(self._out)
        except (OSError, KeyError, ValueError) as error:
            return self._finish(context, {'ERROR'}, f"Could not read GEM-X's result: {error}")
        return self._finish(context, {'INFO'}, f"GEM-X tracked {frames} frames. Press Animate, then Retarget (Hip Height: From Floor, "
                                                   "GEM-X's floor is levelled)")

    def _finish(self, context, level, message):
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
        self._status(context, message)
        self.report(level, message)
        return {'FINISHED'} if 'INFO' in level else {'CANCELLED'}


def load_gemx_capture(path):
    """Fill the capture cache with a GEM-X take (see gemx.py), as tracking + Combine Data would."""
    import numpy as np
    with np.load(path) as data:
        take = gemx.to_capture({key: data[key] for key in data.files})
    for lst in (gvar.positionList, gvar.position_times, gvar.position_vis, gvar.position_meta,
                gvar.position_missing_times, gvar.R_body_hand_positionList, gvar.L_body_hand_positionList,
                gvar.R_body_hand_times, gvar.L_body_hand_times, gvar.sword_front, gvar.sword_axis):
        lst.clear()
    gvar.positionList.extend(take["frames"])
    gvar.position_times.extend(take["times"])
    gvar.position_vis.extend(take["vis"])
    gvar.position_meta.extend(take["meta"])
    gvar.positionList_raw = None
    gvar.position_meta_raw = None
    gvar.front_fps = gvar.fps = take["fps"]
    gvar.front_width, gvar.front_height = take["width"], take["height"]
    gvar.L_body_hand_positionList.extend(take["hands"]["Left"])
    gvar.R_body_hand_positionList.extend(take["hands"]["Right"])
    gvar.L_body_hand_times.extend(take["times"])
    gvar.R_body_hand_times.extend(take["times"])
    gvar.body_hands_3d = True
    gvar.body_spine = take["spine"]
    return len(take["frames"])


classes = [StartTracking, StopTracking, TrackSword, CombineMotionData, AnimatePose, AnimateHand, BakeMotion,
           ClearMoCapCache, SaveCapture, LoadCapture, VideoSelect,  ClearVideoPath, RemoveExtraView,
           ToggleCopyRotationConstraints, HideSkeleton, TrackGEMX]
