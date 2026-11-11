import time
import threading
import bpy
import numpy as np
from . import globalVariables as gvar
from . import PackageInstaller
from .capture_data import CaptureClock



# -------------------------------------------------------------
# IMPORTING EXTERNAL LIBRARIES
# -------------------------------------------------------------
# Imported lazily so the add-on can be enabled before the dependencies are installed,
# and picks them up right after "Install Dependencies" without restarting Blender.
cv2 = None
mp = None
mp_vision = None
mp_base_options = None


def load_libraries():
    """
        Import OpenCV and MediaPipe from the add-on's private package folder.
        :return: True if both are available.
    """
    global cv2, mp, mp_vision, mp_base_options
    if cv2 is not None and mp_vision is not None:
        return True
    PackageInstaller.add_lib_to_path()
    try:
        import cv2 as _cv2
        import mediapipe as _mp
        from mediapipe.tasks.python import vision as _vision
        from mediapipe.tasks.python import BaseOptions as _BaseOptions
    except ImportError as e:
        print(f"[Open Mocap] Could not import OpenCV/MediaPipe: {e}")
        return False
    cv2, mp, mp_vision, mp_base_options = _cv2, _mp, _vision, _BaseOptions
    return True



# -------------------------------------------------------------
# MEDIAPIPE TASKS DETECTORS
# Drop-in replacements for the cvzone PoseDetector / HandDetector that the add-on
# used with mediapipe 0.10.14. Landmarks are returned in the same format cvzone used:
# [int(x * width), int(y * height), int(z * width)] in pixels.
# -------------------------------------------------------------
def close_in_background(detector):
    """
        Close a MediaPipe detector without blocking Blender. MediaPipe 0.10.33+ uploads usage logs
        when a landmarker closes and waits for the upload (~1 s, ~40 s if the connection hangs).
    """
    def _close():
        try:
            detector.close()
        except Exception as e:  # noqa: BLE001
            print(f"[Open Mocap] Closing the tracker failed: {e}")
    threading.Thread(target=_close, daemon=True, name="OpenMocapCloseTracker").start()


class _TimestampClock:
    """VIDEO running mode needs strictly increasing timestamps in milliseconds."""

    def __init__(self, fps=None):
        self.fps = fps
        self.frame = 0
        self.last = -1
        self.start = time.monotonic()

    def next(self, seconds=None):
        if seconds is not None:
            ts = int(round(seconds * 1000.0))
        elif self.fps:
            ts = int(self.frame * 1000.0 / self.fps)
        else:
            ts = int((time.monotonic() - self.start) * 1000.0)
        self.frame += 1
        ts = max(ts, self.last + 1)
        self.last = ts
        return ts


def _capture_time(tracker):
    """Called immediately after acquisition, before detector initialization/inference."""
    if not hasattr(tracker, 'capture_clock'):
        tracker.capture_clock = CaptureClock(gvar.fps, tracker.mode.startswith('rt_'))
    pts = tracker.cap.get(cv2.CAP_PROP_POS_MSEC) if not tracker.mode.startswith('rt_') else None
    return tracker.capture_clock.next(pts)


def _to_mp_image(img_bgr):
    rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def _draw_landmarks(img, points, connections, line_color, point_color):
    px = [(int(round(p[0])), int(round(p[1]))) for p in points]      # landmarks are sub-pixel floats
    for c in connections:
        if c.start < len(px) and c.end < len(px):
            cv2.line(img, px[c.start], px[c.end], line_color, 2)
    for p in px:
        cv2.circle(img, p, 4, point_color, cv2.FILLED)


# Pose depth. MediaPipe gives two depths: the image landmarks' z, and world landmarks in metres around
# the hip centre. Both describe the same motion (correlation 0.99 on test footage); the world one
# jitters about 12% less and is in true proportion to x/y once scaled to the image, where the image z
# is about 4x too large (the add-on used to divide it by 400 instead of 100 for that reason).
# Landmark z is now the world depth in image pixels, so it shares the x/y scale (POSE_DEPTH_SCALE).
# A frame without world landmarks uses the image z brought to the same size (IMAGE_Z_TO_WORLD,
# measured 0.26..0.29).
POSE_DEPTH_SCALE = 100
IMAGE_Z_TO_WORLD = 0.27


def _world_depth(image_points, world_landmarks, visibility):
    """
        Depth of each landmark in image pixels from the world landmarks. The pixels-per-metre scale
        is the one that fits the world x/y onto this frame's image x/y (both about the hip centre).
        :return: list of 33 depths relative to the hip centre, or None.
    """
    P = np.asarray(image_points, dtype=np.float64)[:, :2]
    W = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks], dtype=np.float64)
    if len(P) != len(W) or len(P) < 25:
        return None
    w = np.asarray(visibility, dtype=np.float64) + 1e-3
    P = P - (P[23] + P[24]) / 2
    Wc = W - (W[23] + W[24]) / 2
    den = float(np.sum(w[:, None] * Wc[:, :2] ** 2))
    if den < 1e-9:
        return None
    scale = float(np.sum(w[:, None] * P * Wc[:, :2])) / den
    if not np.isfinite(scale) or scale <= 0:
        return None
    return (Wc[:, 2] * scale).tolist()


class PoseDetector:
    hands = {}          # only the RTMW tracker finds whole hands (see RTMWDetector)

    def __init__(self, fps=None, detection_con=0.5, track_con=0.5):
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_base_options(model_asset_path=PackageInstaller.model_path("pose")),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=detection_con,
            min_pose_presence_confidence=detection_con,
            min_tracking_confidence=track_con,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self.clock = _TimestampClock(fps)
        self.visibility = []
        self.depth_confidence = []
        self.xy_source = 'mediapipe'
        self.depth_source = 'missing'

    def find_pose(self, img, draw=True, timestamp=None):
        """
            :return: (img, landmarks). landmarks is a list of 33 [x, y, z] pixel
                     coordinates (floats), or an empty list if no person was found.
                     self.visibility then holds the 33 visibility scores (0..1).
                     z is the depth from the world landmarks, in pixels (see POSE_DEPTH_SCALE).
        """
        result = self.landmarker.detect_for_video(_to_mp_image(img), self.clock.next(timestamp))
        if not result.pose_landmarks:
            self.visibility = []
            self.depth_confidence = []
            self.depth_source = 'missing'
            return img, []
        h, w = img.shape[:2]
        landmarks = [[lm.x * w, lm.y * h, lm.z * w * IMAGE_Z_TO_WORLD] for lm in result.pose_landmarks[0]]
        self.visibility = [min(lm.visibility if lm.visibility is not None else 1.0,
                               lm.presence if lm.presence is not None else 1.0)
                           for lm in result.pose_landmarks[0]]
        world = getattr(result, "pose_world_landmarks", None)
        depth = _world_depth(landmarks, world[0], self.visibility) if world else None
        self.depth_source = 'mediapipe_world' if depth is not None else 'mediapipe_image'
        self.depth_confidence = [v * (1.0 if depth is not None else 0.5) for v in self.visibility]
        if depth is not None:
            for p, z in zip(landmarks, depth):
                p[2] = z
        if draw:
            _draw_landmarks(img, landmarks, mp_vision.PoseLandmarksConnections.POSE_LANDMARKS,
                            (255, 255, 255), (255, 0, 255))
        return img, landmarks

    def close(self):
        self.landmarker.close()


# COCO-WholeBody (133 points, RTMW) -> MediaPipe pose layout (33 points) used by the add-on.
# COCO body/feet/hands are the subject's own left/right, like MediaPipe.
_WB_LHAND, _WB_RHAND = 91, 112
_WB_TO_MP = [0, 1, 1, 1, 2, 2, 2, 3, 4, 23 + 54, 23 + 48,        # nose, eyes, ears, mouth corners
             5, 6, 7, 8, 9, 10,                                  # shoulders, elbows, wrists
             _WB_LHAND + 17, _WB_RHAND + 17,                     # pinky knuckles
             _WB_LHAND + 5, _WB_RHAND + 5,                       # index knuckles
             _WB_LHAND + 4, _WB_RHAND + 4,                       # thumb tips
             11, 12, 13, 14, 15, 16,                             # hips, knees, ankles
             19, 22, 17, 20]                                     # heels, big toes
# RTMW scores are not 0..1. Calibrated against MediaPipe visibility on test footage:
# ~2.4 -> <0.3, ~3.8 -> 0.5, ~4.9 -> 0.8, ~5.8 -> ~1.
_RTMW_SCORE_LOW, _RTMW_SCORE_RANGE = 2.0, 3.5
BODY_HAND_MIN_VIS = 0.3     # mean visibility of a hand's 21 points below which the hand is not recorded


class RTMWDetector:
    """
        RTMW whole-body pose (rtmlib, ONNX Runtime) for x/y, with MediaPipe running alongside for
        the depth estimate (RTMW has none). Same interface as PoseDetector.
    """

    def __init__(self, fps=None, device='dml'):
        import rtmlib
        from rtmlib.tools import base as rtm_base
        rtm_base.RTMLIB_SETTINGS['onnxruntime'].setdefault('dml', 'DmlExecutionProvider')
        self.model = rtmlib.Wholebody(
            det=PackageInstaller.rtmw_model_path("det"), det_input_size=(640, 640),
            pose=PackageInstaller.rtmw_model_path("pose"), pose_input_size=(288, 384),
            backend='onnxruntime', device=device)
        self.depth = PoseDetector(fps)
        self.previous = None            # hip centre of the tracked person, to stay on the same one
        self.visibility = []
        self.hands = {}                 # {"Left"/"Right": (21 [x, y] pixel points, 21 scores 0..1)}
        self.depth_confidence = []
        self.xy_source = 'rtmw'
        self.depth_source = 'missing'

    def _pick_person(self, keypoints, scores):
        body = keypoints[:, :23]
        if self.previous is not None and len(keypoints) > 1:
            hips = (keypoints[:, 11] + keypoints[:, 12]) / 2
            return int(np.argmin(np.linalg.norm(hips - self.previous, axis=1)))
        size = np.ptp(body[:, :, 0], axis=1) * np.ptp(body[:, :, 1], axis=1)
        return int(np.argmax(size * scores[:, :23].mean(axis=1)))

    def find_pose(self, img, draw=True, timestamp=None):
        """:return: (img, 33 [x, y, z] pixel landmarks or []), self.visibility like PoseDetector."""
        _img, mp_landmarks = self.depth.find_pose(img, draw=False, timestamp=timestamp)
        mp_visibility = self.depth.visibility
        keypoints, scores = self.model(img)
        self.hands = {}
        if len(keypoints) == 0:
            # RTMW found nobody: fall back to MediaPipe for this frame
            self.visibility = mp_visibility
            self.depth_confidence = list(self.depth.depth_confidence)
            self.xy_source = 'mediapipe'
            self.depth_source = self.depth.depth_source
            if draw and mp_landmarks:
                _draw_landmarks(img, mp_landmarks, mp_vision.PoseLandmarksConnections.POSE_LANDMARKS,
                                (255, 255, 255), (255, 0, 255))
            return img, mp_landmarks
        k = self._pick_person(keypoints, scores)
        pts, sc = keypoints[k][_WB_TO_MP], scores[k][_WB_TO_MP]
        self.previous = (keypoints[k][11] + keypoints[k][12]) / 2
        landmarks = [[float(x), float(y), (mp_landmarks[i][2] if mp_landmarks else 0.0)]
                     for i, (x, y) in enumerate(pts)]
        self.visibility = np.clip((sc - _RTMW_SCORE_LOW) / _RTMW_SCORE_RANGE, 0.0, 1.0).tolist()
        self.xy_source = 'rtmw'
        # Independent models can select different actors. Only borrow depth when their torsos agree.
        matched = False
        if mp_landmarks:
            torso = np.array([11, 12, 23, 24])
            mp_xy = np.asarray(mp_landmarks)[:, :2]
            size = max(float(np.linalg.norm(pts[torso[:2]].mean(axis=0) - pts[torso[2:]].mean(axis=0))), 1.0)
            reliable = np.minimum(np.asarray(self.visibility)[torso], np.asarray(mp_visibility)[torso]) > 0.3
            matched = reliable.sum() >= 3 and float(np.median(
                np.linalg.norm(pts[torso][reliable] - mp_xy[torso][reliable], axis=1))) < 0.35 * size
        self.depth_source = self.depth.depth_source if matched else 'missing'
        self.depth_confidence = ([min(a, b) for a, b in zip(self.visibility, self.depth.depth_confidence)]
                                 if matched else [0.0] * 33)
        if not matched:
            for point in landmarks:
                point[2] = 0.0
        # Whole hands: 21 points each, in MediaPipe's hand order (wrist, thumb, index ... pinky).
        # RTMW has no depth, so these are x/y only.
        for name, first in (("Left", _WB_LHAND), ("Right", _WB_RHAND)):
            conf = np.clip((scores[k][first:first + 21] - _RTMW_SCORE_LOW) / _RTMW_SCORE_RANGE, 0.0, 1.0)
            self.hands[name] = (keypoints[k][first:first + 21].astype(float).tolist(), conf.tolist())
        if draw:
            _draw_landmarks(img, landmarks, mp_vision.PoseLandmarksConnections.POSE_LANDMARKS,
                            (255, 255, 255), (0, 200, 0))
            for name, (points, conf) in self.hands.items():
                if float(np.mean(conf)) >= BODY_HAND_MIN_VIS:
                    _draw_landmarks(img, points, mp_vision.HandLandmarksConnections.HAND_CONNECTIONS,
                                    (200, 200, 200), (0, 200, 255))
        return img, landmarks

    def close(self):
        self.depth.close()
        self.model = None


def create_pose_detector(fps=None):
    """The pose tracker chosen in the add-on preferences (falls back to MediaPipe)."""
    if PackageInstaller.pose_tracker() == 'RTMW':
        missing = PackageInstaller.rtmw_missing()
        if not missing:
            try:
                det = RTMWDetector(fps, PackageInstaller.rtmw_device())
                print(f"[Open Mocap] Pose tracker: RTMW ({PackageInstaller.rtmw_device()}) + MediaPipe depth")
                return det
            except Exception as e:  # noqa: BLE001
                print(f"[Open Mocap] RTMW failed to start ({e}); using MediaPipe.")
        else:
            print(f"[Open Mocap] RTMW is not installed ({', '.join(missing)}); using MediaPipe. "
                  "Click Install Dependencies in the add-on preferences.")
    return PoseDetector(fps)


class HandDetector:
    def __init__(self, max_hands=2, fps=None, detection_con=0.5, track_con=0.5):
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_base_options(model_asset_path=PackageInstaller.model_path("hand")),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_con,
            min_hand_presence_confidence=detection_con,
            min_tracking_confidence=track_con,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self.clock = _TimestampClock(fps)

    def find_hands(self, img, draw=True, timestamp=None):
        """
            :return: (hands, img). Each hand is {"type": "Right"/"Left", "lmList": 21 [x, y, z]}.
        """
        result = self.landmarker.detect_for_video(_to_mp_image(img), self.clock.next(timestamp))
        h, w = img.shape[:2]
        hands = []
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            lm_list = [[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks]
            # The Tasks API reports the subject's own hand for a non-mirrored image,
            # which is what cvzone produced after its flipType swap.
            hand_type = handedness[0].category_name if handedness else "Right"
            hands.append({"type": hand_type, "lmList": lm_list})
            if draw:
                _draw_landmarks(img, lm_list, mp_vision.HandLandmarksConnections.HAND_CONNECTIONS,
                                (255, 255, 255), (255, 0, 255))
                xs = [p[0] for p in lm_list]
                ys = [p[1] for p in lm_list]
                x0, y0, x1, y1 = int(min(xs)) - 20, int(min(ys)) - 20, int(max(xs)) + 20, int(max(ys)) + 20
                cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 255), 2)
                cv2.putText(img, hand_type, (x0, y0 - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        return hands, img

    def close(self):
        self.landmarker.close()



# -------------------------------------------------------------
# SOME USEFUL MATH UTILITIES
# -------------------------------------------------------------
class MathUtility:
    @staticmethod
    def centroid(list_of_points):
        """
            :param list_of_points: List of points whose centroid is to be calculated.
            :return: Centroid of the points in the list.
        """
        avg_x = sum(p[0] for p in list_of_points) / len(list_of_points)
        avg_y = sum(p[1] for p in list_of_points) / len(list_of_points)
        avg_z = sum(p[2] for p in list_of_points) / len(list_of_points)
        return [avg_x,avg_y,avg_z]
    
    @staticmethod
    def normalize_coordinates(coordinates,height,width,scale):
        """
            :param coordinates: Original [x,y,z] (Unnormalized)
            :param height: Height of the video. 
            :param width: Width of the video.
            :param scale: Scales z to cover up for inaccuracy in z value tracked by the tracker.
            :return: Normalized [x,y,z] 
        """
        x = (coordinates[0] - (width/2))/100 #centre x
        y = (coordinates[1]-height)/100 #translate y up
        z = coordinates[2]/scale
        return [x,y,z]
    
    @staticmethod
    def cv2blender_coordinates(cv_coordinates):
        """
            :param cv_coordinates: [x,y,z] as per Opencv's convention.
            :return: [x,y,z] as per Blender's convention.
        """
        x = cv_coordinates[0]       #blender.x = cv2.x
        y = cv_coordinates[2]       #blender.y = cv2.z
        z = -cv_coordinates[1]      #blender.z = cv2.(-y)
        return [x,y,z]



# -------------------------------------------------------------
# SOME USEFUL BLENDER UTILITIES
# -------------------------------------------------------------
class BlenderUtility:   
    @staticmethod
    def create_collection(context,Name):
        collection_name = Name
        """
            Creates collections, with a given name, in the Blender's Outliner.
            :param collection_name: Name of the collection.
        """
        if collection_name in bpy.data.collections:
            named_collection = bpy.data.collections[collection_name]
        else:
            named_collection = bpy.data.collections.new(collection_name)
            context.scene.collection.children.link(named_collection)
        return named_collection
    


# -------------------------------------------------------------
# SOME CUSTOM UTILITIES
# -------------------------------------------------------------
class VideoPlaneManager:
    plane_obj = None
    image = None
    """
        Creates a plane object in Blender scene to display reatime tracking video.
        :param plane_obj: The plane object created for displaying.
        :param image: Frame from the video to be displayed in the plane.
    """

    @classmethod
    def create_plane(cls, context, width=640, height=480):
        cls.width, cls.height = width, height
        aspect_ratio = width / height

        """
            :param width: Width of the video stream. (Default = 640)
            :param height: Height of the video stream. (Default = 480)
        """

        # Remove old plane if exists
        if cls.plane_obj and cls.plane_obj.name in bpy.data.objects:
            bpy.data.objects.remove(cls.plane_obj, do_unlink=True)
            cls.plane_obj = None

        # Remove old image if exists
        if "CameraFeed" in bpy.data.images:
            bpy.data.images.remove(bpy.data.images["CameraFeed"], do_unlink=True)

        cls.image = bpy.data.images.new(
            "CameraFeed", width=width, height=height, alpha=True, float_buffer=False
        )

        # Create material with image texture
        mat = bpy.data.materials.new("CameraFeed_Mat")
        if bpy.app.version < (5, 0, 0):
            mat.use_nodes = True  # always on (and deprecated) since Blender 5.0
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        for n in list(nodes):
            nodes.remove(n)

        tex_image = nodes.new("ShaderNodeTexImage")
        tex_image.image = cls.image
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        output = nodes.new("ShaderNodeOutputMaterial")
        links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
        links.new(tex_image.outputs["Color"], bsdf.inputs["Base Color"])


        # Create plane and apply material
        bpy.ops.mesh.primitive_plane_add(size=4)
        cls.plane_obj = context.active_object
        cls.plane_obj.name = "VideoPlane"
        cls.plane_obj.data.materials.append(mat)
        cls.plane_obj.scale.x = aspect_ratio * 2
        cls.plane_obj.scale.y = 2
        cls.plane_obj.location.z += 2
        cls.plane_obj.location.y += 6
        #cls.plane_obj.location.x += 6
        cls.plane_obj.scale *= 0.5
        cls.plane_obj.rotation_euler[0] = np.radians(90)

        # Set 3D view to Material mode
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces[0].shading.type = 'MATERIAL'

        # Setup scene view transform and look
        try:
            bpy.context.scene.view_settings.view_transform = 'Standard'
            bpy.context.scene.view_settings.look = 'Medium High Contrast'
        except TypeError as e:
            print(f"[Open Mocap] Could not set the view transform: {e}")
        
        print("Video plane ready for realtime feed.")
        return cls.plane_obj
    
    @classmethod
    def remove_plane(cls):
        """Cleanly remove the video plane and image."""
        if cls.plane_obj and cls.plane_obj.name in bpy.data.objects:
            bpy.data.objects.remove(cls.plane_obj, do_unlink=True)
            cls.plane_obj = None

        if "CameraFeed" in bpy.data.images:
            bpy.data.images.remove(bpy.data.images["CameraFeed"], do_unlink=True)

        print("Video plane removed.")

    @classmethod
    def update_frame(cls, frame):
        """:param image: Frame from the video to be displayed in the plane."""

        if cv2 is None:
            print("OpenCV is not installed. Cannot update video plane.")
            return
        
        if cls.image is None:
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = np.flipud(frame_rgb)

        h, w, _ = frame_rgb.shape
        rgba = np.concatenate(
            [frame_rgb, np.ones((h, w, 1), dtype=np.uint8) * 255], axis=2
        ).astype(np.float32).ravel() / 255.0

        if cls.image.size[0] != w or cls.image.size[1] != h:
            cls.image.scale(w, h)

        cls.image.pixels.foreach_set(rgba)
        cls.image.update()



# -------------------------------------------------------------
# POSE TRACKING UTILITY
# -------------------------------------------------------------
class PoseTracking:
    def __init__(self, mode):
        self.detector = None
        self.mode = mode
        self.running = False
        """
            It uses the MediaPipe Tasks PoseLandmarker to do pose tracking.
        """
        if not load_libraries():
            print("OpenCV/MediaPipe are not installed! Install via the Addon Preferences first.")
            return

        # Determine video path or realtime mode
        if self.mode == "front_pose":
            self.path = gvar.front_video_path
        elif self.mode == "side_pose":
            self.path = gvar.side_video_path
        else:
            self.path = ""

        # A new take replaces the earlier one of the same view (appending a second run to the
        # first would double the take and restart its frame times half way)
        if self.mode == "side_pose":
            for lst in (gvar.zlist, gvar.zlist_times, gvar.zlist_vis, gvar.zlist_meta, gvar.zlist_missing_times):
                lst.clear()
        else:
            for lst in (gvar.positionList, gvar.position_times, gvar.position_vis,
                        gvar.position_meta, gvar.position_missing_times,
                        gvar.R_body_hand_positionList, gvar.L_body_hand_positionList,
                        gvar.R_body_hand_times, gvar.L_body_hand_times):
                lst.clear()
            gvar.positionList_raw = None
            gvar.position_meta_raw = None

        # --- Handle realtime webcam tracking ---
        if self.mode == "rt_pose":
            self.cap = cv2.VideoCapture(bpy.context.scene.cam_index)
            if not self.cap.isOpened():
                print("Failed to open camera.")
                return

            gvar.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if not gvar.fps or gvar.fps <= 0:
                gvar.fps = 30
            gvar.delay = 1.0 / gvar.fps

            # Create video plane for live feed
            width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            VideoPlaneManager.create_plane(bpy.context, width=width, height=height)

            # Start playback
            if not bpy.context.screen.is_animation_playing:
                bpy.ops.screen.animation_play()

            self.running = True
            bpy.app.timers.register(self.update_frame, first_interval=0.0)
            print("Realtime pose tracking started.")
            return

        # --- Handle video-based tracking ---
        if self.path:
            self.cap = cv2.VideoCapture(self.path)
            gvar.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if not gvar.fps or gvar.fps <= 0:
                gvar.fps = 30
            gvar.delay = 1.0 / gvar.fps

            # Create OpenCV window
            cv2.namedWindow("Pose Tracking", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Pose Tracking", 640, 480)
            cv2.moveWindow("Pose Tracking", 0, 0)

            self.running = True
            bpy.app.timers.register(self.update_frame, first_interval=0.0)
            print("Pose tracking started.")

    def detect_pose(self, img):
        """
            :param img: Image on which pose dectection is to be performed.
            :return: Image with pose landmarks and lines connecting the landmarks drawn on it.
        """
        if mp_vision is None:
            print("MediaPipe is not installed. Install via the Addon Preferences first. Skipping pose detection.")
            return img

        frame_time = _capture_time(self)
        if self.detector is None:
            self.detector = create_pose_detector(fps=None if self.mode == "rt_pose" else gvar.fps)
            self.video_fps = gvar.fps or 30
            self.frame_index = -1
            if self.mode == "side_pose":
                gvar.side_fps = self.video_fps
            else:
                gvar.front_fps = self.video_fps
        self.frame_index += 1          # every call is one video frame, detected or not

        height, width = img.shape[:2]
        img, landmarksList = self.detector.find_pose(img, timestamp=frame_time)

        if not landmarksList:
            missing = gvar.zlist_missing_times if self.mode == 'side_pose' else gvar.position_missing_times
            missing.append(frame_time)
            return img

        # Per-landmark visibility (33 + the 7 centroids below, which take the lower of their pair)
        vis = list(self.detector.visibility)
        depth_conf = list(self.detector.depth_confidence)

        # --- Calculate custom points (centroids) ---
        point_pairs = [
            (23, 24),  # hips           (point 32)
            (11, 12),  # shoulders      (point 33)
            (18, 20),  # right hand     (point 34)
            (17, 19),  # left hand      (point 35)
            (7, 8),    # ears           (point 36)
            (9, 10),   # mouth          (point 37)
            (1, 4)     # eyes           (point 38)
        ]

        extra_points = []
        for a, b in point_pairs:
            centroid = MathUtility.centroid([landmarksList[a], landmarksList[b]])
            normalized = MathUtility.normalize_coordinates(centroid, height, width, POSE_DEPTH_SCALE)
            extra_points.append(normalized)
            vis.append(min(vis[a], vis[b]))
            depth_conf.append(min(depth_conf[a], depth_conf[b]))

        meta = {'depth_confidence': depth_conf, 'xy_source': self.detector.xy_source,
                'depth_source': self.detector.depth_source,
                'depth_state': ['observed' if c > 0 else 'missing' for c in depth_conf]}

        # --- Handle mode: front, side, or realtime ---
        if self.mode in ["front_pose", "rt_pose"]:
            listOf32Landmarks = []
            for landmark in landmarksList:
                normalized = MathUtility.normalize_coordinates(landmark, height, width, POSE_DEPTH_SCALE)
                listOf32Landmarks.extend(normalized)

            for point in extra_points:
                listOf32Landmarks.extend(point)

            gvar.positionList.append(listOf32Landmarks)
            gvar.position_times.append(frame_time)
            gvar.position_vis.append(vis)
            gvar.position_meta.append(meta)

            # Whole hands from the body video (RTMW tracker only), in the pose's own coordinates.
            # They have no depth: Animate Hand puts each one at the depth of the pose wrist.
            for name, (points, conf) in self.detector.hands.items():
                if sum(conf) / len(conf) < BODY_HAND_MIN_VIS:
                    continue                    # hand hidden in this frame: no entry
                flat = []
                for x, y in points:
                    flat.extend(MathUtility.normalize_coordinates([x, y, 0.0], height, width, POSE_DEPTH_SCALE))
                side = "R" if name == "Right" else "L"
                getattr(gvar, f"{side}_body_hand_positionList").append(flat)
                getattr(gvar, f"{side}_body_hand_times").append(frame_time)

        elif self.mode == "side_pose":
            zframe = []
            for landmark in landmarksList:
                normalized = MathUtility.normalize_coordinates(landmark, height, width, POSE_DEPTH_SCALE)
                zframe.extend(normalized)

            for point in extra_points:
                zframe.extend(point)

            gvar.zlist.append(zframe)
            gvar.zlist_times.append(frame_time)
            gvar.zlist_vis.append(vis)
            gvar.zlist_meta.append(meta)

        return img

    def update_frame(self):
        """Updates the image in the CV window or Plane in the Blender Scene."""

        if cv2 is None:
            print("OpenCV (cv2) is not installed! Install via the Addon Preferences first.")
            return
        
        if not self.running or not self.cap.isOpened():
            self.stop()
            return None

        # Stop if the window was closed (for non-realtime)
        if self.mode != "rt_pose":
            if cv2.getWindowProperty("Pose Tracking", cv2.WND_PROP_VISIBLE) < 1:
                self.stop()
                return None

        success, img = self.cap.read()
        if not success:
            self.stop()
            return None

        img = self.detect_pose(img)

        # --- Realtime: show on video plane ---
        if self.mode == "rt_pose":
            VideoPlaneManager.update_frame(img)
        else:
            # --- Video file: show in OpenCV window ---
            cv2.imshow("Pose Tracking", img)
            cv2.waitKey(1)
            return 0.0          # a video file is tracked as fast as the tracker runs, not at its frame rate

        return gvar.delay

    def stop(self):
        """Stops pose tracking."""
        if self.running:
            print("Pose tracking stopped.")
        self.running = False

        if self.detector is not None:
            close_in_background(self.detector)
            self.detector = None

        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

        if self.mode == "rt_pose":
            VideoPlaneManager.remove_plane()
        else:
            try:
                cv2.destroyWindow("Pose Tracking")
            except:
                pass

        # Stop animation playback if running
        if bpy.context.screen.is_animation_playing:
            bpy.ops.screen.animation_play()

        return None



# -------------------------------------------------------------
# HAND TRACKING UTILITY
# -------------------------------------------------------------
class HandTracking:
    def __init__(self, num_hands, mode):
        self.detector = None
        self.num_hands = num_hands
        self.mode = mode
        self.running = False
        """
            It uses the MediaPipe Tasks HandLandmarker to do hand tracking.
            :param num_hands: Numbers of hands to track (1 or 2).
            :param mode: Different modes like realtime, offline: (front view / side view).
        """

        if not load_libraries():
            print("OpenCV/MediaPipe are not installed! Install via the Addon Preferences first.")
            return

        # ---- Realtime Hand Tracking Mode ----
        if self.mode == "rt_hand":
            self.cap = cv2.VideoCapture(bpy.context.scene.cam_index)
            if not self.cap.isOpened():
                print("Camera failed to open.")
                return
            for lst in (gvar.R_hand_positionList, gvar.L_hand_positionList, gvar.R_hand_times, gvar.L_hand_times):
                lst.clear()
            gvar.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if not gvar.fps or gvar.fps <= 0:
                gvar.fps = 30
            gvar.delay = 1.0 / gvar.fps
            self.running = True
            # Create video plane for live feed
            width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            VideoPlaneManager.create_plane(bpy.context, width=width, height=height)

        # ---- Offline Video Modes (front_hand / side_hand) ----
        if self.mode == "front_hand":
            self.path = gvar.hand_front_video_path
        elif self.mode == "side_hand":
            self.path = gvar.hand_side_video_path
        else:
            self.path = ""

        if self.path:
            self.cap = cv2.VideoCapture(self.path)
            gvar.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if not gvar.fps or gvar.fps <= 0:
                gvar.fps = 30
            gvar.delay = 1.0 / gvar.fps
            cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Hand Tracking", 640, 480)
            cv2.moveWindow("Hand Tracking", 0, 0)
            # A new take replaces the earlier one of the same view
            if self.mode == "front_hand":
                for lst in (gvar.R_hand_positionList, gvar.L_hand_positionList, gvar.R_hand_times, gvar.L_hand_times):
                    lst.clear()
            else:
                for lst in (gvar.R_hand_zlist, gvar.L_hand_zlist, gvar.R_hand_ztimes, gvar.L_hand_ztimes):
                    lst.clear()
            self.running = True
            bpy.app.timers.register(self.update_frame, first_interval=0.0)

    def detect_hand(self, img):
        """
            :param img: Image on which hand dectection is to be performed.
            :return: Image with hand landmarks and lines connecting the landmarks drawn on it.
        """
        
        if mp_vision is None:
            print("MediaPipe is not installed. Install via the Addon Preferences first. Skipping hand detection.")
            return img

        frame_time = _capture_time(self)
        if self.detector is None:
            self.detector = HandDetector(max_hands=self.num_hands,
                                         fps=None if self.mode == "rt_hand" else gvar.fps,
                                         detection_con=0.8)
            self.video_fps = gvar.fps or 30
            self.frame_index = -1
            gvar.hand_fps = self.video_fps
            if self.mode == 'side_hand':
                gvar.hand_side_fps = self.video_fps
            else:
                gvar.hand_front_fps = self.video_fps
        self.frame_index += 1          # every call is one video frame, detected or not

        height, width = img.shape[:2]
        hands, img = self.detector.find_hands(img, timestamp=frame_time)
        if not hands:
            return img

        seen = set()
        for hand in hands:
            handType = hand['type']  # 'Right' or 'Left'
            if handType in seen:
                continue               # two detections with one label: keep the first (one entry per frame)
            seen.add(handType)
            landmarksList = hand['lmList']
            if handType == "Right":
                position_list = gvar.R_hand_positionList
                z_list = gvar.R_hand_zlist
                rt_hand_lmlist = gvar.rt_hand_lmlist_R
                times, ztimes = gvar.R_hand_times, gvar.R_hand_ztimes

            else:
                position_list = gvar.L_hand_positionList
                z_list = gvar.L_hand_zlist
                rt_hand_lmlist = gvar.rt_hand_lmlist_L
                times, ztimes = gvar.L_hand_times, gvar.L_hand_ztimes

            normalized_landmarks = []
            for landmark in landmarksList:
                normalized = MathUtility.normalize_coordinates(landmark, height, width, 40)
                normalized_landmarks.append(normalized)

            if self.mode in ["front_hand", "rt_hand"]:
                flat_list = []
                rt_hand_lmlist.clear()
                for normalized in normalized_landmarks:
                    rt_hand_lmlist.append(normalized)
                    flat_list.extend(normalized)
                position_list.append(flat_list)
                times.append(frame_time)

            elif self.mode == "side_hand":
                zframe = []
                for normalized in normalized_landmarks:
                    zframe.extend(normalized)
                z_list.append(zframe)
                ztimes.append(frame_time)

        return img

    def update_frame(self):
        """Updates the image in the CV window."""

        if cv2 is None:
            print("OpenCV (cv2) is not installed! Install via the Addon Preferences first.")
            return

        if not self.running or not self.cap.isOpened():
            self.stop()
            return None

        # Stop if CV window closed
        if self.mode != "rt_hand":
            if cv2.getWindowProperty("Hand Tracking", cv2.WND_PROP_VISIBLE) < 1:
                self.stop()
                return None

        success, img = self.cap.read()
        if not success:
            self.stop()
            return None

        img = self.detect_hand(img)

        cv2.imshow("Hand Tracking", img)
        cv2.waitKey(1)
        return 0.0              # a video file is tracked as fast as the tracker runs

    def stop(self):
        """Stops hand tracking."""
        if self.running:
            print("Hand tracking stopped.")
        self.running = False
        if self.detector is not None:
            close_in_background(self.detector)
            self.detector = None
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if self.mode != "rt_hand":
            try:
                cv2.destroyWindow("Hand Tracking")
            except:
                pass
        return None



# -----------------------------------------------------------------------
# A CUSTOM UTILITY TO CREATE A VIRTUAL SKELETON FROM TRACKED MOTION DATA
# Custom method developed by the author (Larenju Rai)
# Commonly referred to as "Larenju's Algorithm"
# -----------------------------------------------------------------------
class Skeleton:
    @staticmethod
    def create_skeleton(context, bone_pairs, mode):
        """
        Creates bones.
        Bone = Line (with skin modifier) between specified pair of empties 'empty_a' and 'empty_b'.
        Each bone has a DAMPED_TRACK constraint to its target 'empty_b' with default Y-axis tracking. This is head of the bone.
        Each bone is paranted to other empty: 'empty_a'. This is tail of the bone.
        :param bone_pairs: It is a list of tuples (a,b) where a and b are index of 'empty_a' and 'empty_b'.
        :return: A list of (a, b, obj) tuples for further configuration. obj is the bone object.
        """
        bones_created = []

        for a, b in bone_pairs:
            if mode == "pose":
                empty_a = gvar.object_list[a]
                empty_b = gvar.object_list[b]
                bone_list = gvar.bones_list
                # --- Get or create collection ---
                skeleton_collection = BlenderUtility.create_collection(context,"PoseSkeleton")
            elif mode == "R_hand":
                empty_a = gvar.R_hand_object_list[a]
                empty_b = gvar.R_hand_object_list[b]
                bone_list = gvar.R_hand_bones_list
                # --- Get or create collection ---
                skeleton_collection = BlenderUtility.create_collection(context,"RightHandSkeleton")
            else:
                empty_a = gvar.L_hand_object_list[a]
                empty_b = gvar.L_hand_object_list[b]
                bone_list = gvar.L_hand_bones_list
                # --- Get or create collection ---
                skeleton_collection = BlenderUtility.create_collection(context,"LeftHandSkeleton")
            loc_a = empty_a.matrix_world.translation
            loc_b = empty_b.matrix_world.translation

            # --- Create cube and extrude towards target ---
            bpy.ops.mesh.primitive_cube_add(size=1, location=loc_a)
            obj = context.active_object
            skeleton_collection.objects.link(obj)
            obj.name = f"hand_bone_{a}_{b}"

            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.merge(type='CENTER')

            dir_vec = loc_b - loc_a
            bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value": dir_vec})
            bpy.ops.object.mode_set(mode='OBJECT')

            # --- Hook tip vertex to target empty ---
            mesh = obj.data
            verts = [v.co.copy() for v in mesh.vertices]
            tip_index = max(range(len(verts)), key=lambda i: (verts[i] - loc_a).length)

            obj.parent = empty_a
            hook = obj.modifiers.new("HookToB", 'HOOK')
            hook.object = empty_b
            hook.vertex_indices_set([tip_index])

            # --- Add modifiers ---
            skin = obj.modifiers.new("Skin", 'SKIN')
            skin.use_smooth_shade = True
            subsurf = obj.modifiers.new("Subdivision", 'SUBSURF')
            subsurf.levels = 2

            # Resize skin
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.transform.skin_resize(value=(0.3, 0.3, 0.3))
            bpy.ops.object.mode_set(mode='OBJECT')

            # --- Add default DAMPED_TRACK constraint (Y-axis) ---
            constraint = obj.constraints.new('DAMPED_TRACK')
            constraint.target = empty_b
            constraint.track_axis = 'TRACK_Y'

            # Record
            bone_list.append(obj)
            bones_created.append((a, b, obj))

        return bones_created
