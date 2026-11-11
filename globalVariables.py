fps = 24 
delay = 0


# -------------------------------------------------------------
# GLOBAL VARIABLES FOR POSE TRACKING
# -------------------------------------------------------------
front_video_path = "" 
side_video_path = "" 
positionList = [] #each element of this list = list of [x,y,z] positions of 40 body landmarks in a given frame (front view)
object_list = [] #list of 40 empty objects corresponding to 40 pose landmarks
zlist = [] #each element of this list = list of [x,y,z] positions of 40 body landmarks in a given frame (side view)
bones_list = [] #list of bone objects
# Recorded alongside positionList / zlist (one entry per tracked frame), used by Combine Data
position_times = [] #time of each front frame in its video (seconds)
position_vis = [] #40 visibility scores (0..1) per front frame
zlist_times = [] #time of each side frame in its video (seconds)
zlist_vis = [] #40 visibility scores per side frame
front_fps = 0
side_fps = 0
positionList_raw = None #front data before Combine Data, so it can be re-combined with other settings
position_meta = [] #per-frame depth confidence and measurement sources
zlist_meta = []
position_meta_raw = None
position_missing_times = [] #explicit failed detections, in video/capture seconds
zlist_missing_times = []
root = None #root empty to which the skeleton rig is parented
pose_tracker = None




# -------------------------------------------------------------
# GLOBAL VARIABLES FOR HAND TRACKING
# -------------------------------------------------------------
hand_front_video_path = "" 
hand_side_video_path = "" 
hand_tracker = None
hand_fps = 0
hand_front_fps = 0
hand_side_fps = 0
# Time (seconds, in its video) of each tracked frame of the lists below. A hand that is not found in a
# frame gets no entry, so without these the frames after a miss would play too early.
R_hand_times = [] #front view, right hand
L_hand_times = []
R_hand_ztimes = [] #side view
L_hand_ztimes = []

# RIGHT HAND
R_hand_positionList = [] #each element of this list = list of [x,y,z] positions of 21 right hand landmarks in a given frame (front view)
R_hand_object_list = [] #list of 21 empty objects corresponding to 21 right hand landmarks
R_hand_zlist = [] #each element of this list = list of [x,y,z] positions of 21 right hand landmarks in a given frame (side view)
R_hand_bones_list = [] #list of bone objects
R_hand_root = None #root empty to which the right hand skeleton rig is parented
rt_hand_lmlist_R = [] #List to hold 21 landmarks' coordinates in a given frame for realtime hand tracking


# LEFT HAND
L_hand_positionList = [] 
L_hand_object_list = [] 
L_hand_zlist = []
L_hand_bones_list = []
L_hand_root = None
rt_hand_lmlist_L = []


# WHOLE HANDS FROM THE BODY VIDEO (RTMW pose tracker only)
# 21 points per hand in the pose's own coordinates, x/y only (flat list of 21 * [x, y, 0]), one entry
# per front-video frame in which the hand was seen, with its time. Animate Hand uses these when no
# hand video was tracked.
R_body_hand_positionList = []
L_body_hand_positionList = []
R_body_hand_times = []
L_body_hand_times = []
