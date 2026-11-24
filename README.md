🎬 Open Mocap — AI-Powered Motion Capture for Blender

Open Mocap is an open-source Blender add-on for AI-powered motion capture.
It supports offline full-body pose tracking, offline + real-time hand tracking, and easy retargeting to any rig.
Designed for artists, animators, and researchers who want fast, accessible motion capture inside Blender—no external software required.

---

⚠️ Blender & Dependencies Compatibility

Tested with Blender 5.1 (Python 3.13).

Dependencies (installed by the add-on: Edit > Preferences > Add-ons > Open Mocap > Install Dependencies):

MediaPipe (brings OpenCV as opencv-contrib-python), chosen in the add-on preferences:
- Stable: 0.10.32 (default, tested).
- Latest: the newest release on PyPI, with a "Check for MediaPipe Update" button. Versions from 0.10.33 on (incl. 1.x) upload usage statistics to Google every time tracking stops; this cannot be turned off. Open Mocap closes trackers in the background, so Blender does not wait for that upload.
Switching versions after tracking has run needs a Blender restart (the MediaPipe library stays loaded).

Optional pose tracker (add-on preferences > Pose Tracker > "RTMW + MediaPipe depth"):
- OpenMMLab RTMW whole-body model through rtmlib and ONNX Runtime (Apache-2.0), with MediaPipe running alongside for the depth estimate RTMW does not provide. Steadier joints in fast moves (about 28% less jitter on a sword clip), real hand knuckles (better hand roll) and heel/toe points.
- Install Dependencies also installs rtmlib and onnxruntime-directml and downloads the RTMW-x and YOLOX-m models (about 330 MB) from OpenMMLab.
- RTMW Device: GPU (DirectML, any DirectX 12 card, about 11 frames per second including MediaPipe) or CPU (about 3 frames per second).

Experimental pose tracker (Pose Tracker > "RTMW3D (experimental)"):
- OpenMMLab RTMW3D-x (Apache-2.0) gives joint positions and its own depth, without MediaPipe. Install Dependencies downloads the RTMW3D-x ONNX file (about 370 MB, the copy rtmlib uses) and RTMW's YOLOX-m detector.
- On four views of one performance its depth agreed with a 90 degree camera far less than MediaPipe's (correlation 0.12-0.55 against 0.69-0.81) and jittered about twice as much, so Combine Data trusts its depth half as much. Use it to compare, not as the default.
- "RTMW3D + MediaPipe depth (experimental)" runs MediaPipe alongside and mixes the two depths; RTMW3D Depth Share (default 0.1) sets RTMW3D's part. At 0.1 the katana take's depth agreed slightly better with the 90 degree cameras (0.772 -> 0.781) and the Whirlpool take's stayed the same (0.732 -> 0.730); shares above 0.3 made both worse.

Optional Depth Lifter (add-on preferences > Depth Lifter, works with every pose tracker):
- After a video is tracked, MotionAGFormer-B (TaatiTeam/MotionAGFormer, Apache-2.0, 11.7M parameters) rebuilds the body depth from the whole 2D movement, 243 frames at a time, and mixes it with the tracker's depth (Lifter Share, default 0.5). Face, hand and foot points move with the joint they belong to. It runs on the CPU, about 2 s per 120 frames.
- Install Dependencies downloads the model as ONNX (about 48 MB, checked by SHA-256) and installs ONNX Runtime if no RTMW tracker brought it already.
- Measured with RTMW on the katana and Whirlpool takes (four cameras each, depth against the camera 90 degrees to the side): agreement 0.745 -> 0.817 and 0.720 -> 0.796 at share 0.5, the nearer shoulder/hip/wrist told right 89.0 -> 91.8% and 91.9 -> 95.2%, depth jitter about halved (share 1: a fifth, agreement 0.783 / 0.770).

Optional GEM-X (add-on preferences > GEM-X, then Pose Tracking > Track with GEM-X):
- NVIDIA GEM-X (github.com/NVlabs/GEM-X; code Apache-2.0, weights under the NVIDIA Open Model License) tracks the whole body, fingers included, in 3D from one video of a fixed camera. It is not installed by the add-on: set up its repository, weights and SOMA assets in an environment of its own (PyTorch with CUDA) as its INSTALL.md says, then point GEM-X Python (that environment's python.exe) and GEM-X Folder at them.
- Track with GEM-X runs it outside Blender (Esc cancels) and fills the tracked pose as Start Pose Tracking + Combine Data would: press Animate, then Retarget. No side video, no Combine Data. FOV next to the button sets the camera's horizontal field of view for GEM-X's distances (0: its own guess).
- The world-space joints are used, turned to the camera's level right/forward, with the floor levelled where GEM-X finds a foot standing still (its world height drifted 0.19 m over a 3.4 s take). SOMA has no ears, nose, mouth or heels: they are placed from the head and the foot. Fingers use the 3D hand joints. Retarget with Hip Height From Floor: From First Frame pins the hips to their standing height, and a take that starts crouched then stands below the floor. The heel is placed in the foot's own plane, so the foot turns as one piece, and the retarget keeps GEM-X's own foot direction (no turning toward the knee). The spine follows GEM-X's own: each Mixamo spine bone (Hips, Spine, Spine1, Spine2, Neck) turns as its SOMA segment turns away from the actor's rest skeleton, so a rounded upper back over a straight lower back comes through, not only one lean between hips and shoulders. Takes tracked before this need Track with GEM-X again (or the runner's --reuse) for the rest skeleton.
- On an RTX 4070 Laptop (8 GB) a 3.4 s 1080p take took about 2 minutes, at most 3.9 GB of graphics memory; clips can be up to 60 s.

MediaPipe Tasks models: pose_landmarker_full.task and hand_landmarker.task (downloaded from Google's MediaPipe model storage)

They are installed into `<Blender user folder>/<version>/datafiles/open_mocap/` (`lib` and `models`), not into Blender's own Python, so Blender's bundled numpy is never replaced and no admin rights are needed.

> **Developer Note:**  
> MediaPipe 0.10.30 and newer (the only versions available for Python 3.13) removed the legacy `mediapipe.solutions` API that CVZone was built on. The add-on now uses the MediaPipe Tasks API (PoseLandmarker / HandLandmarker) directly, so CVZone is no longer needed. Landmarks keep the same pixel format CVZone produced, so tracking data and skeletons behave as before.

---

🚀 Features

✔️ Offline full-body motion capture

Video of the actor below belongs to: https://www.youtube.com/@endlessreference

![PoseTracking GIF](assets/PoseTracking.gif)

✔️ Easy Retargeting of captured motion data to any 3D character

3D character used in video below:
“The Amazing Spiderman” (https://skfb.ly/6sRDt) by iqbalmulyawan is licensed under CC BY 4.0.

![AnimRetargeting GIF](assets/PoseAnimRetarget.gif)

✔️ Offline hand tracking

Video created by Larenju Rai (https://github.com/Larenju-Rai).

![HandTracking GIF](assets/HandTracking.gif)

✔️ Realtime hand tracking

Video created by Larenju Rai (https://github.com/Larenju-Rai).

![RT_HandTracking GIF](assets/RealTimeHandTracking.gif)

🚀 Other Features:

✔️ Sword tracking: "Track Sword" finds the blade in every tracked video (static camera), Combine Data turns it into a 3D direction, and the retarget turns the hands to its handle (option "Sword"), so the hand roll the pose tracker cannot see on a two-handed grip comes from the blade
✔️ Animation layers: Retarget and Bake Motion work on a rig with animation layers (NLA tracks, e.g. the NLA Anim Layers add-on), even while a layer is being edited. Retarget leaves layer editing and switches the layers off while the mocap drives the rig (Remove Retarget switches them back on). Bake Motion bakes the mocap on its own and, with Layers set to New Base Layer, adds it as a layer right above the base and mutes the old base, so the layers above keep playing on top; New Top Layer puts it above all of them; Active Action bakes as on a rig without layers.
✔️ Perspective in Combine Data: an actor walking toward or away from the front camera keeps one size. "Find FOV" (on by default) picks the field of view that keeps the actor's size steady while the hips move in depth and sets FOV to it; a take where the actor stays at one distance keeps FOV as set. The hips' travel toward the camera is measured at one scale, so an actor walking from near the camera to the back of the room (up to 2.5 times nearer or farther than their middle distance) keeps one size. On a back-to-front take that shrank 20% at 65 degrees it found 44 degrees (-6%, bone length variation 15.0 -> 12.6%); on its front-to-back twin bone length variation went 22.5 -> 15.3%
**Lock Feet on Baked Action** is in Open Mocap → Pose Tracking → Animation Baking. Select the skinned Mixamo rig and its baked action, set **Floor Z** to the height of the ground, then click the button. The corrected action becomes active and the original is kept in the file. The pass uses the heel and toe surfaces of the skinned mesh, allows the heel to rise during a toe pivot, and checks the sole between keys. New GEM-X takes can also supply contact confidence. For an NLA stack, bake the full rig with **Layers → Active Action** first. The companion Mixamo Rig O3DE exporter writes four samples per scene frame for foot-locked actions.

Foot lock options (contact rules after RGB MoCap's motion correction, github.com/toyxyz/toyxyz_RGBMocap, MIT):
- **Flat Ground / Multi Level**: Flat Ground pins every contact to Floor Z. Multi Level pins each stance at its own height (stairs, steps, slopes, a foot on a box) and has no floor check; a stance can stand up to 0.3 leg lengths above the other foot.
- **Gentle / Normal / Firm**: Gentle takes fewer contacts and pins them at 60%, keeping more of the original foot motion. Normal is the default. Firm takes looser contacts and bridges dropouts up to 80 ms where the foot stayed in place; avoid it on deliberate foot slides.
- **Lower Hips** (on by default) keys the hips down, smoothly and at most 0.3 leg lengths, where a planted foot cannot reach its pin with the leg straight. Without it that foot stays short of the ground. Locked or constrained hip locations are left alone.
- Jumps never plant: where both feet rise well above their surroundings and come down again, nothing is pinned.
- **Contact markers**: timeline markers named `L+` / `L-` and `R+` / `R-` start and end a contact by hand (inclusive). A foot with markers takes its contacts only from them; the other foot stays automatic.

✔️ Blender-native workflow
✔️ Easy installation
✔️ Works fully offline

---

🎓 Tutorial

A complete installation + usage tutorial is available here:

[![Watch the video](https://img.youtube.com/vi/NWfkF-sr-dY/0.jpg)](https://youtu.be/NWfkF-sr-dY)


---

🤝 Contributing

Contributions are welcome!

You can help by:

reporting issues

fixing bugs

improving performance

adding new features

contributing documentation


Feel free to open:

Issues → for bugs, questions, feature requests

Pull Requests → for code contributions



---

## Credits

The following test videos included in this repository are the property of the author:

- `Front5.mp4`  
- `Side5.mp4`  

Other test videos were obtained from:  

- [Endless Refernce] https://endlessreference.com https://youtube.com/@endlessreference

These videos are used for testing and demonstration purposes only.

---

📄 License

This project is licensed under the MIT License.
See the LICENSE file for full details.


---

💬 Support

For doubts, improvements, or help with animation/mocap:

📧 Email: larenju23@gmail.com
📂 GitHub Issues: Feel free to open one anytime.


---

🎉 Thank You for Using Open Mocap!

If this add-on helps you, consider ⭐ starring the repository!

