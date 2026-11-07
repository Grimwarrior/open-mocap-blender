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

