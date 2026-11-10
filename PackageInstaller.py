"""
    Dependency management for Open Mocap.

    The add-on needs MediaPipe (which brings OpenCV as opencv-contrib-python) and two
    MediaPipe Tasks model files. Everything is installed into a private folder in
    Blender's user data directory, never into Blender's own site-packages, so it works
    for installs in Program Files and never replaces the numpy that ships with Blender:

        <Blender user dir>/<version>/datafiles/open_mocap/lib      python packages
        <Blender user dir>/<version>/datafiles/open_mocap/models   .task model files

    MediaPipe 0.10.30 and newer (the only versions that exist for Python 3.13, i.e.
    Blender 5.1+) dropped the legacy `mediapipe.solutions` API that cvzone used, so the
    add-on now talks to the MediaPipe Tasks API directly and cvzone is not needed.
"""

import os
import sys
import json
import shutil
import tempfile
import threading
import subprocess
import importlib
import importlib.util
import importlib.metadata
import urllib.request
import zipfile

import bpy


# Two choices in the add-on preferences:
#   Stable  0.10.32, tested with Blender 5.1 (Python 3.13).
#   Latest  the newest release on PyPI. From 0.10.33 on (incl. 1.x) libmediapipe.dll uploads usage
#           logs to https://play.googleapis.com/log whenever a tracker is closed, and close() waits
#           for that upload (~1 s, or ~40 s when the connection hangs). Open Mocap closes trackers on a
#           background thread so Blender does not freeze, but the upload itself cannot be turned off.
STABLE_VERSION = "0.10.32"
MEDIAPIPE_VERSION = STABLE_VERSION          # kept for older references
MEDIAPIPE_REQUIREMENT = f"mediapipe=={STABLE_VERSION}"

MODELS = {
    "pose": (
        "pose_landmarker_full.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    ),
    # Larger pose model (about 30 MB instead of 9), chosen with "Pose Model" in the preferences
    "pose_heavy": (
        "pose_landmarker_heavy.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    ),
    "hand": (
        "hand_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task",
    ),
}

# Optional RTMW tracker (OpenMMLab RTMW whole-body pose via rtmlib + ONNX Runtime, Apache-2.0).
# rtmlib is installed without its dependencies: it asks for opencv-python, which would clash
# with the opencv-contrib-python that MediaPipe brings (both install the cv2 package).
RTMLIB_REQUIREMENT = "rtmlib==0.0.16"
ORT_PACKAGE = "onnxruntime-directml" if sys.platform == "win32" else "onnxruntime"   # DirectML: any DX12 GPU
RTMW_MODELS = {
    "det": ("yolox_m_8xb8-300e_humanart-c2c7a14a",
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
            "yolox_m_8xb8-300e_humanart-c2c7a14a.zip"),
    "pose": ("rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122",
             "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
             "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip"),
}

# Packages Blender already ships. If pip ever wants to install these into our folder,
# they would shadow Blender's own copy, so we warn about it.
_BLENDER_OWNED = {"numpy"}

status = {"busy": False, "message": ""}


# -------------------------------------------------------------
# PATHS
# -------------------------------------------------------------
def data_dir():
    return bpy.utils.user_resource('DATAFILES', path="open_mocap", create=True)


def lib_dir():
    return os.path.join(data_dir(), "lib")


def models_dir():
    return os.path.join(data_dir(), "models")


def pose_model():
    """:return: 'FULL' or 'HEAVY', from the add-on preferences."""
    return _pref("pose_model", 'FULL')


def needed_models(model=None):
    """:return: keys of MODELS that the chosen pose model needs."""
    return ["pose_heavy" if (model or pose_model()) == 'HEAVY' else "pose", "hand"]


def model_path(kind):
    """
        :param kind: "pose" (the pose model chosen in the preferences; the full one if the heavy
                     one is not downloaded yet), "pose_heavy" or "hand".
        :return: Path of the .task model file.
    """
    if kind == "pose" and pose_model() == 'HEAVY':
        heavy = os.path.join(models_dir(), MODELS["pose_heavy"][0])
        if os.path.isfile(heavy):
            return heavy
    return os.path.join(models_dir(), MODELS[kind][0])


def _model_file(kind):
    return os.path.join(models_dir(), MODELS[kind][0])


def rtmw_model_path(kind):
    """:param kind: "det" or "pose". :return: Path of the RTMW .onnx file, or None if missing."""
    folder = os.path.join(models_dir(), "rtmw", RTMW_MODELS[kind][0])
    for root, _dirs, files in os.walk(folder):
        if "end2end.onnx" in files:
            return os.path.join(root, "end2end.onnx")
    return None


def add_lib_to_path():
    """Make the private package folder importable (first, so its OpenCV wins)."""
    path = lib_dir()
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)
    importlib.invalidate_caches()


# -------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------
def is_installed(module_name):
    """Check if a Python module can be imported in Blender."""
    add_lib_to_path()
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def installed_mediapipe_version():
    """:return: Version of the MediaPipe that would be imported, or None."""
    add_lib_to_path()
    try:
        spec = importlib.util.find_spec("mediapipe")
    except (ImportError, ValueError):
        spec = None
    # Read __version__ from the package itself: metadata folders of a replaced version can linger
    if spec and spec.origin and os.path.isfile(spec.origin):
        with open(spec.origin, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=", 1)[1].strip().strip("'\"")
    try:
        return importlib.metadata.version("mediapipe")
    except importlib.metadata.PackageNotFoundError:
        return None


def _remove_stale_metadata(pins):
    """Delete <name>-<other version>.dist-info folders that pip --target leaves behind on upgrade."""
    lib = lib_dir()
    if not os.path.isdir(lib):
        return
    wanted = {}
    for pin in pins:
        name, _, version = pin.partition("==")
        wanted[name.lower().replace("-", "_")] = version
    for entry in os.listdir(lib):
        if not entry.endswith(".dist-info"):
            continue
        name, _, version = entry[:-len(".dist-info")].rpartition("-")
        key = name.lower().replace("-", "_")
        if key in wanted and version != wanted[key]:
            shutil.rmtree(os.path.join(lib, entry), ignore_errors=True)


def channel():
    """:return: 'STABLE' or 'LATEST', from the add-on preferences."""
    try:
        return bpy.context.preferences.addons[__package__].preferences.mediapipe_channel
    except (AttributeError, KeyError):
        return 'STABLE'


def _pref(name, default):
    try:
        return getattr(bpy.context.preferences.addons[__package__].preferences, name)
    except (AttributeError, KeyError):
        return default


def pose_tracker():
    """:return: 'MEDIAPIPE' or 'RTMW', from the add-on preferences."""
    return _pref("pose_tracker", 'MEDIAPIPE')


def rtmw_device():
    """:return: 'dml' (GPU via DirectML) or 'cpu' for ONNX Runtime."""
    return 'dml' if _pref("rtmw_device", 'GPU') == 'GPU' and sys.platform == "win32" else 'cpu'


def rtmw_missing():
    """:return: list of missing RTMW pieces (packages and models)."""
    missing = []
    if not is_installed("rtmlib"):
        missing.append("rtmlib")
    if not is_installed("onnxruntime"):
        missing.append(ORT_PACKAGE)
    for kind in RTMW_MODELS:
        if rtmw_model_path(kind) is None:
            missing.append(f"RTMW {kind} model")
    return missing


def requirement(ch):
    return MEDIAPIPE_REQUIREMENT if ch == 'STABLE' else "mediapipe"


def _version_tuple(v):
    parts = []
    for p in v.split("."):
        digits = "".join(c for c in p if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def mediapipe_ok(ch=None):
    """Is the MediaPipe for the chosen channel installed? Latest = anything newer than Stable."""
    ch = ch or channel()
    v = installed_mediapipe_version()
    if v is None or not is_installed("mediapipe"):
        return False
    if ch == 'STABLE':
        return v == STABLE_VERSION
    return _version_tuple(v) > _version_tuple(STABLE_VERSION)


def mediapipe_loaded():
    """MediaPipe's DLL stays locked on Windows once tracking ran in this Blender session."""
    return "mediapipe" in sys.modules


def check_required_packages(ch=None, tracker=None, model=None):
    """Return a list of missing requirements (packages and model files)."""
    ch = ch or channel()
    tracker = tracker or pose_tracker()
    missing = rtmw_missing() if tracker == 'RTMW' else []
    if not mediapipe_ok(ch):
        found = installed_mediapipe_version()
        want = MEDIAPIPE_REQUIREMENT if ch == 'STABLE' else "mediapipe (latest)"
        missing.append(want + (f" (found {found})" if found else ""))
    if not is_installed("cv2"):
        missing.append("opencv-contrib-python")
    for kind in needed_models(model):
        if not os.path.isfile(_model_file(kind)):
            missing.append(MODELS[kind][0])
    return missing


def _set_status(message, busy=None):
    status["message"] = message
    if busy is not None:
        status["busy"] = busy
    print(f"[Open Mocap] {message}")


def _pip(*args):
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    cmd = [sys.executable, "-m", "pip", "--disable-pip-version-check", *args]
    subprocess.check_call(cmd, env=env)


def _resolve_missing_requirements(*reqs):
    """
        Ask pip which distributions are missing from Blender's own Python to satisfy the
        requirements. Packages Blender already ships (numpy, packaging, ...) are left out,
        so they are not installed a second time.
        :return: List of "name==version" strings.
    """
    reqs = reqs or (MEDIAPIPE_REQUIREMENT,)
    fd, report = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        _pip("install", "--dry-run", "--quiet", "--report", report, *reqs)
        with open(report, encoding="utf-8") as f:
            data = json.load(f)
    finally:
        os.remove(report)

    pins = []
    for item in data.get("install", []):
        name = item["metadata"]["name"]
        version = item["metadata"]["version"]
        if name.lower() in _BLENDER_OWNED:
            print(f"[Open Mocap] Warning: {name} {version} will be installed next to Blender's own copy.")
        pins.append(f"{name}=={version}")
    return pins


def _download_models():
    os.makedirs(models_dir(), exist_ok=True)
    for kind in needed_models():
        filename, url = MODELS[kind]
        target = _model_file(kind)
        if os.path.isfile(target):
            continue
        _set_status(f"Downloading {filename}...")
        tmp = target + ".part"
        with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as out:
            shutil.copyfileobj(response, out)
        os.replace(tmp, target)


def _download_rtmw_models():
    base = os.path.join(models_dir(), "rtmw")
    os.makedirs(base, exist_ok=True)
    for kind, (name, url) in RTMW_MODELS.items():
        if rtmw_model_path(kind) is not None:
            continue
        _set_status(f"Downloading RTMW {kind} model ({name})...")
        archive = os.path.join(base, name + ".zip.part")
        with urllib.request.urlopen(url, timeout=120) as response, open(archive, "wb") as out:
            shutil.copyfileobj(response, out)
        # The archives hold the model plus sample images in deep folders: take only the .onnx
        # file, flat, so long Windows paths are no problem
        target = os.path.join(base, name)
        os.makedirs(target, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            member = next((m for m in zf.namelist() if m.endswith("end2end.onnx")), None)
            if member is None:
                raise RuntimeError(f"No end2end.onnx in {url}")
            with zf.open(member) as src, open(os.path.join(target, "end2end.onnx.part"), "wb") as dst:
                shutil.copyfileobj(src, dst)
        os.replace(os.path.join(target, "end2end.onnx.part"), os.path.join(target, "end2end.onnx"))
        os.remove(archive)


def _install_rtmw():
    if not (is_installed("rtmlib") and is_installed("onnxruntime")):
        _set_status("Resolving RTMW packages...")
        # rtmlib itself without dependencies (see RTMLIB_REQUIREMENT); its runtime needs are ONNX
        # Runtime and tqdm, the rest (numpy, OpenCV) is already there
        pins = [p for p in _resolve_missing_requirements(ORT_PACKAGE, "tqdm")
                if not p.lower().startswith("opencv-python==")]
        pins.append(RTMLIB_REQUIREMENT)
        _set_status(f"Installing RTMW ({len(pins)} packages) into {lib_dir()} ...")
        os.makedirs(lib_dir(), exist_ok=True)
        _pip("install", "--no-deps", "--upgrade", "--target", lib_dir(), *pins)
        _remove_stale_metadata(pins)
        importlib.invalidate_caches()
    _download_rtmw_models()


def install_packages(ch='STABLE', update=False, tracker='MEDIAPIPE'):
    """
        :param update: for 'LATEST', reinstall even if a newer-than-stable MediaPipe is present.
        :param tracker: 'RTMW' also installs rtmlib, ONNX Runtime and the RTMW models.
    """
    _set_status("Installing dependencies...", busy=True)
    try:
        if update or not (mediapipe_ok(ch) and is_installed("cv2")):
            _set_status("Resolving packages...")
            pins = _resolve_missing_requirements(requirement(ch))
            current = installed_mediapipe_version()
            target = next((p.split("==")[1] for p in pins if p.lower().startswith("mediapipe==")), current)
            if update and target == current and mediapipe_ok(ch):
                _download_models()
                if tracker == 'RTMW':
                    _install_rtmw()
                _set_status(f"MediaPipe {current} is already the latest version.", busy=False)
                return
            if mediapipe_loaded() and current and target != current:
                _set_status(f"MediaPipe {current} is in use. Restart Blender, then click Install again "
                            f"to switch to {target}.", busy=False)
                return
            if pins:
                _set_status(f"Installing {len(pins)} packages into {lib_dir()} ...")
                os.makedirs(lib_dir(), exist_ok=True)
                _pip("install", "--no-deps", "--upgrade", "--target", lib_dir(), *pins)
                _remove_stale_metadata(pins)
                importlib.invalidate_caches()
        _download_models()
        if tracker == 'RTMW':
            _install_rtmw()
        add_lib_to_path()
        missing = check_required_packages(ch, tracker)
        if missing:
            _set_status(f"Install finished but still missing: {', '.join(missing)}", busy=False)
        else:
            extra = " and RTMW" if tracker == 'RTMW' else ""
            _set_status(f"All dependencies are installed (MediaPipe {installed_mediapipe_version()}{extra}).",
                        busy=False)
    except Exception as e:  # noqa: BLE001 - report anything to the user
        _set_status(f"Error installing dependencies: {e}", busy=False)


def uninstall_packages():
    _set_status("Removing dependencies...", busy=True)
    try:
        if lib_dir() in sys.path:
            sys.path.remove(lib_dir())
        # Loaded OpenCV/MediaPipe binaries stay locked on Windows until Blender restarts.
        shutil.rmtree(lib_dir(), ignore_errors=True)
        shutil.rmtree(models_dir(), ignore_errors=True)
        _set_status("Dependencies removed. Restart Blender to finish.", busy=False)
    except Exception as e:  # noqa: BLE001
        _set_status(f"Error removing dependencies: {e}", busy=False)


def _redraw_preferences():
    """Timer: redraw the Preferences window while the install thread runs."""
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'PREFERENCES':
                area.tag_redraw()
    return 0.5 if status["busy"] else None



# -------------------------------------------------------------
# OPERATORS
# -------------------------------------------------------------
class InstallAddonDependenciesOperator(bpy.types.Operator):
    """Install required Python dependencies"""
    bl_idname = "addon.install_dependencies"
    bl_label = "Install Dependencies"
    bl_description = "Install MediaPipe and OpenCV and download the pose and hand tracking models"

    update: bpy.props.BoolProperty(
        name="Update", default=False, options={'HIDDEN', 'SKIP_SAVE'},
        description="Install the newest MediaPipe release even if a newer-than-stable one is present")

    @classmethod
    def poll(cls, context):
        return not status["busy"]

    def execute(self, context):
        self.report({'INFO'}, "Installing dependencies... Check Blender console for progress.")
        status["busy"] = True
        threading.Thread(target=install_packages, args=(channel(), self.update, pose_tracker()),
                         daemon=True).start()
        bpy.app.timers.register(_redraw_preferences, first_interval=0.5)
        return {'FINISHED'}


class UninstallAddonDependenciesOperator(bpy.types.Operator):
    """Uninstall Python dependencies"""
    bl_idname = "addon.uninstall_dependencies"
    bl_label = "Uninstall Dependencies"
    bl_description = "Remove the MediaPipe and OpenCV packages and models installed by Open Mocap"

    @classmethod
    def poll(cls, context):
        return not status["busy"]

    def execute(self, context):
        self.report({'INFO'}, "Uninstalling dependencies... Check Blender console for progress.")
        status["busy"] = True
        threading.Thread(target=uninstall_packages, daemon=True).start()
        bpy.app.timers.register(_redraw_preferences, first_interval=0.5)
        return {'FINISHED'}


classes = [InstallAddonDependenciesOperator, UninstallAddonDependenciesOperator]
