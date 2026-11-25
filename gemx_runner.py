"""
    Runs NVIDIA GEM-X on one video and writes its joints for Open Mocap (see gemx.py).
    Not imported by Blender: Track with GEM-X starts it with the Python of GEM-X's own environment
    (PyTorch, CUDA), so Blender needs none of that.

    python gemx_runner.py --repo D:/mocap/GEM-X --video take.mp4 --out take_gemx.npz [--fov 50]
                          [--camera static|background] [--device cuda]

    Progress lines start with "STAGE " for Blender to show; the last line is "DONE <npz>" or
    "ERROR <message>".
"""
import argparse
import math
import os
import subprocess
import sys


def stage(text):
    print("STAGE " + text, flush=True)


# FFmpeg exits with its negative AVERROR code, which Windows shows as a large unsigned number.
FFMPEG_ERRORS = {
    b"EXT ": "an external library failed (libx264 could not start its encoder, usually because memory ran out)",
    12: "it ran out of memory",
    b"INDA": "the video has invalid data",
    2: "a file was not found",
    13: "permission was denied",
}


def ffmpeg_meaning(code):
    """What an ffmpeg exit code means, or None. 3752568763 (0xDFABA7BB) is AVERROR_EXTERNAL."""
    code = int(code)
    if code >= 2**31:
        code -= 2**32
    if code >= 0:
        return None
    tag = (-code).to_bytes(4, "little")
    return FFMPEG_ERRORS.get(tag) or FFMPEG_ERRORS.get(-code)


def free_commit_gb():
    """Memory Windows can still hand out (RAM plus page file), in GB; None elsewhere."""
    try:
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + \
                       [(n, ctypes.c_ulonglong) for n in ("total_phys", "avail_phys", "total_page", "avail_page",
                                                          "total_virtual", "avail_virtual", "avail_ext")]
        s = Status()
        s.dwLength = ctypes.sizeof(Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
            return s.avail_page / 2**30
    except (AttributeError, OSError):
        pass
    return None


def explain_failure(tail, free_gb=None):
    """A plain message for GEM-X failures we recognise in its last output lines, else None."""
    import re
    text = "\n".join(tail)
    match = re.search(r"Command '\['ffmpeg'.*returned non-zero exit status (\d+)", text, re.S)
    if not match:
        return None
    code = int(match.group(1))
    meaning = ffmpeg_meaning(code)
    msg = (f"GEM-X could not re-encode the clip to 30 fps: ffmpeg stopped with exit code {code} "
           f"(0x{code & 0xFFFFFFFF:08X})" + (f", which means {meaning}" if meaning else "") + ".")
    if free_gb is not None:
        msg += f" Windows has {free_gb:.1f} GB of memory (RAM plus page file) free now."
    if meaning and "memory" in meaning:
        msg += " Close other Blender instances or large programs, then run Track with GEM-X again."
    return msg


def fov_config(repo, base, width, height, fov):
    """A copy of the video configuration with the camera's focal length set from a horizontal field of
    view, next to the original (its paths are relative to its own folder)."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(base)
    f = (width / 2.0) / math.tan(math.radians(fov) / 2.0)
    cfg.camera.intrinsics = [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]]
    path = os.path.join(os.path.dirname(base), "_openmocap_fov.yaml")
    OmegaConf.save(cfg, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fov", type=float, default=0.0, help="horizontal field of view in degrees; 0: GEM-X's guess")
    ap.add_argument("--camera", default="static")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reuse", action="store_true", help="read GEM-X's saved .pt again instead of tracking")
    a = ap.parse_args()
    repo = os.path.abspath(a.repo)
    out_npz = os.path.abspath(a.out)
    out_pt = os.path.splitext(out_npz)[0] + ".pt"
    env_bin = os.path.join(os.path.dirname(sys.executable), "Library", "bin")   # a conda env's ffmpeg
    env = dict(os.environ)
    if os.path.isdir(env_bin):
        env["PATH"] = env_bin + os.pathsep + env.get("PATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    import cv2
    cap = cv2.VideoCapture(a.video)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if width <= 0 or height <= 0:
        print(f"ERROR cannot read the video {a.video}", flush=True)
        return 1
    config = os.path.join(repo, "docs", "configs", "body_video_native77.yaml")
    if a.fov > 0:
        config = fov_config(repo, config, width, height, a.fov)

    if a.reuse and os.path.exists(out_pt):
        return write_joints(a, repo, out_pt, out_npz, width, height)
    cmd = [sys.executable, "-m", "gem.inference", "motion-video", "--task", "body", "--input", a.video,
           "--config", config, "--camera-mode", a.camera, "--device", a.device, "--postprocess",
           "--soma-assets", os.path.join(repo, "assets", "soma"), "--output", out_pt]
    stage("Starting GEM-X")
    proc = subprocess.Popen(cmd, cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    tail = []
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        tail = (tail + [line])[-20:]
        if line[:1].isupper() and not line.startswith(("Warning", "Warp", "Module", "UserWarning", "Traceback")) \
                and "Warning" not in line and len(line) < 120:
            stage(line)
    if proc.wait() != 0 or not os.path.exists(out_pt):
        known = explain_failure(tail, free_commit_gb())
        print("ERROR " + (known or "GEM-X failed: " + " | ".join(tail[-4:])), flush=True)
        return 1
    return write_joints(a, repo, out_pt, out_npz, width, height)


def write_joints(a, repo, out_pt, out_npz, width, height):
    stage("Reading the joints")
    import numpy as np
    import torch
    from gem.motion.geometry import MotionGeometry
    res = torch.load(out_pt, map_location="cpu", weights_only=False)
    geo = MotionGeometry(os.path.join(repo, "assets", "soma"), device=a.device, apply_correctives=False)
    joints = {}
    for coord in ("global", "incam"):
        chunks = [c["joints"][0].numpy() for _, c in geo.iter_chunks(res[f"pred_soma_params_{coord}"])]
        joints[coord] = np.concatenate(chunks, axis=0)
    names = list(getattr(geo.body_model.runtime, "public_joint_names", []) or [])
    # The actor's own skeleton at rest (no joint rotations, shape of the first frame): the spine's
    # neutral curve, which the retarget takes off before giving the rig GEM-X's spine bends
    params = res["pred_soma_params_global"]
    rest = {k: v[:, :1].clone() for k, v in params.items()}
    for k in ("body_pose", "global_orient", "transl"):
        rest[k].zero_()
    joints_rest = next(iter(geo.iter_chunks(rest)))[1]["joints"][0, 0].numpy()
    cond = res["conditions"]
    kp = cond["kp2d_native77"].numpy()
    out = {"joints_global": joints["global"], "joints_incam": joints["incam"], "joint_names": np.array(names),
           "R_w2c": res["camera"]["R_w2c"].numpy().reshape(-1, 3, 3),
           "K": cond["K_fullimg"].numpy().reshape(-1, 3, 3)[0],
           "frame_times": cond["frame_times"].numpy().reshape(-1), "fps": np.array(float(res.get("sampling_fps", 30.0))),
           "width": np.array(width), "height": np.array(height),
           "obs_conf": kp.reshape(-1, kp.shape[-2], kp.shape[-1])[:, :, 2],
           "static_conf": torch.sigmoid(res["static_conf_logits"][0]).numpy(), "joints_rest": joints_rest,
           # the 2D keypoints GEM-X saw (pixels, confidence): Track Sword searches the video around them
           "kp2d": kp.reshape(-1, kp.shape[-2], kp.shape[-1])}
    np.savez(out_npz, **out)
    print("DONE " + out_npz, flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:                      # report to Blender instead of a bare traceback
        print(f"ERROR {type(error).__name__}: {error}", flush=True)
        raise
