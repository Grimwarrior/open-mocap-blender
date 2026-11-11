# Capture pipeline regression checks

Run from the add-on directory, with Python and NumPy:

```powershell
python -B -m unittest discover -s tests -v
python -B tests/synthetic_combine.py
```

These are interpreted Python checks, with bytecode file generation disabled. They do not
build the add-on, open Blender, load detector models, or change a Blender scene. The unit
tests use mock Blender objects; they do not validate Blender's dependency graph or UI.

## First capture improvement batch

- Live timestamps measure acquisition time before inference. Files use OpenCV presentation
  timestamps, with a monotonic frame-duration fallback for unsupported/repeated timestamps.
  MediaPipe receives the same timeline, rounded to strictly increasing milliseconds.
- Animation retains subframe times instead of collapsing closely spaced samples to integers.
- Capture format 2 adds `position_meta`, `zlist_meta`, `position_meta_raw`,
  `position_missing_times`, `zlist_missing_times`, and separate front/side hand frame rates.
  Metadata stores depth confidence, source labels, and per-joint depth states. Existing
  visibility arrays remain the image-position confidence. Version 1 remains readable,
  including captures truncated by the old original-combine path.
- RTMW depth is accepted only when at least three reliable torso landmarks agree with
  MediaPipe within 35% of RTMW torso length. This is a conservative geometric gate,
  not a persistent actor identity tracker. Missing/rejected depth has zero confidence.
- Failed whole-body detections retain explicit timestamps beside the sparse observations.
  Between observations, interpolation confidence decays across short dropouts and is zero
  for gaps over 120 ms. This intentionally treats very low-rate captures conservatively.
  Display/keyframe curves may still span a dropout; that does not turn it into a measurement.
- Improved Combine rejects untrusted automatic synchronization and insufficient overlap
  without replacing the current capture. A manual offset can override automatic rejection.
  Missing metadata or short takes no longer silently invoke original combine; the original
  method remains an explicit selection. Successful re-combines use raw front metadata.
- Combined confidence accounts for the side root and front depth fallback. It is stored
  with the animated source, so later cache changes cannot affect that source's confidence.
  Arm/head direction weighting and foot-contact acceptance use it; legacy sources retain
  the previous behavior. This does not implement a new full-body/contact solver.
- Arm and leg solvers measure each side independently. Overall leg scale uses both legs.
- Helper collections carry a rig ID owner reference. Cleanup preserves other rigs' helper
  dependencies, including duplicated rigs and the old shared helper collection.

## Manual Blender checks still needed

1. Track front/side files with MediaPipe and RTMW, including an occlusion; save, reload,
   combine, animate, retarget, and bake. Confirm the final motion visually.
2. Test live capture under slow inference and a variable-frame-rate video; compare duration
   against the original capture. Decoder support determines whether real PTS are available.
3. Try unrelated or static takes: auto combine should leave the current result unchanged;
   set an explicit offset to combine footage whose synchronization is known.
4. Retarget two characters, rename one, rebuild/clear either, and verify the other still moves.
   Repeat with a duplicated rig that shares helper targets and with an older saved scene.
5. Test a rig with deliberately different left/right limb lengths and inspect reach/foot placement.

The synthetic benchmark measures approximate orthographic reconstruction, not real footage
quality. Its existing depth metric permits a depth-sign correction. It does not validate the
retargeted character, foot sliding, or hand grip accuracy.

## Bake fix validation

`test_bake.py` covers source action preservation while sampling, failed/empty bakes,
invalid ranges, and muting only mocap constraints on the bones actually baked. Full-rig
bakes ignore bone selection; selected-only bakes display a partial-bake warning and reject
an empty selection. Full-rig playback disables NLA evaluation because its visual result
has already been captured; tracks and prior actions remain available.

The bake was also exercised in the user's running Blender 5.1: 65 bones, frames 1–200,
with all 36 OM constraints muted afterward. Nine sampled poses matched the driven source
to a maximum matrix-component difference of approximately 0.00057. This validates that
specific bake, not every rig or the broader tracking pipeline. The blend file was not saved.

## More than two views

- `combine.combine_multi` takes the front video and any number of other views. With one view it
  is `combine_improved`. `tests/test_multi_view.py` checks on the synthetic actor that a second
  side camera lowers the depth error, that an opposite camera places joints the front view
  hides, that an opposite view of another performance is left out, that a working view stands
  in for a useless side video, and that views without depth are refused.
- Each further view's direction is fitted to a first depth, not entered. A view whose motion no
  direction explains is left out and named in the report (`report["views"]`).
- Position (as opposed to depth) is taken from another view only for joints hidden in the front
  view, and only when the views' heights agree like one performance filmed by several cameras.
- Further views are tracked into `globalVariables.extra_views` (one entry per video; tracking a
  video again replaces its entry) and saved in capture files as `extra_views`. The unit tests do
  not cover the tracking, the panel or the operators.

## Perspective, the leg solver and the skeleton solve

- `synthetic_combine.make_take(camera_distance=...)` films with real cameras: nearer is bigger
  and, below the camera, lower in the picture. The earlier scenarios have no perspective and
  could not show errors that come from it (false leg swaps, a floor that moves in the picture).
- `tests/synthetic_retarget.py` is the same idea for the retarget's leg solver: an actor who
  steps, lunges, crouches and turns on the ball of one foot, with known contacts and knee
  directions. Run it directly for the table; `tests/test_retarget_solver.py` holds the limits.
  It drives `mocap_floor` and `leg_directions` with mocked Blender; it does not build
  constraints, pose a rig or bake.
- Foot contacts use heel and toe, each with separate thresholds for coming down and lifting
  off, relative to where that point last stood. Flat feet are held; a foot on its ball keeps its
  toe in place. Knee and elbow direction is weighted by how far the joint is off the limb's line
  and carried through straight stretches (`bend_direction`).
- `combine._solve_skeleton` replaces bone-by-bone clipping (`SKELETON_SOLVE`): all joint depths
  are solved together per frame against confidence and bone lengths. Bones that are too short
  are not pushed out (`SOLVE_LENGTHEN = 0`): that helps without perspective and hurts with a close camera.
