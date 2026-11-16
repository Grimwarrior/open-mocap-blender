"""
    Retarget and Bake Motion on a rig with animation layers (NLA tracks, as the NLA Anim Layers
    add-on makes them).

    A rig in NLA tweak mode (a layer being edited) refuses any change of its action, so Retarget
    and Bake would stop with an error: both leave tweak mode first. While the retarget drives the
    rig its layers are switched off, as its action is set aside: the constraints aim the bones but
    keep each bone's twist, and leave bones they do not drive, so the layers would show through
    (and be baked in). Bake Motion then puts the baked mocap into the layer stack:

      BASE    a new layer right above the lowest playing one (the base), which is muted on a
              full-rig bake; the layers above keep playing on top of the new mocap
      TOP     a new layer on top of all of them, everything else as it was
      ACTION  the rig's active action with the layers off (as without layers)

    The layers' own track list is the NLA stack, so this works with or without that add-on; with
    it, its own helpers make the layer (identity timing, HOLD, fitted to the scene and keys) and
    its layer list is pointed at the new layer.
"""
import sys

import bpy

NLA_PROP = "open_mocap_nla"         # the rig's use_nla before Retarget switched its layers off
LAYER_MODES = [
    ('BASE', "New Base Layer", "Add the baked mocap as a layer right above the base (the lowest playing "
                               "layer) and mute the old base; the layers above keep playing on top of it"),
    ('TOP', "New Top Layer", "Add the baked mocap as a new layer on top of all layers (it covers them; "
                             "the layers are kept)"),
    ('ACTION', "Active Action", "Bake into the rig's active action and turn its layers off"),
]


def layers_core():
    """The NLA Anim Layers add-on's core module, if that add-on is enabled."""
    for name, module in list(sys.modules.items()):
        if name.endswith("nla_anim_layers.core") and hasattr(module, "add_layer"):
            return module
    return None


def has_layers(rig):
    ad = getattr(rig, "animation_data", None)
    return bool(ad is not None and len(getattr(ad, "nla_tracks", ())))


def leave_tweak(context, rig):
    """Stop editing a layer (NLA tweak mode). :return: True if the rig was in tweak mode."""
    ad = getattr(rig, "animation_data", None)
    if ad is None or not getattr(ad, "use_tweak_mode", False):
        return False
    core = layers_core()
    if core is not None:
        try:
            core.exit_edit(rig, context.scene, context)     # through the NLA operator: clears the scene flag too
        except Exception as error:                           # its RNA fallback below still works
            print(f"[Open Mocap] leaving layer edit through NLA Anim Layers failed: {error}")
    if ad.use_tweak_mode:
        ad.use_tweak_mode = False
    return True


def layers_off(context, rig):
    """Retarget: leave tweak mode and switch the layers off while the constraints drive the rig,
    remembering whether they played. :return: number of layers switched off."""
    leave_tweak(context, rig)
    if not has_layers(rig):
        return 0
    ad = rig.animation_data
    if NLA_PROP not in rig:
        rig[NLA_PROP] = bool(ad.use_nla)
    ad.use_nla = False
    return len(ad.nla_tracks)


def layers_back(rig):
    """Remove Retarget: the layers play again if they did before. :return: True if restored."""
    if rig is None or NLA_PROP not in rig:
        return False
    if rig.animation_data is not None:
        rig.animation_data.use_nla = bool(rig[NLA_PROP])
    del rig[NLA_PROP]
    return True


def forget(rig):
    if rig is not None and hasattr(rig, "keys") and NLA_PROP in rig:
        del rig[NLA_PROP]


def base_track(ad):
    """The lowest track that plays."""
    return next((t for t in ad.nla_tracks if not t.mute), None)


def unique_name(ad, base):
    names = {t.name for t in ad.nla_tracks}
    if base not in names:
        return base
    n = 2
    while f"{base} {n}" in names:
        n += 1
    return f"{base} {n}"


def _new_track(context, rig, name, above, action, slot):
    core = layers_core()
    if core is not None:
        return core.add_layer(context, rig, name=name, blend='REPLACE', above=above, action=action, slot=slot)
    ad = rig.animation_data
    track = ad.nla_tracks.new(prev=above)
    track.name = name
    track.select = False
    start, end = action.frame_range
    strip = track.strips.new(name, int(start), action)
    if slot is not None and strip.action_slot != slot:
        strip.action_slot = slot
    strip.blend_type = 'REPLACE'
    strip.use_auto_blend = False
    strip.blend_in = strip.blend_out = 0.0
    strip.extrapolation = 'HOLD'
    return track


def add_to_layers(context, rig, action, mode, name, full=True):
    """
        Moves the rig's active (baked) action into the layer stack (see the top of this module).
        :param mode: 'BASE' or 'TOP'. :param full: the whole rig was baked (only then is the old base
        muted: a partial bake covers only its bones). :return: (new track, muted old base or None).
    """
    ad = rig.animation_data
    slot = getattr(ad, "action_slot", None)
    ad.action = None
    base = base_track(ad) if mode == 'BASE' else None
    above = base if base is not None else (ad.nla_tracks[-1] if len(ad.nla_tracks) else None)
    track = _new_track(context, rig, unique_name(ad, name), above, action, slot)
    muted = None
    if base is not None and full:
        base.mute = True
        muted = base
    ad.use_nla = True
    forget(rig)
    core = layers_core()
    if core is not None and hasattr(rig, "anim_layers_index"):
        index = next((i for i, t in enumerate(ad.nla_tracks) if t == track), -1)
        if index >= 0:
            try:
                core.set_active(context, rig, index, edit=False)
            except Exception:
                pass
    return track, muted
