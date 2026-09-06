"""Independent ComfyUI MiniMax H3 runtime patches for Director segment continuity.

Enables interior keyframe time coordinates and keyframe+reference payload
coexistence so the Director can pin a previous segment's tail into the next
segment. Behavior is inspired by community Motion Context work; this module is
an original Apache-2.0 implementation for AIMixer/ComfyUI_MiniMaxH3_Director.

Does not copy third-party GPL sources. Refuses to stack on foreign H3 layout /
payload wrappers (including standalone Motion Context packs).
"""

from __future__ import annotations

import logging

import torch

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.h3_context_patches")

# Director-owned markers (not shared with third-party packs).
CTX_FRAME_KEY = "director_context_index"
CTX_AUDIO_END_KEY = "director_context_audio_end"
# 段间衔接 (join): marks a conditioning anchor that lives in NEGATIVE time —
# placed before the target origin so the output latent is never extended.
CTX_NEG_KEY = "director_negative_context_index"
# Optional float (pixel count) by which the whole negative-time block stack is
# shifted further back: when the phase-aligned context window ends ``gap`` px
# before the previous segment's export end, the nearest block would otherwise
# collide with the identity pin at frame 0 showing a DIFFERENT frame (blur).
CTX_NEG_SHIFT_KEY = "director_negative_context_px_shift"
# 稀疏取帧: per-block REAL negative distance (px back from the previous export
# end). When present, the layout rewrite places that block at
# ``origin - FRAME_RESCALE * back_px`` instead of the whole-stack shift +
# reversed-cycle spacing. Dense blocks never carry this key -> byte-identical.
CTX_NEG_BACK_KEY = "director_negative_context_back_px"
LAYOUT_MARKER = "_h3_director_continuity_layout_patch"
PAYLOAD_MARKER = "_h3_director_continuity_payload_patch"

# Known foreign markers — stand down / refuse rather than double-wrap.
_FOREIGN_LAYOUT_MARKERS = (
    LAYOUT_MARKER,
    "_h3_motion_context_layout_patch",
)
_FOREIGN_PAYLOAD_MARKERS = (
    PAYLOAD_MARKER,
    "_h3_motion_context_payload_patch",
)

_layout_orig = None
_layout_applied = False
_payload_orig = None
_payload_applied = False

_REF_KINDS = ("ref_img", "ref_audio")


def _mm():
    import comfy.ldm.minimax.model as mm

    return mm


def layout_patch_applied() -> bool:
    return _layout_applied


def payload_patch_applied() -> bool:
    return _payload_applied


def _has_marker(fn, markers: tuple[str, ...]) -> str | None:
    if fn is None:
        return None
    for name in markers:
        if getattr(fn, name, False):
            return name
    return None


def _target_origin(layout) -> float:
    """Time coordinate where the target video segment begins."""
    a, b, kind = layout.segments[-1]
    if kind != "video" or b <= a:
        raise RuntimeError(
            "Director continuity: expected final layout segment to be target video "
            f"(got {kind!r} spanning {b - a} rows)."
        )
    return float(layout.position_ids[a, 0])


def _keyframe_time(mm, text_len: float, pixel_index: int) -> float:
    """Stock-compatible time for pixel frame ``pixel_index`` on the target clip."""
    return float(text_len) + mm.FRAME_RESCALE * float(int(pixel_index))


def join_negative_distance(steps_back: int) -> float:
    """RoPE-time distance back from the target origin for a 段间衔接 context slot.

    ``steps_back`` = 0 is the last context frame, which sits exactly at the
    target origin; each step further back consumes
    ``FRAME_RESCALE · FRAME_PER_TOKEN`` on the reversed (1,4,4,4,4) cycle —
    the same negative-time placement Extend uses (``_context_k_distance``).
    """
    mm = _mm()
    m = max(0, int(steps_back))
    return sum(mm.FRAME_RESCALE * mm.FRAME_PER_TOKEN[(-i) % 5] for i in range(1, m + 1))


def _expected_ref_kinds(block: dict) -> tuple[str, ...]:
    kind = block.get("kind")
    if kind == "image":
        return ("ref_img",)
    if kind == "audio":
        return ("ref_audio",) if int(block.get("ref_audio_t", 0) or 0) > 0 else ()
    if kind in ("video", "video_audio"):
        if int(block.get("ref_audio_t", 0) or 0) > 0:
            return ("ref_audio", "ref_img")
        return ("ref_img",)
    raise RuntimeError(f"Director continuity: unknown reference kind {kind!r}.")


def _ref_segments(layout, refs: list | None) -> dict[int, dict[str, tuple[int, int]]]:
    ref_segs = [(a, b, k) for a, b, k in layout.segments if k in _REF_KINDS]
    wanted = [(i, k) for i, blk in enumerate(refs or []) for k in _expected_ref_kinds(blk)]
    if len(wanted) != len(ref_segs):
        raise RuntimeError(
            "Director continuity: reference segment count mismatch "
            f"(expected {len(wanted)}, layout has {len(ref_segs)})."
        )
    out: dict[int, dict[str, tuple[int, int]]] = {}
    for (i, kind), (a, b, got) in zip(wanted, ref_segs):
        if got != kind:
            raise RuntimeError(
                f"Director continuity: reference block {i} expected {kind}, got {got}."
            )
        out.setdefault(i, {})[kind] = (a, b)
    return out


def _rewrite_keyframe_times(layout, text_len, keyframes, refs=None):
    del refs  # origin is read from the built layout
    mm = _mm()
    origin = _target_origin(layout)
    offset = origin - float(text_len)
    if offset and any(
        kf.get(CTX_FRAME_KEY) is None and kf.get(CTX_NEG_KEY) is None
        for kf in keyframes
    ):
        raise RuntimeError(
            "Director continuity: cannot mix unmarked stock keyframes with "
            "director context anchors when references shift the target origin."
        )
    cond_spans = [(a, b) for a, b, kind in layout.segments if kind == "cond"]
    if len(cond_spans) != len(keyframes):
        raise RuntimeError(
            f"Director continuity: expected {len(keyframes)} cond segments, "
            f"layout has {len(cond_spans)}."
        )
    # 段间衔接: context blocks carry only CTX_NEG_KEY; the identity pin carries
    # both CTX_FRAME_KEY and CTX_NEG_KEY (hard lock at frame 0). When the
    # phase-aligned context window ends ``gap`` pixels before the previous
    # segment's export end, the whole block stack shifts back by that gap so the
    # nearest block shows the frames immediately before the seam instead of
    # colliding with the pin at frame 0 (which would blend two different frames
    # into a blurry first frame).
    neg_count = sum(
        1
        for kf in keyframes
        if kf.get(CTX_NEG_KEY) is not None and CTX_FRAME_KEY not in kf
    )
    neg_shift_px = float(
        next((kf[CTX_NEG_SHIFT_KEY] for kf in keyframes if CTX_NEG_SHIFT_KEY in kf), 0.0)
    )
    for (a, b), kf in zip(cond_spans, keyframes):
        if CTX_NEG_KEY in kf:
            if CTX_FRAME_KEY in kf:
                # Identity pin: previous segment's true last exported pixel at
                # frame 0 (zero RoPE distance).
                layout.position_ids[a:b, 0] = origin
            elif CTX_NEG_BACK_KEY in kf:
                # 稀疏取帧: per-block REAL negative distance (px back from the
                # previous export end) — placed at its own true negative RoPE
                # time, so the far ~1s anchors land on the actual scene moments.
                back_px = float(kf[CTX_NEG_BACK_KEY])
                layout.position_ids[a:b, 0] = origin - mm.FRAME_RESCALE * back_px
            else:
                # Negative-time context: p == neg_count-1 is the frame closest
                # to origin; each earlier frame steps back on the FRAME_PER_TOKEN
                # cycle (Extend's `_context_k_distance`, transposed to pixels).
                steps_back = (neg_count - 1) - int(kf[CTX_NEG_KEY])
                layout.position_ids[a:b, 0] = (
                    origin
                    - mm.FRAME_RESCALE * neg_shift_px
                    - join_negative_distance(steps_back)
                )
            continue
        p = kf.get(CTX_FRAME_KEY)
        if p is None:
            continue
        layout.position_ids[a:b, 0] = _keyframe_time(
            mm, text_len, int(p)
        ) + offset


def _rewrite_audio_timeline(layout, text_len, refs):
    del text_len
    mm = _mm()
    marked = [i for i, r in enumerate(refs or []) if r.get(CTX_AUDIO_END_KEY) is not None]
    if len(marked) != 1:
        raise RuntimeError(
            "Director continuity: audio continuation requires exactly one marked "
            f"audio reference (found {len(marked)})."
        )
    idx = marked[0]
    blk = refs[idx]
    if blk.get("kind") != "audio":
        raise RuntimeError(
            f"Director continuity: {CTX_AUDIO_END_KEY} set on non-audio ref "
            f"{blk.get('kind')!r}."
        )
    rt = int(blk.get("ref_audio_t", 0) or 0)
    if rt <= 0:
        return
    seg = _ref_segments(layout, refs).get(idx, {}).get("ref_audio")
    if seg is None:
        raise RuntimeError("Director continuity: marked audio ref produced no layout rows.")
    a, b = seg
    if b - a != 2 * rt:
        raise RuntimeError(
            f"Director continuity: audio ref rows {b - a} != expected stereo {2 * rt}."
        )
    origin = _target_origin(layout)
    slot_start = float(layout.position_ids[a, 0])
    end_frame = float(blk[CTX_AUDIO_END_KEY])
    desired_start = origin + mm.FRAME_RESCALE * end_frame - float(rt)
    layout.position_ids[a:b, 0] = layout.position_ids[a:b, 0] + (desired_start - slot_start)


def _director_layout_init(
    self,
    text_len,
    latent_t,
    latent_h,
    latent_w,
    audio_t,
    keyframes=None,
    refs=None,
):
    # Stock accepts only first/last; pass interior anchors as index 0, then rewrite.
    stock_keyframes = None
    if keyframes:
        stock_keyframes = []
        for kf in keyframes:
            entry = dict(kf)
            if CTX_FRAME_KEY in entry:
                entry["resolved_frame_index"] = 0
            stock_keyframes.append(entry)
    _layout_orig(
        self,
        text_len,
        latent_t,
        latent_h,
        latent_w,
        audio_t,
        keyframes=stock_keyframes,
        refs=refs,
    )
    has_ctx_kf = bool(keyframes) and any(kf.get(CTX_FRAME_KEY) is not None for kf in keyframes)
    has_ctx_audio = bool(refs) and any(r.get(CTX_AUDIO_END_KEY) is not None for r in refs)
    if has_ctx_kf:
        _rewrite_keyframe_times(self, text_len, keyframes, refs)
    if has_ctx_audio:
        _rewrite_audio_timeline(self, text_len, refs)


setattr(_director_layout_init, LAYOUT_MARKER, True)


def _self_test_layout() -> None:
    mm = _mm()
    text_len, latent_t, lh, lw, audio_t = 7, 7, 22, 38, 16
    last_px = sum(mm.FRAME_PER_TOKEN[k % 5] for k in range(latent_t)) - 1
    # Official creates cond rows only for keyframes carrying a video latent; use a
    # one-step block like the real continuity anchors (h3_motion_context:393-400).
    dummy_lat = torch.zeros(1, 16, 1, lh, lw)

    def build(keyframes=None, refs=None, rewrite=False):
        lay = mm.PackedLayout.__new__(mm.PackedLayout)
        stock_kf = None
        if keyframes:
            stock_kf = []
            for kf in keyframes:
                entry = dict(kf)
                if rewrite and CTX_FRAME_KEY in entry:
                    entry["resolved_frame_index"] = 0
                stock_kf.append(entry)
        _layout_orig(
            lay,
            text_len,
            latent_t,
            lh,
            lw,
            audio_t,
            keyframes=stock_kf,
            refs=refs,
        )
        if rewrite:
            _rewrite_keyframe_times(lay, text_len, keyframes, refs)
        return lay

    stock = build(
        keyframes=[
            {"resolved_frame_index": 0, "latent": dummy_lat},
            {"resolved_frame_index": last_px, "latent": dummy_lat},
        ]
    )
    ours = build(
        keyframes=[
            {"resolved_frame_index": 0, CTX_FRAME_KEY: 0, "latent": dummy_lat},
            {"resolved_frame_index": 0, CTX_FRAME_KEY: last_px, "latent": dummy_lat},
        ],
        rewrite=True,
    )
    if not torch.equal(stock.position_ids, ours.position_ids):
        raise RuntimeError("Director continuity layout self-test: endpoint mismatch vs stock")

    run = [{"resolved_frame_index": 0, CTX_FRAME_KEY: i, "latent": dummy_lat} for i in range(4)]
    interior = build(keyframes=run, rewrite=True)
    times = [float(interior.position_ids[a, 0]) for a, _, k in interior.segments if k == "cond"]
    if len(times) != 4 or any(times[i] >= times[i + 1] for i in range(3)):
        raise RuntimeError("Director continuity layout self-test: interior times not increasing")

    # 段间衔接 (negative time): context blocks first, identity pin LAST — mirrors
    # apply_negative_time_context / Extend. p=0 is the furthest-back block
    # (steps_back = neg_count-1-p); the nearest block and the pin both land on
    # the target origin, a legitimate tie (Extend's k=0 == first-frame anchor).
    neg_keyframes = [
        {"resolved_frame_index": 0, CTX_NEG_KEY: 0, "latent": dummy_lat},
        {"resolved_frame_index": 0, CTX_NEG_KEY: 1, "latent": dummy_lat},
        {"resolved_frame_index": 0, CTX_NEG_KEY: 2, "latent": dummy_lat},
        {"resolved_frame_index": 0, CTX_NEG_KEY: 3, "latent": dummy_lat},
        {"resolved_frame_index": 0, CTX_FRAME_KEY: 0, CTX_NEG_KEY: 0, "latent": dummy_lat},
    ]
    neg = build(keyframes=neg_keyframes, rewrite=True)
    neg_times = [float(neg.position_ids[a, 0]) for a, _, k in neg.segments if k == "cond"]
    if len(neg_times) != 5:
        raise RuntimeError("Director join layout self-test: expected 5 cond rows")
    neg_origin = _target_origin(neg)
    expected = [neg_origin - join_negative_distance(m) for m in (3, 2, 1, 0)]
    if any(abs(neg_times[i] - expected[i]) > 1e-6 for i in range(4)):
        raise RuntimeError("Director join layout self-test: negative context times mismatch")
    if abs(neg_times[4] - neg_origin) > 1e-6:
        raise RuntimeError("Director join layout self-test: identity pin not at origin")
    # Non-decreasing: each earlier block strictly behind, the final tie (nearest
    # block + pin) at the origin is expected by design.
    if any(neg_times[i] > neg_times[i + 1] + 1e-6 for i in range(4)):
        raise RuntimeError("Director join layout self-test: negative times not increasing")

    # Gap-shifted variant: when the context window ends ``gap`` px before the
    # export end, the block stack moves back by FRAME_RESCALE*gap while the pin
    # stays at origin — so frame 0 is never a blend of two different frames.
    gap = 3
    shifted_kf = [dict(k, **{CTX_NEG_SHIFT_KEY: float(gap)}) for k in neg_keyframes]
    shifted = build(keyframes=shifted_kf, rewrite=True)
    shifted_times = [
        float(shifted.position_ids[a, 0]) for a, _, k in shifted.segments if k == "cond"
    ]
    if len(shifted_times) != 5:
        raise RuntimeError("Director join layout self-test: expected 5 shifted cond rows")
    shifted_origin = _target_origin(shifted)
    exp_shifted = [
        shifted_origin - mm.FRAME_RESCALE * gap - join_negative_distance(m)
        for m in (3, 2, 1, 0)
    ]
    if any(abs(shifted_times[i] - exp_shifted[i]) > 1e-6 for i in range(4)):
        raise RuntimeError("Director join layout self-test: shifted block times mismatch")
    if abs(shifted_times[4] - shifted_origin) > 1e-6:
        raise RuntimeError("Director join layout self-test: shifted pin not at origin")
    if any(shifted_times[i] > shifted_times[i + 1] + 1e-6 for i in range(4)):
        raise RuntimeError("Director join layout self-test: shifted negative times not increasing")


def _classify_layout_owner() -> str | None:
    """Return None, 'ours', 'foreign_mc', or 'foreign_other'."""
    mm = _mm()
    init = getattr(getattr(mm, "PackedLayout", None), "__init__", None)
    if init is None:
        return None
    if getattr(init, LAYOUT_MARKER, False):
        return "ours"
    if getattr(init, "_h3_motion_context_layout_patch", False):
        return "foreign_mc"
    if getattr(init, "__name__", "") in {"_patched_init", "_director_layout_init"}:
        return "foreign_other"
    if hasattr(init, "__wrapped__"):
        return "foreign_other"
    home = getattr(mm.PackedLayout, "__module__", None)
    where = getattr(init, "__module__", None)
    if home and where and where != home:
        return "foreign_other"
    return None


def _is_transparent_foreign_layout_wrapper(init) -> bool:
    """True for SolAttn-style pass-through wrappers: a module-level ``def __init__``
    from a foreign module, carrying no Director/MC markers and no functools chain.

    Cheap pre-filter only — the self-test in ensure_layout_patch is the real gate
    (it verifies the wrapper accepts index-0 interior anchors and that Director's
    time rewrite stays correct through it)."""
    if init is None or getattr(init, "__name__", "") != "__init__":
        return False
    if getattr(init, LAYOUT_MARKER, False) or getattr(
        init, "_h3_motion_context_layout_patch", False
    ):
        return False
    if hasattr(init, "__wrapped__"):
        return False
    return True


def ensure_layout_patch() -> bool:
    """Install layout patch on first continuity use. Returns True if usable."""
    global _layout_orig, _layout_applied
    if _layout_applied:
        return True
    mm = _mm()
    owner = _classify_layout_owner()
    if owner == "ours":
        _layout_applied = True
        return True
    if owner == "foreign_mc":
        raise RuntimeError(
            "Director continuity: standalone ComfyUI-H3-Motion-Context (or a fork) "
            "already patched MiniMax H3 layout. Disable that custom node pack and "
            "restart ComfyUI — both packs cannot own PackedLayout.__init__."
        )
    if owner == "foreign_other":
        # SolAttn-style packs wrap PackedLayout.__init__ with a transparent
        # pass-through (call original init, then read the built span). That can
        # coexist: our rewrite sits on top of it. Any wrapper that mutates
        # keyframes (Motion Context forks etc.) fails the self-test and is refused.
        init = getattr(getattr(mm, "PackedLayout", None), "__init__", None)
        if _is_transparent_foreign_layout_wrapper(init):
            _layout_orig = init
            try:
                _self_test_layout()
            except Exception as exc:
                _layout_orig = None
                raise RuntimeError(
                    "Director continuity: foreign layout wrapper failed the "
                    f"coexistence self-test ({exc}). Disable the other pack and "
                    "restart ComfyUI."
                ) from exc
            mm.PackedLayout.__init__ = _director_layout_init
            _layout_applied = True
            log.info(
                "Director continuity: interior keyframe anchors enabled "
                "(stacked on a foreign transparent layout wrapper)"
            )
            return True
        raise RuntimeError(
            "Director continuity: another pack already patched MiniMax H3 "
            "PackedLayout.__init__. Disable the other pack and restart ComfyUI."
        )
    if not hasattr(mm, "PackedLayout") or not hasattr(mm, "FRAME_RESCALE"):
        raise RuntimeError("Director continuity: MiniMax H3 model module incomplete.")
    _layout_orig = mm.PackedLayout.__init__
    try:
        _self_test_layout()
    except Exception as exc:
        _layout_orig = None
        raise RuntimeError(
            f"Director continuity: layout self-test failed ({exc}). "
            "Interior keyframe anchors unavailable."
        ) from exc
    mm.PackedLayout.__init__ = _director_layout_init
    _layout_applied = True
    log.info("Director continuity: interior keyframe anchors enabled")
    return True


def _director_extra_conds(self, **kwargs):
    out = _payload_orig(self, **kwargs)
    keyframes = kwargs.get("minimax_keyframes", None)
    refs = kwargs.get("minimax_refs", None)
    if not keyframes or not refs:
        return out
    if not (
        any(CTX_FRAME_KEY in kf for kf in keyframes)
        or any(CTX_AUDIO_END_KEY in r for r in refs)
    ):
        return out
    cond = out.get("minimax_payload", None)
    payload = getattr(cond, "cond", None) if cond is not None else None
    if not isinstance(payload, dict):
        log.warning("Director continuity: could not reach H3 payload for keyframe+ref merge")
        return out
    kf_video = [kf["latent"] for kf in keyframes if "latent" in kf]
    ref_video = [r["latent"] for r in refs if "latent" in r]
    payload["cond_video_latents"] = kf_video + ref_video
    payload["cond_audio_latents"] = [
        r["audio_latent"] for r in refs if r.get("audio_latent") is not None
    ]
    return out


setattr(_director_extra_conds, PAYLOAD_MARKER, True)


def _classify_payload_owner() -> str | None:
    import comfy.model_base as model_base

    cls = getattr(model_base, "MiniMaxH3", None)
    fn = getattr(cls, "extra_conds", None) if cls is not None else None
    if fn is None:
        return None
    if getattr(fn, PAYLOAD_MARKER, False):
        return "ours"
    if getattr(fn, "_h3_motion_context_payload_patch", False):
        return "foreign_mc"
    if getattr(fn, "__name__", "") in {"_patched_extra_conds", "_director_extra_conds"}:
        return "foreign_other"
    if hasattr(fn, "__wrapped__"):
        return "foreign_other"
    home = getattr(cls, "__module__", None)
    where = getattr(fn, "__module__", None)
    if home and where and where != home:
        return "foreign_other"
    return None


def ensure_payload_patch() -> bool:
    """Install payload merge patch when audio refs coexist with keyframes."""
    global _payload_orig, _payload_applied
    if _payload_applied:
        return True
    owner = _classify_payload_owner()
    if owner == "ours":
        _payload_applied = True
        return True
    if owner == "foreign_mc":
        raise RuntimeError(
            "Director continuity: standalone ComfyUI-H3-Motion-Context already patched "
            "MiniMaxH3.extra_conds. Disable that pack and restart ComfyUI."
        )
    if owner == "foreign_other":
        raise RuntimeError(
            "Director continuity: another pack already patched MiniMaxH3.extra_conds. "
            "Disable the other pack and restart ComfyUI."
        )
    import comfy.model_base as model_base

    cls = getattr(model_base, "MiniMaxH3", None)
    if cls is None or not hasattr(cls, "extra_conds"):
        raise RuntimeError("Director continuity: MiniMaxH3.extra_conds not found.")
    _payload_orig = cls.extra_conds
    cls.extra_conds = _director_extra_conds
    _payload_applied = True
    log.info("Director continuity: keyframe/ref coexistence enabled")
    return True
