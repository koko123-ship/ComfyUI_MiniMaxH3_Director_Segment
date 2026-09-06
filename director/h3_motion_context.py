"""Director-owned MiniMax H3 segment motion/audio continuation helpers.

Pins the previous segment's tail into the next segment as never-denoised
conditioning, then trims that prefix from decoded output. Inspired by the
community Motion Context approach; original Apache-2.0 code for this Director.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from .h3_context_patches import (
    CTX_AUDIO_END_KEY,
    CTX_FRAME_KEY,
    CTX_NEG_KEY,
    CTX_NEG_SHIFT_KEY,
    CTX_NEG_BACK_KEY,
    ensure_layout_patch,
    ensure_payload_patch,
)

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.h3_motion_context")

FPS = 24.0
AUDIO_HZ = 40.0
FRAME_RESCALE = 5.0 / 3.0
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)

# Pixel windows that map to a whole number of VAE latent steps from cycle 0.
# 17k+5 grid up to 226 (≈10s @ 24fps: 226 = 17*13 + 5).
CONTEXT_FRAME_CHOICES = (5, 22, 39, 56, 73, 90, 107, 124, 141, 158, 175, 192, 209, 226)
DEFAULT_CONTEXT_FRAMES = 22
VIDEO_RUN_GRID = (124, 107, 90, 73, 56, 39, 22, 5, 1)

# 段间衔接 (join) only: the continuity grid extended to ~15s (17k+5 up to
# 17*21+5 = 362px ≈ 15.1s @24fps). 段间引导 keeps CONTEXT_FRAME_CHOICES (≤226).
JOIN_FRAME_CHOICES = CONTEXT_FRAME_CHOICES + (
    243, 260, 277, 294, 311, 328, 345, 362,
)
# Dense join window snap. Legacy choices (≤226) keep VIDEO_RUN_GRID for
# byte-identical behaviour; the new extended reach adds whole-step entries so a
# dense ~15s window is actually honoured (option A), not truncated to 124.
JOIN_RUN_GRID = (
    362, 345, 328, 311, 294, 277, 260, 243, 226, 209, 192, 175, 158,
    141, 124, 107, 90, 73, 56, 39, 22, 5, 1,
)

# 稀疏取帧 constants (fixed, per design): the tail keeps 7 contiguous steps
# hugging the seam — one ALIGNED 5+2=7-token window up against the pin,
# mirroring how the identity pin itself is decoded from [total-7:total]; a
# 6-step tail lands one token short of that window (the Extend failure mode
# documented at the pin). Beyond the tail a single 1-token step every ~1s
# (@24fps) carries scene identity out to the chosen reach.
SPARSE_CTX_TAIL_STEPS = 7
SPARSE_CTX_ANCHOR_FRAMES = 24
# 稀疏取帧·时间折叠 (join): the real tail keeps its TRUE RoPE time (motion into
# the seam); the far 1/s anchors (scene appearance only) have their claimed time
# FOLDED into a compact near-simultaneous still pile hugging the tail's deep
# edge, so the whole reference reads as ONE last small segment instead of a
# stretched deep history. Far frames sit sub-frame apart (≈ the same instant) so
# they act as appearance stills, never as motion-bearing history.
SPARSE_CTX_FOLD_GAP_PX = 1.5   # clear claimed gap (px) tail-deep-edge -> far pile
SPARSE_CTX_FOLD_SUB_PX = 0.7   # claimed spacing (px) between far pile frames (sub-frame)

CONTINUITY_TASK_KEYS = frozenset({"t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"})
# v8: v7 + export audio cache + fps in fingerprint + trim hydrate on partial re-run.
# Single source of truth — imported by segment_cache.segment_cache_fingerprint.
CONTINUITY_PIPELINE_ID = "minimax_h3_motion_context_v8"
# Example workflow tested value (NikoDemon80): audio_context_length=24 with video=22.
DEFAULT_AUDIO_CONTEXT_FRAMES = 24


def snap_context_frames(raw: int | float | None) -> int:
    """Snap UI/plan overlap to a supported context window (official baseline 22)."""
    try:
        n = int(raw or DEFAULT_CONTEXT_FRAMES)
    except (TypeError, ValueError):
        n = DEFAULT_CONTEXT_FRAMES
    chosen = min(CONTEXT_FRAME_CHOICES, key=lambda g: (abs(g - n), -g))
    return int(chosen)


def snap_join_context_frames(raw: int | float | None) -> int:
    """段间衔接 snap: the continuity grid extended to ~15s (JOIN_FRAME_CHOICES).

    Join-only — 段间引导 callers must keep ``snap_context_frames`` so their
    dropdown never exceeds 226.
    """
    try:
        n = int(raw or DEFAULT_CONTEXT_FRAMES)
    except (TypeError, ValueError):
        n = DEFAULT_CONTEXT_FRAMES
    chosen = min(JOIN_FRAME_CHOICES, key=lambda g: (abs(g - n), -g))
    return int(chosen)


def _join_run_snap(frames: int) -> int:
    """Largest whole-step window <= ``frames`` for a dense join ctx.

    Legacy reach (≤226) uses VIDEO_RUN_GRID byte-identically with the old code;
    the extended reach (>226) is snapped on JOIN_RUN_GRID so option A dense can
    actually run a ~15s window.
    """
    grid = VIDEO_RUN_GRID if int(frames) <= max(CONTEXT_FRAME_CHOICES) else JOIN_RUN_GRID
    return next(g for g in grid if g <= int(frames))


def _dense_join_reach_fits(
    total_steps: int, end_frame: int | None, want_px: int
) -> int:
    """Deepest whole-step dense reach (a JOIN choice) <= ``want_px`` that fits.

    Dense 段间衔接 no longer caps at the legacy 124px (VIDEO_RUN_GRID) depth, but
    it still cannot exceed what fits STRICTLY BEFORE the seam step: the identity
    pin alone owns the previous export's last pixel at frame 0. When the chosen
    reach cannot physically fit in the previous sample, degrade to the deepest
    feasible JOIN choice instead of raising. ``end_frame`` is the previous
    export's pixel end (None = phase-aligned whole-sample path, left to the
    caller's existing fit checks).
    """
    if end_frame is None:
        return max(1, int(want_px))
    total = int(total_steps)
    end_limit = min(int(end_frame), pixel_frames_for_latent_t(total))
    seam = _seam_step_for(total, end_limit)
    if seam < 1:
        raise RuntimeError(
            f"Director join: no latent step contains frame {end_limit - 1}."
        )
    max_steps = int(seam)  # steps 0..seam-1 fit strictly before the seam step
    want = max(1, int(want_px))
    for cand in reversed(JOIN_FRAME_CHOICES):
        if cand > want:
            continue
        steps = steps_for_frames(cand)
        if steps is not None and steps <= max_steps:
            return cand
    raise RuntimeError(
        f"Director join: no whole-step reach <= {want}px fits before the seam "
        f"step of a {total}-step previous sample (max {max_steps} steps)."
    )


def recommended_context_frames(task_key: str | None = None) -> int:
    """Official Motion Context baseline (22) for all continuity tasks."""
    del task_key
    return DEFAULT_CONTEXT_FRAMES


def pixel_frames_for_latent_t(latent_t: int) -> int:
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(int(latent_t)))


def steps_for_frames(n: int) -> int | None:
    k, covered = 0, 0
    while covered < n:
        covered += FRAME_PER_TOKEN[k % 5]
        k += 1
    return k if covered == n else None


def step_offsets(latent_t: int) -> list[int]:
    out, acc = [], 0
    for k in range(int(latent_t)):
        out.append(acc)
        acc += FRAME_PER_TOKEN[k % 5]
    return out


def _streams_from_latent(latent: dict) -> list[torch.Tensor]:
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    else:
        raise ValueError(
            f"Director continuity: expected MiniMax H3 AV NestedTensor, got {type(samples)!r}"
        )
    if not parts:
        raise ValueError("Director continuity: AV latent has no streams")
    return parts


def video_from_latent(latent: dict) -> torch.Tensor:
    video = _streams_from_latent(latent)[0]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError(
            f"Director continuity: expected video latent [B,C,T,H,W], got {tuple(video.shape)}"
        )
    return video


def _resize_frames(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    import comfy.utils

    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled")
    return samples.movedim(1, -1)


def _phase_aligned_tail_start(
    total_steps: int, n_steps: int, end_frame: int | None
) -> tuple[int, int, int]:
    """Pick a 5-cycle-aligned step start whose pixel window ends at/before ``end_frame``.

    Returns ``(start_step, pin_end_px, gap_after_pin)``.

    When ``end_frame`` is None, use the absolute latent end (official Motion Context).
    Director passes the *exported* end so align() overshoot beyond the visible
    segment is never pinned into the next clip. ``gap_after_pin`` is how many
    exported frames sit *after* the pin window — those must be dropped from the
    previous export before concat, or the next clip's opening will echo them.
    """
    if n_steps > total_steps:
        raise ValueError(
            f"Director continuity: need {n_steps} latent steps, context has {total_steps}."
        )
    if end_frame is None:
        start = total_steps - n_steps
        if start % 5 != 0:
            raise RuntimeError(
                f"Director continuity: tail start cycle {start % 5} != 0; refusing shifted join."
            )
        pin_end = pixel_frames_for_latent_t(total_steps)
        return start, pin_end, 0

    end_limit = int(end_frame)
    best_start = None
    best_end_px = -1
    for start in range(0, total_steps - n_steps + 1, 5):
        start_px = pixel_frames_for_latent_t(start)
        end_px = start_px + pixel_frames_for_latent_t(n_steps)
        if end_px <= end_limit and end_px >= best_end_px:
            best_start = start
            best_end_px = end_px
    if best_start is None:
        raise RuntimeError(
            f"Director continuity: no phase-aligned {n_steps}-step window ending "
            f"at or before frame {end_limit}."
        )
    gap = max(0, end_limit - best_end_px)
    if gap > 0:
        log.info(
            "Director continuity: pin window ends %df before export end "
            "(phase align; export_end=%d, pin_end=%d) — prev export tail will be trimmed",
            gap,
            end_limit,
            best_end_px,
        )
    return best_start, best_end_px, gap


def _seam_tail_start(
    total_steps: int, n_steps: int, end_frame: int | None
) -> tuple[int, int, int]:
    """Join-only tail window: the last ``n_steps`` FULL latent steps that end
    STRICTLY BEFORE the seam step.

    Returns ``(start_step, pin_end_px, gap_after_pin)``.

    The pixel frame at the export end (``end_limit - 1``) always lives in a
    latent "seam step" -- a compressed blend of up to FRAME_PER_TOKEN pixel
    frames ending at/around the pinned frame. If that step also became the
    window's last ctx block, the layout would tie it to the identity pin AT
    origin: two cond blocks with different content share frame 0, and the
    model denoises it toward their average (blur/white wash under motion).
    Official i2v keeps frame 0 owned by a single first-frame keyframe, and
    Extend only gets away with the tie because its default context is 2 steps
    of near-identical content. So the join window stops at ``seam_step - 1``
    and ``gap`` is forced >= 1: the pin is origin's sole occupant, exactly
    like a first-frame image.
    """
    if n_steps > total_steps:
        raise ValueError(
            f"Director join: need {n_steps} latent steps, context has {total_steps}."
        )
    if end_frame is None:
        return _phase_aligned_tail_start(total_steps, n_steps, None)
    end_limit = min(int(end_frame), pixel_frames_for_latent_t(total_steps))
    # The seam pixel (end_limit-1) lives in one latent step. That step is a
    # compressed blend of up to FRAME_PER_TOKEN pixel frames ending at/around
    # the pinned frame — if it also ends at/before the export end it becomes the
    # window's LAST step, which the layout would tie to the identity pin AT
    # origin: two cond blocks with different content then share frame 0 and the
    # model denoises it toward their average (blur/wash on motion). Official i2v
    # keeps frame 0 owned by the single first-frame keyframe, so the window must
    # stop BEFORE the seam step and let the pin be origin's only occupant.
    seam_step = -1
    for k in range(total_steps):
        if pixel_frames_for_latent_t(k) <= end_limit - 1 < pixel_frames_for_latent_t(k + 1):
            seam_step = k
            break
    if seam_step < 0:
        raise RuntimeError(f"Director join: no latent step contains frame {end_limit - 1}.")
    last_full = seam_step - 1  # window ends at the step before the seam step
    # Sanity: every step before seam_step must end at/before end_limit.
    if last_full < n_steps - 1:
        raise RuntimeError(
            f"Director join: fewer than {n_steps} full latent steps before the "
            f"seam step (step {seam_step}, frame {end_limit - 1})."
        )
    start = last_full - (n_steps - 1)
    pin_end_px = pixel_frames_for_latent_t(last_full + 1)
    return start, pin_end_px, max(1, end_limit - pin_end_px)


def _video_tail_blocks(
    latent: dict,
    n: int,
    *,
    end_frame: int | None = None,
    seam: bool = False,
) -> tuple[list[torch.Tensor], list[int], int, int, int]:
    """Return ``(blocks, offsets, covered, pin_end_px, gap_after_pin)``.

    ``seam`` switches the join (段间衔接) to ``_seam_tail_start`` so the ctx
    window hugs the export end; the continuity default keeps the phase-aligned
    window byte-identical.
    """
    video = video_from_latent(latent)
    total = int(video.shape[2])
    steps = steps_for_frames(n)
    if steps is None:
        raise ValueError(
            f"Director continuity: {n} frames is not a whole number of latent steps "
            f"(use {', '.join(str(x) for x in CONTEXT_FRAME_CHOICES)})."
        )
    if seam and end_frame is not None:
        start, pin_end_px, gap = _seam_tail_start(total, steps, end_frame)
    else:
        start, pin_end_px, gap = _phase_aligned_tail_start(total, steps, end_frame)
    covered = pixel_frames_for_latent_t(steps)
    if covered != n:
        raise RuntimeError(
            f"Director continuity: {steps} steps cover {covered} frames, expected {n}."
        )
    blocks = [video[:1, :, start + k : start + k + 1].clone() for k in range(steps)]
    return blocks, step_offsets(steps), covered, pin_end_px, gap


def _seam_step_for(total_steps: int, end_px: int) -> int:
    """Latent step containing pixel frame (``end_px`` - 1); -1 if none."""
    end_limit = max(1, int(end_px))
    for k in range(int(total_steps)):
        if pixel_frames_for_latent_t(k) <= end_limit - 1 < pixel_frames_for_latent_t(k + 1):
            return k
    return -1


def sparse_join_anchors(
    total_steps: int,
    end_px: int | None,
    reach_px: int,
    *,
    tail_steps: int = SPARSE_CTX_TAIL_STEPS,
    spacing_px: int = SPARSE_CTX_ANCHOR_FRAMES,
) -> list[tuple[int, int]]:
    """Pick sparse ctx steps for 段间衔接.

    Returns ``[(step, back_px), ...]`` sorted oldest-first (ascending step).
    ``back_px`` (how many exported pixel frames lie AFTER ``step``'s content on
    the previous SAMPLE timeline, export end ``end_px`` exclusive) drives which
    steps to sample and their real spacing; it no longer drives RoPE placement
    — the caller time-FOLDS the far anchors' claimed times into a compact still
    pile behind the real tail (时间折叠), keeping only the real tail's backs.

    The ``tail_steps`` contiguous steps immediately before the seam step carry
    the motion arc at the seam; further back a single 1-token step is taken
    roughly every ``spacing_px`` frames (1s @24fps) out to ``reach_px``, for
    scene identity only. Every step ends strictly before the seam step, so the
    identity pin stays origin's sole occupant.
    """
    total = int(total_steps)
    total_px = pixel_frames_for_latent_t(total)
    end_limit = total_px if end_px is None else min(max(1, int(end_px)), total_px)
    reach = max(1, int(reach_px))
    seam = _seam_step_for(total, end_limit)
    if seam < 0:
        raise RuntimeError(
            f"Director join: no latent step contains frame {end_limit - 1}."
        )
    eligible: list[tuple[int, int]] = []
    for s in range(0, seam):
        back = end_limit - pixel_frames_for_latent_t(s + 1)
        if back >= 1 and back <= reach:
            eligible.append((s, back))
    if not eligible:
        raise RuntimeError(
            f"Director join: reach {reach}f is too shallow before the seam step "
            f"(step {seam}, frame {end_limit - 1})."
        )
    # Tail: up to `tail_steps` steps nearest the seam (largest s / smallest back).
    tail = eligible[-tail_steps:]
    chosen: dict[int, int] = {s: b for s, b in tail}
    tail_min_s = tail[0][0]
    tail_max_back = max(b for _, b in tail)
    # Anchors: candidate steps strictly beyond the tail, backs in (tail, reach].
    # 远距离锚点：每隔 10 个 latent 步取连续 2 步
    far_cands = [(s, b) for s, b in eligible if s < tail_min_s]
    if far_cands and reach > tail_max_back:
        # 从尾部往前，每隔 10 个 latent 步取一个目标起始步
        anchor_step_stride = 10
        anchor_pair_count = 2
        step_targets = []
        current_step = tail_min_s - 1
        while current_step >= 0:
            step_targets.append(current_step)
            current_step -= anchor_step_stride

        # 对每个目标步索引，尝试取连续 anchor_pair_count 步
        for target_step in step_targets:
            candidate_steps = []
            for offset in range(anchor_pair_count):
                s = target_step - offset
                if s < 0:
                    break
                if s in chosen:
                    break
                # 检查这个步是否在 far_cands 中
                back = next((b for _s, b in far_cands if _s == s), None)
                if back is None:
                    break
                candidate_steps.append((s, back))
            # 如果取到了完整的连续步，全部加入 chosen
            if len(candidate_steps) == anchor_pair_count:
                for s, b in candidate_steps:
                    chosen[s] = b
                # 跳过已经选取的步，避免重叠
                # 由于我们是按照 target_step 倒序取，后面遇到重叠的 step 会因为 s in chosen 而跳过
    return sorted((s, b) for s, b in chosen.items())


def _sparse_tail_blocks(
    latent: dict,
    reach_frames: int,
    *,
    end_frame: int | None,
) -> tuple[list[torch.Tensor], list[float], int]:
    """Slice sparse 段间衔接 blocks ``(blocks_oldest_first, backs, span)``.

    ``reach_frames`` is the chosen reach (px back on the export timeline);
    ``end_frame`` is the previous export's exclusive sample-pixel end.
    """
    video = video_from_latent(latent)
    total = int(video.shape[2])
    end_limit = None if end_frame is None else int(end_frame)
    entries = sparse_join_anchors(total, end_limit, int(reach_frames))
    if not entries:
        raise RuntimeError(
            f"Director join: sparse reach {reach_frames}f selected no context steps."
        )
    blocks = [video[:1, :, s : s + 1].clone() for s, _b in entries]
    backs = [float(b) for _s, b in entries]
    span = int(reach_frames)
    return blocks, backs, span


def _audio_tail_from_latent(
    latent: dict,
    a_frames: int,
    *,
    end_frame: int | None = None,
) -> tuple[torch.Tensor, int, float]:
    parts = _streams_from_latent(latent)
    if len(parts) < 2:
        raise ValueError("Director continuity: context latent has no audio stream.")
    video, audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if audio.ndim != 4:
        raise ValueError(
            f"Director continuity: expected audio latent [B,C,2,T], got {tuple(audio.shape)}"
        )
    total_t = int(audio.shape[-1])
    frames = pixel_frames_for_latent_t(int(video.shape[2]))
    overhang = total_t - FRAME_RESCALE * frames
    if not (0.0 <= overhang < 1.0):
        log.warning(
            "Director continuity: unexpected audio grid (%d steps / %d frames); "
            "assuming no overhang.",
            total_t,
            frames,
        )
        overhang = 0.0
    rt = int(round(a_frames / float(FPS) * AUDIO_HZ))
    if rt > total_t:
        log.warning(
            "Director continuity: asked for %d audio steps, latent has %d; pinning all.",
            rt,
            total_t,
        )
        rt = total_t
    if rt < 1:
        raise ValueError("Director continuity: empty audio window")
    if end_frame is None:
        audio_end = total_t
    else:
        # Match the video pin window end (export end), not the sample overshoot.
        audio_end = int(round(float(end_frame) / float(FPS) * AUDIO_HZ))
        audio_end = max(rt, min(total_t, audio_end))
    audio_start = audio_end - rt
    if audio_start < 0:
        audio_start = 0
        rt = audio_end
    return audio[:1, ..., audio_start:audio_end].clone(), rt, float(overhang)


def _encode_tail_audio(audio_vae, audio: dict, seconds: float) -> tuple[torch.Tensor, int]:
    try:
        import torchaudio
    except ImportError:
        torchaudio = None
    waveform = audio["waveform"]
    sr = int(audio["sample_rate"])
    vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if sr != vae_sr:
        if torchaudio is None:
            raise RuntimeError(
                f"Director continuity: context audio is {sr} Hz, VAE wants {vae_sr} Hz, "
                "and torchaudio is unavailable."
            )
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    want = int(round(seconds * vae_sr))
    have = int(waveform.shape[-1])
    if have >= want:
        waveform = waveform[..., have - want :]
    z = audio_vae.encode(waveform[:1].movedim(1, -1))
    return z, int(z.shape[-1])


def _existing_keyframes(positive) -> list[dict]:
    """Best-effort read of keyframes already on conditioning (e.g. fl2v last frame)."""
    try:
        if not positive:
            return []
        meta = positive[0][1] if isinstance(positive[0], (list, tuple)) else None
        if not isinstance(meta, dict):
            return []
        kfs = meta.get("minimax_keyframes") or []
        return [dict(kf) for kf in kfs if isinstance(kf, dict)]
    except Exception:
        return []


def apply_motion_context(
    positive,
    latent: dict,
    *,
    vae,
    context_length: int,
    context_latent: dict | None = None,
    context_frames: torch.Tensor | None = None,
    context_audio: dict | None = None,
    audio_vae=None,
    continue_audio: bool = True,
    keep_existing_keyframes: bool = True,
    context_end_frame: int | None = None,
    audio_context_length: int | None = None,
) -> tuple[Any, int, int]:
    """Inject previous-segment motion (and optional audio) into conditioning.

    Returns ``(positive, trim_frames, prev_export_trim_tail)``.

    ``trim_frames`` is the pinned head length to remove after decode.
    ``prev_export_trim_tail`` is how many frames to drop from the *previous*
    segment's export before concat (phase-align pin often ends a few frames
    before the export end; leaving them causes a visible ~5f echo at the seam).

    ``context_end_frame``: exclusive pixel index on the previous *sample*
    timeline to pin up to. With official sample semantics (no post-trim crop)
    this is usually ``None`` (absolute latent end). Kept for legacy overshoot
    caches.

    ``audio_context_length``: official example uses 24 with video context 22.
    """
    import node_helpers

    ensure_layout_patch()
    # r2v/v2v/rv2v already carry minimax_refs; stock payload overwrites keyframe
    # video latents unless coexistence merge is installed.
    ensure_payload_patch()
    context_length = snap_context_frames(context_length)
    if audio_context_length is None:
        audio_ctx = DEFAULT_AUDIO_CONTEXT_FRAMES
    else:
        try:
            audio_ctx = max(0, int(audio_context_length))
        except (TypeError, ValueError):
            audio_ctx = DEFAULT_AUDIO_CONTEXT_FRAMES

    video = video_from_latent(latent)
    width = int(video.shape[4]) * 16
    height = int(video.shape[3]) * 16
    frame_count = pixel_frames_for_latent_t(int(video.shape[2]))

    if context_latent is not None:
        src = video_from_latent(context_latent)
        src_w, src_h = int(src.shape[4]) * 16, int(src.shape[3]) * 16
        if src_w != width or src_h != height:
            raise ValueError(
                f"Director continuity: context latent is {src_w}x{src_h} but this "
                f"segment is {width}x{height}. Regenerate the previous segment at "
                "this resolution."
            )
        available = pixel_frames_for_latent_t(int(src.shape[2]))
        if context_end_frame is not None:
            available = min(available, max(0, int(context_end_frame)))
        video_src = "latent"
    else:
        if context_frames is None or int(context_frames.shape[0]) < 1:
            raise ValueError(
                "Director continuity: need previous segment latent or decoded frames."
            )
        available = int(context_frames.shape[0])
        video_src = "pixels"

    n = min(int(context_length), available)
    if n < 1:
        raise ValueError("Director continuity: no frames available to pin")
    run = next(g for g in VIDEO_RUN_GRID if g <= n)
    if run != n:
        log.warning(
            "Director continuity: %d frames off VAE grid; pinning last %d.", n, run
        )
        n = run
    if n >= frame_count:
        raise ValueError(
            f"Director continuity: cannot pin {n} frames into a {frame_count}-frame clip."
        )

    pin_end_px: int | None = None
    prev_export_trim_tail = 0
    if video_src == "latent":
        blocks, offsets, covered, pin_end_px, prev_export_trim_tail = _video_tail_blocks(
            context_latent, n, end_frame=context_end_frame
        )
        span = covered
    else:
        # Decoded frames are already the export — absolute tail is correct.
        tail = _resize_frames(context_frames[available - n :], width, height)
        enc = vae.encode(tail)
        if getattr(enc, "ndim", 0) != 5:
            raise ValueError(
                f"Director continuity: VAE encode returned shape "
                f"{tuple(getattr(enc, 'shape', ()))}, expected [B,C,T,H,W]."
            )
        steps = int(enc.shape[2])
        offsets = step_offsets(steps)
        covered = pixel_frames_for_latent_t(steps)
        if covered != n:
            raise RuntimeError(
                f"Director continuity: {n} frames encoded to {steps} steps covering "
                f"{covered}; VAE grid mismatch."
            )
        blocks = [enc[:, :, k : k + 1] for k in range(steps)]
        span = covered
        pin_end_px = available
        prev_export_trim_tail = 0

    ctx_keyframes = [
        {
            "resolved_frame_index": 0,
            CTX_FRAME_KEY: int(p),
            "latent": blk,
        }
        for p, blk in zip(offsets, blocks)
    ]

    merged = list(ctx_keyframes)
    if keep_existing_keyframes:
        for kf in _existing_keyframes(positive):
            # Drop stock first-frame at 0 — replaced by context head.
            if int(kf.get("resolved_frame_index", -1)) == 0 and CTX_FRAME_KEY not in kf:
                continue
            # Avoid duplicating director context markers.
            if CTX_FRAME_KEY in kf:
                continue
            merged.append(kf)

    values: dict[str, Any] = {
        "minimax_keyframes": merged,
    }
    out = node_helpers.conditioning_set_values(positive, values)

    if continue_audio and (context_latent is not None or context_audio is not None):
        # Official: audio window independent; 0 follows video span. Example WF uses 24.
        a_frames = int(audio_ctx) if audio_ctx > 0 else int(span)
        # Align audio pin end with the video pin window (not export overshoot).
        audio_end_limit = pin_end_px if pin_end_px is not None else context_end_frame
        if context_latent is not None:
            audio_latent, ref_audio_t, overhang = _audio_tail_from_latent(
                context_latent, a_frames, end_frame=audio_end_limit
            )
        else:
            if audio_vae is None:
                raise ValueError(
                    "Director continuity: context_audio requires audio_vae "
                    "(or pass previous AV latent)."
                )
            audio_latent, ref_audio_t = _encode_tail_audio(
                audio_vae, context_audio, a_frames / float(FPS)
            )
            overhang = 0.0
        end_frame = float(span) + float(overhang) / FRAME_RESCALE
        end_coord = round(FRAME_RESCALE * end_frame)
        end_frame = end_coord / FRAME_RESCALE
        audio_ref = {
            "kind": "audio",
            "ref_audio_t": ref_audio_t,
            "audio_latent": audio_latent,
            CTX_AUDIO_END_KEY: end_frame,
        }
        out = node_helpers.conditioning_set_values(
            out, {"minimax_refs": [audio_ref]}, append=True
        )
        log.info(
            "Director continuity: pinned %d video frames (%s) + %d audio steps"
            "%s%s",
            span,
            video_src,
            ref_audio_t,
            f" (context_end={context_end_frame})" if context_end_frame is not None else "",
            f", trim_prev_export={prev_export_trim_tail}f" if prev_export_trim_tail else "",
        )
    else:
        log.info(
            "Director continuity: pinned %d video frames (%s), audio off%s%s",
            span,
            video_src,
            f" (context_end={context_end_frame})" if context_end_frame is not None else "",
            f", trim_prev_export={prev_export_trim_tail}f" if prev_export_trim_tail else "",
        )

    return out, int(span), int(prev_export_trim_tail)


def trim_context_prefix(
    images: torch.Tensor,
    audio: dict | None,
    trim_frames: int,
    *,
    fps: float = FPS,
    match_tail: bool = True,
) -> tuple[torch.Tensor, dict | None]:
    """Remove pinned head from decoded images/audio; optionally match audio duration."""
    trim = max(0, int(trim_frames))
    if trim > 0:
        if int(images.shape[0]) <= trim:
            raise ValueError(
                f"Director continuity: cannot trim {trim} frames from "
                f"{int(images.shape[0])}-frame decode."
            )
        images = images[trim:]
    if not isinstance(audio, dict) or audio.get("waveform") is None:
        return images, audio
    waveform = audio["waveform"]
    sr = int(audio.get("sample_rate") or 32000)
    drop = int(round((trim / float(fps)) * sr)) if trim > 0 else 0
    if drop > 0 and int(waveform.shape[-1]) > drop:
        waveform = waveform[..., drop:]
    if match_tail:
        want = int(round((int(images.shape[0]) / float(fps)) * sr))
        if int(waveform.shape[-1]) > want:
            waveform = waveform[..., :want]
    return images, {"waveform": waveform, "sample_rate": sr}


def trim_export_tail(
    images: torch.Tensor,
    audio: dict | None,
    trim_frames: int,
    *,
    fps: float = FPS,
) -> tuple[torch.Tensor, dict | None]:
    """Drop trailing frames from a previous export so it ends at the pin window."""
    trim = max(0, int(trim_frames))
    if trim <= 0:
        return images, audio
    keep = int(images.shape[0]) - trim
    if keep < 1:
        raise ValueError(
            f"Director continuity: cannot drop {trim} tail frames from "
            f"{int(images.shape[0])}-frame export."
        )
    images = images[:keep]
    if not isinstance(audio, dict) or audio.get("waveform") is None:
        return images, audio
    waveform = audio["waveform"]
    sr = int(audio.get("sample_rate") or 32000)
    want = int(round((keep / float(fps)) * sr))
    if int(waveform.shape[-1]) > want:
        waveform = waveform[..., :want]
    return images, {"waveform": waveform, "sample_rate": sr}


def generation_frame_budget(visible_frames: int, context_frames: int) -> tuple[int, int]:
    """Return ``(sample_length, trim_frames)`` for Director continuity.

    Director contract: UI segment duration == exported frames.

    Standalone Motion Context sets ``length`` to the sample and delivers
    ``length - context`` (shorter than the UI seconds). That produced the
    27s-vs-30s result. Here we instead:

    1. ``sample = align(visible + context)`` so the pin fits in the head
    2. Trim ``context`` frames after decode
    3. Keep exactly ``visible`` frames for export
    4. Next pin uses ``context_end_frame = trim + visible`` (not the sample
       absolute end, which includes align overshoot beyond the export)
    5. If phase-align places the pin a few frames before that export end,
       drop those frames from the previous export before concat (v7)
    """
    from .frame_align import minimax_align_frame_count

    visible = minimax_align_frame_count(max(5, int(visible_frames)))
    ctx = snap_context_frames(context_frames) if context_frames else 0
    if ctx <= 0:
        return visible, 0
    sample = minimax_align_frame_count(visible + ctx)
    if ctx >= sample:
        raise ValueError(
            f"Director continuity: context {ctx}f must be smaller than sample "
            f"length {sample}f."
        )
    return sample, ctx


def handoff_end_frame(*, trim_frames: int, export_frames: int) -> int:
    """Sample-timeline pixel index where the exported segment ends (exclusive)."""
    return max(0, int(trim_frames)) + max(0, int(export_frames))


def negative_time_frame_budget(visible_frames: int, context_frames: int) -> tuple[int, int]:
    """Return ``(sample_length, trim_frames)`` for Director 段间衔接 (negative time).

    The previous segment's tail is placed in NEGATIVE-time conditioning, so the
    output latent is never extended: ``sample = align(visible)`` and
    ``trim_frames = 0``. Context lives only in conditioning keyframes.
    """
    from .frame_align import minimax_align_frame_count

    visible = minimax_align_frame_count(max(5, int(visible_frames)))
    # Join reach can extend to ~15s (join-only grid); 段间引导 uses
    # generation_frame_budget / snap_context_frames above.
    ctx = snap_join_context_frames(context_frames) if context_frames else 0
    if ctx <= 0:
        return visible, 0
    if ctx >= visible:
        raise ValueError(
            f"Director join: context {ctx}f must be smaller than segment "
            f"length {visible}f."
        )
    return visible, 0


def segment_export_end_frame(plan, seg) -> int:
    """Sample-timeline pixel index (exclusive) where ``seg``'s exported frames
    end, derived from the plan alone so the loop-driven node can pin the
    previous segment's TRUE last exported pixel without the prev handoff cache.

    Mirrors the node's R2VTrim math (``images[trim : trim + target_frames]``):
      * continuity prev (master on, from_prev, idx>0): head-trim ctx, export
        ``frame_count`` (raw, unaligned) → ctx + frame_count
      * join / lead / no-master prev: trim 0, export ``frame_count`` → frame_count
    ``end_px`` is on the prev segment's SAMPLE timeline (the context latent
    passed to the join pin), so it must be the sample pixel where the export
    ends, NOT an aligned frame count.
    """
    visible = max(1, int(getattr(seg, "frame_count", 0) or 0))
    if (
        int(getattr(seg, "index", -1)) > 0
        and bool(getattr(seg, "continuity_from_prev", True))
        and getattr(plan, "continuity_enabled", False)
    ):
        ctx = snap_context_frames(getattr(plan, "continuity_overlap_frames", 0))
        return ctx + visible
    return visible


def _pin_last_export_pixel(vae, context_latent, *, end_px, width, height):
    """Decode the latent window covering pixel (end_px-1) and return that exact
    pixel as an IMAGE, so the join's frame-0 identity pin equals the previous
    segment's true last exported pixel even when each latent token spans
    1-4 pixel frames and the sample overshoots the export. ``end_px`` is the
    exclusive export end on the previous SAMPLE timeline (pixel frames)."""
    video = video_from_latent(context_latent)
    total = int(video.shape[2])
    max_px = pixel_frames_for_latent_t(total)
    if end_px is None:
        end_px = max_px
    end_px = max(1, min(int(end_px), max_px))
    if end_px == max_px:
        # Seam == true sample end (the 10s/10s case): the pinned frame is the
        # terminal one. The ViT3D decoder chunks at 5 latent tokens (decode
        # windows of 5+2=7), so the export's TRUE last frame always comes from
        # its final chunk [total-7:total]. Extend's 6-step tail ([total-6:total])
        # lands one token early and pads a duplicated last token -> an OOD chunk
        # -> the pinned frame decodes soft/washed (共性问题 with Extend). Decode
        # exactly the last 7 tokens (one aligned chunk) and take ``[:,-1]``:
        # byte-identical to what the previous export displayed.
        start = max(0, total - 7)
        decoded = vae.decode(video[:1, :, start:total].clone())
        rel = -1
    else:
        # Export cuts the sample mid-step (overshoot prev): the pinned pixel is
        # NOT the sample's terminal frame, so a partial sub-window decode cannot
        # reproduce it — ViT3D chunk boundaries move with the window start and
        # the interior frame lands at an unverifiable offset (can come back soft
        # or one step off). Decode the WHOLE source exactly as the previous
        # segment's own export was decoded, then index the true pixel directly.
        decoded = vae.decode(video[:1])
        rel = int(end_px) - 1
    if isinstance(decoded, (tuple, list)):
        decoded = decoded[0]
    # MiniMax H3 Video VAE decodes to [B,F,H,W,C]; unwrap the single-batch dim
    # so frame indexing below works on the frame axis (Extend's pin does the
    # same via ``decoded[:, -1]``).
    if getattr(decoded, "ndim", 0) == 5 and int(decoded.shape[0]) == 1:
        decoded = decoded[0]
    if getattr(decoded, "ndim", 0) != 4:
        raise ValueError(
            f"Director join: identity pin VAE decode returned shape "
            f"{tuple(getattr(decoded, 'shape', ()))}, expected [F,H,W,C]."
        )
    if rel < 0:
        rel = int(decoded.shape[0]) - 1
    rel = max(0, min(int(decoded.shape[0]) - 1, rel))
    return _resize_frames(decoded[rel : rel + 1], width, height)


def apply_negative_time_context(
    positive,
    latent: dict,
    *,
    vae,
    context_length: int,
    context_latent: dict | None = None,
    context_frames: torch.Tensor | None = None,
    context_audio: dict | None = None,
    audio_vae=None,
    continue_audio: bool = True,
    keep_existing_keyframes: bool = True,
    context_end_frame: int | None = None,
    audio_context_length: int | None = None,
    sparse_context: bool = False,
) -> tuple[Any, int, int]:
    """Inject the previous segment's tail as NEGATIVE-time conditioning (段间衔接).

    Returns ``(positive, trim_frames, prev_export_trim_tail)`` with both always 0:
    the previous tail lives only in conditioning keyframes at negative RoPE
    times before frame 0, plus an identity pin of the previous segment's TRUE
    last exported pixel at frame 0 (zero RoPE distance). The output latent is
    never extended. ``context_end_frame`` clamps the tail window to the
    previous segment's EXPORT end (not the sample overshoot).

    ``sparse_context`` switches the ctx block stack to 稀疏取帧 (when the
    previous AV latent is available): the 6 contiguous steps hugging the seam
    plus a single 1-token step every ~1s out to the chosen reach. The real
    tail-6 keeps its TRUE RoPE back (motion into the seam); the far anchors are
    time-FOLDED (时间折叠) so their claimed RoPE times land in a compact
    near-simultaneous still pile hugging the tail's deep edge — the whole
    reference reads as ONE last small segment, far frames acting as appearance
    constraints only. Real-time-spread far anchors (true back) washed every
    render, so their claimed time is collapsed rather than left deep. Off keeps
    the dense stack byte-identical.
    """
    import node_helpers

    ensure_layout_patch()
    ensure_payload_patch()
    context_length = snap_join_context_frames(context_length)
    if audio_context_length is None:
        audio_ctx = DEFAULT_AUDIO_CONTEXT_FRAMES
    else:
        try:
            audio_ctx = max(0, int(audio_context_length))
        except (TypeError, ValueError):
            audio_ctx = DEFAULT_AUDIO_CONTEXT_FRAMES

    video = video_from_latent(latent)
    width = int(video.shape[4]) * 16
    height = int(video.shape[3]) * 16
    frame_count = pixel_frames_for_latent_t(int(video.shape[2]))

    if context_latent is not None:
        src = video_from_latent(context_latent)
        src_w, src_h = int(src.shape[4]) * 16, int(src.shape[3]) * 16
        if src_w != width or src_h != height:
            raise ValueError(
                f"Director join: context latent is {src_w}x{src_h} but this "
                f"segment is {width}x{height}. Regenerate the previous segment at "
                "this resolution."
            )
        available = pixel_frames_for_latent_t(int(src.shape[2]))
        if context_end_frame is not None:
            available = min(available, max(0, int(context_end_frame)))
        video_src = "latent"
    else:
        if context_frames is None or int(context_frames.shape[0]) < 1:
            raise ValueError(
                "Director join: need previous segment latent or decoded frames."
            )
        available = int(context_frames.shape[0])
        video_src = "pixels"

    n = min(int(context_length), available)
    if n < 1:
        raise ValueError("Director join: no frames available to pin")
    sparse_mode = bool(sparse_context) and video_src == "latent"
    if sparse_mode:
        # 稀疏取帧: tail 6 contiguous steps + ~1/s anchors. The reach ``n`` is
        # already whole/off-grid agnostic; each block carries its OWN per-block
        # claimed back (tail=real, far=time-folded) so no whole-stack grid snap
        # is needed here.
        if n >= frame_count:
            log.info(
                "Director join (sparse): reach %df >= segment %df; capping to "
                "the previous export only.",
                n,
                frame_count,
            )
    else:
        # Dense join honors the full chosen reach (no legacy 124px cap). The
        # negative-time ctx is pure conditioning (trim stays 0), so the only
        # bound is the previous sample's seam step; UI always sends a whole-step
        # JOIN choice, the snap below only guards programmatic off-grid values.
        if steps_for_frames(n) is None:
            run = _join_run_snap(n)
            if run != n:
                log.warning(
                    "Director join: %d frames off VAE grid; pinning last %d.", n, run
                )
            n = run
        if video_src == "latent" and context_end_frame is not None:
            fitted = _dense_join_reach_fits(
                int(src.shape[2]), context_end_frame, n
            )
            if fitted != n:
                log.warning(
                    "Director join: reach %df cannot fit before the previous "
                    "segment's seam; using deepest feasible %df.",
                    n,
                    fitted,
                )
                n = fitted
        elif n >= frame_count:
            log.info(
                "Director join: dense reach %df >= this clip %df; negative-time "
                "conditioning only, no extension.",
                n,
                frame_count,
            )

    if video_src == "latent":
        if sparse_mode:
            blocks, backs, span = _sparse_tail_blocks(
                context_latent, n, end_frame=context_end_frame
            )
            offsets = []
            # Bookkeeping mirroring the dense seam window (used by the audio tail
            # when context_end_frame is None): the pin's true export end pixel.
            src_total = int(src.shape[2])
            end_eff = (
                min(int(context_end_frame), pixel_frames_for_latent_t(src_total))
                if context_end_frame is not None
                else pixel_frames_for_latent_t(src_total)
            )
            seam_step = _seam_step_for(src_total, end_eff)
            pin_end_px = pixel_frames_for_latent_t(seam_step)
            gap = max(1, end_eff - pin_end_px)
        else:
            blocks, offsets, covered, pin_end_px, gap = _video_tail_blocks(
                context_latent, n, end_frame=context_end_frame, seam=True
            )
            span = covered
        pin_img = _pin_last_export_pixel(
            vae, context_latent, end_px=context_end_frame, width=width, height=height
        )
        # 调试保存 pin_img（自增序号）
        try:
            import os
            import re
            import torchvision
            import folder_paths

            debug_dir = os.path.join(folder_paths.get_output_directory(), "minimax_seg_cache", "debug")
            os.makedirs(debug_dir, exist_ok=True)

            # 查找现有文件，确定下一个序号
            existing = [f for f in os.listdir(debug_dir) if f.startswith("pin_img_") and f.endswith(".png")]
            max_num = 0
            for fname in existing:
                # 提取数字部分，如 pin_img_00001.png
                match = re.search(r"pin_img_(\d+)\.png", fname)
                if match:
                    num = int(match.group(1))
                    if num > max_num:
                        max_num = num
            next_num = max_num + 1
            filename = f"pin_img_{next_num:05d}.png"  # 5位数字，如 00001
            save_path = os.path.join(debug_dir, filename)

            torchvision.utils.save_image(pin_img.permute(0, 3, 1, 2), save_path)
            print(f"[DEBUG] Saved pin image to {save_path}")

        except Exception as e:
            print(f"[DEBUG] Failed to save pin image: {e}")
    else:
        tail = _resize_frames(context_frames[available - n :], width, height)
        enc = vae.encode(tail)
        if getattr(enc, "ndim", 0) != 5:
            raise ValueError(
                f"Director join: VAE encode returned shape "
                f"{tuple(getattr(enc, 'shape', ()))}, expected [B,C,T,H,W]."
            )
        steps = int(enc.shape[2])
        offsets = step_offsets(steps)
        covered = pixel_frames_for_latent_t(steps)
        if covered != n:
            raise RuntimeError(
                f"Director join: {n} frames encoded to {steps} steps covering "
                f"{covered}; VAE grid mismatch."
            )
        blocks = [enc[:, :, k : k + 1] for k in range(steps)]
        span = covered
        pin_img = _resize_frames(context_frames[-1:], width, height)
        pin_end_px = available
        gap = 0

    if sparse_mode:
        # 时间折叠 (模拟成最后一小段): the real contiguous tail keeps its TRUE
        # RoPE back (motion into the seam); every far ~1/s anchor has its claimed
        # time FOLDED into a compact still pile hugging the tail's deep edge
        # (sub-frame spacing ≈ the same instant), so the whole reference reads as
        # one SHORT final small segment whose far frames act as appearance-only
        # image constraints instead of a stretched deep history.
        tail_count = min(SPARSE_CTX_TAIL_STEPS, len(blocks))
        far_count = len(blocks) - tail_count
        if far_count > 0:
            tail_max_back = max(backs[p] for p in range(far_count, len(blocks)))
            far_base = tail_max_back + SPARSE_CTX_FOLD_GAP_PX
        ctx_keyframes = []
        for p, blk in enumerate(blocks):
            if far_count > 0 and p < far_count:
                # Oldest anchor (p=0) deepest in the pile; sub-frame spacing.
                claimed = far_base + (far_count - 1 - p) * SPARSE_CTX_FOLD_SUB_PX
            else:
                claimed = float(backs[p])
            ctx_keyframes.append(
                {
                    "resolved_frame_index": 0,
                    CTX_NEG_KEY: int(p),
                    CTX_NEG_BACK_KEY: claimed,
                    "latent": blk,
                }
            )
    else:
        ctx_keyframes = [
            {
                "resolved_frame_index": 0,
                CTX_NEG_KEY: int(p),
                # Whole-stack backward shift (px) so the nearest block shows the
                # frames right before the seam; the identity pin alone owns frame 0.
                CTX_NEG_SHIFT_KEY: float(gap),
                "latent": blk,
            }
            for p, blk in enumerate(blocks)
        ]
    pin_latent = vae.encode(pin_img)
    if getattr(pin_latent, "ndim", 0) != 5:
        raise ValueError(
            f"Director join: identity pin VAE encode returned shape "
            f"{tuple(getattr(pin_latent, 'shape', ()))}, expected [B,C,T,H,W]."
        )
    pin_kf = {
        "resolved_frame_index": 0,
        CTX_FRAME_KEY: 0,
        CTX_NEG_KEY: 0,
        "latent": pin_latent,
    }

    merged = list(ctx_keyframes) + [pin_kf]
    if keep_existing_keyframes:
        for kf in _existing_keyframes(positive):
            if (
                int(kf.get("resolved_frame_index", -1)) == 0
                and CTX_FRAME_KEY not in kf
                and CTX_NEG_KEY not in kf
            ):
                continue
            if CTX_FRAME_KEY in kf or CTX_NEG_KEY in kf:
                continue
            merged.append(kf)

    out = node_helpers.conditioning_set_values(
        positive, {"minimax_keyframes": merged}
    )

    if continue_audio and (context_latent is not None or context_audio is not None):
        a_frames = int(audio_ctx) if audio_ctx > 0 else int(span)
        # Join audio ends exactly at the seam (frame 0 of this segment).
        audio_end_limit = (
            context_end_frame if context_end_frame is not None else pin_end_px
        )
        if context_latent is not None:
            audio_latent, ref_audio_t, overhang = _audio_tail_from_latent(
                context_latent, a_frames, end_frame=audio_end_limit
            )
        else:
            if audio_vae is None:
                raise ValueError(
                    "Director join: context_audio requires audio_vae "
                    "(or pass previous AV latent)."
                )
            audio_latent, ref_audio_t = _encode_tail_audio(
                audio_vae, context_audio, a_frames / float(FPS)
            )
            overhang = 0.0
        audio_ref = {
            "kind": "audio",
            "ref_audio_t": ref_audio_t,
            "audio_latent": audio_latent,
            CTX_AUDIO_END_KEY: 0.0,
        }
        out = node_helpers.conditioning_set_values(
            out, {"minimax_refs": [audio_ref]}, append=True
        )
        log.info(
            "Director join: pinned %d video frames (%s) + %d audio steps ending "
            "at frame 0 (context_end=%s)",
            span,
            video_src,
            ref_audio_t,
            context_end_frame,
        )
    else:
        log.info(
            "Director join: pinned %d video frames (%s), audio off (context_end=%s)",
            span,
            video_src,
            context_end_frame,
        )

    return out, 0, 0
