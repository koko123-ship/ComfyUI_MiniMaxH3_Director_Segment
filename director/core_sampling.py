"""Single-stage sampling for MiniMax H3 (SigmaShift + KSampler)."""

from __future__ import annotations

import logging
from typing import Any, Callable

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.core_sampling")

PhaseCallback = Callable[[str, float], None]
StepPreviewCallback = Callable[[int, int, Any], None]


def _unpack_node_output(out):
    if hasattr(out, "args"):
        args = out.args
        if args:
            return args
    if isinstance(out, (tuple, list)):
        return out
    raise RuntimeError(f"Unexpected node output type: {type(out)!r}")


def sample_single_stage(
    *,
    model,
    positive,
    negative,
    latent,
    seed: int,
    cfg: float,
    steps: int,
    sampler_name: str,
    scheduler: str,
    shift_video: float = 12.0,
    shift_audio: float = 3.0,
    on_phase: PhaseCallback | None = None,
    on_step_preview: StepPreviewCallback | None = None,
    preview_every: int = 1,
):
    import comfy.sample
    import comfy.utils
    import latent_preview
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift

    def notify(phase: str, value: float) -> None:
        if on_phase:
            on_phase(phase, value)

    notify("sample", 0)
    shifted = MiniMaxH3SigmaShift.execute(model, float(shift_video), float(shift_audio))
    model_shifted = _unpack_node_output(shifted)[0]

    neg = negative if negative else []
    steps = int(steps)
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model_shifted,
        latent_image,
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None),
    )

    noise = comfy.sample.prepare_noise(
        latent_image,
        int(seed),
        latent.get("batch_index", None),
    )
    noise_mask = latent.get("noise_mask", None)

    base_cb = latent_preview.prepare_callback(model_shifted, steps)
    every = max(1, int(preview_every))

    def callback(step, x0, x, total_steps):
        if on_step_preview is not None:
            try:
                if step % every == 0 or step >= max(0, int(total_steps) - 1):
                    on_step_preview(int(step), int(total_steps), x0)
            except Exception as exc:
                log.debug("Step preview callback skipped: %s", exc)
        if base_cb is not None:
            base_cb(step, x0, x, total_steps)

    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(
        model_shifted,
        noise,
        steps,
        float(cfg),
        sampler_name,
        scheduler,
        positive,
        neg,
        latent_image,
        denoise=1.0,
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=int(seed),
    )
    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    notify("sample", 1)
    return out
