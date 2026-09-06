"""Incremental per-segment MP4 export for「分段导出」runs.

Best-effort: encode failures must never abort generation. Each run uses a
timestamp folder: ``output/minimax_seg_export/<YYYYMMDD_HHMMSS>/seg_XXXX.mp4``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import folder_paths
import torch

from .audio_export import prepare_segment_audio_for_file_export
from .plan import DirectorPlan, SegmentPlan

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.mp4_export")

VIDEO_EXPORT_TASKS = frozenset({"t2v", "i2v", "r2v", "fl2v", "v2v", "rv2v"})


def new_segment_mp4_run_dir(plan: DirectorPlan) -> Path | None:
    """Create ``minimax_seg_export/<YYYYMMDD_HHMMSS>/`` for one Director execute.

    Returns None when not in segments mode or the output dir is unavailable.
    """
    if getattr(plan, "export_mode", "all") != "segments":
        return None
    try:
        base = Path(folder_paths.get_output_directory()) / "minimax_seg_export"
        base.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = base / stamp
        if root.exists():
            # Same-second collision (rare): append a short suffix.
            for i in range(1, 1000):
                candidate = base / f"{stamp}_{i:03d}"
                if not candidate.exists():
                    root = candidate
                    break
        root.mkdir(parents=True, exist_ok=False)
        log.info("MiniMax H3 Director segment mp4 run dir: %s", root)
        return root
    except OSError as exc:
        log.warning("Segment mp4 export dir unavailable (%s); skipped.", exc)
        return None


def segment_mp4_path(run_dir: Path, seg: SegmentPlan) -> Path:
    return Path(run_dir) / f"seg_{int(seg.index):04d}.mp4"


def maybe_export_segment_mp4(
    run_dir: Path | None,
    plan: DirectorPlan,
    seg: SegmentPlan,
    frames: torch.Tensor,
    audio_dict: dict[str, Any] | None = None,
) -> str | None:
    """Write one segment mp4 into ``run_dir``. Never raises.

    Returns the absolute path string on success, otherwise None.
    """
    if run_dir is None or getattr(plan, "export_mode", "all") != "segments":
        return None
    task = str(getattr(seg, "task_key", "") or getattr(plan, "global_task_key", "") or "")
    if task not in VIDEO_EXPORT_TASKS:
        return None
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or int(frames.shape[0]) <= 0:
        return None

    dest = segment_mp4_path(run_dir, seg)

    try:
        from ..lib.video_export import write_frames_to_mp4

        audio = prepare_segment_audio_for_file_export(
            plan,
            seg,
            audio_dict=audio_dict,
            frame_count=int(frames.shape[0]),
        )
        path = write_frames_to_mp4(
            dest,
            frames.detach().cpu().float(),
            fps=float(getattr(plan, "frame_rate", 24) or 24),
            audio=audio,
        )
        log.info(
            "MiniMax H3 Director segment #%d mp4 saved: %s",
            int(seg.index) + 1,
            path,
        )
        return str(path)
    except Exception as exc:
        log.warning(
            "Segment #%d mp4 export failed (generation continues): %s",
            int(seg.index) + 1,
            exc,
        )
        return None
