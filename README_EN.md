# ComfyUI MiniMax H3 Director — External-Sampling Edition

> This repository is a **modified fork** of [AIMixer/ComfyUI_MiniMaxH3_Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director). The original plugin was written by **AIMixer**. This edition adds **external custom sampling / decoding** support and several helper nodes. For the original introduction and full feature set, please see the original author's repository.

**中文文档** → [README.md](README.md)

**Thanks to the original author:** [AIMixer · ComfyUI_MiniMaxH3_Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director)

## What changed in this edition

The original `MiniMaxH3Director` is a single-node "full pipeline director" — timeline, conditioning, sampling, decode and export all inside one node. This edition **moves sampling → decode → export outside** for t2v / i2v / fl2v / r2v / v2v / rv2v:

- The director (`MiniMaxH3Segment`) only does **conditioning**, outputting `positive` + `latent`;
- Sampling / decoding / saving use **standard ComfyUI nodes** built freely outside (custom samplers, tiled decode, `SaveLatent`, …);
- Segment continuity (段间引导), run selection, reference material and local two-sample video are still managed by the timeline UI.

## Modified files

| File | Content |
|---|---|
| `nodes/r2v_segment.py` | **New**: `MiniMaxH3Segment` (per-segment conditioning, 6 tasks), `MiniMaxH3R2VTrim` (head/tail trim), `MiniMaxH3AudioSelect` (generate/source/mute), local two-sample video outputs `frames_0` / `audio_0` |
| `__init__.py` | Register the nodes above (`MiniMaxH3R2VSegment` kept as a legacy alias so old workflows load) |
| `web/js/minimax_timeline.js` | Timeline editor mounted on the new node; hide export/audio/max-frames panels; `twoVideo` persistence; static 「本地二采不可用」 hint |
| `web/js/minimax_image_batch.js` | t2v/i2v preview area → local two-sample upload slot; r2v row under audio; upload/render/remove logic |
| `web/js/minimax_fl2v.js` | two-sample slot under each shot's duration; `twoVideo` preserved in shots serialize/parse |
| `example_workflows/minimax_h3_r2v_external_demo.json` | **New**: external-sampling example workflow (for-loop + Trim + AudioSelect + forLoopEnd closure) |

## New features & usage

### 1. `MiniMaxH3Segment` — external-sampling conditioning node

Returns the conditioning (`positive` + `latent`) for segment N by timeline `index` (loop counter); sampling/decoding happen outside.

```
[forLoopStart] ──index──▶ MiniMaxH3Segment ──positive──▶ BasicGuider ─▶ SamplerCustomAdvanced ─▶ VAEDecode ─▶ ...
                 └─value1──▶ prev_latent(continuity feedback)  └─latent──▶         └─▶ VAEDecodeAudio ─▶ CreateVideo
```

- **Tasks**: t2v / i2v / fl2v / r2v / v2v / rv2v (`task_type` on the node; the editor switches the matching timeline);
- **`trim_frames` / `target_frames`**: pinned-head frames / target export frames; use with `MiniMaxH3R2VTrim` after decode so the export length is exact;
- **`source_audio`**: the segment's source-video audio (v2v/rv2v) — feed `MiniMaxH3AudioSelect.source_audio` for "source" mode;
- **Segment continuity**: enabled when `prev_latent` is connected and the timeline's 段间引导 is on (i2v with a source image auto-skips);
- **Run select**: the compact loop index is mapped to the checked segments (`forLoopStart.total` = number checked).

### 2. `MiniMaxH3R2VTrim` — head + tail trim

Removes the pinned continuity head and the 17k+5 grid alignment overshoot after decode, so each segment exports exactly.

```
VAEDecode.images ──▶ MiniMaxH3R2VTrim ──▶ CreateVideo
VAEDecodeAudio.audio ──▶ (audio trimmed together)
MiniMaxH3Segment.trim_frames / target_frames ──▶ connect
```

### 3. `MiniMaxH3AudioSelect` — generate / source / mute

Switch the final audio between Trim and CreateVideo.audio.

```
Trim.audio ──▶ MiniMaxH3AudioSelect ──▶ CreateVideo.audio
MiniMaxH3Segment.source_audio ──▶ source_audio (for "source" mode)
mode: generate (default) / source / mute
```

### 4. Local two-sample video (`frames_0` / `audio_0`)

Except v2v/rv2v, each group/shot can upload a local two-sample video; the node outputs its frames `frames_0` (IMAGE) and audio `audio_0` (AUDIO) **for external custom sampling only — not used in internal conditioning**.

- t2v/i2v: the group's preview area; fl2v: under each shot's duration; r2v: under each group's audio row;
- When not uploaded, outputs placeholders (1 gray frame + segment-length silence) so downstream nodes never get `None`.

### 5. Trimmed export UI

The new node hides "export mode / audio / max frames" (export happens outside); keeps resolution and segment continuity; shows a static 「本地二采不可用」 hint after the continuity control.

## Installation

Same as the original: put it under `ComfyUI/custom_nodes/`, `pip install -r requirements.txt`, restart ComfyUI.

- **Dependency**: ComfyUI ≥ 0.30.0 with official MiniMax H3 nodes;
- Optional `imageio-ffmpeg` (ffmpeg needed to extract `source_audio`).

Models are the same as the original — see the original author's repo or [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3).

## Original author

- **AIMixer**: [github.com/AIMixer/ComfyUI_MiniMaxH3_Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director)
- Tutorials / community info: see the original author's repository.

## Credits

- Original author [AIMixer](https://github.com/AIMixer) — [ComfyUI_MiniMaxH3_Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director)
- [Comfy-Org / ComfyUI](https://github.com/Comfy-Org/ComfyUI) — official MiniMax H3 support
- [MiniMax-AI](https://github.com/MiniMax-AI) — MiniMax H3 model
- [NikoDemon80/ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context) — cross-segment motion/audio continuation approach

## License

Apache-2.0
