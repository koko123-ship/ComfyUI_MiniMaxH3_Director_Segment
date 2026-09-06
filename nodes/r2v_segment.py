"""MiniMax H3 per-segment conditioning node (loop-driven, external sampling).

The Director role is reduced to "input management": given a loop counter
``index``, this node returns the conditioning (``positive`` + ``latent``) for
that segment of the timeline, for gen tasks t2v / i2v / fl2v / r2v. Sampling,
decode and video assembly happen OUTSIDE this node with standard ComfyUI nodes
(the official advanced sampler chain + VAEDecode/VAEDecodeAudio + CreateVideo),
so no sampling widgets exist here and ``negative`` is not produced (MiniMax H3
has no negative).

Segment continuity (段间引导) reads from the timeline UI (plan.continuity_*),
so this node carries no continuity widgets; it only pins the previous segment's
latent when ``prev_latent`` is connected (loop value feedback).
"""

from __future__ import annotations

import logging
import os

import torch

import comfy.model_management

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.r2v_segment")

from ..director.h3_motion_context import (
    _streams_from_latent,
    apply_motion_context,
    apply_negative_time_context,
    generation_frame_budget,
    negative_time_frame_budget,
    segment_export_end_frame,
    snap_context_frames,
    snap_join_context_frames,
)
from ..director.plan import (
    build_director_plan,
    ref_audios_to_dict,
    ref_videos_to_dict,
    reinforce_r2v_prompt,
    reinforce_rv2v_prompt,
    reinforce_v2v_prompt,
)
from ..director.backfill_store import get_backfill
from ..director.segment_cache import _av_latent_to_cpu, prune_stale_segment_cache
from ..director.segment_runtime import resolve_segment_raw_clip
from ..lib.audio_io import extract_timeline_audio, load_reference_audio
from ..lib.image_prep import fit_canvas, fit_video_long_edge
from ..lib.task_prompts import TASK_PROMPT_BY_KEY, resolve_task_key, task_type_option_label
from ..lib.video_io import load_reference_video_clip, probe_video_clip, resolve_video_path
from .conditioning import run_minimax_conditioning
from .director_common import timeline_required_inputs

# 可在节点外自定义采样的任务(t2v / i2v / fl2v / r2v / v2v / rv2v)。
TASK_KEYS = ("t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v")
TASK_LABELS = [
    task_type_option_label(TASK_PROMPT_BY_KEY[k]) for k in TASK_KEYS
]


def segment_timeline_inputs() -> dict:
    """Timeline widget group for the segment node.

    Reuses the Director editor's widget set but drops sampling-only widgets
    (cfg / seed / 采样设置 group) — sampling happens outside this node.
    ``task_type`` offers the gen tasks t2v / i2v / fl2v / r2v.
    """
    inputs = timeline_required_inputs()
    inputs["task_type"] = (
        list(TASK_LABELS),
        {
            "default": TASK_LABELS[0],
            "tooltip": "t2v / i2v / fl2v / r2v / v2v / rv2v。采样/解码/合成在节点外完成。",
        },
    )
    for key in ("cfg", "seed", "bd_grp_sample"):
        inputs.pop(key, None)
    return inputs


def _ref_tensor_from_seg_refs(refs, index):
    """Return the ``index``-th reference image as a single frame (or None)."""
    for ref in refs or []:
        if int(getattr(ref, "index", -1)) == index and ref.tensor is not None:
            t = ref.tensor
            if int(t.shape[0]) > 0:
                return t[:1]
    return None


def _silent_audio(target_frames: int, fps: float, sr: int = 32000) -> dict:
    """生成一段目标时长的立体声静音 AUDIO(避免下游节点收到 None 崩溃)。"""
    n = max(0, int(round((max(0, int(target_frames)) / float(fps or 24.0)) * sr)))
    return {"waveform": torch.zeros(1, 2, n, dtype=torch.float32), "sample_rate": sr}


def _seg_raw(plan, seg) -> dict:
    """取时间轴原始 JSON 中该段对应的 dict(gen 批量用 segments;fl2v 用 shots)。"""
    if plan is None or not plan.raw:
        return {}
    raw_segs = plan.raw.get("segments") or plan.raw.get("shots") or []
    if 0 <= seg.index < len(raw_segs) and isinstance(raw_segs[seg.index], dict):
        return raw_segs[seg.index]
    return {}


def _load_latent_file(ref) -> tuple[dict | None, bool]:
    """从段 raw 的 ref dict 加载本地 latent 文件,返回 (latent_dict, has_audio)。

    支持插件 .av.pt 缓存同款格式(torch.save({"samples": NestedTensor((video, audio))})),
    也兼容裸 5D 视频张量(自动包成单流 NestedTensor)与 .safetensors(取第一个 tensor)。
    未上传/加载失败/格式无法识别 → (None, False)。
    """
    if not isinstance(ref, dict):
        return None, False
    fname = (ref.get("videoFile") or ref.get("fileName") or ref.get("file") or "").strip()
    if not fname:
        return None, False
    path = resolve_video_path(ref)
    if not os.path.isfile(path):
        log.warning("MiniMax H3 Segment: 本地 latent 文件不存在: %s", path)
        return None, False
    try:
        if fname.lower().endswith(".safetensors"):
            import safetensors.torch

            payload = safetensors.torch.load_file(path)
            if not isinstance(payload, dict) or not payload:
                return None, False
            payload = next(iter(payload.values()))
        else:
            payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        log.warning("MiniMax H3 Segment: 本地 latent 加载失败: %s", exc)
        return None, False

    if isinstance(payload, dict) and "samples" in payload:
        latent = _av_latent_to_cpu(payload)
        try:
            has_audio = len(_streams_from_latent(latent)) >= 2
        except Exception:
            has_audio = False
        return latent, has_audio
    if isinstance(payload, torch.Tensor) and payload.ndim == 5:
        import comfy.nested_tensor

        video = payload.detach().cpu().contiguous()
        return {"samples": comfy.nested_tensor.NestedTensor((video,))}, False
    log.warning(
        "MiniMax H3 Segment: 本地 latent 格式无法识别(%s),期望 "
        "{'samples': NestedTensor} 或 5D 视频张量。",
        type(payload).__name__,
    )
    return None, False


def _latent_to_device(latent: dict, device) -> dict:
    """把 AV latent dict 整体搬到指定设备(流级重建 NestedTensor,镜像 _av_latent_to_cpu)。"""
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        import comfy.nested_tensor

        parts = [p.to(device) for p in samples.unbind()]
        samples_out = comfy.nested_tensor.NestedTensor(tuple(parts))
    elif isinstance(samples, (tuple, list)):
        samples_out = tuple(p.to(device) for p in samples)
    elif torch.is_tensor(samples):
        samples_out = samples.to(device)
    else:
        samples_out = samples
    out = {"samples": samples_out}
    for key, value in latent.items():
        if key == "samples":
            continue
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def _load_two_video(plan, seg):
    """读取该段的「本地二采视频」(twoVideo),返回 (frames_0, audio_0)。

    纯输出用:不参与内部条件编码,只把上传视频的帧和音频取出来供外部采样。
    没有上传/加载失败时返回有效占位(1 帧灰图 + 段时长静音),不给 None。
    """
    seg_raw = _seg_raw(plan, seg)
    # 二采主存 refVideos(index>=3,参考视频持久化最稳);兼容旧 twoVideo 字段。
    two_video: dict = {}
    for rv in (seg_raw.get("refVideos") or seg_raw.get("ref_videos") or []):
        if isinstance(rv, dict) and int(rv.get("index", rv.get("slot", -1)) or -1) >= 3:
            two_video = rv
            break
    if not (two_video.get("videoFile") or two_video.get("fileName") or "").strip():
        two_video = seg_raw.get("twoVideo") or seg_raw.get("two_video") or {}
    fps = float((plan.frame_rate if plan is not None else 0) or 24)
    frames = torch.zeros((1, 16, 16, 3), dtype=torch.float32)  # 1 帧灰占位
    audio = _silent_audio(int(seg.frame_count), fps)
    if not (two_video.get("videoFile") or two_video.get("fileName") or "").strip():
        return frames, audio

    try:
        probe = probe_video_clip(two_video)
        n = max(
            1,
            int(
                probe.get("frame_count")
                or probe.get("frameCount")
                or 124
            ),
        )
        # 用二采视频的「原生 fps」加载,否则按时间轴 fps 重采样会拉伸时长 → 播放变慢。
        native_fps = float(probe.get("native_fps") or probe.get("nativeFps") or 0)
        tl = dict(plan.raw) if plan is not None and plan.raw else {}
        if native_fps and native_fps > 0:
            tl["frameRate"] = native_fps
        frames = load_reference_video_clip(two_video, tl, n, start_frame=0)
    except Exception as exc:
        log.warning("MiniMax H3 Segment: 二采视频帧加载失败: %s", exc)
    try:
        audio = load_reference_audio(resolve_video_path(two_video))
    except Exception as exc:
        log.warning("MiniMax H3 Segment: 二采视频音频加载失败: %s", exc)
    return frames, audio


class MiniMaxH3Segment:
    """Loop-driven per-segment conditioning provider (t2v/i2v/fl2v/r2v)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": (
                    "CLIP",
                    {"tooltip": "CLIPLoader type=minimax (qwen3vl)."},
                ),
                "vae": (
                    "VAE",
                    {"tooltip": "MiniMax H3 video VAE (minimax_h3_video_vae)."},
                ),
                "audio_vae": (
                    "VAE",
                    {"tooltip": "MiniMax H3 audio VAE (minimax_h3_audio_vae). r2v 必需。"},
                ),
                "index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "tooltip": (
                            "loop 计数器:取时间轴第 index 段(0 起)。"
                            "loop 的 total 应等于时间轴的段数(或「选择运行」勾选的段数)。"
                        ),
                    },
                ),
                **segment_timeline_inputs(),
            },
            "optional": {
                "prev_latent": (
                    "LATENT",
                    {
                        "tooltip": (
                            "段间连贯:上一段采样器输出的 latent(循环 value 回传)。"
                            "接了且时间轴开启段间引导时,把上一段末尾运动钉入本段头部。"
                        ),
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "dynprompt": "DYNPROMPT",
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "INT", "INT", "AUDIO", "IMAGE", "AUDIO", "LATENT", "INT")
    RETURN_NAMES = ("positive", "latent", "trim_frames", "target_frames", "source_audio", "frames_0", "audio_0", "latent_0", "loop_index")
    FUNCTION = "execute"
    CATEGORY = "MinimaxH3_Segment"
    DESCRIPTION = (
        "MiniMax H3 Segment: 根据 loop 计数器返回时间轴第 index 段的条件"
        "(positive + latent),支持 t2v / i2v / fl2v / r2v / v2v / rv2v。采样/解码/合成全部"
        "在节点外(官方高级采样器链 + CreateVideo)。无 negative。段间连贯读取时间轴设置,"
        "接 prev_latent 生效。trim_frames = 钉进本段头部的帧数(第1段=0),"
        "target_frames = 本段目标导出帧数;解码后用 MiniMaxH3R2VTrim 裁头+裁尾。"
        "source_audio = 该段源视频的原声音频(v2v/rv2v 用「使用原声」时接 MiniMaxH3AudioSelect)。"
        "frames_0 / audio_0 = 该段「本地二采视频」的帧与音频(仅供外部自定义采样使用)。"
        "latent_0 = 该段「本地二采 latent」(纯透传,不参与内部条件)。若上传了本地二采 latent,"
        "且时间轴勾选「段间引导 + 引用上段」,trim_frames 仍按重叠帧输出,target_frames = 段目标帧数,"
        "供解码后用 MiniMaxH3R2VTrim 裁出与段时长一致的对齐视频。"
        "loop_index = 本次循环 index 原样透出(供 MiniMaxH3SaveLatent「回填」接线定位该段素材组)。"
        "本地二采 latent 也可由回填自动注入:Save Latent 勾回填后,下次跑该行自动用最近一次保存的 latent。"
    )

    def execute(
        self,
        clip,
        vae,
        audio_vae,
        index,
        task_type,
        global_prompt,
        frame_rate,
        width,
        height,
        ref_max_size,
        total_frames,
        timeline_data,
        prev_latent=None,
        unique_id=None,
        dynprompt=None,
        **kwargs,
    ):
        task_key = resolve_task_key(task_type)
        if task_key not in TASK_KEYS:
            raise ValueError(
                f"MiniMax H3 Segment: 不支持任务 '{task_key}'"
                f"(支持 {'/'.join(TASK_KEYS)})。"
            )

        if not timeline_data or not str(timeline_data).strip():
            raise ValueError(
                "MiniMax H3 Segment: 时间轴为空,请在节点时间轴 UI 里添加至少一段。"
            )

        plan = build_director_plan(
            timeline_data,
            global_task_type=task_type,
            global_prompt=(global_prompt or "").strip(),
            total_frames=total_frames,
            frame_rate=frame_rate,
            width=width,
            height=height,
            ref_max_size=ref_max_size,
        )
        if plan.global_task_key != task_key:
            raise ValueError(
                f"MiniMax H3 Segment: 时间轴任务类型是 '{plan.global_task_key}',"
                f"与节点选择的 '{task_key}' 不一致。请在时间轴里改为 {task_key}。"
            )
        # 清理已删素材的磁盘缓存残留(seg_{idx} 且 idx >= 当前段数)。
        if unique_id:
            prune_stale_segment_cache(unique_id, plan.segment_count)

        index = int(index)
        # 「选择运行」:把循环的紧凑序号(0..勾选数-1)映射到实际勾选的段。
        run_set = plan.run_indices
        if run_set is not None:
            ordered = sorted(run_set)
            if index < 0 or index >= len(ordered):
                raise ValueError(
                    f"MiniMax H3 Segment: 循环 index={index} 越界(时间轴共 {plan.segment_count} 段,"
                    f"「选择运行」勾选了 {len(ordered)} 段:第 {[i + 1 for i in ordered]} 段)。"
                    "forLoopStart 的 total 应设为勾选的段数。"
                )
            seg = plan.segments[ordered[index]]
        else:
            if index < 0 or index >= plan.segment_count:
                raise ValueError(
                    f"MiniMax H3 Segment: index={index} 越界(时间轴共 {plan.segment_count} 段)。"
                    "loop 的 total 应设为时间轴的段数。"
                )
            seg = plan.segments[index]

        # 源片段:gen 任务取 seg.source_clip,v2v/rv2v 从源视频时间轴取该段画面。按输出画布归一。
        clip_frames = resolve_segment_raw_clip(plan, seg)
        if clip_frames is not None and int(clip_frames.shape[0]) > 0:
            if plan.output_mode == "fixed":
                clip_frames = fit_canvas(clip_frames, plan.width, plan.height)
            else:
                clip_frames = fit_video_long_edge(clip_frames, plan.ref_max_size)
                if (
                    int(clip_frames.shape[1]) != int(plan.height)
                    or int(clip_frames.shape[2]) != int(plan.width)
                ):
                    clip_frames = fit_canvas(clip_frames, plan.width, plan.height)

        # ── 按任务构建条件输入 ──
        first_frame = None
        last_frame = None
        ref_images = None
        ref_videos = None
        ref_audios = None

        if task_key == "fl2v":
            first_frame = _ref_tensor_from_seg_refs(seg.refs, 0)
            last_frame = _ref_tensor_from_seg_refs(seg.refs, 1)
            if first_frame is None and last_frame is None and clip_frames is not None:
                if int(clip_frames.shape[0]) >= 1:
                    first_frame = clip_frames[:1]
                if int(clip_frames.shape[0]) >= 2:
                    last_frame = clip_frames[-1:].clone()
            elif first_frame is not None and last_frame is None and clip_frames is not None:
                if int(clip_frames.shape[0]) >= 2:
                    last_frame = clip_frames[-1:].clone()
        elif task_key == "i2v":
            if clip_frames is not None and int(clip_frames.shape[0]) > 0:
                first_frame = clip_frames[:1]
            else:
                first_frame = _ref_tensor_from_seg_refs(seg.refs, 0)
        elif task_key == "r2v":
            ref_videos = ref_videos_to_dict(
                [
                    (int(v.index), v.tensor)
                    for v in (seg.ref_videos or [])
                    if v is not None and v.tensor is not None and int(v.tensor.shape[0]) > 0
                ]
            )
        elif task_key in ("v2v", "rv2v"):
            # 源视频时间轴编辑:每段源画面作为 <Video 1> 参考。
            if clip_frames is None or int(clip_frames.shape[0]) <= 0:
                raise ValueError(
                    f"MiniMax H3 Segment: {task_key} 段 #{seg.index + 1} 没有源画面。"
                    "请在时间轴上传源视频再运行。"
                )
            ref_videos = {"ref_video_0": clip_frames}

        # r2v / rv2v 共享:参考图 + 参考音频。
        if task_key in ("r2v", "rv2v"):
            ref_images = {}
            for ref in seg.refs or []:
                if ref is None or ref.tensor is None or int(ref.tensor.shape[0]) <= 0:
                    continue
                idx = int(getattr(ref, "index", len(ref_images)))
                t = ref.tensor
                ref_images[f"ref_image_{idx}"] = t[:1] if t.ndim == 4 else t
            ref_images = ref_images or None
            # ref_audios_to_dict 期望 SegmentRefAudio 对象(内部取 .index/.audio)。
            ref_audios = ref_audios_to_dict(
                [a for a in (seg.ref_audios or []) if a is not None]
            )

        # ── 本地 latent:上传段间引导latent / 本地二采latent ──
        seg_raw = _seg_raw(plan, seg)
        guide_latent, guide_has_audio = _load_latent_file(
            seg_raw.get("guideLatent") or seg_raw.get("guide_latent")
        )
        two_ref = seg_raw.get("twoLatent") or seg_raw.get("two_latent")
        # SaveLatent「回填」:该 (段节点, 循环 index) 最近一次保存的 latent 总是
        # 覆盖素材组手动填的两 ref(登记表在 SaveLatent 回填时写、关闭回填时清)。
        # easy-forLoop 每轮给循环体节点换带前缀的临时 id(如 394.1.0.468),用
        # dynprompt 归一成逻辑段节点 id(468),保证与 Save 侧登记 key 一致。
        seg_uid = str(unique_id)
        if dynprompt is not None:
            try:
                seg_uid = str(dynprompt.get_display_node_id(str(unique_id)))
            except Exception:
                pass
        backfilled = get_backfill(seg_uid, int(index))
        if backfilled is not None:
            log.info(
                "MiniMaxH3Segment[node %s]: 行 %s 命中回填登记,input/%s 覆盖素材组两 ref",
                seg_uid,
                int(index),
                backfilled.get("videoFile") or backfilled.get("fileName") or "?",
            )
            two_ref = backfilled
        else:
            log.debug(
                "MiniMaxH3Segment[node %s]: 行 %s 无回填登记(手动两 ref=%r)",
                seg_uid,
                int(index),
                (two_ref or {}).get("videoFile") if isinstance(two_ref, dict) else two_ref,
            )
        two_latent, _ = _load_latent_file(two_ref)
        # r2v 参考图尺寸(统一设置,存于 output):match=按生成画布等比缩放,max=2048 高清保真。
        _out = plan.raw.get("output") or {}
        ref_image_size = str(_out.get("refImageSize") or _out.get("ref_image_size") or "match")
        if ref_image_size not in ("match", "max"):
            ref_image_size = "match"

        # ── 段间连贯/段间衔接: 从时间轴 UI 读取(plan.continuity_* / plan.join_* +
        # seg.continuity_from_prev) ──
        # 上传段间引导latent 优先于接口 prev_latent,且上传即强制启用该段导引。
        # i2v 不特判:只要接了 prev_latent 且主开关+「引用上段」开启,段间连贯即生效
        # (prev_latent 尾部会替换本段首帧 keyframe,即"上段结尾=本段开头")。
        # 段间衔接为主方案时(join_master),上传的 latent 也路由到 join,两方案互斥。
        ctx_latent = guide_latent if guide_latent is not None else prev_latent
        local_force = guide_latent is not None
        join_master = bool(getattr(plan, "join_enabled", False))
        continuity_master = bool(getattr(plan, "continuity_enabled", False))
        from_prev = bool(getattr(seg, "continuity_from_prev", True))
        if join_master:
            # 上传即强制启用(与段间引导一致);否则需主开关+「引用上段」。
            join_on = ctx_latent is not None and (local_force or from_prev)
            continuity_on = False
        else:
            continuity_on = ctx_latent is not None and (
                local_force or (continuity_master and from_prev)
            )
            join_on = False
        if join_on:
            ctx_n = snap_join_context_frames(plan.join_overlap_frames)
            sample_len, _budget_trim = negative_time_frame_budget(
                int(seg.frame_count), ctx_n
            )
        elif continuity_on:
            ctx_n = snap_context_frames(plan.continuity_overlap_frames)
            sample_len, _budget_trim = generation_frame_budget(int(seg.frame_count), ctx_n)
        else:
            ctx_n = 0
            sample_len, _budget_trim = generation_frame_budget(int(seg.frame_count), 0)

        # ── 提示词强化 ──
        prompt = seg.prompt
        if task_key == "fl2v":
            from ..director.fl2v_timeline import reinforce_fl2v_prompt

            has_start = any(int(getattr(r, "index", -1)) == 0 for r in (seg.refs or []))
            has_end = any(int(getattr(r, "index", -1)) == 1 for r in (seg.refs or []))
            if not has_start and not has_end and seg.refs:
                has_start = True
                has_end = len(seg.refs) >= 2
            prompt = reinforce_fl2v_prompt(
                prompt,
                has_end_frame=has_end,
                has_start_frame=has_start,
            )
        elif task_key == "r2v":
            prompt = reinforce_r2v_prompt(
                seg.prompt,
                ref_indices=[int(r.index) for r in (seg.refs or []) if r is not None],
                video_indices=[int(v.index) for v in (seg.ref_videos or []) if v is not None],
                audio_indices=[int(a.index) for a in (seg.ref_audios or []) if a is not None],
            )
        elif task_key == "v2v":
            prompt = reinforce_v2v_prompt(seg.prompt)
        elif task_key == "rv2v":
            prompt = reinforce_rv2v_prompt(
                seg.prompt,
                ref_indices=[int(r.index) for r in (seg.refs or []) if r is not None],
                audio_indices=[int(a.index) for a in (seg.ref_audios or []) if a is not None],
            )

        positive, _negative, latent, _hint = run_minimax_conditioning(
            clip=clip,
            vae=vae,
            audio_vae=audio_vae,
            prompt=prompt,
            width=int(plan.width),
            height=int(plan.height),
            length=sample_len,
            task_key=task_key,
            first_frame=first_frame,
            last_frame=last_frame,
            ref_images=ref_images,
            ref_videos=ref_videos,
            ref_audios=ref_audios,
            ref_image_size=ref_image_size,
        )

        trim_frames = 0
        if join_on:
            # 段间衔接:上一段尾巴放在负时间条件里 + frame0 身份锁,latent 不加长。
            # context_end_frame 用「上段真实导出末像素」,避免钉到 sample overshoot。
            prev_seg = plan.segments[seg.index - 1] if seg.index > 0 else None
            context_end = (
                segment_export_end_frame(plan, prev_seg)
                if prev_seg is not None
                else None
            )
            positive, trim_frames, _ = apply_negative_time_context(
                positive,
                latent,
                vae=vae,
                context_length=ctx_n,
                context_latent=ctx_latent,
                audio_vae=audio_vae,
                # 本地 latent 可能只有视频流;循环 prev_latent 恒有音频流。
                continue_audio=(not local_force) or guide_has_audio,
                keep_existing_keyframes=(task_key == "fl2v"),
                context_end_frame=context_end,
                sparse_context=bool(getattr(plan, "join_sparse_context", False)),
            )
        elif continuity_on:
            # span = 实际钉进本段头部的帧数(== 上下文帧数,除非上一段不足)。
            positive, trim_frames, _ = apply_motion_context(
                positive,
                latent,
                vae=vae,
                context_length=ctx_n,
                context_latent=ctx_latent,
                audio_vae=audio_vae,
                # 本地 latent 可能只有视频流;循环 prev_latent 恒有音频流。
                continue_audio=(not local_force) or guide_has_audio,
                keep_existing_keyframes=(task_key == "fl2v"),
            )
        elif (
            two_latent is not None
            and (continuity_master or join_master)
            and bool(getattr(seg, "continuity_from_prev", True))
        ):
            # 本地二采复用:上传的 latent_0 已采样好,不触发内部 apply_motion_context,
            # 但该段 UI 勾选「段间引导 + 引用上段」——二采视频开头已带上一段尾巴,
            # 仍要按重叠帧裁头,否则 R2VTrim 收不到 trim_frames,裁不出对齐。
            # 帧数按 UI 读到的重叠帧走;snap_context_frames 会兜底到默认上下文。
            # 段间衔接为负时间方案,无正时间头可裁 → trim_frames = 0。
            trim_frames = (
                0
                if join_master
                else snap_context_frames(
                    getattr(plan, "continuity_overlap_frames", 0) or ctx_n
                )
            )

        # 该段源视频的原声音频(v2v/rv2v「使用原声」用;gen 任务无源音频 → None)。
        fps = float(plan.frame_rate or 24)
        source_audio = extract_timeline_audio(
            plan.raw,
            int(seg.start_frame),
            int(seg.end_frame),
            fps,
        )
        if source_audio is None:
            source_audio = _silent_audio(int(seg.frame_count), fps)
        frames_0, audio_0 = _load_two_video(plan, seg)
        # 本地二采latent:纯透传,不进内部条件,直接吐给外部自定义采样。
        latent_0 = (
            _latent_to_device(two_latent, comfy.model_management.intermediate_device())
            if two_latent is not None
            else None
        )
        return (
            positive,
            latent,
            int(trim_frames),
            int(seg.frame_count),
            source_audio,
            frames_0,
            audio_0,
            latent_0,
            # loop_index = 本次循环 index 原样透出,供 MiniMaxH3SaveLatent「回填」
            # 从段节点接 index,自动定位该段素材组(twoLatent)行。
            int(index),
        )


# 旧名兼容(早期 r2v-only 工作流)。
MiniMaxH3R2VSegment = MiniMaxH3Segment


class MiniMaxH3R2VTrim:
    """Trim the pinned motion-context head (trim_frames) from decoded video+audio.

    Segment continuity pins the previous segment's tail into the head of the
    next segment's sample (never-denoised keyframes). After external decode the
    head ``trim_frames`` frames are the previous tail's duplicate and must be
    dropped; the tail alignment overshoot is also cropped to ``target_frames``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fps": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 1.0,
                        "max": 240.0,
                        "tooltip": "视频帧率,用于同步裁音频头部时长.",
                    },
                ),
            },
            "optional": {
                "images": (
                    "IMAGE",
                    {"tooltip": "解码出的画面帧(VAEDecode 输出).在解码后裁头+裁尾对齐."},
                ),
                "trim_frames": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "tooltip": (
                            "头部要裁掉的帧数。接导演台 MiniMaxH3Segment 的 trim_frames"
                            "(第1段=0,第2段起=上下文帧数 22/39/56)。"
                        ),
                    },
                ),
                "target_frames": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "tooltip": (
                            "本段目标导出帧数(接导演台 target_frames)。>0 时裁头后"
                            "再裁掉尾部对齐溢出,导出正好 target_frames 帧;0 则只裁头。"
                        ),
                    },
                ),
                "audio": (
                    "AUDIO",
                    {"tooltip": "对应音频(VAEDecodeAudio 输出),同步裁掉头部时长."},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "trim"
    CATEGORY = "MinimaxH3_Segment"
    DESCRIPTION = (
        "裁掉头部 trim_frames 帧(段间连贯钉进的上一段尾巴),并按 target_frames "
        "裁掉尾部对齐溢出。在解码后对视频/音频裁切:放在 VAEDecode/VAEDecodeAudio "
        "与 CreateVideo 之间。不做 latent 阶段裁切——MiniMax H3 的 latent 时间轴 "
        "一旦裁剪会破坏解码 chunk 网格相位,导致周期性的明暗闪烁(花屏),因此只在 "
        "解码后的像素上裁头裁尾,保证段时长精确且无花屏。"
    )

    def trim(self, images, fps=24.0, trim_frames=0, target_frames=0, audio=None):
        trim = max(0, int(trim_frames))
        target = max(0, int(target_frames))
        # 解码后裁头 + (可选)裁尾对齐溢出到目标长度。只在像素上裁,不碰 latent。
        if images is not None:
            if target > 0:
                images = images[trim : trim + target]
            else:
                images = images[trim:]
        if isinstance(audio, dict) and audio.get("waveform") is not None:
            sr = int(audio.get("sample_rate") or 32000)
            wave = audio["waveform"]
            fps_f = float(fps or 24.0)
            if target > 0:
                start = int(round((trim / fps_f) * sr))
                end = int(round(((trim + target) / fps_f) * sr))
                audio = {"waveform": wave[..., start:end], "sample_rate": sr}
            else:
                drop = int(round((trim / fps_f) * sr))
                if drop > 0 and int(wave.shape[-1]) > drop:
                    audio = {"waveform": wave[..., drop:], "sample_rate": sr}
                elif drop > 0:
                    audio = {"waveform": wave[..., :0], "sample_rate": sr}
        return images, audio


class MiniMaxH3AudioSelect:
    """Pick the final audio: generated / source / mute.

    The Director node no longer owns audio mode (sampling/export are external),
    so this node lets you choose the final audio right before CreateVideo.audio.
    ``generated_audio`` should be the already-trimmed output of MiniMaxH3R2VTrim.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_audio": (
                    "AUDIO",
                    {"tooltip": "VAEDecodeAudio 经 MiniMaxH3R2VTrim 裁好的生成音频."},
                ),
                "mode": (
                    ["generate", "source", "mute"],
                    {"default": "generate"},
                ),
            },
            "optional": {
                "source_audio": (
                    "AUDIO",
                    {"tooltip": "接导演台 MiniMaxH3Segment 的 source_audio 输出(该段源视频原声)."},
                ),
                "target_frames": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "tooltip": "段目标导出帧数(接导演台 target_frames),用于把 source/mute 裁到段时长.",
                    },
                ),
                "fps": (
                    "FLOAT",
                    {"default": 24.0, "min": 1.0, "max": 240.0},
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "select_audio"
    CATEGORY = "MinimaxH3_Segment"
    DESCRIPTION = (
        "在 生成声音 / 使用原声 / 静音 之间切换最终音频。"
        "放在 MiniMaxH3R2VTrim 与 CreateVideo.audio 之间。"
    )

    def select_audio(self, generated_audio=None, mode="generate", source_audio=None, target_frames=0, fps=24.0):
        target = max(0, int(target_frames))
        fps_f = float(fps or 24.0)
        mode = str(mode or "generate").lower()

        if mode == "mute":
            return (self._silence(generated_audio, source_audio, target, fps_f),)
        if mode == "source":
            if (
                isinstance(source_audio, dict)
                and source_audio.get("waveform") is not None
                and int(source_audio["waveform"].numel()) > 0
            ):
                fitted, _pad_n = self._fit(source_audio, target, fps_f)
                return (fitted,)
            # 没接原声或原声为空时退回生成音频。
            return (generated_audio,)
        return (generated_audio,)

    @staticmethod
    def _fit(audio, target, fps_f):
        """把原声 截短/补静音 到恰好 target 帧时长,返回 (audio, 补了多少采样)。"""
        if target <= 0:
            return audio, 0
        sr = int(audio.get("sample_rate") or 32000)
        wave = audio["waveform"]
        want = int(round((target / fps_f) * sr))
        have = int(wave.shape[-1])
        if have == want:
            return audio, 0
        if have > want:
            return {"waveform": wave[..., :want], "sample_rate": sr}, 0
        pad = torch.zeros((*wave.shape[:-1], want - have), dtype=wave.dtype, device=wave.device)
        return {"waveform": torch.cat([wave, pad], dim=-1), "sample_rate": sr}, want - have

    @staticmethod
    def _silence(generated_audio, source_audio, target, fps_f):
        ref = generated_audio if isinstance(generated_audio, dict) else source_audio
        if isinstance(ref, dict) and ref.get("waveform") is not None:
            sr = int(ref.get("sample_rate") or 32000)
            wave = ref["waveform"]
            n = int(round((target / fps_f) * sr))
            if target > 0 and int(wave.shape[-1]) != n:
                if int(wave.shape[-1]) > n:
                    wave = wave[..., :n]
                else:
                    pad = n - int(wave.shape[-1])
                    wave = torch.cat(
                        [wave, torch.zeros((*wave.shape[:-1], pad), dtype=wave.dtype)], dim=-1
                    )
            return {"waveform": torch.zeros_like(wave), "sample_rate": sr}
        # 无参考:返回默认立体声静音。
        n = int(round((target / fps_f) * 32000))
        return {"waveform": torch.zeros((1, 2, n), dtype=torch.float32), "sample_rate": 32000}
