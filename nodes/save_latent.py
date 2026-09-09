"""MiniMax H3 AV latent 保存节点(核心 SaveLatent 无法处理 NestedTensor)。

核心 ``SaveLatent``(``nodes.py``)对 ``samples["samples"]`` 直接调
``.contiguous()``——AV latent 的 samples 是 ``comfy.nested_tensor.NestedTensor``
(视频+音频双流),没有该属性,保存即崩
(``AttributeError: 'NestedTensor' object has no attribute 'contiguous'``)。

本节点改用 ``_av_latent_to_cpu`` 逐流搬到 CPU 再 ``torch.save``,产物与插件
``.av.pt`` 缓存同格式(``{"samples": NestedTensor((video, audio))}``),可被
``nodes/r2v_segment.py`` 的 ``_load_latent_file`` 直接读回,作为
「上传段间引导latent」/「本地二采latent」再上传复用。

只有一个 ``latent`` 输入(与其它 LATENT 节点同一种类型)。为避免误接:
保存前做 H3 格式校验——视频流必须形如 5D ``[B,24,T,H,W]``(AV 双流或纯视频
张量均可)。接了非 H3 的 latent(如图像 4D latent)时直接报错提示,不会静默
存出无法回读的文件。
"""

from __future__ import annotations

import logging
import os

import torch

import folder_paths

from ..director.segment_cache import _av_latent_to_cpu
from ..director.segment_runtime import get_current_segment

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.save_latent")

_DEFAULT_PREFIX = "minimax_h3_latent"

# backfill widget:「回填」= 保存后把路径填入当前素材组的本地二采latent槽。
BACKFILL_OFF = "不回填"
BACKFILL_ON = "回填"
BACKFILL_OPTIONS = [BACKFILL_OFF, BACKFILL_ON]


def _rel_ref(filename: str, subfolder: str) -> dict:
    """构造素材组「本地二采latent」的 ref dict(与前端 uploadLocalLatent 同结构)。

    ``type="output"`` 让 ``resolve_video_path`` 优先在 output/ 目录找文件
    (SaveLatent 存的是 output/,普通上传存的是 input/)。
    """
    sub = (subfolder or "").strip().replace("\\", "/").strip("/")
    return {
        "videoFile": f"{sub}/{filename}" if sub else filename,
        "fileName": filename,
        "subfolder": sub,
        "type": "output",
    }


def _report_latent_backfill(node_id, segment_index: int, ref: dict) -> None:
    """推送回填事件给前端,让时间轴 UI 把路径写入对应素材组并显示。"""
    try:
        from server import PromptServer

        srv = PromptServer.instance
        if not srv:
            return
        srv.send_sync(
            "minimax_director_latent_backfill",
            {
                "node_id": str(node_id),
                "segment_index": int(segment_index),
                "latent": ref,
            },
            srv.client_id,
        )
    except Exception as exc:
        log.warning("MiniMaxH3SaveLatent: 回填事件推送失败: %s", exc)


def _assert_h3_latent(payload: dict) -> None:
    """校验 latent 是 H3 音视频/视频格式(视频流 5D [B,C,T,H,W]),否则报清晰错误。"""
    samples = payload["samples"]
    if hasattr(samples, "unbind"):
        streams = list(samples.unbind())
        video = streams[0] if streams else None
    elif isinstance(samples, (tuple, list)):
        video = samples[0] if samples else None
    else:
        video = samples
    if video is None or not torch.is_tensor(video) or video.ndim != 5:
        raise ValueError(
            "MiniMaxH3SaveLatent: 接入的 latent 不是 H3 音视频/视频 latent"
            "(视频流应形如 [B,24,T,H,W])。"
            f"实际: {tuple(video.shape) if torch.is_tensor(video) else type(video).__name__}。"
            "请接入 MiniMaxH3Segment 的 latent / latent_0、拆分节点输出的视频流、"
            "或 MiniMaxH3LatentUpscaleTo 的输出。"
        )


class MiniMaxH3SaveLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "filename_prefix": ("STRING", {"default": _DEFAULT_PREFIX}),
                "backfill": (
                    BACKFILL_OPTIONS,
                    {
                        "default": BACKFILL_OPTIONS[0],
                        "tooltip": (
                            "回填:保存后把文件路径自动填入「当前跑的素材组」的本地二采latent槽,"
                            "时间轴 UI 会同步显示。需要先经过 MiniMaxH3Segment 节点才能定位素材组。"
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "MinimaxH3_Segment"
    DESCRIPTION = (
        "MiniMax H3 专用 latent 保存。只有 latent 输入,与其它 LATENT 节点同类型。"
        "存为 .av.pt,可回读作为「上传段间引导latent」/「本地二采latent」。"
        "backfill=回填 时,保存后自动把路径填入当前跑的素材组的「本地二采latent」槽,"
        "并在时间轴 UI 上显示(下次运行该组即可直接复用这个 latent,无需手动上传)。"
        "保存前校验 H3 格式,误接非 H3 latent 会直接报错,防止存出没法回读的文件。"
    )

    def save(self, latent, filename_prefix=_DEFAULT_PREFIX, backfill=BACKFILL_OPTIONS[0]):
        _assert_h3_latent(latent)
        # 与核心 SaveLatent 一致,存到 output/。回填的 ref 带 type="output",
        # resolve_video_path 会优先在 output/ 找文件,无需手动拷进 input/。
        full_folder, base_name, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory()
        )
        filename = f"{base_name}_{counter:05}_.av.pt"
        file_path = os.path.join(full_folder, filename)

        # 逐流搬到 CPU 再保存(NestedTensor 可正常 pickle),兼容 _load_latent_file。
        payload = _av_latent_to_cpu(latent)
        torch.save(payload, file_path)

        results = [
            {
                "filename": filename,
                "subfolder": subfolder,
                "type": "output",
            }
        ]

        if backfill == BACKFILL_ON:
            # 当前跑的素材组由 MiniMaxH3Segment.execute 在执行时登记。
            seg_node_id, seg_index = get_current_segment()
            if seg_node_id is None or seg_index is None:
                raise ValueError(
                    "MiniMaxH3SaveLatent: 回填失败 —— 定位不到「当前跑的素材组」。"
                    "本节点必须接在 MiniMaxH3Segment 之后(采样链上游要有 Segment 节点),"
                    "由它登记当前素材组;或把 backfill 改回「不回填」。"
                )
            _report_latent_backfill(
                seg_node_id, seg_index, _rel_ref(filename, subfolder)
            )

        return {"ui": {"latents": results}, "result": (latent,)}
