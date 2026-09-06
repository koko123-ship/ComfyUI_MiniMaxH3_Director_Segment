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

「回填」:勾选后,把本次保存的文件拷进 ComfyUI ``input/`` 并登记为
(段节点, index) 行的本地二采latent,下次跑循环该段自动用这份 latent 起采样
(总是覆盖成最新一次保存;关闭回填会清掉旧登记)。``index`` 从
``MiniMaxH3Segment/R2VSegment`` 新增的 ``loop_index`` 输出接(同一根循环计数),
接线即声明了回填给哪个段节点,无需手填节点号。
"""

from __future__ import annotations

import logging
import os

import torch

import folder_paths

from ..director.backfill_store import backfill_from_output_file, clear_backfill
from ..director.segment_cache import _av_latent_to_cpu

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.save_latent")

_DEFAULT_PREFIX = "minimax_h3_latent"


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


def _resolve_seg_node(prompt, dynprompt, unique_id) -> str | None:
    """从本节点 ``index`` 输入连线的源头找段节点逻辑 id。

    ComfyUI api-格式里已连线输入值为 ``[源节点id, 输出槽位]``(``is_link`` 判定,
    见 comfy_execution/graph_utils.py;``get_input_data`` 取 ``[0]`` 为源节点)。
    回填按方案②从段节点 ``loop_index`` 输出接 index,所以 ``index`` 连线的源头
    就是该段节点。未接线(index 是普通 widget 值时)→ None。

    循环体注意:comfyui-easy-use 的 for/while 每轮把循环体节点整体用新前缀 id
    重实例化(第 1 轮跑原 id、第 2+ 轮跑 ``394.x.0.<id>`` 这类临时 id),而传给
    节点的 ``PROMPT`` 永远是原图 dict(不含临时 id),所以 ``prompt[unique_id]``
    第 2+ 轮查不到自己。改用 hidden ``dynprompt`` 拿当前轮 def(
    ``dynprompt.get_node(unique_id)``),其 ``inputs["index"]`` 已被重接到本轮
    段节点克隆,再 ``get_display_node_id`` 还原成逻辑段节点 id。非循环场景退化为
    原图查法。
    """
    if unique_id is None:
        return None
    uid = str(unique_id)
    # 首选:当前轮 def(dynprompt 含临时/克隆节点)。
    if dynprompt is not None:
        try:
            cur = dynprompt.get_node(uid)
            idx_in = ((cur or {}).get("inputs") or {}).get("index")
            if isinstance(idx_in, (list, tuple)) and len(idx_in) >= 2 and idx_in[0] is not None:
                return str(dynprompt.get_display_node_id(str(idx_in[0])))
        except Exception:
            pass  # 回退到 PROMPT 原图查法
    # 回退:原图 PROMPT(仅原 id 那轮可达)。
    if prompt:
        me = (prompt.get(uid) or {}).get("inputs") or {}
        idx_in = me.get("index")
        if isinstance(idx_in, (list, tuple)) and len(idx_in) >= 2 and idx_in[0] is not None:
            return str(idx_in[0])
    return None


class MiniMaxH3SaveLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "filename_prefix": ("STRING", {"default": _DEFAULT_PREFIX}),
                "backfill": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "回填(自动填本段素材组二采latent)",
                        "label_off": "不回填",
                        "tooltip": (
                            "勾选后:本次保存的文件拷进 input/ 并登记为 (段节点, index) "
                            "行的本地二采latent,下次跑该段自动用最近一次保存的 latent 起采样"
                            "(总是覆盖)。index 需从 MiniMaxH3Segment 的 loop_index 输出接入。"
                        ),
                    },
                ),
            },
            "optional": {
                "index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "tooltip": (
                            "回填行号(循环第几段)。从 MiniMaxH3Segment / R2VSegment 的 "
                            "loop_index 输出接同一根循环计数;接线即声明回填给该段节点。"
                        ),
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "dynprompt": "DYNPROMPT",
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
        "保存前校验 H3 格式,误接非 H3 latent 会直接报错,防止存出没法回读的文件。"
        "「回填」:勾选后自动把本次保存登记为该段素材组的本地二采latent"
        "(index 从段节点 loop_index 输出接入),下次跑该段自动复用,免手动上传。"
    )

    def save(
        self,
        latent,
        filename_prefix=_DEFAULT_PREFIX,
        backfill=False,
        index=0,
        unique_id=None,
        prompt=None,
        dynprompt=None,
    ):
        _assert_h3_latent(latent)
        # 与核心 SaveLatent 一致,存到 output/。
        full_folder, base_name, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory()
        )
        filename = f"{base_name}_{counter:05}_.av.pt"
        file_path = os.path.join(full_folder, filename)

        # 逐流搬到 CPU 再保存(NestedTensor 可正常 pickle),兼容 _load_latent_file。
        payload = _av_latent_to_cpu(latent)
        torch.save(payload, file_path)

        seg_node = _resolve_seg_node(prompt, dynprompt, unique_id)
        if backfill:
            if seg_node is None:
                log.warning(
                    "MiniMaxH3SaveLatent[node %s]: 回填打开但 index 未接线/源非段节点"
                    "(index 需直连 MiniMaxH3Segment/R2VSegment 的 loop_index 输出,"
                    "勿经 Get_/Set_ 虚拟隧道;本执行 id=%s),已跳过回填"
                    "(文件已保存到 output/%s)。",
                    unique_id,
                    unique_id,
                    filename,
                )
            else:
                two_ref = backfill_from_output_file(seg_node, int(index), file_path)
                if two_ref is not None:
                    log.info(
                        "MiniMaxH3SaveLatent[node %s]: 回填段节点 %s 第 %s 段素材组二采latent ← input/%s",
                        unique_id,
                        seg_node,
                        int(index),
                        two_ref["videoFile"],
                    )
        elif seg_node is not None:
            # 关闭回填 = 停止自动复用,清掉该 (段节点, index) 的旧登记。
            clear_backfill(seg_node, int(index))

        results = [
            {
                "filename": filename,
                "subfolder": subfolder,
                "type": "output",
            }
        ]
        return {"ui": {"latents": results}, "result": (latent,)}
