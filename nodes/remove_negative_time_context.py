"""MiniMax H3 负时间条件移除节点。

「段间衔接」(负时间)由 ``director/h3_motion_context.py`` 的
``apply_negative_time_context`` 往 CONDITIONING 的 meta 里注入三样东西:

1. **纯负时间 context blocks** —— ``minimax_keyframes`` 中只带 ``CTX_NEG_KEY``
   的项,内容是上一段尾巴的 latent token,RoPE 位置在 origin 之前(负时间);
2. **身份 pin** —— 同时带 ``CTX_FRAME_KEY:0`` 与 ``CTX_NEG_KEY:0``,内容是上一段
   真实导出末像素,RoPE 位置落在 **origin(frame 0)**,是本段首帧的身份锚点;
3. **衔接音频 ref** —— ``minimax_refs`` 中带 ``CTX_AUDIO_END_KEY`` 的项,由
   ``conditioning_set_values(..., append=True)`` 追加。

本节点移除 1 与 3、**保留 2**:pin 的时间位置在 origin 而非负时间,且注入时上游
已把 stock 首帧 keyframe 丢弃(见 ``apply_negative_time_context`` 的
``keep_existing_keyframes`` 过滤),删掉 pin 会让 frame 0 失去唯一锚点且无法恢复。

移除后 layout / payload patch 的行为(均已核对):

* ``_rewrite_keyframe_times`` 仍被调用(pin 带 ``CTX_FRAME_KEY``),``neg_count``
  归零,pin 按 ``origin`` 放置 —— 与移除前完全一致;
* ``_rewrite_audio_timeline`` 不再被调用(``has_ctx_audio`` 为 False),因此不会
  触发 "audio continuation requires exactly one marked audio reference" 断言;
* ``_director_extra_conds`` 在 ``minimax_refs`` 缺失/为空时直接返回,不做 payload
  合并(此时 stock payload 自行处理 pin);r2v / v2v / rv2v 的参考图 ref 不带
  ``CTX_AUDIO_END_KEY``,会被保留,合并逻辑照旧带上 pin + 参考图。

未注入负时间条件时(段间衔接未开启、或接错上游)本节点原样透传,可在工作流中常开。
"""

from __future__ import annotations

import logging

from ..director.h3_context_patches import (
    CTX_AUDIO_END_KEY,
    CTX_FRAME_KEY,
    CTX_NEG_KEY,
)

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.remove_neg_ctx")


def _is_negative_context_keyframe(kf) -> bool:
    """纯负时间 block:带 ``CTX_NEG_KEY`` 且不带 ``CTX_FRAME_KEY``。

    身份 pin 两个标记都有(RoPE 在 origin,是 frame 0 的锚点),不算负时间块。
    该判据对稠密(``CTX_NEG_SHIFT_KEY`` 整栈平移)与稀疏
    (``CTX_NEG_BACK_KEY`` 逐块真实距离)两种注入方式同样成立。
    """
    if not isinstance(kf, dict):
        return False
    return kf.get(CTX_NEG_KEY) is not None and CTX_FRAME_KEY not in kf


def _is_join_audio_ref(ref) -> bool:
    """衔接音频 ref:带 ``CTX_AUDIO_END_KEY`` 标记。

    r2v / v2v / rv2v 的参考图与参考视频 ref 由官方节点注入,不带该 Director
    标记,因此不会被误删。
    """
    if not isinstance(ref, dict):
        return False
    return ref.get(CTX_AUDIO_END_KEY) is not None


class MiniMaxH3RemoveNegativeTimeContext:
    """Strip 段间衔接 negative-time context blocks from a CONDITIONING."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": (
                    "CONDITIONING",
                    {
                        "tooltip": (
                            "接 MiniMaxH3Segment 的 positive。节点会滤掉段间衔接注入的"
                            "负时间 keyframe 与衔接音频 ref,保留身份 pin、提示词与参考图。"
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "remove"
    CATEGORY = "MinimaxH3_Segment"
    DESCRIPTION = (
        "移除「段间衔接」(负时间)注入的条件:滤除所有纯负时间 keyframe(上一段尾巴的 "
        "latent block)与衔接音频 ref,保留 frame 0 的身份 pin、提示词与 r2v 参考图/视频。"
        "没有负时间条件时原样透传。"
    )

    def remove(self, conditioning):
        if not conditioning:
            return (conditioning,)

        removed_kf = 0
        removed_ref = 0
        out = []
        for item in conditioning:
            # CONDITIONING 元素形如 [tensor, meta] 或 [tensor, meta, extra...]。
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                out.append(item)
                continue
            meta = item[1]
            if not isinstance(meta, dict):
                out.append(item)
                continue

            new_meta = dict(meta)

            keyframes = meta.get("minimax_keyframes")
            if isinstance(keyframes, (list, tuple)) and keyframes:
                kept = [
                    kf for kf in keyframes if not _is_negative_context_keyframe(kf)
                ]
                removed_kf += len(keyframes) - len(kept)
                new_meta["minimax_keyframes"] = kept

            refs = meta.get("minimax_refs")
            if isinstance(refs, (list, tuple)) and refs:
                kept_refs = [r for r in refs if not _is_join_audio_ref(r)]
                removed_ref += len(refs) - len(kept_refs)
                if kept_refs:
                    new_meta["minimax_refs"] = kept_refs
                else:
                    # 全是衔接音频 ref:直接删键,回到「从未注入 refs」的状态,
                    # 避免下游 payload patch 在空列表与 None 之间产生行为分叉。
                    new_meta.pop("minimax_refs", None)

            new_item = list(item)
            new_item[1] = new_meta
            out.append(new_item)

        if removed_kf or removed_ref:
            log.info(
                "RemoveNegativeTimeContext: 移除 %d 个负时间 keyframe、%d 个衔接音频 ref"
                "(保留 frame 0 身份 pin)。",
                removed_kf,
                removed_ref,
            )
        else:
            log.info(
                "RemoveNegativeTimeContext: 未发现负时间条件,原样透传"
                "(段间衔接未开启或上游未经过 apply_negative_time_context)。"
            )
        return (out,)
