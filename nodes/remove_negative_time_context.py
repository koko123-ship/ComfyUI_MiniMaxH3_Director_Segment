"""MiniMax H3 段间衔接条件移除节点。

「段间衔接」(负时间)由 ``director/h3_motion_context.py`` 的
``apply_negative_time_context`` 往 CONDITIONING 的 meta 里注入三样东西:

1. **纯负时间 context blocks** —— ``minimax_keyframes`` 中只带 ``CTX_NEG_KEY``
   的项,内容是上一段尾巴的 latent token,RoPE 位置在 origin 之前(负时间);
2. **身份 pin** —— 同时带 ``CTX_FRAME_KEY:0`` 与 ``CTX_NEG_KEY:0``,内容是上一段
   真实导出末像素,RoPE 位置落在 **origin(frame 0)**;
3. **衔接音频 ref** —— ``minimax_refs`` 中带 ``CTX_AUDIO_END_KEY`` 的项,由
   ``conditioning_set_values(..., append=True)`` 追加。

本节点负责清掉它们,让 CONDITIONING 回到「从未注入段间衔接」的状态。典型用法:把
段间衔接的 positive 接到另一次采样上(典型是放大二采)。

**身份 pin 是什么**:pin 是 ``vae.encode(pin_img)``,``pin_img`` 来自
``_pin_last_export_pixel`` —— 上一段**真实导出**(而非 sample overshoot)的最后一
像素;它在 layout 里被 ``_rewrite_keyframe_times`` 钉在 ``origin``(t=0,零 RoPE
距离),就是段间衔接的「frame0 身份锁」(见 ``nodes/r2v_segment.py`` 的
"prev_latent 尾部会替换本段首帧 keyframe,即'上段结尾=本段开头'")。fl2v 时上游
``keep_existing_keyframes`` 已经把 stock 首帧 keyframe 丢弃了,所以这个 pin 是接缝
处 frame 0 的**唯一**锚点。

``pin`` 选项(默认「清除首帧pin」):

* **清除首帧pin**(默认)—— 三样全清。**改变了分辨率的二次采样(如放大二采)必须
  用这个**:pin 的空间网格锁死在**段编码分辨率**,而 ``PackedLayout`` 按**本次采样的
  目标网格**给每个 keyframe 算 cond span 行数、``_cond_video_rows`` 按各 latent 自己
  的网格 patchify,``_forward`` 的 ``all_video_rows[~img_update] = cond_video_rows``
  要求两者严格相等 —— 目标网格一变就必然 shape mismatch。按新网格重编码 pin 需要
  上一段的真实导出末像素(只在采样链上游手上有),本节点给不出降级方案。代价:frame 0
  自此没有任何锚点,且无法从下游恢复。
* **保留首帧pin** —— 只清负时间 blocks 与衔接音频 ref,留着 pin 当 frame 0 锚点。
  只对**同分辨率**的续接(采样1 / 采样2)成立;接到放大二采上仍会因网格不匹配报错。

判据用 ``CTX_NEG_KEY``:衔接注入的两类 keyframe(负时间 block 与 pin)都带它;另一类
Director 正时间锚点(``h3_motion_context`` 的 ``CTX_FRAME_KEY:<px>`` 内部锚点)只有
``CTX_FRAME_KEY``,因此不会被误删。

清掉后 layout / payload patch 的行为(均已核对):

* 保留 pin 时 ``_rewrite_keyframe_times`` 照常被调用(pin 带 ``CTX_FRAME_KEY``),
  ``neg_count`` 归零,pin 按 ``origin`` 放置 —— 与移除前完全一致;
* keyframe 全清时 ``_rewrite_keyframe_times`` 整体跳过;若还剩正时间锚点则照常调用,
  锚点按 ``origin`` 放置;
* ``_rewrite_audio_timeline`` 不再被调用(``has_ctx_audio`` 为 False),因此不会
  触发 "audio continuation requires exactly one marked audio reference" 断言;
* keyframe 清空后 ``_director_extra_conds`` 直接返回,stock payload 自行处理剩余内容;
  r2v / v2v / rv2v 的参考图 ref 不带 ``CTX_AUDIO_END_KEY``,会被保留,合并逻辑照旧。

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

# pin widget:清除 = 连身份 pin 一起清(换分辨率采样用),保留 = 只清负时间块。
PIN_CLEAR = "清除首帧pin"
PIN_KEEP = "保留首帧pin"
PIN_OPTIONS = [PIN_CLEAR, PIN_KEEP]


def _is_negative_context_keyframe(kf) -> bool:
    """纯负时间 block:带 ``CTX_NEG_KEY`` 且不带 ``CTX_FRAME_KEY``。

    身份 pin 两个标记都有(RoPE 在 origin,是 frame 0 的锚点),不算负时间块。
    该判据对稠密(``CTX_NEG_SHIFT_KEY`` 整栈平移)与稀疏
    (``CTX_NEG_BACK_KEY`` 逐块真实距离)两种注入方式同样成立。
    """
    if not isinstance(kf, dict):
        return False
    return kf.get(CTX_NEG_KEY) is not None and CTX_FRAME_KEY not in kf


def _is_identity_pin(kf) -> bool:
    """身份 pin(段间衔接的 frame0 身份锁):两个标记都有。

    正时间的 Director 内部锚点只有 ``CTX_FRAME_KEY``,不会被误判成 pin。
    """
    if not isinstance(kf, dict):
        return False
    return kf.get(CTX_NEG_KEY) is not None and CTX_FRAME_KEY in kf


def _is_join_audio_ref(ref) -> bool:
    """衔接音频 ref:带 ``CTX_AUDIO_END_KEY`` 标记。

    r2v / v2v / rv2v 的参考图与参考视频 ref 由官方节点注入,不带该 Director
    标记,因此不会被误删。
    """
    if not isinstance(ref, dict):
        return False
    return ref.get(CTX_AUDIO_END_KEY) is not None


class MiniMaxH3RemoveNegativeTimeContext:
    """Strip 段间衔接 artifacts (negative-time blocks, optional identity pin, audio ref)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": (
                    "CONDITIONING",
                    {
                        "tooltip": (
                            "接 MiniMaxH3Segment 的 positive。节点会清掉段间衔接注入的"
                            "负时间 keyframe 与衔接音频 ref,保留提示词、参考图与"
                            "Director 正时间锚点;首帧身份 pin 由 pin 选项决定去留。"
                        ),
                    },
                ),
                "pin": (
                    PIN_OPTIONS,
                    {
                        "default": PIN_OPTIONS[0],
                        "tooltip": (
                            "清除首帧pin:连身份 pin 一起清,用于改变分辨率的二次采样"
                            "(放大二采)——pin 编码于段分辨率,换网格必报 shape 错误;"
                            "代价是 frame 0 不再有锚点(上游已丢弃 stock 首帧 keyframe)。"
                            "保留首帧pin:只清负时间块与衔接音频 ref,保住「上段结尾="
                            "本段开头」的 frame0 身份锁,仅适用于同分辨率的续接。"
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
        "清空「段间衔接」(负时间)注入的条件:负时间 keyframe block、衔接音频 ref,"
        "以及(可选)帧0身份 pin;保留提示词、r2v 参考图/视频与正时间锚点。"
        "改变分辨率的二次采样(如放大二采)需选「清除首帧pin」——pin 编码于段分辨率,"
        "换分辨率采样会因网格不一致直接报错。没有段间衔接条件时原样透传。"
    )

    def remove(self, conditioning, pin=PIN_OPTIONS[0]):
        if not conditioning:
            return (conditioning,)

        drop_pin = pin != PIN_KEEP
        removed_kf = 0  # 负时间 context block
        removed_pin = 0  # 身份 pin
        removed_ref = 0
        kept_anchor = 0
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
                kept = []
                n_kf = n_pin = 0
                for kf in keyframes:
                    if _is_negative_context_keyframe(kf):
                        n_kf += 1
                    elif _is_identity_pin(kf):
                        if drop_pin:
                            n_pin += 1
                        else:
                            kept.append(kf)
                    else:
                        kept.append(kf)
                # 没有要删的 keyframe 时完全不动 meta(连列表对象都保持原样),
                # 保证「未注入 → 原样透传」是逐字节的。
                if n_kf or n_pin:
                    removed_kf += n_kf
                    removed_pin += n_pin
                    kept_anchor += sum(
                        1 for kf in kept if isinstance(kf, dict) and CTX_FRAME_KEY in kf
                    )
                    if kept:
                        new_meta["minimax_keyframes"] = kept
                    else:
                        # 全被清空:直接删键,回到「从未注入 keyframes」的状态,避免下游
                        # payload patch 在空列表与 None 之间产生行为分叉。
                        new_meta.pop("minimax_keyframes", None)

            refs = meta.get("minimax_refs")
            if isinstance(refs, (list, tuple)) and refs:
                kept_refs = [r for r in refs if not _is_join_audio_ref(r)]
                if len(kept_refs) != len(refs):
                    removed_ref += len(refs) - len(kept_refs)
                    if kept_refs:
                        new_meta["minimax_refs"] = kept_refs
                    else:
                        # 全是衔接音频 ref:同样删键。
                        new_meta.pop("minimax_refs", None)

            new_item = list(item)
            new_item[1] = new_meta
            out.append(new_item)

        if removed_kf or removed_pin or removed_ref:
            pin_note = (
                "已清除首帧 pin,frame 0 自此无锚点(上游已丢弃 stock 首帧 keyframe)。"
                if removed_pin
                else f"身份 pin 保留(frame0 身份锁),另有 {kept_anchor} 个正时间锚点。"
            )
            log.info(
                "RemoveNegativeTimeContext: 移除 %d 个负时间 block、%d 个身份 pin、"
                "%d 个衔接音频 ref;%s",
                removed_kf,
                removed_pin,
                removed_ref,
                pin_note,
            )
        else:
            log.info(
                "RemoveNegativeTimeContext: 未发现负时间条件,原样透传"
                "(段间衔接未开启或上游未经过 apply_negative_time_context)。"
            )
        return (out,)
