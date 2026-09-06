"""MiniMax H3 专用 latent 空间放大节点(按目标像素宽高)。

以 ComfyUI 官方 LatentUpscaleBy 为基础,对齐逻辑参照
ComfyUI-H3LatentUpscale-jingchen573:

- 输入目标像素 width / height(代替 scale_by 倍率);
- 自动把二采 latent 宽高对齐到偶数,保证像素宽高可被 32 整除;
- 输出尺寸不超过用户指定的 width / height(contain:短边顶格,长边取
  最接近短边实际倍率、且不超过自身目标的合法偶数值);
- 尽可能保持输入宽高比;
- 支持 AV NestedTensor:只放大视频流,音频流原样保留(音频不能做空间
  放大);也支持纯视频 5D latent(如拆分节点输出的视频流)。

放大后用于二次采样时注意:conditioning 不要携带原生分辨率的段间导引
关键帧(关掉 MiniMaxH3Segment 的「段间导引」即可)。分辨率不匹配会让
H3 模型 ``all_video_rows[~img_update] = cond_video_rows`` 出现 token
数错配(``shape mismatch [4082, 96] ...``)。
"""

from __future__ import annotations

import math

import torch

import comfy.utils

from ..director.h3_motion_context import _streams_from_latent

# H3 视频 VAE 的空间压缩率为 16;DiT 使用 2×2 latent patch。
# 因此像素宽高要被 32 整除,latent 宽高就必须被 2 整除。
LATENT_ALIGNMENT = 2
H3_VAE_SPATIAL_DOWNSCALE = 16


def _floor_aligned(value: float) -> int:
    """向下对齐到偶数 latent 单位,且至少保留一个完整对齐单位。"""
    return max(
        LATENT_ALIGNMENT,
        math.floor(value / LATENT_ALIGNMENT) * LATENT_ALIGNMENT,
    )


class MiniMaxH3LatentUpscaleTo:
    """按目标像素宽高安全放大 H3 视频 latent。"""

    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "upscale_method": (cls.upscale_methods,),
                "width": (
                    "INT",
                    {"default": 1280, "min": 32, "max": 8192, "step": 8},
                ),
                "height": (
                    "INT",
                    {"default": 720, "min": 32, "max": 8192, "step": 8},
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "LATENT",
        "width",
        "height",
        "effective_scale_x",
        "effective_scale_y",
        "alignment_info",
    )
    FUNCTION = "upscale"
    CATEGORY = "MinimaxH3_Segment"
    DESCRIPTION = (
        "MiniMax H3 专用 latent 放大,按目标像素 width/height 输出。"
        "自动使二采宽高可被 32 整除,避免 H3 因奇数 latent 轴进行内部 "
        "circular padding;输出不超过目标尺寸,尽可能保持输入宽高比。"
        "AV NestedTensor 输入时只放大视频流,音频原样保留。"
    )

    @classmethod
    def _calculate_aligned_size(
        cls,
        latent_width: int,
        latent_height: int,
        target_width_lat: int,
        target_height_lat: int,
    ):
        """计算不超过目标宽高的 H3 合法 latent 尺寸。

        短边先向下对齐到偶数目标;长边再寻找最接近短边实际倍率的合法
        偶数值,同时不允许长边超过其自身的目标上限。效果等同于 contain。
        """
        if latent_width <= 0 or latent_height <= 0:
            raise ValueError("输入 latent 的宽高必须大于 0")

        if latent_width >= latent_height:
            long_is_width = True
            long_in, short_in = latent_width, latent_height
            long_target, short_target = target_width_lat, target_height_lat
        else:
            long_is_width = False
            long_in, short_in = latent_height, latent_width
            long_target, short_target = target_height_lat, target_width_lat

        short_out = _floor_aligned(short_target)
        short_effective_scale = short_out / short_in
        ideal_long = long_in * short_effective_scale
        long_cap = _floor_aligned(long_target)

        # 比较理想长边两侧的合法偶数值;只保留不超过长边目标上限的值。
        lower = _floor_aligned(ideal_long)
        upper = lower + LATENT_ALIGNMENT
        candidates = {
            candidate
            for candidate in (lower, upper, long_cap)
            if LATENT_ALIGNMENT <= candidate <= long_cap
        }
        long_out = min(
            candidates,
            key=lambda candidate: (abs(candidate - ideal_long), candidate),
        )

        if long_is_width:
            return long_out, short_out
        return short_out, long_out

    @classmethod
    def _replace_video_stream(cls, samples, upscaled: torch.Tensor) -> torch.Tensor:
        """把放大后的视频流写回原 latent 的 samples。

        支持 AV NestedTensor / (tuple, list) / 裸视频张量;非视频流(音频)
        原样保留,只替换第 0 个流。
        """
        if hasattr(samples, "unbind"):
            parts = list(samples.unbind())
            if not parts:
                raise ValueError("AV latent 没有任何流")
            parts[0] = upscaled
            import comfy.nested_tensor

            return comfy.nested_tensor.NestedTensor(tuple(parts))
        if isinstance(samples, (tuple, list)):
            parts = list(samples)
            if not parts:
                raise ValueError("AV latent 没有任何流")
            parts[0] = upscaled
            return tuple(parts)
        return upscaled

    def upscale(self, latent, upscale_method, width, height):
        stream = latent["samples"]
        if hasattr(stream, "unbind") or isinstance(stream, (tuple, list)):
            video = _streams_from_latent(latent)[0]
        else:
            video = stream
        if video.ndim not in (4, 5):
            raise ValueError(
                f"期望视频 latent [B,24,T,H,W] 或 [B,C,H,W],got {tuple(video.shape)}"
            )

        latent_width = int(video.shape[-1])
        latent_height = int(video.shape[-2])
        target_width_lat = max(1, int(width) // H3_VAE_SPATIAL_DOWNSCALE)
        target_height_lat = max(1, int(height) // H3_VAE_SPATIAL_DOWNSCALE)
        output_latent_width, output_latent_height = self._calculate_aligned_size(
            latent_width, latent_height, target_width_lat, target_height_lat
        )

        upscaled = comfy.utils.common_upscale(
            video,
            output_latent_width,
            output_latent_height,
            upscale_method,
            "disabled",
        )

        result = latent.copy()
        result["samples"] = self._replace_video_stream(latent["samples"], upscaled)

        input_pixel_width = latent_width * H3_VAE_SPATIAL_DOWNSCALE
        input_pixel_height = latent_height * H3_VAE_SPATIAL_DOWNSCALE
        output_pixel_width = output_latent_width * H3_VAE_SPATIAL_DOWNSCALE
        output_pixel_height = output_latent_height * H3_VAE_SPATIAL_DOWNSCALE
        effective_scale_x = output_latent_width / latent_width
        effective_scale_y = output_latent_height / latent_height

        alignment_info = (
            f"输入: {input_pixel_width} x {input_pixel_height} | "
            f"目标: {int(width)} x {int(height)} | "
            f"输出: {output_pixel_width} x {output_pixel_height} | "
            f"实际倍率 X/Y: {effective_scale_x:.6f} / {effective_scale_y:.6f} | "
            "32 像素对齐: 是"
        )

        return (
            result,
            output_pixel_width,
            output_pixel_height,
            effective_scale_x,
            effective_scale_y,
            alignment_info,
        )


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3LatentUpscaleTo": MiniMaxH3LatentUpscaleTo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3LatentUpscaleTo": "MiniMax H3 Latent Upscale To",
}
