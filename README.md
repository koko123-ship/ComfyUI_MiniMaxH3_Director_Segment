# ComfyUI MiniMax H3 Director — 外部采样改造版

> 本仓库是 [AIMixer/ComfyUI_MiniMaxH3_Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director) 的**修改版**。原插件由原作者 **AIMixer（AI搅拌手）** 开发，本版在其基础上新增了 **外部自定义采样 / 解码** 支持，并新增若干辅助节点。原版介绍与完整功能请以原作者仓库为准。


**感谢原作者：** [AIMixer · ComfyUI_MiniMaxH3_Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director)

## 本版改动简介

原版 `MiniMaxH3Director` 是"单节点全流程导演台"——时间轴、条件编码、采样、解码、导出全在一个节点内。本版把 **t2v / i2v / fl2v / r2v / v2v / rv2v 的"采样→解码→导出"从导演台拆出来**，改为：

- 导演台（`MiniMaxH3Segment`）只做**条件编码**，输出 `positive` + `latent`；
- 采样 / 解码 / 保存用 **ComfyUI 标准节点**在节点外自由搭建（可用自定义采样器、tiled 解码、`MiniMaxH3SaveLatent` 等）；
- 段间衔接（段间引导）、选择运行、参考素材、本地二采视频仍由时间轴 UI 管理；
- 每个素材组新增「上传段间引导latent」与「本地二采latent」槽，配合 `MiniMaxH3SaveLatent` 的「回填」可自动写回，实现段间 latent 的二采闭环。
- 段间衔接目前支持 15s 上下文记忆。更快的速度选择稀疏，更好的质量使用连续。
稀疏具有全部的空间信息和尾部的运动轨迹及音频，连续具有上段所有信息。

## 节点

### 1. `MiniMaxH3Segment` —— 外部采样条件节点

参考示例工作流，推荐使用采样1可以有效避免画质劣化

### 2. `MiniMaxH3R2VTrim` —— 裁头 + 裁尾

去掉解码后段间连贯钉的头部帧、以及 17k+5 网格对齐溢出，使每段导出时长正确。

```
VAEDecode.images ──▶ MiniMaxH3R2VTrim ──▶ CreateVideo
VAEDecodeAudio.audio ──▶ (音频同步裁)
MiniMaxH3Segment.trim_frames / target_frames ──▶ 接上
```

### 3. `MiniMaxH3AudioSelect` —— 生成 / 原声 / 静音

在 Trim 与 CreateVideo.audio 之间一键切换最终音频。

```
Trim.audio ──▶ MiniMaxH3AudioSelect ──▶ CreateVideo.audio
MiniMaxH3Segment.source_audio ──▶ source_audio（使用原声时接）
mode: generate（默认）/ source / mute
```

### 4. `MiniMax H3 Save Latent` —— 保存latent

保存视频latent，并支持回填到素材组。


### 5. `MiniMaxH3RemoveNegativeTimeContext` —— 移除负时间条件

移除「段间衔接」注入的纯负时间 context blocks 与衔接音频 ref，**保留**首帧身份 pin。未注入时原样透传，可在工作流中常开。


## 安装

放入 `ComfyUI/custom_nodes/`，`pip install -r requirements.txt`，重启 ComfyUI。

- **依赖**：ComfyUI ≥ 0.33.3 + 官方 MiniMax H3 节点；
- 可选 `imageio-ffmpeg`（`source_audio` 原声抽取需要 ffmpeg）。

模型下载与原版一致，见原作者仓库或 [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)。

## 原作者

- **AIMixer（AI搅拌手）**：[github.com/AIMixer/ComfyUI_MiniMaxH3_Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director)
- 原版教程/交流群信息见原作者仓库。

## 致谢

- 原作者 [AIMixer](https://github.com/AIMixer) — 原版 [ComfyUI_MiniMaxH3_Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director)
- [Comfy-Org / ComfyUI](https://github.com/Comfy-Org/ComfyUI) — 官方 MiniMax H3 支持
- [MiniMax-AI](https://github.com/MiniMax-AI) — MiniMax H3 模型
- [kat3ri/ComfyUI-MiniMax-H3-Extend](https://github.com/kat3ri/ComfyUI-MiniMax-H3-Extend) — 段间运动/音频续接思路


## License

Apache-2.0
