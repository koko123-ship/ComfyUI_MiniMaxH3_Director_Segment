# ComfyUI MiniMax H3 Director — 外部采样改造版

> 本仓库是 [AIMixer/ComfyUI_MiniMaxH3_Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director) 的**修改版**。原插件由原作者 **AIMixer（AI搅拌手）** 开发，本版在其基础上新增了 **外部自定义采样 / 解码** 支持，并新增若干辅助节点。原版介绍与完整功能请以原作者仓库为准。

**English** → [README_EN.md](README_EN.md)

**感谢原作者：** [AIMixer · ComfyUI_MiniMaxH3_Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director)

## 本版改动简介

原版 `MiniMaxH3Director` 是"单节点全流程导演台"——时间轴、条件编码、采样、解码、导出全在一个节点内。本版把 **t2v / i2v / fl2v / r2v / v2v / rv2v 的"采样→解码→导出"从导演台拆出来**，改为：

- 导演台（`MiniMaxH3Segment`）只做**条件编码**，输出 `positive` + `latent`；
- 采样 / 解码 / 保存用 **ComfyUI 标准节点**在节点外自由搭建（可用自定义采样器、tiled 解码、`SaveLatent` 等）；
- 段间连贯（段间引导）、选择运行、参考素材、本地二采视频仍由时间轴 UI 管理。

## 修改的文件

| 文件 | 内容 |
|---|---|
| `nodes/r2v_segment.py` | **新增**：`MiniMaxH3Segment`（分段条件编码，6 任务）、`MiniMaxH3R2VTrim`（裁头+裁尾对齐）、`MiniMaxH3AudioSelect`（生成/原声/静音）、本地二采视频输出 `frames_0`/`audio_0` |
| `__init__.py` | 注册上述节点（`MiniMaxH3R2VSegment` 保留为旧名别名，旧工作流可加载） |
| `web/js/minimax_timeline.js` | 时间轴编辑器挂到新节点；隐藏导出/声音/最大帧数面板；`twoVideo` 持久化；「本地二采不可用」静态提示 |
| `web/js/minimax_image_batch.js` | t2v/i2v 预览区 → 本地二采上传位；r2v 音频下方加二采行；二采上传/渲染/移除逻辑 |
| `web/js/minimax_fl2v.js` | fl2v 每镜时长下加二采行；shots 序列化/解析保留 `twoVideo` |
| `example_workflows/minimax_h3_r2v_external_demo.json` | **新增**：外部采样示例工作流（for 循环 + Trim + AudioSelect + forLoopEnd 闭合） |

## 新功能与使用方法

### 1. `MiniMaxH3Segment` —— 外部采样条件节点

按时间轴 `index`（loop 计数器）返回第 N 段的条件 `positive` + `latent`，采样/解码全在节点外。

```
[forLoopStart] ──index──▶ MiniMaxH3Segment ──positive──▶ BasicGuider ─▶ SamplerCustomAdvanced ─▶ VAEDecode ─▶ ...
                 └─value1──▶ prev_latent(段间连贯回传)  └─latent──▶            └─▶ VAEDecodeAudio ─▶ CreateVideo
```

- **任务**：t2v / i2v / fl2v / r2v / v2v / rv2v（节点 `task_type` 切换，编辑器自动切对应时间轴）；
- **`trim_frames` / `target_frames`**：段间连贯钉头帧数 / 段目标导出帧数，解码后用 `MiniMaxH3R2VTrim` 裁头+裁尾，导出时长才精确；
- **`source_audio`**：v2v/rv2v 该段源视频原声（接 `MiniMaxH3AudioSelect` 的 source_audio 用「使用原声」）；
- **段间连贯**：接 `prev_latent` 且时间轴开启段间引导时生效（i2v 有源图时自动跳过）；
- **选择运行**：循环紧凑序号自动映射到勾选段（`forLoopStart.total` = 勾选段数）。

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

### 4. 本地二采视频（`frames_0` / `audio_0`）

除 v2v/rv2v 外，每组/镜可上传一个本地二采视频，节点输出其帧 `frames_0`（IMAGE）与音频 `audio_0`（AUDIO），**仅供外部自定义采样使用，不参与内部条件编码**。

- t2v/i2v：每组预览区上传位；fl2v：每镜时长下方；r2v：每组音频下方；
- 未上传时输出占位（1 帧灰图 + 段时长静音），避免下游节点收到 None。

### 5. 导出 UI 精简

新节点隐藏「导出方式 / 声音 / 最大帧数」（导出在外部），保留分辨率与段间引导；段间引导后显示静态「本地二采不可用」提示。

## 安装

与原版相同：放入 `ComfyUI/custom_nodes/`，`pip install -r requirements.txt`，重启 ComfyUI。

- **依赖**：ComfyUI ≥ 0.30.0 + 官方 MiniMax H3 节点；
- 可选 `imageio-ffmpeg`（`source_audio` 原声抽取需要 ffmpeg）。

模型下载与原版一致，见原作者仓库或 [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)。

## 原作者

- **AIMixer（AI搅拌手）**：[github.com/AIMixer/ComfyUI_MiniMaxH3_Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director)
- 原版教程/交流群信息见原作者仓库。

## 致谢

- 原作者 [AIMixer](https://github.com/AIMixer) — 原版 [ComfyUI_MiniMaxH3_Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director)
- [Comfy-Org / ComfyUI](https://github.com/Comfy-Org/ComfyUI) — 官方 MiniMax H3 支持
- [MiniMax-AI](https://github.com/MiniMax-AI) — MiniMax H3 模型
- [NikoDemon80/ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context) — 段间运动/音频续接思路

## License

Apache-2.0
