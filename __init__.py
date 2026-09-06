"""ComfyUI MiniMax H3 Director — timeline plugin for MiniMax-H3 AV generation.

Based on ComfyUI official MiniMax H3 support (PR #15224 / #15228).
Licensed under the Apache License, Version 2.0. See LICENSE.
"""

from .nodes.conditioning import (
    MiniMaxH3DirectorConditioning,
    MiniMaxH3DirectorPlannerConditioning,
)
from .nodes.director import MiniMaxH3Director
from .nodes.director_groups import (
    MiniMaxH3DirectorGroupImageToVideo,
    MiniMaxH3DirectorGroupReferenceToVideo,
    MiniMaxH3DirectorGroupsCombine,
)
from .nodes.r2v_segment import (
    MiniMaxH3AudioSelect,
    MiniMaxH3R2VSegment,
    MiniMaxH3R2VTrim,
    MiniMaxH3Segment,
)
from .nodes.h3_latent_upscale import MiniMaxH3LatentUpscaleTo
from .nodes.save_latent import MiniMaxH3SaveLatent

NODE_CLASS_MAPPINGS = {
    # 本地改版身份:与上游 ComfyUI_MiniMaxH3_Director 同装时互不覆盖,节点 ID 加 Local 后缀。
    "MiniMaxH3DirectorLocal": MiniMaxH3Director,
    # Legacy type id kept so older workflows still load.
    "ComfyMiniMaxH3DirectorLocal": MiniMaxH3Director,
    "MiniMaxH3DirectorConditioningLocal": MiniMaxH3DirectorConditioning,
    "MiniMaxH3DirectorPlannerConditioningLocal": MiniMaxH3DirectorPlannerConditioning,
    "MiniMaxH3DirectorGroupImageToVideoLocal": MiniMaxH3DirectorGroupImageToVideo,
    "MiniMaxH3DirectorGroupReferenceToVideoLocal": MiniMaxH3DirectorGroupReferenceToVideo,
    # Must stay in NODE_CLASS_MAPPINGS: ComfyUI skips comfy_entrypoint when
    # NODE_CLASS_MAPPINGS is present (if/elif in load_custom_node).
    "MiniMaxH3DirectorGroupsCombineLocal": MiniMaxH3DirectorGroupsCombine,
    "MiniMaxH3Segment": MiniMaxH3Segment,
    # Legacy r2v-only type id kept so older workflows still load.
    "MiniMaxH3R2VSegment": MiniMaxH3R2VSegment,
    "MiniMaxH3R2VTrim": MiniMaxH3R2VTrim,
    "MiniMaxH3AudioSelect": MiniMaxH3AudioSelect,
    "MiniMaxH3SaveLatent": MiniMaxH3SaveLatent,
    "MiniMaxH3LatentUpscaleTo": MiniMaxH3LatentUpscaleTo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3DirectorLocal": "MiniMax H3 Director (Local)",
    "ComfyMiniMaxH3DirectorLocal": "MiniMax H3 Director (Local)",
    "MiniMaxH3DirectorConditioningLocal": "MiniMax H3 Director Conditioning (Local)",
    "MiniMaxH3DirectorPlannerConditioningLocal": "MiniMax H3 Director Planner Conditioning (Local)",
    "MiniMaxH3DirectorGroupImageToVideoLocal": "MiniMax H3 Director Group (Image to Video) (Local)",
    "MiniMaxH3DirectorGroupReferenceToVideoLocal": "MiniMax H3 Director Group (Reference to Video) (Local)",
    "MiniMaxH3DirectorGroupsCombineLocal": "MiniMax H3 Director Groups Combine (Local)",
    "MiniMaxH3Segment": "MiniMax H3 Segment",
    "MiniMaxH3R2VSegment": "MiniMax H3 Segment",
    "MiniMaxH3R2VTrim": "MiniMax H3 R2V Trim",
    "MiniMaxH3AudioSelect": "MiniMax H3 Audio Select",
    "MiniMaxH3SaveLatent": "MiniMax H3 Save Latent",
    "MiniMaxH3LatentUpscaleTo": "MiniMax H3 Latent Upscale To",
}

WEB_DIRECTORY = "./web/js"

import logging

_log = logging.getLogger("ComfyUI-MiniMaxH3-Director")

try:
    from .director.http_routes import register_routes as _register_director_routes

    if not _register_director_routes():
        _log.warning(
            "MiniMax H3 Director HTTP routes deferred (PromptServer not ready). "
            "Restart ComfyUI if /minimax/director_local/* returns 404."
        )
except Exception as _director_routes_exc:
    _log.warning("MiniMax H3 Director HTTP routes failed to load: %s", _director_routes_exc)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
