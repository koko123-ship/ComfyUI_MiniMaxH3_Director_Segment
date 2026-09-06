"""SaveLatent「回填」登记表(本地二采 latent 自动复用)。

段节点素材组里「本地二采 latent」槽(``twoLatent``)在 timeline_data 中存的是
ComfyUI ``input/`` 相对文件名(editor 上传后即此形态,``_load_latent_file`` 也只
认 input/)。而 ``MiniMaxH3SaveLatent`` 存到 ``output/`` —— 手动流程是"存好→
素材组手动上传(拷进 input/)"。

回填 = 省掉手动上传:把本次保存的 ``.av.pt`` 拷进 input/,并把
``{videoFile, fileName, subfolder:"", type:"input"}`` 登记到本表;段节点下次跑
同一 ``(段节点 node id, 循环 index)`` 时,用登记值覆盖该行 ``twoLatent``
(总是覆盖成最新一次保存)。

登记表持久化到 ComfyUI ``output/minimax_h3_backfill.json``,跨 ComfyUI 重启保留。
key 只含段节点 node id + 行号;不同工作流 node id 可能撞号,但回填是"总是覆盖
最新",每次带回填的保存都会重写同 key 条目,自洽可接受。
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import shutil
import threading

import folder_paths

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.backfill")

_lock = threading.Lock()
_cache: dict | None = None
_STORE_NAME = "minimax_h3_backfill.json"
_SAFE_RE = re.compile(r"[^A-Za-z0-9._\-()]+")


def _store_path() -> str:
    return os.path.join(folder_paths.get_output_directory(), _STORE_NAME)


def _load() -> dict:
    global _cache
    if _cache is None:
        try:
            with open(_store_path(), "r", encoding="utf-8") as fh:
                data = json.load(fh)
            _cache = data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            _cache = {}
    return _cache


def _save() -> None:
    path = _store_path()
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(_cache or {}, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except OSError as exc:
        log.warning("MiniMax H3 backfill: 写入登记表失败: %s", exc)


def _key(node_id, index) -> tuple[str, str] | None:
    if not node_id:
        return None
    try:
        return str(node_id), str(int(index))
    except (TypeError, ValueError):
        return None


def set_backfill(node_id, index, two_ref: dict | None) -> None:
    """登记/清除回填。``two_ref=None`` 表示清除该 (node, row) 条目。"""
    key = _key(node_id, index)
    if key is None:
        return
    nid, row = key
    with _lock:
        data = _load()
        node_map = data.setdefault(nid, {})
        if two_ref is None:
            node_map.pop(row, None)
            if not node_map:
                data.pop(nid, None)
        else:
            node_map[row] = dict(two_ref)
        _save()


def clear_backfill(node_id, index) -> None:
    set_backfill(node_id, index, None)


def get_backfill(node_id, index) -> dict | None:
    """取回填登记;无则 None。返回的是 twoLatent 形态的 dict(可直接喂 _load_latent_file)。"""
    key = _key(node_id, index)
    if key is None:
        return None
    nid, row = key
    with _lock:
        entry = _load().get(nid, {}).get(row)
        return dict(entry) if isinstance(entry, dict) and entry else None


def snapshot(node_id=None) -> dict:
    """返回登记表(单节点子表或整表)的深拷贝,供 HTTP/前端读取,与写入无竞态。

    ``node_id`` 给定 → ``{row: two_ref}``;未给 → ``{node_id: {row: two_ref}}``。
    """
    with _lock:
        data = _load()
        if node_id is None:
            return copy.deepcopy(data or {})
        return copy.deepcopy((data or {}).get(str(node_id), {}))


def _safe_basename(name: str) -> str:
    base = os.path.basename(str(name or "").replace("\\", "/"))
    base = _SAFE_RE.sub("_", base).strip("._")
    return base or "backfill.av.pt"


def copy_into_input(src_path: str) -> str | None:
    """把 output/ 的保存文件拷进 ComfyUI input/(同名冲突 ``_N`` 去重)。

    返回 input 相对文件名(subfolder 恒为 '')。
    """
    try:
        src_path = os.path.realpath(src_path)
        if not os.path.isfile(src_path):
            log.warning("MiniMax H3 backfill: 源文件不存在: %s", src_path)
            return None
        name = _safe_basename(os.path.basename(src_path))
        input_dir = folder_paths.get_input_directory()
        out = os.path.join(input_dir, name)
        if os.path.exists(out):
            stem, ext = os.path.splitext(name)
            for n in range(1, 1000):
                cand = f"{stem}_{n}{ext}"
                if not os.path.exists(os.path.join(input_dir, cand)):
                    out = os.path.join(input_dir, cand)
                    name = cand
                    break
        shutil.copy2(src_path, out)
        return name
    except OSError as exc:
        log.warning("MiniMax H3 backfill: 拷贝进 input/ 失败: %s", exc)
        return None


def backfill_from_output_file(node_id, index, src_path: str) -> dict | None:
    """把本次保存文件拷进 input/ 并登记为该 (node, row) 的回填 twoLatent。

    返回登记的两 ref dict;拷贝/登记失败返回 None。
    """
    name = copy_into_input(src_path)
    if not name:
        return None
    two_ref = {
        "videoFile": name,
        "fileName": name,
        "subfolder": "",
        "type": "input",
    }
    set_backfill(node_id, index, two_ref)
    log.info(
        "MiniMax H3 backfill: 段节点 %s 第 %s 段 二采latent ← input/%s",
        node_id,
        index,
        name,
    )
    return two_ref
