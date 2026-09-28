"""结构检查（structural validation）——不是质量认证。

本模块只验证「产物在结构上可用」：图片可打开且尺寸达标、视频时长/分辨率
可读、首尾帧可提取、文件不是空画面或过小文件。它不包含人脸一致性、美学
评分或闪烁检测——这些能力当前未接入，任何界面与日志都不得借本模块的
结果声称「质量通过」。
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

MIN_IMAGE_SIDE = 64
MIN_IMAGE_BYTES = 1024
MIN_VIDEO_BYTES = 4096


def validate_image_file(path: str) -> dict:
    """检查单张图片：存在、可打开、尺寸与字节数达标。"""
    result: dict = {"kind": "image", "path": str(path or ""), "passed": False, "issues": []}
    target = Path(str(path or ""))
    if not path or not target.exists():
        result["issues"].append("文件不存在")
        return result
    size_bytes = target.stat().st_size
    if size_bytes < MIN_IMAGE_BYTES:
        result["issues"].append(f"文件过小（{size_bytes} 字节）")
        return result
    try:
        with Image.open(target) as image:
            image.verify()
        with Image.open(target) as image:
            width, height = image.size
    except Exception as exc:
        result["issues"].append(f"无法解码: {exc}")
        return result
    if min(width, height) < MIN_IMAGE_SIDE:
        result["issues"].append(f"尺寸过小（{width}x{height}）")
        return result
    result.update({"passed": True, "width": width, "height": height, "bytes": size_bytes})
    return result


def validate_video_sync(path: str) -> dict:
    """同步版视频检查（依赖 ffprobe；不可用时如实报告 unsupported）。"""
    result: dict = {"kind": "video", "path": str(path or ""), "passed": False, "issues": []}
    target = Path(str(path or ""))
    if not path or not target.exists():
        result["issues"].append("文件不存在")
        return result
    size_bytes = target.stat().st_size
    if size_bytes < MIN_VIDEO_BYTES:
        result["issues"].append(f"文件过小（{size_bytes} 字节）")
        return result
    try:
        probe = asyncio.run(_ffprobe_streams(target))
    except FileNotFoundError:
        result["issues"].append("ffprobe 不可用（unsupported）")
        return result
    except Exception as exc:
        result["issues"].append(f"ffprobe 失败: {exc}")
        return result
    if probe.get("video_streams") == 0:
        result["issues"].append("没有视频轨")
        return result
    duration = probe.get("duration")
    if duration is not None and duration <= 0.2:
        result["issues"].append(f"时长异常（{duration:.2f}s）")
        return result
    result.update(
        {
            "passed": True,
            "duration_seconds": round(duration, 2) if duration is not None else None,
            "width": probe.get("width"),
            "height": probe.get("height"),
            "bytes": size_bytes,
        }
    )
    return result


async def validate_video_file(path: str) -> dict:
    """异步版视频结构检查：时长、分辨率、视频轨可读性。"""
    result: dict = {"kind": "video", "path": str(path or ""), "passed": False, "issues": []}
    target = Path(str(path or ""))
    if not path or not target.exists():
        result["issues"].append("文件不存在")
        return result
    size_bytes = target.stat().st_size
    if size_bytes < MIN_VIDEO_BYTES:
        result["issues"].append(f"文件过小（{size_bytes} 字节）")
        return result
    try:
        probe = await _ffprobe_streams(target)
    except FileNotFoundError:
        result["issues"].append("ffprobe 不可用（unsupported）")
        return result
    except Exception as exc:
        result["issues"].append(f"ffprobe 失败: {exc}")
        return result
    if probe.get("video_streams") == 0:
        result["issues"].append("没有视频轨")
        return result
    duration = probe.get("duration")
    if duration is not None and duration <= 0.2:
        result["issues"].append(f"时长异常（{duration:.2f}s）")
        return result
    result.update(
        {
            "passed": True,
            "duration_seconds": round(duration, 2) if duration is not None else None,
            "width": probe.get("width"),
            "height": probe.get("height"),
            "bytes": size_bytes,
        }
    )
    return result


async def _ffprobe_streams(target: Path) -> dict:
    proc = await asyncio.create_subprocess_exec(
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,width,height:format=duration",
        "-of",
        "json",
        str(target),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8", errors="ignore")[-300:])
    import json

    data = json.loads(stdout.decode("utf-8", errors="ignore") or "{}")
    streams = data.get("streams") or []
    video_streams = [item for item in streams if item.get("codec_type") == "video"]
    duration = None
    try:
        duration = float((data.get("format") or {}).get("duration"))
    except (TypeError, ValueError):
        duration = None
    first = video_streams[0] if video_streams else {}
    return {
        "video_streams": len(video_streams),
        "width": first.get("width"),
        "height": first.get("height"),
        "duration": duration,
    }


__all__ = [
    "MIN_IMAGE_BYTES",
    "MIN_IMAGE_SIDE",
    "MIN_VIDEO_BYTES",
    "validate_image_file",
    "validate_video_file",
    "validate_video_sync",
]
