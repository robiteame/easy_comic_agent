"""结构检查（structural validation）——不是质量认证。

本模块只验证「产物在结构/技术上可用」：图片可打开且尺寸达标、视频可播放、
时长与执行计划匹配、分辨率/比例合理、没有大段黑帧/空帧或冻结、音频不超过
画面、尾帧可提取。它不包含人脸/身份一致性、场景识别、动作识别或美学评分——
这些能力当前未接入，任何界面与日志都不得借本模块的结果声称「视觉质量通过」
或「角色一致性通过」。

视频检查结果按三个维度返回：

1. ``structural_validity`` —— 文件可读、可解码、有视频轨、时长 sane；
2. ``technical_quality`` —— 时长/分辨率/比例/黑帧/空帧/冻结/音画时长/尾帧/文件过小；
3. ``visual_quality_pending`` —— 视觉质量待审：五项视觉维度恒为 pending，
   永不给出通过或失败结论（未接入视觉模型时不得伪造为 passed）。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import re
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

MIN_IMAGE_SIDE = 64
MIN_IMAGE_BYTES = 1024
MIN_VIDEO_BYTES = 4096

# 三类检查的稳定键名；critic/契约/前端都使用同一组词。
CATEGORY_STRUCTURAL = "structural_validity"
CATEGORY_TECHNICAL = "technical_quality"
CATEGORY_VISUAL_PENDING = "visual_quality_pending"

# 技术检查阈值（刻意保守：只拦截确定性缺陷，不做未经校准的视觉评分硬阻断）。
MIN_VIDEO_SIDE = 240
DEFAULT_DURATION_TOLERANCE_S = 0.5
ASPECT_WARNING_DEVIATION = 0.02
ASPECT_ERROR_DEVIATION = 0.15
BLACK_ERROR_INTERVAL_S = 1.0
BLACK_ERROR_RATIO = 0.3
FREEZE_DETECT_MIN_S = 1.0
FREEZE_ERROR_INTERVAL_S = 2.0
FREEZE_ERROR_RATIO = 0.5
AUDIO_OVER_PICTURE_TOLERANCE_S = 0.5
# 空帧（blank/empty frame）：整帧亮度几乎无变化（纯黑、纯白、纯灰、纯色底）。
# 只拦截「确定性空帧」——signalstats 的 YMAX-YMIN 落在极窄范围内才算，
# 正常画面（即使很暗或很亮）动态范围都远大于该阈值，不会误报。
BLANK_LUMA_RANGE = 12
BLANK_ERROR_INTERVAL_S = 1.0
BLANK_ERROR_RATIO = 0.3

# ffmpeg 7+ 把检测日志改为 lavfi.<filter>.<marker> 元数据格式，旧版是同行
# black_start/black_end；两种都兼容，freeze 到 EOF 时可能只有 start 无 duration。
_DETECTOR_MARKER_RE = re.compile(r"(black|freeze)_(start|end|duration)\s*:\s*(\d+(?:\.\d+)?)")

# 视觉质量待审的固定说明：未接入识别模型前不评估、不宣称通过。
VISUAL_PENDING_REASON = (
    "未接入身份识别、场景识别或动作识别模型；本检查不评估角色一致性、"
    "构图、美学或连续性，视觉质量保持待审状态"
)

# 视觉质量的五项稳定维度。它们**永远**是 pending：没有真实视觉模型时既不能
# 判通过也不能判失败，只能如实标为待审，并把「未验证」写进最终报告的未解决
# 风险里。键名与 label 是跨 critic/契约/前端复用的稳定词，不得随意改名。
VISUAL_DIMENSIONS: tuple[dict[str, str], ...] = (
    {
        "key": "first_frame_storyboard_similarity",
        "label": "首帧与故事板相似度",
        "reason": "未接入图像相似度/视觉模型，无法比对首帧与定稿故事板",
    },
    {
        "key": "reference_match",
        "label": "角色和场景参考匹配度",
        "reason": "未接入身份 embedding 与场景识别模型，无法比对角色/场景参考图",
    },
    {
        "key": "motion_stability",
        "label": "运动稳定度",
        "reason": "未接入运动/光流分析模型，无法评估抖动、闪烁与主体变形",
    },
    {
        "key": "shot_continuity",
        "label": "镜头连续性",
        "reason": "未接入跨镜头视觉比对能力，无法评估与上一镜头的衔接跳变",
    },
    {
        "key": "action_completion",
        "label": "动作完成度",
        "reason": "未接入动作识别模型，无法判断 character_action 是否完整呈现",
    },
)

VISUAL_DIMENSION_KEYS: tuple[str, ...] = tuple(item["key"] for item in VISUAL_DIMENSIONS)


def visual_pending_dimensions() -> list[dict[str, Any]]:
    """五项视觉维度的 pending 快照（每次返回新对象，调用方可安全改写）。"""

    return [
        {
            "key": item["key"],
            "label": item["label"],
            "status": "pending",
            "passed": None,
            "reason": item["reason"],
        }
        for item in VISUAL_DIMENSIONS
    ]


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


def _issue(code: str, message: str, recommendation: str) -> dict[str, str]:
    return {"code": code, "message": message, "recommendation": recommendation}


def _add_warning(result: dict, code: str, message: str, recommendation: str) -> None:
    """非阻断观察项：黑边余量、轻微比例偏差、扫描未完成等，只提示不失败。"""

    result.setdefault("warnings", []).append(_issue(code, message, recommendation))


def _empty_categories() -> dict[str, Any]:
    return {
        CATEGORY_STRUCTURAL: {"passed": False, "issues": []},
        CATEGORY_TECHNICAL: {"passed": None, "issues": [], "skipped": []},
        CATEGORY_VISUAL_PENDING: {
            "status": "pending",
            "passed": None,
            "reason": VISUAL_PENDING_REASON,
            # 五项视觉维度逐项 pending：未接入视觉模型时不评估、不宣称通过。
            "dimensions": visual_pending_dimensions(),
            "issues": [_issue("visual_quality_pending", "视觉质量未评估（pending）", "接入视觉审核模型或转人工审核后再判断视觉质量")],
        },
    }


def _force_visual_pending(category: dict[str, Any]) -> dict[str, Any]:
    """视觉维度不可被任何调用方改写为 passed：这里做最后一次强制归位。"""

    category["status"] = "pending"
    category["passed"] = None
    dimensions = [
        {**item, "status": "pending", "passed": None}
        for item in (category.get("dimensions") or visual_pending_dimensions())
        if isinstance(item, dict)
    ]
    category["dimensions"] = dimensions or visual_pending_dimensions()
    return category


def _finalize_video_result(result: dict, categories: dict[str, Any] | None = None) -> dict:
    """汇总三分类结果；总体 passed 只代表结构与技术可用，不代表视觉质量。

    ``None`` 表示技术维度仍有未检测项，不能被当作通过；调用方可以据此把
    产物留在 pending，而不是把检查能力缺失误报成质量通过。
    """

    merged = categories or result.get("categories") or _empty_categories()
    structural = bool(merged[CATEGORY_STRUCTURAL]["passed"])
    technical = merged[CATEGORY_TECHNICAL]["passed"]
    merged[CATEGORY_STRUCTURAL]["passed"] = structural
    # 视觉质量恒为 pending：即使调用方手工塞入 passed=True 也在这里被纠正。
    merged[CATEGORY_VISUAL_PENDING] = _force_visual_pending(merged[CATEGORY_VISUAL_PENDING])
    result["categories"] = merged
    result["passed"] = structural and technical is True
    result["visual_quality_pending"] = True
    result["visual_quality_dimensions"] = [item["key"] for item in merged[CATEGORY_VISUAL_PENDING]["dimensions"]]
    return result


def validate_video_sync(path: str, **options: Any) -> dict:
    """同步版视频检查（依赖 ffprobe/ffmpeg；不可用时如实报告 unsupported）。

    在事件循环内被同步调用（例如 LangGraph 节点里的 critic）时，放到独立
    线程的新事件循环执行，避免 ``asyncio.run`` 嵌套崩溃。
    """

    return _run_coro_sync(validate_video_file(path, **options))


def _run_coro_sync(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


async def validate_video_file(
    path: str,
    *,
    expected_duration_s: float | None = None,
    expected_aspect_ratio: float | None = None,
    audio_duration_s: float | None = None,
    first_frame_path: str | None = None,
    tail_frame_path: str | None = None,
    expect_audio: bool | None = None,
    deep_scan: bool = True,
) -> dict:
    """三分类视频检查。

    只读 ``path`` 一个必选参数时等价于旧的结构检查（外加可解码性验证）。
    可选参数携带执行计划上下文（计划时长、目标比例、外部 TTS 音频时长、首尾帧
    路径、是否期望内嵌音轨）；不提供的维度记为 skipped，不假装通过也不误报失败。

    ``expect_audio`` 为 ``None`` 时音轨存在性记为 skipped：外部 TTS 模式下配音
    在合成阶段混入，单镜头视频没有音轨是正常的，不能当成缺陷。
    """
    result: dict = {
        "kind": "video",
        "path": str(path or ""),
        "passed": False,
        "issues": [],
        "categories": _empty_categories(),
        "checks": {"deep_scan": bool(deep_scan)},
    }
    categories = result["categories"]
    structural_issues: list[dict[str, str]] = categories[CATEGORY_STRUCTURAL]["issues"]
    technical: dict[str, Any] = categories[CATEGORY_TECHNICAL]

    target = Path(str(path or ""))
    if not path or not target.exists():
        structural_issues.append(_issue("video_file_missing", "视频文件不存在", "重新生成该镜头视频或恢复存储中的产物"))
        result["issues"].append("文件不存在")
        return _finalize_video_result(result)
    size_bytes = target.stat().st_size
    if size_bytes < MIN_VIDEO_BYTES:
        structural_issues.append(_issue("video_file_too_small", f"文件过小（{size_bytes} 字节）", "重新生成该镜头视频；疑似 Provider 返回了占位内容"))
        # 同一事实在技术维度也如实记录：过小是输出质量的确定性缺陷，不只是可读性。
        technical["issues"].append(
            _issue(
                "video_output_too_small",
                f"输出文件过小（{size_bytes} 字节，低于 {MIN_VIDEO_BYTES} 字节）",
                "重新生成该镜头视频；持续过小时切换 Provider 或提高分辨率/时长档位",
            )
        )
        technical["checks"] = {**technical.get("checks", {}), "min_bytes": False}
        result["issues"].append(f"文件过小（{size_bytes} 字节）")
        return _finalize_video_result(result)
    try:
        probe = await _ffprobe_streams(target)
    except FileNotFoundError:
        structural_issues.append(_issue("ffprobe_unavailable", "ffprobe 不可用（unsupported）", "安装 ffmpeg/ffprobe 后重跑检查；渲染链路本身也依赖它"))
        result["issues"].append("ffprobe 不可用（unsupported）")
        return _finalize_video_result(result)
    except Exception as exc:
        structural_issues.append(_issue("video_unreadable", f"ffprobe 无法读取: {exc}", "重新生成该镜头视频；文件可能损坏或格式不受支持"))
        result["issues"].append(f"ffprobe 失败: {exc}")
        return _finalize_video_result(result)
    if probe.get("video_streams") == 0:
        structural_issues.append(_issue("video_stream_missing", "没有视频轨", "重新生成该镜头视频；检查 Provider 是否返回了纯音频/空内容"))
        result["issues"].append("没有视频轨")
        return _finalize_video_result(result)
    duration = probe.get("duration")
    if duration is not None and duration <= 0.2:
        structural_issues.append(_issue("video_duration_invalid", f"时长异常（{duration:.2f}s）", "重新生成该镜头视频；疑似截断产物"))
        result["issues"].append(f"时长异常（{duration:.2f}s）")
        return _finalize_video_result(result)

    result.update(
        {
            "duration_seconds": round(duration, 2) if duration is not None else None,
            "video_duration_seconds": round(probe["video_duration"], 2) if probe.get("video_duration") is not None else None,
            "width": probe.get("width"),
            "height": probe.get("height"),
            "bytes": size_bytes,
            "fps": probe.get("fps"),
            "audio": {
                "streams": int(probe.get("audio_streams") or 0),
                "duration_seconds": round(probe["audio_duration"], 2) if probe.get("audio_duration") is not None else None,
            },
        }
    )

    # --- 可播放性 + 黑帧/冻结（一次 ffmpeg 解码完成两项检查） ---
    picture_duration = probe.get("video_duration") or duration
    if deep_scan and duration:
        try:
            scan = await _ffmpeg_frame_scan(target, total_duration=picture_duration)
        except FileNotFoundError:
            # 扫描工具缺失不算产物缺陷：记为 warning，不阻断候选。
            technical["skipped"].append("frame_scan")
            technical["unknown"] = True
            _add_warning(result, "frame_scan_unavailable", "ffmpeg 不可用，跳过黑帧/冻结扫描（unsupported）", "安装 ffmpeg 后重跑；本项未检测不代表通过")
            result["issues"].append("ffmpeg 不可用，跳过黑帧/冻结扫描（unsupported）")
            scan = None
        except Exception as exc:
            technical["skipped"].append("frame_scan")
            _add_warning(result, "frame_scan_inconclusive", f"黑帧/冻结扫描未完成: {exc}", "重跑检查；连续失败时人工抽查该镜头画面")
            result["issues"].append(f"黑帧/冻结扫描未完成: {exc}")
            scan = None
        if scan is not None:
            result["frame_scan"] = scan
            if scan.get("returncode") != 0:
                structural_issues.append(
                    _issue("video_not_playable", "视频无法完整解码播放", "重新生成该镜头视频；本地重mux 无效时切换 Provider")
                )
                result["issues"].append("视频无法完整解码播放")
            else:
                result["checks"]["decoded_ok"] = True
                # 解码器已成功读取视频开头；这是真实首帧可解码证据，即使调用方
                # 没有另存首帧文件，也不能把首帧检查误报成 skipped。
                result["first_frame_ok"] = True
                if "first_frame" in technical["skipped"]:
                    technical["skipped"].remove("first_frame")
                _assess_frame_scan(result, technical, picture_duration or duration)
    else:
        technical["skipped"].append("frame_scan")

    # --- 音视频轨存在性 ---
    # 视频轨缺失在上面的 ``video_stream_missing`` 已拦截；音轨是否「应该有」
    # 取决于音频路由（外部 TTS 在合成阶段混音，单镜头视频无音轨是正常的），
    # 因此只有调用方显式声明 expect_audio 时才判定，否则记为 skipped。
    audio_streams = int(probe.get("audio_streams") or 0)
    if expect_audio is True and audio_streams == 0:
        structural_issues.append(
            _issue(
                "audio_stream_missing",
                "期望内嵌音轨但视频没有音频轨",
                "重新生成该镜头视频（native audio），或确认音频路由为外部 TTS 并在合成阶段混音",
            )
        )
        result["checks"]["audio_track"] = False
    elif expect_audio is None:
        technical["skipped"].append("audio_track")
    else:
        result["checks"]["audio_track"] = True

    # --- 时长与执行计划匹配 ---
    if expected_duration_s is not None and duration is not None and expected_duration_s > 0:
        picture_duration = probe.get("video_duration") or duration
        tolerance = max(DEFAULT_DURATION_TOLERANCE_S, float(expected_duration_s) * 0.1)
        if picture_duration + tolerance < float(expected_duration_s):
            technical["issues"].append(
                _issue(
                    "video_duration_shorter_than_plan",
                    f"实际时长 {picture_duration:.2f}s 短于执行计划 {float(expected_duration_s):.2f}s（容差 {tolerance:.2f}s）",
                    "重新生成该镜头，或按实际时长收紧执行计划的裁剪区间再合成",
                )
            )
            technical["checks"] = {**technical.get("checks", {}), "duration_match": False}
        else:
            technical["checks"] = {**technical.get("checks", {}), "duration_match": True}
    else:
        technical["skipped"].append("duration_match")

    # --- 分辨率与画幅比例 ---
    width, height = probe.get("width"), probe.get("height")
    if width and height:
        if min(int(width), int(height)) < MIN_VIDEO_SIDE:
            technical["issues"].append(
                _issue(
                    "video_resolution_below_minimum",
                    f"分辨率过低（{width}x{height}，最低边 < {MIN_VIDEO_SIDE}px）",
                    "以更高分辨率重生成该镜头，或检查 Provider 分辨率档位设置",
                )
            )
        if expected_aspect_ratio and float(expected_aspect_ratio) > 0:
            actual_ratio = float(width) / float(height)
            deviation = abs(actual_ratio - float(expected_aspect_ratio)) / float(expected_aspect_ratio)
            if deviation > ASPECT_ERROR_DEVIATION:
                technical["issues"].append(
                    _issue(
                        "video_aspect_mismatch",
                        f"画幅比例偏差过大（实际 {actual_ratio:.3f}，预期 {float(expected_aspect_ratio):.3f}）",
                        "检查目标画幅与 Provider 分辨率配置后重生成，避免成片出现大面积裁切/黑边",
                    )
                )
            elif deviation > ASPECT_WARNING_DEVIATION:
                # 轻微比例偏差不阻断：合成阶段会适配，只提示。
                _add_warning(
                    result,
                    "video_aspect_slight_mismatch",
                    f"画幅比例轻微偏差（实际 {actual_ratio:.3f}，预期 {float(expected_aspect_ratio):.3f}）",
                    "合成会做适配；若不可接受请按目标比例重生成",
                )
    else:
        technical["skipped"].append("resolution")

    # --- 音频与画面时长偏差（内嵌轨与外部 TTS 都检查两个方向） ---
    if picture_duration is not None and picture_duration > 0:
        muxed_audio = probe.get("audio_duration")
        if muxed_audio is not None and int(probe.get("audio_streams") or 0) > 0:
            audio_delta = float(muxed_audio) - float(picture_duration)
            if audio_delta > AUDIO_OVER_PICTURE_TOLERANCE_S:
                technical["issues"].append(
                    _issue(
                        "audio_exceeds_picture",
                        f"内嵌音轨比画面长 {audio_delta:.2f}s（音轨 {float(muxed_audio):.2f}s / 画面 {picture_duration:.2f}s）",
                        "按画面裁剪音轨尾部，或延长视频时长使对白完整呈现",
                    )
                )
            elif audio_delta < -AUDIO_OVER_PICTURE_TOLERANCE_S:
                technical["issues"].append(
                    _issue(
                        "audio_shorter_than_picture",
                        f"内嵌音轨比画面短 {abs(audio_delta):.2f}s（音轨 {float(muxed_audio):.2f}s / 画面 {picture_duration:.2f}s）",
                        "补齐或延长配音，或按实际音频时长收紧画面，避免镜头后段无声",
                    )
                )
        if audio_duration_s is not None and float(audio_duration_s) > 0:
            external_delta = float(audio_duration_s) - float(picture_duration)
            if external_delta > AUDIO_OVER_PICTURE_TOLERANCE_S:
                technical["issues"].append(
                    _issue(
                        "audio_exceeds_picture",
                        f"配音 {float(audio_duration_s):.2f}s 超过画面 {picture_duration:.2f}s（超出 {external_delta:.2f}s）",
                        "拆分镜头、延长视频生成时长，或裁剪配音尾部并复核台词完整性",
                    )
                )
            elif external_delta < -AUDIO_OVER_PICTURE_TOLERANCE_S:
                technical["issues"].append(
                    _issue(
                        "audio_shorter_than_picture",
                        f"配音 {float(audio_duration_s):.2f}s 短于画面 {picture_duration:.2f}s（短 {abs(external_delta):.2f}s）",
                        "补齐或延长配音，或按实际音频时长收紧画面，避免镜头后段无声",
                    )
                )
    else:
        technical["skipped"].append("audio_duration")

    # --- 首帧/尾帧是否可用（下一阶段连续性参考依赖它们） ---
    if first_frame_path is not None:
        frame_check = validate_image_file(str(first_frame_path))
        if not frame_check.get("passed"):
            technical["issues"].append(
                _issue(
                    "first_frame_missing",
                    f"首帧提取失败或不可用（{first_frame_path}）",
                    "用 ffmpeg 从视频开头重新提取首帧，并确认故事板首帧文件可读",
                )
            )
        else:
            result["first_frame_ok"] = True
    else:
        # 未要求外部首帧文件时，视频解码成功已经验证了首帧可读性；只有
        # 扫描被跳过/不可用时才保留 skipped，避免把可播放视频误报为未知。
        if "frame_scan" in technical["skipped"]:
            technical["skipped"].append("first_frame")

    if tail_frame_path is not None:
        frame_check = validate_image_file(str(tail_frame_path))
        if not frame_check.get("passed"):
            technical["issues"].append(
                _issue(
                    "tail_frame_missing",
                    f"尾帧提取失败或不可用（{tail_frame_path}）",
                    "用 ffmpeg 从视频末尾重新提取尾帧（本地操作，无需重新生成视频）",
                )
            )
        else:
            result["tail_frame_ok"] = True
    else:
        technical["skipped"].append("tail_frame")

    categories[CATEGORY_STRUCTURAL]["passed"] = not structural_issues
    contextual_checks = any(
        value is not None
        for value in (expected_duration_s, expected_aspect_ratio, audio_duration_s, first_frame_path, tail_frame_path)
    ) or not deep_scan
    required_skips = {
        item
        for item in technical.get("skipped", [])
        if item in {"frame_scan", "blank_scan"}
        or (item == "duration_match" and expected_duration_s is not None)
        or (item == "resolution" and bool(width is None or height is None))
        or (item == "audio_duration" and audio_duration_s is not None)
        or (item == "first_frame" and first_frame_path is not None)
        or (item == "tail_frame" and tail_frame_path is not None)
    }
    technical["passed"] = False if technical["issues"] else (None if required_skips else True)
    result["issues"].extend(f"{item['code']}: {item['message']}" for item in technical["issues"])
    return _finalize_video_result(result)


def _assess_frame_scan(result: dict, technical: dict, duration: float) -> None:
    """把 blackdetect/freezedetect/signalstats 结果转成技术问题；只报确定性缺陷。"""

    scan = result.get("frame_scan") or {}
    blacks = list(scan.get("black_intervals") or [])
    freezes = list(scan.get("freeze_intervals") or [])
    blanks = list(scan.get("blank_intervals") or [])
    total_black = sum(max(0.0, float(item.get("end", 0)) - float(item.get("start", 0))) for item in blacks)
    black_ratio = total_black / duration if duration > 0 else 0.0
    worst_black = max((float(item.get("end", 0)) - float(item.get("start", 0)) for item in blacks), default=0.0)
    if worst_black >= BLACK_ERROR_INTERVAL_S or black_ratio >= BLACK_ERROR_RATIO:
        technical["issues"].append(
            _issue(
                "video_black_frames",
                f"检测到大段黑帧（最长 {worst_black:.2f}s，占比 {black_ratio:.0%}）",
                "重新生成该镜头视频；若 Provider 持续输出黑帧则切换 Provider",
            )
        )
    elif total_black > 0:
        _add_warning(result, "video_black_frames_partial", f"存在少量黑帧（共 {total_black:.2f}s）", "常见于转场余量；若影响观感再重生成")
    # 空帧：整帧无内容（纯黑/纯白/纯灰/纯色底）。黑帧已由 blackdetect 覆盖，
    # 这里扣除与黑帧重叠的时长，避免同一缺陷重复计数。
    blank_only = [
        item
        for item in blanks
        if not any(
            float(item.get("start", 0)) < float(black.get("end", 0)) and float(black.get("start", 0)) < float(item.get("end", 0))
            for black in blacks
        )
    ]
    worst_blank = max((float(item.get("duration", 0)) for item in blank_only), default=0.0)
    total_blank = sum(float(item.get("duration", 0)) for item in blank_only)
    blank_ratio = total_blank / duration if duration > 0 else 0.0
    if worst_blank >= BLANK_ERROR_INTERVAL_S or blank_ratio >= BLANK_ERROR_RATIO:
        technical["issues"].append(
            _issue(
                "video_empty_frames",
                f"检测到长时间空帧（最长 {worst_blank:.2f}s，占比 {blank_ratio:.0%}）：整帧无画面内容",
                "重新生成该镜头视频；持续输出空帧说明 Provider 返回了未渲染内容，切换 Provider 或降低复杂度",
            )
        )
    elif total_blank > 0:
        _add_warning(result, "video_empty_frames_partial", f"存在少量空帧（共 {total_blank:.2f}s）", "低于阻断阈值；频繁出现时检查 Provider 稳定性")
    if scan.get("blank_scan_skipped"):
        technical["skipped"].append("blank_scan")
        _add_warning(result, "blank_scan_unavailable", "空帧检测未执行（临时文件不可用）", "本项未检测不代表通过；修复临时目录后重跑检查")
    result["frame_scan"]["blank_seconds"] = round(total_blank, 2)
    worst_freeze = max((float(item.get("duration", 0)) for item in freezes), default=0.0)
    total_freeze = sum(float(item.get("duration", 0)) for item in freezes)
    freeze_ratio = total_freeze / duration if duration > 0 else 0.0
    if worst_freeze >= FREEZE_ERROR_INTERVAL_S or freeze_ratio >= FREEZE_ERROR_RATIO:
        technical["issues"].append(
            _issue(
                "video_frozen",
                f"检测到长时间冻结画面（最长 {worst_freeze:.2f}s，占比 {freeze_ratio:.0%}）",
                "重新生成该镜头并在 Prompt 中要求持续动作；频繁冻结时切换 Provider",
            )
        )
    elif worst_freeze >= FREEZE_DETECT_MIN_S:
        _add_warning(result, "video_freeze_partial", f"存在短暂静止画面（最长 {worst_freeze:.2f}s）", "低于阻断阈值；频繁出现时再调整 Prompt 要求持续动作")
    result["frame_scan"]["black_seconds"] = round(total_black, 2)
    result["frame_scan"]["freeze_seconds"] = round(total_freeze, 2)


async def probe_media_duration(path: str) -> float | None:
    """读取任意媒体的容器时长（秒）；用于配音等外部音频的时长检查。"""

    target = Path(str(path or ""))
    if not path or not target.exists():
        return None
    try:
        probe = await _ffprobe_streams(target)
    except Exception:
        return None
    duration = probe.get("duration")
    if duration is None:
        duration = probe.get("audio_duration")
    return round(float(duration), 3) if duration is not None else None


def probe_media_duration_sync(path: str) -> float | None:
    return _run_coro_sync(probe_media_duration(path))


async def _ffprobe_streams(target: Path) -> dict:
    proc = await asyncio.create_subprocess_exec(
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,width,height,duration,avg_frame_rate:format=duration",
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
    audio_streams = [item for item in streams if item.get("codec_type") == "audio"]

    def _stream_duration(item: dict) -> float | None:
        try:
            value = float(item.get("duration"))
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    duration = None
    try:
        duration = float((data.get("format") or {}).get("duration"))
    except (TypeError, ValueError):
        duration = None
    audio_duration = next((_stream_duration(item) for item in audio_streams if _stream_duration(item) is not None), None)
    if duration is None or duration <= 0:
        # 容器级时长缺失时退回视频流时长，避免误报 unsupported。
        duration = next((_stream_duration(item) for item in video_streams if _stream_duration(item) is not None), None)
    fps = None
    first = video_streams[0] if video_streams else {}
    rate = str(first.get("avg_frame_rate") or "")
    if "/" in rate:
        numerator, _, denominator = rate.partition("/")
        try:
            fps = round(float(numerator) / float(denominator), 3) if float(denominator) else None
        except (TypeError, ValueError):
            fps = None
    return {
        "video_streams": len(video_streams),
        "audio_streams": len(audio_streams),
        "width": first.get("width"),
        "height": first.get("height"),
        "fps": fps,
        "duration": duration if duration and duration > 0 else None,
        # 画面可用时长以视频流为准：容器时长会被更长的音轨撑大。
        "video_duration": next((_stream_duration(item) for item in video_streams if _stream_duration(item) is not None), None),
        "audio_duration": audio_duration,
    }


async def _ffmpeg_frame_scan(target: Path, *, total_duration: float | None = None) -> dict:
    """一次解码同时完成：可播放性验证、blackdetect、freezedetect 与空帧检测。

    空帧（blank/empty）用 ``signalstats`` 的逐帧亮度动态范围判定：``YMAX-YMIN``
    极小的帧说明整帧几乎没有内容（纯黑/纯白/纯灰/纯色底），与 blackdetect 只认
    黑不同。signalstats 结果经 ``metadata=print`` 写入临时文件，避免与 null
    muxer 的 stdout 互相干扰；临时文件不可用时只跳过空帧检测，不影响其余检查。
    """

    metadata_path: str | None = None
    handle = None
    filters = f"blackdetect=d=0.2:pix_th=0.05,freezedetect=n=0.001:d={FREEZE_DETECT_MIN_S},signalstats"
    try:
        handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 - 需要路径交给 ffmpeg
            mode="w", suffix=".txt", prefix="frame-stats-", delete=False, encoding="utf-8"
        )
        metadata_path = handle.name
        filters = f"{filters},metadata=print:file={metadata_path}"
    except OSError as exc:  # 临时目录不可写：降级为不含空帧检测的扫描
        logger.warning("空帧检测不可用（无法创建临时文件）: %s", exc)
        handle = None
        metadata_path = None
    finally:
        if handle is not None:
            with suppress(OSError):
                handle.close()

    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-nostdin",
            "-v",
            "info",
            "-i",
            str(target),
            "-vf",
            filters,
            "-an",
            "-f",
            "null",
            "-",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
    finally:
        stats_text = ""
        if metadata_path:
            with suppress(OSError):
                stats_text = Path(metadata_path).read_text(encoding="utf-8", errors="ignore")
            with suppress(OSError):
                Path(metadata_path).unlink()
    text = stderr.decode("utf-8", errors="ignore")
    black_intervals, freeze_intervals = _parse_detector_markers(text, total_duration)
    blank_intervals = _parse_blank_intervals(stats_text, total_duration)
    result = {
        "returncode": proc.returncode,
        "black_intervals": black_intervals,
        "freeze_intervals": freeze_intervals,
        "blank_intervals": blank_intervals,
        "error_tail": text[-300:],
    }
    if metadata_path is None:
        result["blank_scan_skipped"] = True
    return result


def _parse_blank_intervals(text: str, total_duration: float | None) -> list[dict[str, float]]:
    """把 signalstats 逐帧元数据转成连续空帧区间。

    ``frame:`` 行给出 ``pts_time``，紧随其后的 ``lavfi.signalstats.YMIN/YMAX``
    给出该帧亮度范围。``YMAX-YMIN <= BLANK_LUMA_RANGE`` 视为空帧；连续空帧合并
    为一个区间。缺任一键的帧按「非空」处理，宁可漏报不误报。
    """

    if not text:
        return []
    intervals: list[dict[str, float]] = []
    open_start: float | None = None
    last_ts: float | None = None
    frame_re = re.compile(r"^frame:.*?pts_time:(\d+(?:\.\d+)?)", re.M)
    key_re = re.compile(r"^lavfi\.signalstats\.(YMIN|YMAX)=(\d+(?:\.\d+)?)", re.M)
    blocks = re.split(r"(?=^frame:)", text, flags=re.M)
    for block in blocks:
        frame_match = frame_re.search(block)
        if not frame_match:
            continue
        timestamp = float(frame_match.group(1))
        values = {key: float(value) for key, value in key_re.findall(block)}
        if "YMIN" not in values or "YMAX" not in values:
            is_blank = False
        else:
            is_blank = (values["YMAX"] - values["YMIN"]) <= BLANK_LUMA_RANGE
        if is_blank:
            if open_start is None:
                open_start = timestamp
            last_ts = timestamp
        elif open_start is not None:
            intervals.append({"start": open_start, "end": last_ts if last_ts is not None else open_start})
            open_start = None
            last_ts = None
    if open_start is not None:
        end = last_ts if last_ts is not None else open_start
        # 空帧持续到文件末尾时用总时长闭合；无法得知时保持最后一帧时间戳。
        if total_duration and float(total_duration) > end:
            end = float(total_duration)
        intervals.append({"start": open_start, "end": end})
    for item in intervals:
        item["duration"] = max(0.0, float(item["end"]) - float(item["start"]))
    return intervals


def _parse_detector_markers(text: str, total_duration: float | None) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """解析 blackdetect/freezedetect 日志（兼容经典同行格式与 lavfi 元数据格式）。

    freeze 持续到文件结束时新版 ffmpeg 只输出 start 不输出 duration/end，
    此时用总时长闭合区间；无法得知总时长则按 0 处理（宁可漏报不误报）。
    """

    black: list[dict[str, float]] = []
    freeze: list[dict[str, float]] = []
    open_black_start: float | None = None
    open_freeze_start: float | None = None
    pending_freeze_duration: float | None = None

    def _close_freeze_at_eof() -> None:
        nonlocal open_freeze_start, pending_freeze_duration
        if open_freeze_start is not None:
            end = float(total_duration) if total_duration else open_freeze_start
            freeze.append({"start": open_freeze_start, "duration": max(0.0, end - open_freeze_start)})
        open_freeze_start = None
        pending_freeze_duration = None

    for match in _DETECTOR_MARKER_RE.finditer(text):
        kind, action, value = match.group(1), match.group(2), float(match.group(3))
        if kind == "black":
            if action == "start":
                open_black_start = value
            elif action == "end" and open_black_start is not None:
                black.append({"start": open_black_start, "end": max(value, open_black_start)})
                open_black_start = None
        else:
            if action == "start":
                _close_freeze_at_eof()
                open_freeze_start = value
                pending_freeze_duration = None
            elif action == "duration" and open_freeze_start is not None:
                pending_freeze_duration = value
            elif action == "end" and open_freeze_start is not None:
                duration = pending_freeze_duration if pending_freeze_duration is not None else max(0.0, value - open_freeze_start)
                freeze.append({"start": open_freeze_start, "duration": max(0.0, duration)})
                open_freeze_start = None
                pending_freeze_duration = None
    _close_freeze_at_eof()
    return black, freeze


__all__ = [
    "ASPECT_ERROR_DEVIATION",
    "ASPECT_WARNING_DEVIATION",
    "AUDIO_OVER_PICTURE_TOLERANCE_S",
    "BLACK_ERROR_INTERVAL_S",
    "BLACK_ERROR_RATIO",
    "BLANK_ERROR_INTERVAL_S",
    "BLANK_ERROR_RATIO",
    "BLANK_LUMA_RANGE",
    "CATEGORY_STRUCTURAL",
    "CATEGORY_TECHNICAL",
    "CATEGORY_VISUAL_PENDING",
    "DEFAULT_DURATION_TOLERANCE_S",
    "FREEZE_DETECT_MIN_S",
    "FREEZE_ERROR_INTERVAL_S",
    "FREEZE_ERROR_RATIO",
    "MIN_IMAGE_BYTES",
    "MIN_IMAGE_SIDE",
    "MIN_VIDEO_BYTES",
    "MIN_VIDEO_SIDE",
    "VISUAL_DIMENSIONS",
    "VISUAL_DIMENSION_KEYS",
    "VISUAL_PENDING_REASON",
    "probe_media_duration",
    "probe_media_duration_sync",
    "validate_image_file",
    "validate_video_file",
    "validate_video_sync",
    "visual_pending_dimensions",
]
