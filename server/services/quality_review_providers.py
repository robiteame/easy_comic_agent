"""质量审核的 Provider 层：VLM 评分、角色身份 embedding、本地音视频探测。

每类 Provider 都遵循同一条诚实性铁律：
- 能力未配置（缺 endpoint / 缺密钥 / 缺 ffmpeg）时返回 ``unsupported`` 状态与
  原因，绝不返回伪评分；
- 调用失败时抛出或返回 ``error``，由上层按 fail-closed 处理；
- 只有真实执行并拿到结果时才给出分数。

Provider 均可注入替换（测试与离线环境用 stub），QualityReviewService 通过
构造函数接收实例。
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from services.providers.endpoint import get_endpoint

logger = logging.getLogger(__name__)

PROBE_TIMEOUT_SECONDS = 60


# ---------------------------------------------------------------------------
# VLM 评分 Provider
# ---------------------------------------------------------------------------


@dataclass
class VLMCapability:
    supported: bool
    reason: str = ""
    provider: str = ""


class VLMJudge:
    """用 script 端点（openai-chat 协议 + vision_model 参数）做视觉评分。

    依赖 ``LLMService.call_json_with_images``：多图输入 + JSON 输出清洗。
    """

    def __init__(self, llm=None):
        if llm is None:
            from services.llm_service import LLMService

            llm = LLMService()
        self._llm = llm

    def capability(self) -> VLMCapability:
        try:
            if not self._llm.vision_available:
                return VLMCapability(
                    supported=False,
                    reason="VLM 未配置：script 端点缺少 API Key，或适配器声明不支持图片输入",
                )
            return VLMCapability(supported=True, provider=self._llm.vision_provider_label)
        except Exception as exc:  # 端点配置异常视为能力不可用，而不是崩溃
            return VLMCapability(supported=False, reason=f"VLM 端点读取失败: {exc}")

    async def judge(self, system_prompt: str, user_prompt: str, image_paths: list[str]) -> dict:
        """执行一次评分，返回解析后的 JSON dict；失败抛异常（上层记 error）。"""
        return await self._llm.call_json_with_images(system_prompt, user_prompt, image_paths)


# ---------------------------------------------------------------------------
# 角色身份 embedding Provider
# ---------------------------------------------------------------------------


@dataclass
class SimilarityReport:
    status: str  # scored / unsupported / error
    similarities: list[dict] = field(default_factory=list)  # [{label, score}]
    min_score: float | None = None
    error: str = ""
    provider: str = ""


class IdentityEmbeddingProvider:
    """OpenAI 兼容多模态 embedding（identity 端点）算身份相似度。

    请求体形如 ``{"model": ..., "input": [{"type": "image_url",
    "image_url": {"url": data-url}}, ...]}``（SiliconFlow / Jina CLIP 等兼容
    服务）。端点未配置（密钥/地址/模型任一为空）时如实返回 unsupported。
    """

    def __init__(self, http_client_factory=None):
        self._http_client_factory = http_client_factory

    def _endpoint(self):
        return get_endpoint("identity")

    def capability(self) -> VLMCapability:
        endpoint = self._endpoint()
        missing = [
            label
            for label, value in (
                ("base_url", endpoint.base_url),
                ("api_key", endpoint.api_key),
                ("model", endpoint.model),
            )
            if not str(value or "").strip()
        ]
        if missing:
            return VLMCapability(
                supported=False,
                reason="身份 embedding 未配置（identity 端点缺少 " + "、".join(missing) + "）",
            )
        return VLMCapability(
            supported=True,
            provider=f"identity-embedding:{endpoint.protocol}:{endpoint.model}",
        )

    async def similarity(self, subject_path: str, reference_paths: list[dict]) -> SimilarityReport:
        """subject 与各参考图（[{label, path}]）的余弦相似度。"""
        capability = self.capability()
        if not capability.supported:
            return SimilarityReport(status="unsupported", error=capability.reason)
        usable = [item for item in reference_paths if item.get("path")]
        if not usable:
            return SimilarityReport(status="unsupported", error="没有可用的角色参考图，无法计算身份相似度")
        endpoint = self._endpoint()
        inputs: list[dict] = []
        try:
            inputs.append({"kind": "subject", "label": "subject", "path": subject_path})
            inputs.extend({"kind": "reference", "label": item["label"], "path": item["path"]} for item in usable)
            payload = {
                "model": endpoint.model,
                "input": [
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_url(item["path"])},
                    }
                    for item in inputs
                ],
                "encoding_format": "float",
            }
            headers = {"Authorization": f"Bearer {endpoint.api_key}"}
            if endpoint.auth_style == "api-key-header":
                headers["api-key"] = endpoint.api_key

            import httpx

            factory = self._http_client_factory or (lambda: httpx.AsyncClient(timeout=60))
            async with factory() as client:
                response = await client.post(
                    f"{endpoint.base_url.rstrip('/')}/embeddings", json=payload, headers=headers
                )
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            logger.warning("身份 embedding 调用失败: %s", exc)
            return SimilarityReport(status="error", error=f"embedding 调用失败: {exc}", provider=capability.provider)

        embeddings: list[list[float]] = []
        try:
            for item in data.get("data") or []:
                vector = item.get("embedding") or []
                if vector:
                    embeddings.append([float(value) for value in vector])
        except (TypeError, ValueError) as exc:
            return SimilarityReport(status="error", error=f"embedding 响应格式异常: {exc}")

        if len(embeddings) != len(inputs) or not embeddings:
            return SimilarityReport(
                status="error",
                error=f"embedding 返回数量不符（期望 {len(inputs)}，实际 {len(embeddings)}）",
                provider=capability.provider,
            )
        subject_vector = embeddings[0]
        similarities: list[dict] = []
        for item, vector in zip(inputs[1:], embeddings[1:], strict=True):
            score = _cosine_similarity(subject_vector, vector)
            similarities.append({"label": item["label"], "score": round(score, 4)})
        min_score = min((item["score"] for item in similarities), default=None)
        return SimilarityReport(
            status="scored",
            similarities=similarities,
            min_score=min_score,
            provider=capability.provider,
        )


def _image_data_url(path: str) -> str:
    mime = "image/png" if str(path).lower().endswith(".png") else "image/jpeg"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{encoded}"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if not norm_a or not norm_b:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


# ---------------------------------------------------------------------------
# 本地音视频探测（ffprobe / ffmpeg）
# ---------------------------------------------------------------------------


@dataclass
class MediaProbe:
    status: str  # scored / unsupported / skipped / error
    has_audio: bool = False
    video_duration: float | None = None
    audio_duration: float | None = None
    mean_volume_db: float | None = None
    max_volume_db: float | None = None
    silence_ratio: float | None = None
    issues: list[str] = field(default_factory=list)
    error: str = ""


def _parse_duration(value: object) -> float | None:
    """解析 ffprobe 的 duration 字段（字符串秒数）；缺失或非法时返回 None。

    修复:该辅助函数此前被 probe_media_streams 调用但从未定义,音画同步
    探测路径会直接 NameError。
    """

    try:
        seconds = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return seconds if seconds > 0 else None


async def _run_command(args: list[str], timeout: int = PROBE_TIMEOUT_SECONDS) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    return proc.returncode or 0, stdout.decode("utf-8", errors="ignore"), stderr.decode("utf-8", errors="ignore")


async def probe_media_streams(path: str) -> MediaProbe:
    """音画同步用的结构探测：是否有音轨、视频/音频时长差。"""
    result = MediaProbe(status="scored")
    try:
        code, stdout, stderr = await _run_command(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,duration:format=duration",
                "-of",
                "json",
                str(path),
            ]
        )
        if code != 0:
            return MediaProbe(status="error", error=f"ffprobe 失败: {stderr[-200:]}")
        data = json.loads(stdout or "{}")
    except FileNotFoundError:
        return MediaProbe(status="unsupported", error="ffprobe 不可用（unsupported）")
    except Exception as exc:
        return MediaProbe(status="error", error=f"ffprobe 失败: {exc}")

    streams = data.get("streams") or []
    video_streams = [item for item in streams if item.get("codec_type") == "video"]
    audio_streams = [item for item in streams if item.get("codec_type") == "audio"]
    result.has_audio = bool(audio_streams)
    result.video_duration = _parse_duration(
        video_streams[0].get("duration") if video_streams else None
    ) or _parse_duration((data.get("format") or {}).get("duration"))
    result.audio_duration = _parse_duration(audio_streams[0].get("duration") if audio_streams else None)

    if not result.has_audio:
        result.status = "skipped"
        result.issues.append("无音频轨（该镜头没有配音）")
        return result
    if result.video_duration is None or result.audio_duration is None:
        result.status = "skipped"
        result.issues.append("音视频时长信息不足，无法核对同步")
        return result
    drift = abs(result.video_duration - result.audio_duration)
    if drift > 0.5:
        result.issues.append(
            f"音画时长差 {drift:.2f}s（视频 {result.video_duration:.2f}s / 音频 {result.audio_duration:.2f}s）"
        )
    return result


async def analyze_audio_clarity(path: str) -> MediaProbe:
    """音频清晰度：音量水平 / 静音占比 / 削波（ffmpeg volumedetect + silencedetect）。"""
    result = MediaProbe(status="scored")
    try:
        code, _, stderr = await _run_command(
            [
                "ffmpeg",
                "-hide_banner",
                "-i",
                str(path),
                "-af",
                "volumedetect,silencedetect=noise=-45dB:d=0.8",
                "-f",
                "null",
                "-",
            ]
        )
        if code != 0:
            return MediaProbe(status="error", error=f"ffmpeg 失败: {stderr[-200:]}")
    except FileNotFoundError:
        return MediaProbe(status="unsupported", error="ffmpeg 不可用（unsupported）")
    except Exception as exc:
        return MediaProbe(status="error", error=f"ffmpeg 失败: {exc}")

    def _db(pattern: str) -> float | None:
        match = re.search(pattern, stderr)
        return float(match.group(1)) if match else None

    result.mean_volume_db = _db(r"mean_volume:\s*(-?[\d.]+)\s*dB")
    result.max_volume_db = _db(r"max_volume:\s*(-?[\d.]+)\s*dB")
    durations = [float(value) for value in re.findall(r"silence_duration:\s*([\d.]+)", stderr)]
    total_silence = sum(durations)
    total_match = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", stderr)
    if total_match:
        total = int(total_match.group(1)) * 3600 + int(total_match.group(2)) * 60 + float(total_match.group(3))
        if total > 0:
            result.silence_ratio = round(min(1.0, total_silence / total), 3)

    if result.mean_volume_db is not None and result.mean_volume_db < -45:
        result.issues.append(f"平均音量过低（{result.mean_volume_db:.1f} dB）")
    if result.silence_ratio is not None and result.silence_ratio > 0.6:
        result.issues.append(f"大部分时间是无声（静音占比 {result.silence_ratio:.0%}）")
    if result.max_volume_db is not None and result.max_volume_db > -0.05:
        result.issues.append("音频疑似削波破音（max_volume 触顶）")
    if not result.issues and result.mean_volume_db is None and result.silence_ratio is None:
        result.status = "error"
        result.error = "ffmpeg 未输出音量统计"
    return result


async def extract_video_frames(path: str, count: int = 4, workdir: Path | None = None) -> list[str]:
    """等间隔抽帧供 VLM 审核（motion / continuity / lip_sync 维度）。"""
    probe = await probe_media_streams(path)
    duration = probe.video_duration
    if not duration or duration <= 0:
        return []
    workdir = Path(workdir or tempfile.mkdtemp(prefix="quality-frames-"))
    timestamps = [duration * (index + 0.5) / count for index in range(count)]
    frames: list[str] = []
    for index, position in enumerate(timestamps):
        output = workdir / f"frame_{index:02d}.jpg"
        try:
            code, _, stderr = await _run_command(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-ss",
                    f"{position:.2f}",
                    "-i",
                    str(path),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "4",
                    "-y",
                    str(output),
                ]
            )
        except FileNotFoundError:
            logger.warning("ffmpeg 不可用，视频抽帧失败（相关维度按 unsupported 处理）")
            return frames
        if code == 0 and output.exists():
            frames.append(str(output))
        else:
            logger.debug("抽帧失败 position=%.2f: %s", position, stderr[-120:])
    return frames


__all__ = [
    "IdentityEmbeddingProvider",
    "MediaProbe",
    "SimilarityReport",
    "VLMCapability",
    "VLMJudge",
    "analyze_audio_clarity",
    "extract_video_frames",
    "probe_media_streams",
]
