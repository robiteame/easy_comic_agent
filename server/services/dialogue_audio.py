"""逐句配音编排：按结构化对白生成 TTS 音轨并回填时间轴。

一镜多句对白时逐句调用 TTS（每句按自身 speaker 选择角色音色），再拼接成
镜头的单条 ``audio_path``（下游混音 / 渲染继续假设一镜一条配音，不扩散改
动），最后用实测单句时长回填 ``start_ms`` / ``end_ms``，供字幕与镜头时间线
使用。说话人不在角色列表时记录可追踪警告并使用端点默认音色，绝不静默改用
第一个角色的声音。
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from pathlib import Path

from config import settings
from services.ffmpeg_service import FFmpegService
from services.security import safe_path, validate_identifier
from services.shot_dialogue import (
    DialogueLine,
    assign_line_timings,
    resolve_speaker_voice,
)
from services.storage_service import StorageQuotaExceeded, StorageService
from services.tts_service import TTSService

logger = logging.getLogger(__name__)

__all__ = ["generate_dialogue_track", "ffmpeg_service", "tts_service"]

# 模块级单例与路由层习惯一致：测试可以整体 patch generate_dialogue /
# probe_duration_ms / concat_audio_clips，不需要真实外呼。
tts_service = TTSService()
ffmpeg_service = FFmpegService()
storage_service = StorageService()


def _line_media_id(base: str, index: int) -> str:
    """第 index 句的媒体文件名（同一镜头版本下可追溯）。"""

    suffix = f"_d{index}"
    candidate = f"{base}{suffix}"
    if len(candidate) <= 128:
        return candidate
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:8]
    return f"{digest}{suffix}"


async def generate_dialogue_track(
    lines: list[DialogueLine],
    *,
    characters: list[dict],
    project_id: str,
    media_id: str,
    default_emotion: str = "neutral",
    text_cleaner: Callable[[str], str] | None = None,
) -> tuple[str, list[DialogueLine]]:
    """为一组结构化对白生成配音，返回 (音频路径, 带实测时间轴的对白)。

    - 每句独立调用 TTS：voice_id 取 ``resolve_speaker_voice(line.speaker, ...)``，
      emotion 取该句自身情绪（缺省用镜头情绪）；
    - 单句：直接落成 ``audio/{media_id}.{fmt}``，与历史行为一致；
    - 多句：逐句落 ``audio/{media_id}_dN.{fmt}`` 后拼接为 ``audio/{media_id}.{fmt}``，
      中间件在拼接成功后清理；
    - 返回的对白列表带 ``start_ms`` / ``end_ms``（毫秒，相对音轨起点）。
    """

    safe_project_id = validate_identifier(project_id, "项目 ID")
    safe_media_id = validate_identifier(media_id, "镜头媒体 ID")
    context = f"project={safe_project_id} shot_media={safe_media_id}"

    prepared: list[DialogueLine] = []
    for line in lines:
        text = str(line.line or "").strip()
        if text_cleaner is not None:
            text = text_cleaner(text)
        if not text:
            continue
        prepared.append(
            DialogueLine(
                speaker=str(line.speaker or "").strip(),
                line=text,
                emotion=str(line.emotion or "").strip() or default_emotion,
                action=line.action,
            )
        )
    if not prepared:
        raise RuntimeError("镜头没有可配音的台词文本")

    clip_paths: list[str] = []
    single_line = len(prepared) == 1
    for index, line in enumerate(prepared, start=1):
        voice_id = resolve_speaker_voice(line.speaker, characters, context=context)
        # 单句直接落到最终媒体名（保持版本复用路径判定不变）；多句逐段落
        # ``_dN`` 中间件后拼接成最终媒体名。
        line_media_id = safe_media_id if single_line else _line_media_id(safe_media_id, index)
        clip_paths.append(
            await tts_service.generate_dialogue(
                text=line.line,
                voice_id=voice_id,
                emotion=line.emotion or default_emotion,
                project_id=safe_project_id,
                shot_id=line_media_id,
            )
        )

    # 单句实测时长先于任何清理动作取得；多句逐段探测再拼接。
    durations: list[int] = []
    if len(clip_paths) == 1:
        final_path = Path(clip_paths[0])
        durations.append(max(0, int(await ffmpeg_service.probe_duration_ms(final_path))))
    else:
        suffix = Path(clip_paths[0]).suffix or ".wav"
        audio_dir = safe_path(settings.OUTPUT_DIR / "projects", safe_project_id, "audio", create_parent=True)
        final_path = audio_dir / f"{safe_media_id}{suffix}"
        for path in clip_paths:
            durations.append(max(0, int(await ffmpeg_service.probe_duration_ms(path))))
        try:
            storage_service.ensure_project_capacity(
                project_id=safe_project_id,
                incoming_bytes=sum(Path(path).stat().st_size for path in clip_paths),
                replacing=final_path,
            )
        except StorageQuotaExceeded as exc:
            raise RuntimeError("项目媒体存储空间不足") from exc
        await ffmpeg_service.concat_audio_clips([Path(path) for path in clip_paths], final_path)
        # 拼接成功后清理逐句中间件；失败路径保留现场便于排查。
        for path in clip_paths:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                logger.warning("清理逐句配音中间件失败: path=%s", path)

    timed: list[DialogueLine] = []
    for line, (start_ms, end_ms) in zip(prepared, assign_line_timings(durations), strict=True):
        timed.append(
            DialogueLine(
                speaker=line.speaker,
                line=line.line,
                emotion=line.emotion,
                action=line.action,
                start_ms=start_ms,
                end_ms=max(start_ms, end_ms),
            )
        )
    return str(final_path), timed
