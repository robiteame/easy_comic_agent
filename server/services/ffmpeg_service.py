import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import time
import uuid
from pathlib import Path

from config import settings
from services import usage_service
from services.audio_mix_planner import SAMPLE_RATE, AudioMixPlan, MixTrackInput, manifest_digest, plan_audio_mix
from services.providers.usage import CAPABILITY_FFMPEG
from services.security import existing_file, safe_path, validate_identifier
from services.storage_service import StorageQuotaExceeded, StorageService
from services.subtitle_service import SubtitleCueData, SubtitleStyle, build_ass_document, serialize_srt

logger = logging.getLogger(__name__)


class MixBundle:
    """一次混音的完整输入：预构建的主音轨链 + 规划器产出的轨道拓扑。"""

    def __init__(self, plan: AudioMixPlan, main_inputs: list[str], main_statements: list[str]):
        self.plan = plan
        self.main_inputs = main_inputs
        self.main_statements = main_statements
        self.warnings: list[str] = list(plan.warnings)

    @property
    def inputs(self) -> list[str]:
        return [*self.main_inputs, *self.plan.inputs]

    @property
    def filter_complex(self) -> str:
        return ";".join([*self.main_statements, self.plan.filter_complex])

    def manifest(self) -> dict:
        return {
            "tracks": self.plan.track_manifest,
            "ducked_track_ids": self.plan.ducked_track_ids,
            "filter_sha1": manifest_digest(self.plan.track_manifest),
        }


def _loud(value: float) -> str:
    """响度参数定点化，避免 -0.0 / 科学计数法进入滤镜串。"""

    return f"{float(value):.2f}"


class FFmpegService:
    """Render shot images and optional audio into a short MP4."""

    _subtitles_filter_available: bool | None = None

    def __init__(self):
        self.output_dir = settings.OUTPUT_DIR / "projects"
        self.fps = settings.DEFAULT_FPS
        self.storage = StorageService()

    async def compose_video(
        self,
        shots: list[dict],
        output_format: str = "9:16",
        resolution: str = "1080p",
        project_id: str = "",
        publish: bool = True,
        av_config: dict | None = None,
        plan: dict | None = None,
    ) -> str:
        try:
            safe_project_id = validate_identifier(project_id, "项目 ID")
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        video_dir = safe_path(self.output_dir, safe_project_id, "output", create_parent=True)
        try:
            self.storage.ensure_project_capacity(safe_project_id, settings.FFMPEG_WORKSPACE_RESERVE_BYTES)
        except StorageQuotaExceeded as exc:
            raise RuntimeError("项目媒体存储空间不足，无法开始渲染") from exc
        work_dir = video_dir / f".render-{uuid.uuid4().hex}"
        work_dir.mkdir(parents=True, exist_ok=False)
        width, height = self._get_resolution(resolution, output_format)
        av_tracks = list((av_config or {}).get("audio_tracks") or [])
        subtitle_tracks = list((av_config or {}).get("subtitle_tracks") or [])
        total_duration_s = float((av_config or {}).get("total_duration_s") or 0) or sum(
            max(0.0, float(shot.get("duration") or 0)) for shot in shots
        )
        transition_specs = list((plan or {}).get("transitions") or self._transition_specs_from_shots(shots))
        # FFmpeg 用量：编码时长（成片秒数）+ 输出分辨率 + 真实处理耗时。
        # 本地能力默认零外部费用；若在系统设置里为 ffmpeg 配了单价，则按价目计费。
        started = time.monotonic()
        encoded_seconds = int(round(sum(max(0.0, float(shot.get("duration") or 0)) for shot in shots))) or len(shots)
        output_resolution = f"{width}x{height}"

        try:
            clip_paths: list[Path] = []
            clip_durations: list[float] = []
            media_roots = (settings.OUTPUT_DIR, settings.ASSETS_DIR, settings.DATA_DIR)
            for index, shot in enumerate(shots):
                if shot.get("video_path"):
                    if existing_file(shot["video_path"], minimum_size=4096, allowed_roots=media_roots) is None:
                        raise ValueError(f"镜头视频文件不存在或无效: {shot.get('shot_id', index)}")
                    clip_paths.append(await self._normalize_video_clip(shot, width, height, index, work_dir))
                    clip_durations.append(max(0.5, float(shot.get("duration") or 3.0)))
                    continue
                if not shot.get("image_path"):
                    continue
                if existing_file(shot["image_path"], minimum_size=1, allowed_roots=media_roots) is None:
                    raise ValueError(f"镜头图片文件不存在或无效: {shot.get('shot_id', index)}")
                clip_paths.append(await self._render_shot_clip(shot, width, height, index, work_dir))
                clip_durations.append(max(0.5, float(shot.get("duration") or 3.0)))

            if not clip_paths:
                raise ValueError("没有可渲染的镜头图片")

            rendered = await self._concat_clips(clip_paths, work_dir, transition_specs, clip_durations)
            # 字幕/音频工作台：有任一有效音轨时，混音接管音频（环境床在混音
            # filter 内重建），否则保持旧管线（concat 音轨 + 独立环境床步骤）。
            mix_bundle = await self._prepare_mix(shots, av_tracks, total_duration_s)
            if mix_bundle is not None:
                rendered = await self._apply_audio_mix(rendered, mix_bundle, work_dir, total_duration_s)
            else:
                rendered = await self._add_continuous_ambient_bed(rendered, work_dir)
            # 烧录字幕在混音之后（避免字幕被后续步骤重编码抹掉画质前就叠上）。
            for subtitle_track in subtitle_tracks:
                if not subtitle_track.get("enabled") or not subtitle_track.get("burn_in"):
                    continue
                cues = subtitle_track.get("cues") or []
                if not cues:
                    continue
                rendered = await self._burn_subtitles(rendered, subtitle_track, width, height, work_dir)
            soft_tracks = [
                track
                for track in subtitle_tracks
                if track.get("enabled") and not track.get("burn_in") and (track.get("cues") or [])
            ]
            if soft_tracks:
                rendered = await self._mux_soft_subtitles(rendered, soft_tracks, work_dir)
            final_path = video_dir / ("final.mp4" if publish else f".final-{uuid.uuid4().hex}.candidate")
            os.replace(rendered, final_path)
            if not final_path.exists() or final_path.stat().st_size <= 1024:
                raise RuntimeError("FFmpeg 未生成有效的成片文件")
            usage_service.record_usage(
                CAPABILITY_FFMPEG,
                provider="local",
                model="ffmpeg",
                quantity=encoded_seconds,
                resolution=output_resolution,
                units={"encoding_seconds": encoded_seconds, "shots": len(clip_paths)},
                duration_ms=int((time.monotonic() - started) * 1000),
                scope=usage_service.current_scope(),
            )
            return str(final_path)
        except BaseException as exc:
            # 失败/取消同样留痕：编码耗时是真实发生的资源消耗。
            cancelled = isinstance(exc, asyncio.CancelledError)
            usage_service.record_usage(
                CAPABILITY_FFMPEG,
                provider="local",
                model="ffmpeg",
                quantity=encoded_seconds,
                resolution=output_resolution,
                units={"encoding_seconds": encoded_seconds, "shots": len(shots)},
                status=usage_service.CALL_CANCELLED if cancelled else usage_service.CALL_FAILED,
                error_code="ffmpeg_cancelled" if cancelled else "ffmpeg_failed",
                duration_ms=int((time.monotonic() - started) * 1000),
                scope=usage_service.current_scope(),
            )
            raise
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    async def _render_shot_clip(self, shot: dict, width: int, height: int, index: int, output_dir: Path) -> Path:
        clip_path = output_dir / f"clip_{index:04d}.mp4"
        duration = max(0.5, float(shot.get("duration") or 3.0))
        frames = max(1, int(duration * self.fps))
        image_obj = self._media_path(shot.get("image_path"), minimum_size=1)
        if image_obj is None:
            raise ValueError("镜头图片文件不存在或无效")
        image_path = str(image_obj)
        video_filter = self._clip_filter(
            self._zoom_filter(
                shot.get("shot_type", "medium"), width, height, frames, shot.get("camera_movement", "静止")
            ),
            shot,
            duration,
        )

        audio_path = self._media_path(shot.get("audio_path"), minimum_size=1)
        if audio_path:
            audio_input = ["-i", str(audio_path)]
            audio_filter = ["-af", f"apad=pad_dur={duration}"]
        else:
            audio_input = ["-f", "lavfi", "-t", str(duration), "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]
            audio_filter = []

        await self._run(
            [
                "ffmpeg",
                "-y",
                "-loop",
                "1",
                "-i",
                image_path,
                *audio_input,
                "-vf",
                video_filter,
                "-t",
                str(duration),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                *audio_filter,
                "-shortest",
                str(clip_path),
            ]
        )
        return clip_path

    async def _normalize_video_clip(self, shot: dict, width: int, height: int, index: int, output_dir: Path) -> Path:
        clip_path = output_dir / f"clip_{index:04d}.mp4"
        duration = max(0.5, float(shot.get("duration") or 3.0))
        video_path_obj = self._media_path(shot.get("video_path"), minimum_size=4096)
        if video_path_obj is None:
            raise ValueError("镜头视频文件不存在或无效")
        video_path = str(video_path_obj)
        audio_path = self._media_path(shot.get("audio_path"), minimum_size=1)
        native_audio = self._shot_has_native_audio(shot)
        if audio_path:
            audio_input = ["-i", str(audio_path)]
            map_audio = ["-map", "1:a:0"]
            audio_codec = ["-c:a", "aac", "-af", f"apad=pad_dur={duration}"]
        elif native_audio:
            # 原生音频视频：直接沿用其自带音轨，保证两条音频路径产出契约一致。
            audio_input = []
            map_audio = ["-map", "0:a:0"]
            audio_codec = ["-c:a", "aac"]
        else:
            audio_input = ["-f", "lavfi", "-t", str(duration), "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]
            map_audio = ["-map", "1:a:0"]
            audio_codec = ["-c:a", "aac"]

        await self._run(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                *audio_input,
                "-vf",
                self._clip_filter(
                    f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}", shot, duration
                ),
                "-t",
                str(duration),
                "-map",
                "0:v:0",
                *map_audio,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                *audio_codec,
                "-shortest",
                str(clip_path),
            ]
        )
        return clip_path

    async def _concat_clips(
        self,
        clip_paths: list[Path],
        output_dir: Path,
        transitions: list[dict] | None = None,
        clip_durations: list[float] | None = None,
    ) -> Path:
        """按 PostProductionPlan 的边界拼接镜头。

        cut 直接 concat；fade/dissolve/push/wipe 使用 xfade；white_flash 使用
        dissolve 并在边界叠加白色覆盖，时长严格取计划中的 duration_ms。
        音频随视频边界 acrossfade，避免字幕/音频和画面出现两套时间。
        """

        concat_path = output_dir / "final.mp4"
        if not clip_paths:
            raise ValueError("没有可拼接的镜头片段")
        inputs: list[str] = []
        for path in clip_paths:
            inputs.extend(["-i", str(path)])

        audio_filters = []
        for index in range(len(clip_paths)):
            audio_filters.append(f"[{index}:a:0]aresample=44100,aformat=channel_layouts=stereo[a{index}]")

        specs = list(transitions or [])
        durations = list(clip_durations or [3.0] * len(clip_paths))
        statements: list[str] = []
        flash_filters: list[str] = []
        current_v = "[0:v:0]"
        current_a = "[a0]"
        current_duration = max(0.0, float(durations[0] if durations else 3.0))
        for index in range(1, len(clip_paths)):
            spec = specs[index - 1] if index - 1 < len(specs) else {}
            duration = max(0.0, float(durations[index] if index < len(durations) else 3.0))
            transition_duration = max(0.0, float(spec.get("duration_ms") or 0) / 1000.0)
            effective = str(spec.get("effective") or "cut")
            if effective == "cut" or transition_duration <= 0:
                next_v = f"[{index}:v:0]"
                next_a = f"[a{index}]"
                out_v = f"[cv{index}]"
                out_a = f"[ca{index}]"
                statements.append(f"{current_v}{current_a}{next_v}{next_a}concat=n=2:v=1:a=1{out_v}{out_a}")
                current_v, current_a = out_v, out_a
                current_duration += duration
                continue

            offset = max(0.0, current_duration - transition_duration)
            transition_name = str(spec.get("renderer") or "dissolve")
            if transition_name == "cut":
                transition_name = "dissolve"
            next_v = f"[{index}:v:0]"
            next_a = f"[a{index}]"
            out_v = f"[xv{index}]"
            out_a = f"[xa{index}]"
            statements.append(
                f"{current_v}{next_v}xfade=transition={transition_name}:"
                f"duration={transition_duration:.3f}:offset={offset:.3f}{out_v}"
            )
            statements.append(f"{current_a}{next_a}acrossfade=d={transition_duration:.3f}{out_a}")
            current_v, current_a = out_v, out_a
            current_duration = current_duration + duration - transition_duration
            if effective == "white_flash":
                flash_start = offset
                flash_end = offset + transition_duration
                flash_filters.append(
                    f"{current_v}drawbox=x=0:y=0:w=iw:h=ih:color=white@1:t=fill:"
                    f"enable='between(t,{flash_start:.3f},{flash_end:.3f})'[wf{index}]"
                )
                current_v = f"[wf{index}]"

        filter_complex = ";".join([*audio_filters, *statements, *flash_filters])
        await self._run(
            [
                "ffmpeg",
                "-y",
                *inputs,
                "-filter_complex",
                filter_complex,
                "-map",
                current_v,
                "-map",
                current_a,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(concat_path),
            ]
        )
        return concat_path

    @staticmethod
    def _transition_specs_from_shots(shots: list[dict]) -> list[dict]:
        """未传 PostProductionPlan 时的兼容入口：仍尊重 shot.transition。"""

        specs: list[dict] = []
        for index in range(max(0, len(shots) - 1)):
            previous = shots[index]
            requested = str(previous.get("transition") or "cut")
            supported = requested in {"cut", "fade", "dissolve", "white_flash", "push", "wipe"}
            effective = requested if supported else "cut"
            default_duration = {"cut": 0, "fade": 500, "dissolve": 500, "white_flash": 350, "push": 500, "wipe": 500}
            specs.append(
                {
                    "effective": effective,
                    "duration_ms": default_duration.get(effective, 0),
                    "renderer": {
                        "fade": "fade",
                        "dissolve": "dissolve",
                        "white_flash": "dissolve",
                        "push": "slideleft",
                        "wipe": "wipeleft",
                    }.get(effective, "cut"),
                    "fallback_reason": "" if supported else f"unsupported_transition:{requested}",
                }
            )
        return specs

    async def _add_continuous_ambient_bed(self, video_path: Path, output_dir: Path) -> Path:
        mixed_path = output_dir / "final_with_ambient.mp4"
        await self._run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-f",
                "lavfi",
                "-i",
                "anoisesrc=color=pink:amplitude=0.008:sample_rate=44100",
                "-filter_complex",
                "[0:a:0]aresample=44100,aformat=channel_layouts=stereo[a0];[1:a:0]volume=0.08[amb];[a0][amb]amix=inputs=2:duration=first:dropout_transition=0[a]",
                "-map",
                "0:v:0",
                "-map",
                "[a]",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                "-movflags",
                "+faststart",
                str(mixed_path),
            ]
        )
        mixed_path.replace(video_path)
        return video_path

    # --- 字幕与音频混音工作台 ---------------------------------------------

    async def probe_duration_ms(self, path: str | Path) -> int:
        """探测媒体时长（毫秒）。优先 ffprobe，缺失时退回 ffmpeg 解码解析。"""

        target = str(path)
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                target,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
            except asyncio.CancelledError:
                if proc.returncode is None:
                    proc.kill()
                await proc.communicate()
                raise
            if proc.returncode == 0:
                for line in stdout.decode("utf-8", errors="ignore").splitlines():
                    line = line.strip()
                    if line:
                        try:
                            return int(float(line) * 1000)
                        except ValueError:
                            continue
        except FileNotFoundError:
            pass
        except TimeoutError:
            return 0
        # 退路：完整解码并取进度行的最后一个 time=。
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-hide_banner",
                "-i",
                target,
                "-f",
                "null",
                "-",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=max(30, int(settings.FFMPEG_TIMEOUT_SECONDS))
                )
            except asyncio.CancelledError:
                if proc.returncode is None:
                    proc.kill()
                await proc.communicate()
                raise
            matches = re.findall(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", stderr.decode("utf-8", errors="ignore"))
            if matches:
                hours, minutes, seconds = matches[-1]
                return int((int(hours) * 3600 + int(minutes) * 60 + float(seconds)) * 1000)
        except FileNotFoundError:
            logger.warning("ffprobe 与 ffmpeg 均不可用，无法探测 %s 的时长", target)
        return 0

    async def concat_audio_clips(self, clip_paths: list[Path], output_path: Path) -> None:
        """把逐句 TTS 音频拼接为单条镜头配音（重编码为项目统一的 44.1kHz 立体声）。

        调用方（dialogue_audio）已经逐段探测过时长，这里只负责顺序拼接：
        concat 滤镜按输入顺序串接，天然保证「多句对白按时间顺序播放」。
        """

        if not clip_paths:
            raise RuntimeError("没有可拼接的配音音频")
        inputs: list[str] = []
        statements: list[str] = []
        labels: list[str] = []
        for index, path in enumerate(clip_paths):
            inputs.extend(["-i", str(path)])
            statements.append(f"[{index}:a:0]aresample={SAMPLE_RATE},aformat=channel_layouts=stereo[a{index}]")
            labels.append(f"[a{index}]")
        statements.append(f"{''.join(labels)}concat=n={len(clip_paths)}:v=0:a=1[aout]")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        await self._run(
            [
                "ffmpeg",
                "-y",
                *inputs,
                "-filter_complex",
                ";".join(statements),
                "-map",
                "[aout]",
                "-c:a",
                "pcm_s16le",
                "-ar",
                str(SAMPLE_RATE),
                str(output_path),
            ]
        )

    async def _prepare_mix(
        self,
        shots: list[dict],
        av_tracks: list[dict],
        total_duration_s: float,
        *,
        allow_empty: bool = False,
    ) -> MixBundle | None:
        """校验素材、探测时长并规划混音。无有效轨道时按需返回 None。"""

        tracks: list[MixTrackInput] = []
        for item in av_tracks:
            if item.get("muted"):
                continue
            media = self._media_path(str(item.get("resolved_source_path") or ""), minimum_size=1)
            if media is None:
                logger.warning("音轨 %s 的素材缺失或越出允许目录，已从混音中剔除", item.get("id"))
                continue
            duration_ms = int(item.get("source_duration_ms") or 0)
            if duration_ms <= 0:
                duration_ms = await self.probe_duration_ms(media)
            tracks.append(
                MixTrackInput(
                    id=str(item.get("id") or ""),
                    kind=str(item.get("kind") or "music"),
                    media_path=str(media),
                    media_duration_ms=duration_ms,
                    shot_id=str(item.get("shot_id") or ""),
                    start_ms=int(item.get("start_ms") or 0),
                    volume=float(item.get("volume") or 1.0),
                    pan=float(item.get("pan") or 0.0),
                    fade_in_ms=int(item.get("fade_in_ms") or 0),
                    fade_out_ms=int(item.get("fade_out_ms") or 0),
                    delay_ms=int(item.get("delay_ms") or 0),
                    trim_start_ms=int(item.get("trim_start_ms") or 0),
                    trim_end_ms=int(item.get("trim_end_ms") or 0),
                    loop=bool(item.get("loop")),
                    duck_amount_db=float(item.get("duck_amount_db") or 0.0),
                    duck_attack_ms=int(item.get("duck_attack_ms") or 120),
                    duck_release_ms=int(item.get("duck_release_ms") or 480),
                    clip_limit_ms=int(item.get("clip_limit_ms") or 0),
                )
            )
        # 对白轨绑定的镜头：其 TTS 配音从主音轨拆出（主音轨对应区间置静音）。
        bound_dialogue_shots = {track.shot_id for track in tracks if track.kind == "dialogue" and track.media_path}
        main_inputs, main_statements = self._build_main_audio_chain(shots, bound_dialogue_shots)
        plan = plan_audio_mix(
            tracks,
            total_duration_s,
            main_label="[main0]",
            first_track_input_index=len(main_inputs),
            allow_empty=allow_empty,
        )
        if plan is None:
            return None
        return MixBundle(plan, main_inputs, main_statements)

    def _build_main_audio_chain(self, shots: list[dict], bound_dialogue_shots: set[str]) -> tuple[list[str], list[str]]:
        """按镜头顺序重建主音轨（等价于 concat 视频内嵌的音轨）。

        渲染与预览共用这一重建逻辑：对白拆出区间为静音、TTS 镜头用配音文件
        并 pad 到镜头时长、native 镜头用生成视频自带音轨，其余为静音。因此
        预览听到的主音轨与成片内嵌音轨来自同一段 filter 定义。
        """

        input_files: list[str] = []
        input_index_of: dict[str, int] = {}

        def allocate(path_obj: Path) -> int:
            key = str(path_obj)
            if key not in input_index_of:
                input_files.append(key)
                input_index_of[key] = len(input_files) - 1
            return input_index_of[key]

        format_chain = [f"aresample={SAMPLE_RATE}", "aformat=sample_fmts=fltp:channel_layouts=stereo"]
        statements: list[str] = []
        segment_labels: list[str] = []
        for index, shot in enumerate(shots):
            duration = max(0.5, float(shot.get("duration") or 3.0))
            sec = f"{duration:.3f}"
            label = f"m{index}"
            audio = None
            if str(shot.get("shot_id") or "") not in bound_dialogue_shots:
                audio = self._media_path(shot.get("audio_path"), minimum_size=1)
            # 输入标签后不能跟逗号（否则 ffmpeg 解析出空滤镜名）。
            if str(shot.get("shot_id") or "") in bound_dialogue_shots:
                head, chain = "", [f"aevalsrc=exprs=0|0:d={sec}:s={SAMPLE_RATE}", *format_chain]
            elif audio is not None:
                head, chain = (
                    f"[{allocate(audio)}:a:0]",
                    [*format_chain, f"apad=pad_dur={sec}", f"atrim=duration={sec}", "asetpts=PTS-STARTPTS"],
                )
            elif self._shot_has_native_audio(shot):
                video = self._media_path(shot.get("video_path"), minimum_size=4096)
                if video is not None:
                    head, chain = (
                        f"[{allocate(video)}:a:0]",
                        [*format_chain, f"atrim=duration={sec}", "asetpts=PTS-STARTPTS"],
                    )
                else:
                    head, chain = "", [f"aevalsrc=exprs=0|0:d={sec}:s={SAMPLE_RATE}", *format_chain]
            else:
                head, chain = "", [f"aevalsrc=exprs=0|0:d={sec}:s={SAMPLE_RATE}", *format_chain]
            statements.append(head + ",".join(part for part in chain if part) + f"[{label}]")
            segment_labels.append(label)
        if not segment_labels:
            statements.append(f"aevalsrc=exprs=0|0:d=0.5:s={SAMPLE_RATE}[main0]")
        else:
            statements.append(
                "".join(f"[{label}]" for label in segment_labels) + f"concat=n={len(segment_labels)}:v=0:a=1[main0]"
            )
        return input_files, statements

    async def _apply_audio_mix(
        self, video_path: Path, bundle: MixBundle, work_dir: Path, total_duration_s: float
    ) -> Path:
        normalized = await self._produce_normalized_mix(bundle, work_dir)
        (work_dir / "mix_manifest.json").write_text(
            json.dumps(bundle.manifest(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if bundle.warnings:
            logger.warning("混音警告: %s", "; ".join(bundle.warnings))
        output = work_dir / "final_mixed.mp4"
        await self._run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-i",
                str(normalized),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                "-shortest",
                str(output),
            ]
        )
        return output

    async def _produce_normalized_mix(self, bundle: MixBundle, work_dir: Path) -> Path:
        """执行规划好的 filter_complex，并完成响度归一化与削波防护。

        渲染与预览共用：同一份轨道配置 → 同一条 filter → 同一段归一化链，
        这是「预览与最终输出一致」的保证点。
        """

        raw = work_dir / "mix_raw.wav"
        inputs: list[str] = []
        for path in bundle.inputs:
            inputs.extend(["-i", path])
        await self._run(
            [
                "ffmpeg",
                "-y",
                *inputs,
                "-filter_complex",
                bundle.filter_complex,
                "-map",
                bundle.plan.output_label,
                "-c:a",
                "pcm_s16le",
                "-ar",
                str(SAMPLE_RATE),
                str(raw),
            ]
        )
        measured = await self._measure_loudness(raw)
        normalized = work_dir / "mix_norm.wav"
        # 两遍响度归一化（EBU R128）：先测量再线性套用，结果可复现；测量失败
        # 时退回单遍动态模式，宁可降级也不放弃归一化。
        if measured:
            audio_filter = (
                f"loudnorm=I={_loud(settings.LOUDNESS_TARGET_I)}:TP={_loud(settings.LOUDNESS_TARGET_TP)}"
                f":LRA={_loud(settings.LOUDNESS_TARGET_LRA)}:linear=true"
                f":measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
                f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
                f":offset={measured['target_offset']}"
            )
        else:
            audio_filter = (
                f"loudnorm=I={_loud(settings.LOUDNESS_TARGET_I)}:TP={_loud(settings.LOUDNESS_TARGET_TP)}"
                f":LRA={_loud(settings.LOUDNESS_TARGET_LRA)}"
            )
        limit = 10 ** (max(float(settings.LOUDNESS_TARGET_TP), -6.0) / 20.0)
        audio_filter += f",alimiter=limit={limit:.4f}:level=false"
        await self._run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(raw),
                "-af",
                audio_filter,
                "-c:a",
                "pcm_s16le",
                "-ar",
                str(SAMPLE_RATE),
                str(normalized),
            ]
        )
        stats = await self.detect_volume(normalized)
        max_db = stats.get("max_volume_db")
        if max_db is None or max_db <= -60.0:
            raise RuntimeError("混音输出完全静音：请检查轨道音量、静音与素材配置后重试")
        if max_db > float(settings.CLIPPING_HEADROOM_DB):
            raise RuntimeError(f"混音输出检测到削波（峰值 {max_db:.2f} dB），已中止本次渲染")
        return normalized

    async def _measure_loudness(self, media: Path) -> dict | None:
        """loudnorm 第一遍测量；静音或解析失败返回 None。"""

        _, stderr = await self._run_capture(
            [
                "ffmpeg",
                "-hide_banner",
                "-i",
                str(media),
                "-af",
                f"loudnorm=I={_loud(settings.LOUDNESS_TARGET_I)}:TP={_loud(settings.LOUDNESS_TARGET_TP)}"
                f":LRA={_loud(settings.LOUDNESS_TARGET_LRA)}:print_format=json",
                "-f",
                "null",
                "-",
            ]
        )
        match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", stderr.decode("utf-8", errors="ignore"), re.S)
        if not match:
            return None
        try:
            measured = json.loads(match.group(0))
        except ValueError:
            return None
        required = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
        if any(str(measured.get(key, "")).strip().lower() in {"-inf", "inf", "nan", ""} for key in required):
            return None
        return measured

    async def detect_volume(self, media: Path) -> dict:
        """volumedetect 峰值/均值（dB）。完全静音时值为 -inf，以 None 表示。"""

        _, stderr = await self._run_capture(
            ["ffmpeg", "-hide_banner", "-i", str(media), "-map", "0:a:0", "-af", "volumedetect", "-f", "null", "-"]
        )
        text = stderr.decode("utf-8", errors="ignore")

        def parse(key: str) -> float | None:
            found = re.search(rf"{key}:\s*(-?[\d.]+|-inf)\s*dB", text)
            if not found:
                return None
            if found.group(1) == "-inf":
                return float("-inf")
            try:
                return float(found.group(1))
            except ValueError:
                return None

        return {"max_volume_db": parse("max_volume"), "mean_volume_db": parse("mean_volume")}

    async def capabilities(self) -> dict:
        """探测当前 FFmpeg 的真实滤镜能力，供 UI 禁用/提示不支持参数。"""

        binary = shutil.which("ffmpeg")
        filters: set[str] = set()
        if binary:
            try:
                stdout, _ = await self._run_capture([binary, "-hide_banner", "-filters"])
                for line in stdout.decode("utf-8", errors="ignore").splitlines():
                    match = re.match(r"^\s*[TSC.][A-Z.]+\s+([A-Za-z0-9_]+)", line)
                    if match:
                        filters.add(match.group(1))
            except (RuntimeError, TimeoutError, OSError):
                filters.clear()
        has_xfade = "xfade" in filters
        supported_transitions = ["cut"] + (["fade", "dissolve", "white_flash", "push", "wipe"] if has_xfade else [])
        return {
            "available": bool(binary),
            "binary": binary or "",
            "filters": sorted(filters),
            "supported_transitions": supported_transitions,
            "camera_movement": "zoompan" in filters,
            "burn_in_subtitles": "subtitles" in filters,
            "white_flash": has_xfade and "drawbox" in filters,
        }

    async def _supports_subtitles_filter(self) -> bool:
        """探测当前 ffmpeg 是否带 subtitles 滤镜（依赖 libass；结果进程内缓存）。

        部分 Homebrew / 最小化构建不含 libass，直接调用会以「No such filter」
        失败。提前探测可以把这类环境问题转成可操作的中文提示，而不是让整次
        渲染在半途才报一段 ffmpeg 原始输出。无法判定时按可用处理，交还 ffmpeg
        自身的错误兜底。
        """

        if FFmpegService._subtitles_filter_available is None:
            available = True
            try:
                stdout, _ = await self._run_capture(["ffmpeg", "-hide_banner", "-filters"])
                lines = stdout.decode("utf-8", errors="ignore").splitlines()
                available = any(line.split() and line.split()[-1] == "subtitles" for line in lines)
            except (RuntimeError, TimeoutError, OSError):
                available = True
            FFmpegService._subtitles_filter_available = available
        return FFmpegService._subtitles_filter_available

    async def _burn_subtitles(self, video_path: Path, track: dict, width: int, height: int, work_dir: Path) -> Path:
        if not await self._supports_subtitles_filter():
            raise RuntimeError(
                "当前 FFmpeg 缺少 subtitles 滤镜（libass），无法烧录字幕；"
                "可在字幕与音频工作台把该字幕轨改为「独立字幕轨」后重试"
            )
        cues = [
            SubtitleCueData(
                start_ms=int(cue.get("start_ms") or 0),
                end_ms=int(cue.get("end_ms") or 0),
                text=str(cue.get("text") or ""),
                character_name=str(cue.get("character_name") or ""),
            )
            for cue in track.get("cues") or []
        ]
        style = SubtitleStyle(
            font_family=str(track.get("font_family") or "sans-serif"),
            font_size=int(track.get("font_size") or 54),
            primary_color=str(track.get("primary_color") or "#FFFFFF"),
            outline_color=str(track.get("outline_color") or "#000000"),
            outline_width=int(track.get("outline_width") or 0),
            bold=bool(track.get("bold")),
            position=str(track.get("position") or "bottom"),
            safe_margin=int(track.get("safe_margin") or 0),
        )
        ass_path = work_dir / "burn_subtitles.ass"
        ass_path.write_text(build_ass_document(style, cues, width, height), encoding="utf-8")
        output = work_dir / "final_subtitled.mp4"
        await self._run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vf",
                f"subtitles=filename='{self._escape_filter_path(ass_path)}'",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "copy",
                "-movflags",
                "+faststart",
                str(output),
            ]
        )
        return output

    async def _mux_soft_subtitles(self, video_path: Path, tracks: list[dict], work_dir: Path) -> Path:
        """把未烧录的字幕轨以 mov_text 内封进 MP4（画面与音轨直通）。"""

        inputs: list[str] = []
        maps: list[str] = ["-map", "0:v:0", "-map", "0:a:0"]
        metadata: list[str] = []
        for index, track in enumerate(tracks):
            cues = [
                SubtitleCueData(
                    start_ms=int(cue.get("start_ms") or 0),
                    end_ms=int(cue.get("end_ms") or 0),
                    text=str(cue.get("text") or ""),
                    character_name=str(cue.get("character_name") or ""),
                )
                for cue in track.get("cues") or []
            ]
            srt_path = work_dir / f"soft_subtitles_{index}.srt"
            srt_path.write_text(serialize_srt(cues), encoding="utf-8")
            inputs.extend(["-i", str(srt_path)])
            maps.extend(["-map", f"{index + 1}:s:0"])
            language = re.sub(r"[^A-Za-z0-9_-]", "", str(track.get("language") or "zh")) or "zh"
            metadata.extend([f"-metadata:s:s:{index}", f"language={language}"])
            name = str(track.get("name") or "").replace("\n", " ")[:60]
            if name:
                metadata.extend([f"-metadata:s:s:{index}", f"title={name}"])
        output = work_dir / "final_soft_subtitles.mp4"
        await self._run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                *inputs,
                *maps,
                "-c:v",
                "copy",
                "-c:a",
                "copy",
                "-c:s",
                "mov_text",
                *metadata,
                "-movflags",
                "+faststart",
                str(output),
            ]
        )
        return output

    async def render_mix_preview(
        self,
        shots: list[dict],
        av_tracks: list[dict],
        total_duration_s: float,
        output_path: Path,
        window_s: tuple[float, float] | None = None,
    ) -> dict:
        """生成音频预览（渲染同源的 filter + 归一化链）。"""

        bundle = await self._prepare_mix(shots, av_tracks, total_duration_s, allow_empty=True)
        if bundle is None:  # allow_empty=True 时规划器恒返回结果，防御式兜底
            raise RuntimeError("无法规划音频预览")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        work_dir = output_path.parent / f".preview-{uuid.uuid4().hex}"
        work_dir.mkdir(parents=True, exist_ok=False)
        try:
            normalized = await self._produce_normalized_mix(bundle, work_dir)
            args = ["ffmpeg", "-y", "-i", str(normalized)]
            if window_s is not None:
                start, end = float(window_s[0]), float(window_s[1])
                if start > 0:
                    args.extend(["-ss", f"{start:.3f}"])
                if end > start:
                    args.extend(["-to", f"{end:.3f}"])
            args.extend(["-c:a", "aac", str(output_path)])
            await self._run(args)
            return {"path": str(output_path), "warnings": bundle.warnings, "manifest": bundle.manifest()}
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    @staticmethod
    def _escape_filter_path(path: Path | str) -> str:
        """subtitles 滤镜的文件名转义（引号内转义 ' 与 :）。"""

        text = str(path).replace("\\", "/")
        return text.replace(":", "\\:").replace("'", "\\'")

    @staticmethod
    def _shot_has_native_audio(shot: dict) -> bool:
        if shot.get("native_audio") is True:
            return True
        profile = shot.get("continuity_profile") or {}
        return isinstance(profile, dict) and str(profile.get("audio_source") or "").lower() == "native"

    def _zoom_filter(
        self,
        shot_type: str,
        width: int,
        height: int,
        frames: int,
        camera_movement: str = "静止",
    ) -> str:
        """把镜头运动映射到 zoompan 的真实运动策略。

        生成视频的镜头运动由视频 Prompt 驱动；静态图片回填时由 FFmpeg
        zoompan 执行同一语义，避免 camera_movement 只存在于数据里。
        """

        movement = str(camera_movement or "静止")
        if movement in {"推", "缓慢推进"}:
            zoom = "min(zoom+0.0015,1.28)" if movement == "推" else "min(zoom+0.0007,1.18)"
            x_expr, y_expr = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
        elif movement == "拉":
            zoom = "max(zoom-0.0012,0.88)"
            x_expr, y_expr = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
        elif movement == "摇":
            zoom = "1.08"
            x_expr, y_expr = f"(iw-iw/zoom)*on/{max(1, frames)}", "ih/2-(ih/zoom/2)"
        elif movement == "移":
            zoom = "1.08"
            x_expr, y_expr = f"(iw-iw/zoom)*(0.25+0.5*on/{max(1, frames)})", "ih/2-(ih/zoom/2)"
        elif movement == "跟":
            zoom = "1.12"
            x_expr, y_expr = f"(iw-iw/zoom)*(0.35+0.3*on/{max(1, frames)})", "ih/2-(ih/zoom/2)"
        elif movement == "升降":
            zoom = "1.08"
            x_expr, y_expr = "iw/2-(iw/zoom/2)", f"(ih-ih/zoom)*on/{max(1, frames)}"
        elif movement == "环绕":
            zoom = "1.12"
            x_expr = f"iw/2-(iw/zoom/2)+sin(2*PI*on/{max(1, frames)})*(iw/zoom/6)"
            y_expr = f"ih/2-(ih/zoom/2)+cos(2*PI*on/{max(1, frames)})*(ih/zoom/8)"
        else:
            if shot_type == "wide":
                zoom = "min(zoom+0.001,1.25)"
            elif shot_type in {"close-up", "extreme_close"}:
                zoom = "1.35"
            else:
                zoom = "1.10"
            x_expr, y_expr = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
        return (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},"
            f"zoompan=z='{zoom}':x='{x_expr}':y='{y_expr}':"
            f"d={frames}:s={width}x{height}:fps={self.fps}"
        )

    def _clip_filter(self, base_filter: str, shot: dict, duration: float) -> str:
        # 边界转场统一交给 _concat_clips 的 PostProductionPlan；
        # 这里只保留单镜头后处理，避免白闪/叠化被重复应用。
        filters = [base_filter, self._post_filter(shot)]
        return ",".join(part for part in filters if part)

    def _post_filter(self, shot: dict) -> str:
        profile = shot.get("post_profile") or {}
        scene_group = str(profile.get("scene_group_id") or shot.get("scene_group_id") or "project_scene")
        digest = int(hashlib.sha1(scene_group.encode("utf-8", errors="ignore")).hexdigest()[:8], 16)
        saturation = self._profile_number(profile.get("saturation"), 0.98 + (digest % 7) * 0.01, 0.94, 1.08)
        contrast = 1.00 + ((digest // 7) % 5) * 0.005
        brightness = (((digest // 35) % 5) - 2) * 0.002
        sharpness = self._profile_number(profile.get("sharpness"), 0.34 + ((digest // 175) % 7) * 0.025, 0.25, 0.55)
        return (
            f"eq=saturation={saturation:.3f}:contrast={contrast:.3f}:brightness={brightness:.3f},"
            f"unsharp=5:5:{sharpness:.3f}:3:3:0.0"
        )

    def _cross_scene_transition_duration(self, profile: dict, duration: float) -> float:
        requested = self._profile_number(profile.get("cross_scene_flash_seconds"), 0.35, 0.3, 0.5)
        if duration <= 0:
            return 0.0
        if duration < 0.6:
            return max(0.1, duration / 2)
        return min(requested, duration / 2)

    def _profile_number(self, value, fallback: float, low: float, high: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = fallback
        return min(high, max(low, number))

    def _get_resolution(self, resolution: str, fmt: str) -> tuple[int, int]:
        sizes = {"720p": 720, "1080p": 1080, "2k": 1440, "4k": 2160}
        height = sizes.get(resolution, 1080)
        ratio = {"9:16": 9 / 16, "16:9": 16 / 9, "1:1": 1}.get(fmt, 9 / 16)
        width = int(height * ratio)
        return width - width % 2, height - height % 2

    async def _run(self, args: list[str]) -> None:
        await self._run_capture(args)

    async def _run_capture(self, args: list[str]) -> tuple[bytes, bytes]:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=max(30, int(settings.FFMPEG_TIMEOUT_SECONDS)),
            )
        except asyncio.CancelledError:
            # Cancelling the coroutine does not automatically terminate an
            # asyncio subprocess. Reap it before propagating cancellation so
            # project deletion cannot leave an encoder writing in the background.
            if proc.returncode is None:
                proc.kill()
            await proc.communicate()
            raise
        except TimeoutError as exc:
            proc.kill()
            await proc.communicate()
            raise TimeoutError("FFmpeg 执行超时") from exc
        if proc.returncode != 0:
            message = stderr.decode("utf-8", errors="ignore")[-2000:]
            raise RuntimeError(f"FFmpeg 执行失败: {message}")
        return stdout or b"", stderr or b""

    def _media_path(self, value: str | None, minimum_size: int = 1) -> Path | None:
        if not value:
            return None
        return existing_file(
            value,
            minimum_size=minimum_size,
            allowed_roots=(settings.OUTPUT_DIR, settings.ASSETS_DIR, settings.DATA_DIR),
        )
