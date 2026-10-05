"""音频混音规划器：从轨道参数生成可复现的 FFmpeg filter_complex 与轨道 manifest。

纯函数、零 IO：渲染管线（FFmpegService）与预览管线共用同一个规划结果，
保证「预览听到的 = 成片里听到的」。所有数值先收敛到安全区间再进入滤镜串，
因此相同输入永远产出相同的 filter 与 manifest（字节级可复现）。

主音轨（concat 产物：native 音 + 未拆出的 TTS 对白）由调用方预先构造成
``main_label`` 指向的滤镜链；规划器只负责把各轨道素材与 Ducking 拓扑接上：

    [main] ──(并入拆出的对白轨)──> [voice] ──asplit──> 主输出 + sidechain 源
    各轨道素材（裁剪/循环/音量/声像/淡入淡出/延迟）
    环境床（粉噪，与旧渲染管线的 _add_continuous_ambient_bed 参数一致）
    配置了 Ducking 的轨道经 sidechaincompress 被对白压低；
    最终 amix(normalize=0) 输出 [aout]，响度归一化与削波防护由调用方执行。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

SAMPLE_RATE = 44100

# 与 FFmpegService._add_continuous_ambient_bed 完全一致的环境床参数；混音接管
# 后由规划器在 filter 内重建，保证「配置了轨道」与「未配置轨道」两条渲染路径
# 的听感连续。
AMBIENT_COLOR = "pink"
AMBIENT_AMPLITUDE = 0.008
AMBIENT_VOLUME = 0.08

# Ducking 近似：把「期望压低量」换算成压缩比。假设对白触发电平比阈值高约
# 14 dB，则压低量 ≈ 14 * (1 - 1/ratio)。范围外收敛到 [2, 20]。
_DUCK_TRIGGER_HEADROOM_DB = 14.0


@dataclass
class MixTrackInput:
    """一条参与混音的轨道（由调用方完成媒体路径校验与时长探测）。"""

    id: str
    kind: str  # dialogue / music / ambient / sfx
    media_path: str
    # 素材完整时长（裁剪前，毫秒）。dialogue 轨为 TTS 音频时长。
    media_duration_ms: int
    # dialogue 轨绑定的镜头：渲染时该镜头区间的主音轨置静音，配音走本轨。
    shot_id: str = ""
    # 时间线起点（dialogue 轨 = 镜头区间起点 + delay_ms）。
    start_ms: int = 0
    volume: float = 1.0
    pan: float = 0.0
    fade_in_ms: int = 0
    fade_out_ms: int = 0
    delay_ms: int = 0
    trim_start_ms: int = 0
    trim_end_ms: int = 0
    loop: bool = False
    muted: bool = False
    duck_amount_db: float = 0.0
    duck_attack_ms: int = 120
    duck_release_ms: int = 480
    # dialogue 轨专用：对白不得越过镜头边界（毫秒）。
    clip_limit_ms: int = 0
    # 规划阶段发现的非致命问题（素材过短等），由调用方汇总为用户警告。
    warnings: list[str] = field(default_factory=list)


@dataclass
class AudioMixPlan:
    filter_complex: str
    output_label: str
    inputs: list[str]  # 除主视频外的输入文件，顺序对应 filter 中的 1..n
    track_manifest: list[dict]
    ducked_track_ids: list[str]
    warnings: list[str]


def _num(value: float, digits: int = 3) -> str:
    """格式化滤镜数值：定点小数，杜绝科学计数法进入 filter_complex。"""

    number = float(value)
    if not math.isfinite(number):
        number = 0.0
    text = f"{number:.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".") or "0"
    return text or "0"


def _clamp(value: float, low: float, high: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        return low
    return min(high, max(low, number))


def _duck_ratio(amount_db: float) -> int:
    amount = _clamp(abs(amount_db), 1.0, 13.0)
    if amount >= 12.9:
        return 20
    ratio = 1.0 / (1.0 - amount / _DUCK_TRIGGER_HEADROOM_DB)
    return int(_clamp(round(ratio), 2, 20))


def _track_chain(track: MixTrackInput, input_index: int, label: str, total_duration_s: float) -> str:
    """构造单轨滤镜链：`[i:a:0]滤镜…[label]`（输入标签后不能跟逗号）。"""

    parts: list[str] = [f"aresample={SAMPLE_RATE}", "aformat=sample_fmts=fltp:channel_layouts=stereo"]

    trim_start = max(0, int(track.trim_start_ms))
    trim_end = max(0, int(track.trim_end_ms))
    media_ms = max(0, int(track.media_duration_ms))
    trimmed_ms = max(0, media_ms - trim_start - trim_end)
    if trim_start > 0 or trim_end > 0:
        if trim_start > 0:
            parts.append(f"atrim=start={_num(trim_start / 1000, 3)}")
        if trim_end > 0 and trimmed_ms > 0:
            parts.append(f"atrim=end={_num((media_ms - trim_end) / 1000, 3)}")
        parts.append("asetpts=PTS-STARTPTS")

    if track.loop and trimmed_ms > 0:
        size = max(2, int(trimmed_ms / 1000 * SAMPLE_RATE) + 1)
        parts.append(f"aloop=loop=-1:size={size}")
        parts.append("asetpts=PTS-STARTPTS")

    # 播放长度：循环轨铺满全片，其余为裁剪后的素材长度。
    total_ms = max(0, int(round(total_duration_s * 1000)))
    if track.loop:
        play_ms = total_ms
    else:
        play_ms = trimmed_ms

    volume = _clamp(track.volume, 0.0, 4.0)
    parts.append(f"volume={_num(volume, 4)}")

    pan = _clamp(track.pan, -1.0, 1.0)
    if abs(pan) > 0.001:
        left = _num(1.0 - max(pan, 0.0), 4)
        right = _num(1.0 - max(-pan, 0.0), 4)
        parts.append(f"pan=stereo|c0={left}*c0|c1={right}*c1")

    fade_in = int(_clamp(track.fade_in_ms, 0, 30_000))
    fade_out = int(_clamp(track.fade_out_ms, 0, 30_000))
    if play_ms > 0:
        if fade_in > 0:
            fade_in = min(fade_in, play_ms)
            parts.append(f"afade=t=in:st=0:d={_num(fade_in / 1000, 3)}")
        if fade_out > 0:
            fade_out = min(fade_out, play_ms)
            start = max(0, play_ms - fade_out)
            parts.append(f"afade=t=out:st={_num(start / 1000, 3)}:d={_num(fade_out / 1000, 3)}")

    delay = max(0, int(track.start_ms) + max(0, int(track.delay_ms)))
    if delay > 0:
        parts.append(f"adelay={delay}:all=1")

    # 对白轨不得越过镜头边界。
    if track.clip_limit_ms > 0:
        limit_ms = max(0, track.clip_limit_ms - max(0, int(track.delay_ms)))
        if limit_ms > 0:
            parts.append(f"atrim=duration={_num(limit_ms / 1000, 3)}")
            parts.append("asetpts=PTS-STARTPTS")

    return f"[{input_index}:a:0]" + ",".join(part for part in parts if part) + f"[{label}]"


def _manifest_entry(track: MixTrackInput) -> dict:
    """轨道摘要：只含影响混音结果的字段，用于可复现比对。"""

    return {
        "id": str(track.id),
        "kind": str(track.kind),
        "source": str(track.media_path).rsplit("/", 1)[-1].rsplit("\\", 1)[-1],
        "source_ref": hashlib.sha1(str(track.media_path).encode("utf-8")).hexdigest()[:12],
        "media_duration_ms": int(track.media_duration_ms),
        "start_ms": int(track.start_ms),
        "delay_ms": int(track.delay_ms),
        "volume": round(_clamp(track.volume, 0.0, 4.0), 4),
        "pan": round(_clamp(track.pan, -1.0, 1.0), 4),
        "fade_in_ms": int(_clamp(track.fade_in_ms, 0, 30_000)),
        "fade_out_ms": int(_clamp(track.fade_out_ms, 0, 30_000)),
        "trim_start_ms": int(track.trim_start_ms),
        "trim_end_ms": int(track.trim_end_ms),
        "loop": bool(track.loop),
        "muted": bool(track.muted),
        "duck_amount_db": round(_clamp(track.duck_amount_db, -48.0, 0.0), 2),
        "duck_attack_ms": int(_clamp(track.duck_attack_ms, 1, 5_000)),
        "duck_release_ms": int(_clamp(track.duck_release_ms, 1, 10_000)),
        "clip_limit_ms": int(track.clip_limit_ms),
    }


def _ambient_statement() -> str:
    return (
        f"anoisesrc=color={AMBIENT_COLOR}:amplitude={_num(AMBIENT_AMPLITUDE, 4)}:sample_rate={SAMPLE_RATE},"
        f"volume={_num(AMBIENT_VOLUME, 4)}[amb]"
    )


def plan_audio_mix(
    tracks: Sequence[MixTrackInput],
    total_duration_s: float,
    *,
    main_label: str = "[main0]",
    first_track_input_index: int = 1,
    include_ambient_bed: bool = True,
    allow_empty: bool = False,
) -> AudioMixPlan | None:
    """规划混音。

    - 无有效轨道且 ``allow_empty`` 为 False（渲染路径）：返回 None，调用方沿用
      旧渲染管线（concat 音轨 + 独立环境床步骤），行为与未引入工作台时一致。
    - ``allow_empty`` 为 True（预览路径）：即使没有轨道也产出仅含主音轨的
      filter，让预览能还原「未配置轨道时的成片听感」。
    轨道素材的输入序号从 ``first_track_input_index`` 起分配，主音轨链由调用方
    预构建并占用更靠前的输入序号。
    """

    total = max(0.0, float(total_duration_s or 0.0))
    active: list[MixTrackInput] = []
    warnings: list[str] = []
    for track in tracks:
        if track.muted:
            continue
        media_ms = max(0, int(track.media_duration_ms))
        trim_start = max(0, int(track.trim_start_ms))
        trim_end = max(0, int(track.trim_end_ms))
        trimmed = media_ms - trim_start - trim_end
        if not track.media_path or media_ms <= 0:
            track.warnings.append(f"轨道 {track.id} 缺少有效音频素材，已跳过")
            warnings.extend(track.warnings)
            continue
        if not track.loop and trimmed <= 0:
            track.warnings.append(f"轨道 {track.id} 裁剪后已无可用片段，已跳过")
            warnings.extend(track.warnings)
            continue
        active.append(track)

    if not active and not allow_empty:
        return None

    statements: list[str] = []
    inputs: list[str] = []
    track_labels: list[str] = []
    manifest: list[dict] = []

    # 主音轨已由调用方构建为 main_label；对白轨（从镜头片段拆出的 TTS）并入。
    dialogue_tracks = [track for track in active if track.kind == "dialogue"]
    other_tracks = [track for track in active if track.kind != "dialogue"]

    dialogue_labels: list[str] = []
    for index, track in enumerate(dialogue_tracks):
        input_index = first_track_input_index + index
        label = f"dlg{index}"
        inputs.append(track.media_path)
        statements.append(_track_chain(track, input_index, label, total))
        dialogue_labels.append(label)
        manifest.append(_manifest_entry(track))

    if dialogue_labels:
        # amix duration=first 以第一个输入为准：主音轨必须排在对白轨之前，
        # 否则整条混音会被最短的对白轨提前截断。
        statements.append(
            f"{main_label}"
            + "".join(f"[{label}]" for label in dialogue_labels)
            + f"amix=inputs={len(dialogue_labels) + 1}:duration=first:normalize=0[voicepre]"
        )
    else:
        statements.append(f"{main_label}anull[voicepre]")

    duck_tracks = [track for track in other_tracks if _clamp(track.duck_amount_db, -48.0, 0.0) < -0.01]
    duck_count = len(duck_tracks)

    # voice 一分为 (1 + duck_count) 份：主输出 + 每条 duck 轨一个 sidechain 源。
    if duck_count > 0:
        split_outputs = ["voice_main"] + [f"sc{index}" for index in range(duck_count)]
        statements.append(f"[voicepre]asplit={duck_count + 1}" + "".join(f"[{name}]" for name in split_outputs))
    else:
        statements.append("[voicepre]anull[voice_main]")

    duck_source_map = {track.id: f"sc{index}" for index, track in enumerate(duck_tracks)}
    ducked_ids: list[str] = []

    other_index = 0
    for track in other_tracks:
        input_index = first_track_input_index + len(dialogue_tracks) + other_index
        label = f"t{other_index}"
        inputs.append(track.media_path)
        statements.append(_track_chain(track, input_index, label, total))
        if track.id in duck_source_map:
            amount = _clamp(track.duck_amount_db, -48.0, 0.0)
            attack = int(_clamp(track.duck_attack_ms, 1, 5_000))
            release = int(_clamp(track.duck_release_ms, 1, 10_000))
            ratio = _duck_ratio(amount)
            ducked_label = f"td{other_index}"
            statements.append(
                f"[{label}][{duck_source_map[track.id]}]"
                # makeup 默认为 1（不补偿）：压低量完全由阈值与压缩比决定，保持可复现。
                f"sidechaincompress=threshold=0.02:ratio={ratio}:attack={attack}:release={release}[{ducked_label}]"
            )
            track_labels.append(f"[{ducked_label}]")
            ducked_ids.append(str(track.id))
        else:
            track_labels.append(f"[{label}]")
        manifest.append(_manifest_entry(track))
        other_index += 1

    mix_inputs = ["[voice_main]", *track_labels]
    if include_ambient_bed:
        statements.append(_ambient_statement())
        mix_inputs.append("[amb]")
    tail = f"amix=inputs={len(mix_inputs)}:duration=first:normalize=0"
    if total > 0:
        tail += f",atrim=duration={_num(total, 3)}"
    statements.append("".join(mix_inputs) + tail + "[aout]")

    filter_complex = ";".join(statements)
    return AudioMixPlan(
        filter_complex=filter_complex,
        output_label="[aout]",
        inputs=inputs,
        track_manifest=manifest,
        ducked_track_ids=ducked_ids,
        warnings=warnings,
    )


def manifest_digest(manifest: list[dict]) -> str:
    """轨道 manifest 的稳定摘要：相同配置 → 相同摘要。"""

    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


__all__ = [
    "AudioMixPlan",
    "MixTrackInput",
    "SAMPLE_RATE",
    "manifest_digest",
    "plan_audio_mix",
]
